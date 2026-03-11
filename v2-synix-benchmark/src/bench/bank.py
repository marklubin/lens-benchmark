"""Artifact bank builder — builds checkpoint-scoped Synix projects.

For each scope × checkpoint, creates a Synix project with:
  - Source episodes (prefix-valid only)
  - Chunk transform (1:N text splitting, no LLM)
  - Optional FoldSynthesis for core memory
  - Optional GroupSynthesis + ReduceSynthesis for summaries
  - SearchSurface + SynixSearch for hybrid retrieval

Checkpoint isolation is enforced at the source level: each project
only ingests episodes valid at that checkpoint. This satisfies D014
and the benchmark spec's non-negotiable rule (rule 9 in CLAUDE.md).

Label convention: Synix's ParseTransform requires .txt file extensions and
prepends "t-text-" to the stem when creating artifact labels. So an episode
added as "signal_001.txt" becomes artifact label "t-text-signal_001".
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path

import synix
from synix.core.models import Pipeline, SearchSurface, Source, SynixSearch
from synix.ext.chunk import Chunk

from bench.broker import ModalBroker
from bench.dataset import ScopeData
from bench.schemas import BankManifest, BankStatus, BuildCost, EventType
from bench.state import EventWriter, StateStore

logger = logging.getLogger(__name__)

# Synix's text adapter prepends this to all .txt source labels
SYNIX_LABEL_PREFIX = "t-text-"


def _make_pipeline(
    scope_id: str,
    checkpoint: int,
    families: list[str],
    llm_config: dict,
    embedding_config: dict,
    *,
    chunk_size: int = 1000,
    chunk_overlap: int = 100,
) -> Pipeline:
    """Construct the Synix pipeline DAG for a single checkpoint bank.

    Returns the pipeline with all layers added. The caller is responsible
    for setting it on the project and ingesting source files.
    """
    pipeline = Pipeline(
        f"lens-{scope_id}-cp{checkpoint:02d}",
        llm_config=llm_config,
    )

    episodes = Source("episodes")

    chunks = Chunk(
        "chunks",
        depends_on=[episodes],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        artifact_type="chunk",
    )

    # Always add episodes and chunks
    pipeline.add(episodes, chunks)

    # Search surface over episodes + chunks
    search_surface = SearchSurface(
        "search",
        sources=[episodes, chunks],
        modes=["keyword", "semantic"],
        embedding_config=embedding_config,
    )
    search_output = SynixSearch("search_output", surface=search_surface)
    pipeline.add(search_surface, search_output)

    # Optional: core memory via FoldSynthesis
    if "core_memory" in families:
        from bench.families.core import add_core_memory

        add_core_memory(pipeline, depends_on=episodes)

    # Optional: summary via GroupSynthesis + ReduceSynthesis
    if "summary" in families:
        from bench.families.summary import add_summary

        add_summary(pipeline, depends_on=episodes)

    return pipeline


class BankBuilder:
    """Builds checkpoint-scoped artifact banks using the Synix SDK.

    Each checkpoint gets its own Synix project directory. The builder
    handles resume (skips already-released banks) and emits events
    for the audit trail.
    """

    def __init__(
        self,
        store: StateStore,
        broker: ModalBroker,
        work_dir: Path,
        *,
        chunk_size: int = 1000,
        chunk_overlap: int = 100,
    ) -> None:
        self._store = store
        self._broker = broker
        self._work_dir = Path(work_dir)
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    def build_scope_banks(
        self,
        study_id: str,
        scope: ScopeData,
        families: list[str],
        event_writer: EventWriter,
    ) -> list[BankManifest]:
        """Build one bank per checkpoint for a scope.

        Args:
            study_id: The study this build belongs to.
            scope: Loaded scope data with episodes and questions.
            families: Artifact families to include (e.g. ["chunks", "core_memory", "summary"]).
            event_writer: For emitting build events.

        Returns:
            List of BankManifests, one per checkpoint, in checkpoint order.
        """
        manifests: list[BankManifest] = []

        for checkpoint in scope.checkpoints:
            manifest = self._build_checkpoint_bank(
                study_id=study_id,
                scope=scope,
                checkpoint=checkpoint,
                families=families,
                event_writer=event_writer,
            )
            manifests.append(manifest)

        return manifests

    def _build_checkpoint_bank(
        self,
        study_id: str,
        scope: ScopeData,
        checkpoint: int,
        families: list[str],
        event_writer: EventWriter,
    ) -> BankManifest:
        """Build a single checkpoint bank. Resumes if already released."""
        bank_id = f"bank-{scope.scope_id}-cp{checkpoint:02d}-{study_id[:8]}"

        # Resume check: if this bank is already released, skip
        existing = self._store.get_bank(bank_id)
        if existing is not None and existing.status == BankStatus.released:
            logger.info("Bank %s already released, skipping", bank_id)
            return existing

        # Get prefix-valid episodes
        episodes = scope.episodes_up_to(checkpoint)
        episode_ids = [ep.episode_id for ep in episodes]

        logger.info(
            "Building bank %s: %d episodes, families=%s",
            bank_id,
            len(episodes),
            families,
        )

        event_writer.emit(
            EventType.bank_build_started,
            scope_id=scope.scope_id,
            bank_manifest_id=bank_id,
            payload={"checkpoint": checkpoint, "episode_count": len(episodes), "families": families},
        )

        # Create or reopen Synix project
        project_dir = self._work_dir / scope.scope_id / f"cp{checkpoint:02d}"

        llm_config = {
            "provider": "openai-compatible",
            "model": "Qwen/Qwen3.5-35B-A3B",
            "base_url": self._broker._llm_base_url,
            "api_key": self._broker._llm_api_key,
        }

        embedding_config = {
            "provider": "openai",
            "model": "Xenova/gte-modernbert-base",
            "base_url": self._broker._embed_base_url,
            "api_key": "unused",
        }

        pipeline = _make_pipeline(
            scope.scope_id,
            checkpoint,
            families,
            llm_config,
            embedding_config,
            chunk_size=self._chunk_size,
            chunk_overlap=self._chunk_overlap,
        )

        # Init or open project
        if (project_dir / ".synix").exists():
            project = synix.open_project(project_dir)
        else:
            project_dir.mkdir(parents=True, exist_ok=True)
            project = synix.init(project_dir, pipeline=pipeline)

        project.set_pipeline(pipeline)

        # Ingest episodes into source.
        # Labels must end in .txt for Synix's ParseTransform to pick them up.
        # Use the original filename if available (already .txt), otherwise append .txt.
        source = project.source("episodes")
        source.clear()
        for ep in episodes:
            # If the scope_dir has the original file, use add() for it (preserves filename)
            original_file = scope.scope_dir / "generated" / "episodes" / ep.filename
            if original_file.exists():
                source.add(original_file)
            else:
                source.add_text(ep.content, label=ep.filename)

        # Build
        try:
            result = project.build()
        except Exception as exc:
            logger.error("Bank build failed for %s: %s", bank_id, exc)
            manifest = BankManifest(
                bank_manifest_id=bank_id,
                study_id=study_id,
                scope_id=scope.scope_id,
                checkpoint_id=f"cp{checkpoint:02d}",
                max_episode_ordinal=checkpoint,
                source_episode_ids=episode_ids,
                artifact_families={f: None for f in families},
                dataset_hash=scope.dataset_hash,
                status=BankStatus.failed,
            )
            self._store.save_bank(manifest)
            event_writer.emit(
                EventType.bank_build_failed,
                scope_id=scope.scope_id,
                bank_manifest_id=bank_id,
                error=str(exc),
            )
            raise

        # Release
        project.release_to("bank")

        # Build manifest
        manifest = BankManifest(
            bank_manifest_id=bank_id,
            study_id=study_id,
            scope_id=scope.scope_id,
            checkpoint_id=f"cp{checkpoint:02d}",
            max_episode_ordinal=checkpoint,
            source_episode_ids=episode_ids,
            artifact_families={f: result.snapshot_oid for f in families},
            dataset_hash=scope.dataset_hash,
            synix_build_ref=result.snapshot_oid,
            synix_release_ref="bank",
            synix_build_graph_hash=result.manifest_oid,
            status=BankStatus.released,
            built_at=datetime.now(timezone.utc),
            build_cost=BuildCost(
                wall_time_s=result.total_time,
            ),
        )
        self._store.save_bank(manifest)

        event_writer.emit(
            EventType.bank_build_completed,
            scope_id=scope.scope_id,
            bank_manifest_id=bank_id,
            payload={
                "built": result.built,
                "cached": result.cached,
                "skipped": result.skipped,
                "total_time": result.total_time,
                "snapshot_oid": result.snapshot_oid,
            },
        )

        logger.info(
            "Bank %s released: %d built, %d cached, %.1fs",
            bank_id,
            result.built,
            result.cached,
            result.total_time,
        )

        return manifest

    def open_release(self, manifest: BankManifest) -> synix.Release:
        """Open a Synix release from a bank manifest for runtime access."""
        project_dir = self._work_dir / manifest.scope_id / manifest.checkpoint_id
        project = synix.open_project(project_dir)
        return project.release(manifest.synix_release_ref or "bank")

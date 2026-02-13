from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

from lens.adapters.base import MemoryAdapter
from lens.adapters.registry import get_adapter
from lens.core.config import BudgetConfig, RunConfig
from lens.core.errors import LatencyExceededError, atomic_write
from lens.core.logging import LensLogger, Verbosity
from lens.core.models import (
    CheckpointResult,
    Episode,
    PersonaResult,
    RunResult,
)
from lens.runner.anticheat import EpisodeVault
from lens.runner.budget import BudgetedLLM, BudgetTracker
from lens.runner.validator import OutputValidator


class RunEngine:
    """Orchestrates a benchmark run: episodes -> checkpoints -> artifacts."""

    def __init__(
        self,
        config: RunConfig,
        logger: LensLogger | None = None,
    ) -> None:
        self.config = config
        self.logger = logger or LensLogger(Verbosity.NORMAL)
        self.vault = EpisodeVault()
        self.tracker = BudgetTracker()
        self.budgeted_llm = BudgetedLLM(config.budget, self.tracker)
        self.validator = OutputValidator(
            self.vault,
            max_evidence_episodes=config.budget.max_evidence_episodes,
        )
        self.run_id = uuid.uuid4().hex[:12]

    def run(self, personas: dict[str, list[Episode]]) -> RunResult:
        """Execute the full benchmark run across all personas."""
        self.logger.info(
            f"Starting run [bold]{self.run_id}[/bold] "
            f"adapter={self.config.adapter} budget={self.config.budget.preset}"
        )

        adapter_cls = get_adapter(self.config.adapter)
        adapter = adapter_cls()
        adapter.set_budgeted_llm(self.budgeted_llm)

        persona_results: list[PersonaResult] = []

        for persona_id, episodes in personas.items():
            self.logger.info(f"Persona [bold]{persona_id}[/bold]: {len(episodes)} episodes")
            result = self._run_persona(adapter, persona_id, episodes)
            persona_results.append(result)

        run_result = RunResult(
            run_id=self.run_id,
            adapter=self.config.adapter,
            dataset_version="",  # Set by caller
            budget_preset=self.config.budget.preset,
            personas=persona_results,
        )

        self.logger.success(f"Run {self.run_id} complete")
        return run_result

    def _run_persona(
        self,
        adapter: MemoryAdapter,
        persona_id: str,
        episodes: list[Episode],
    ) -> PersonaResult:
        """Run the benchmark for a single persona."""
        # Sort episodes by timestamp
        episodes = sorted(episodes, key=lambda e: e.timestamp)

        # Reset adapter state
        adapter.reset(persona_id)
        checkpoints_done: list[CheckpointResult] = []

        for idx, episode in enumerate(episodes, start=1):
            # Store in vault for evidence validation
            self.vault.store(episode.episode_id, episode.text)

            # Ingest with timing check
            self.budgeted_llm.set_context("ingest", "ingest")
            self.logger.start_step("ingest")

            t0 = time.monotonic()
            adapter.ingest(
                episode_id=episode.episode_id,
                persona_id=persona_id,
                timestamp=episode.timestamp.isoformat(),
                text=episode.text,
                meta=episode.meta,
            )
            elapsed_ms = (time.monotonic() - t0) * 1000

            self.logger.end_step(
                message=f"episode {episode.episode_id}",
                persona_id=persona_id,
                elapsed=elapsed_ms,
            )

            if elapsed_ms > self.config.budget.ingest.max_latency_ms:
                self.logger.warn(
                    f"Ingest latency {elapsed_ms:.0f}ms exceeds "
                    f"{self.config.budget.ingest.max_latency_ms}ms cap"
                )

            # Check if this is a checkpoint
            if idx in self.config.checkpoints:
                checkpoint_result = self._run_checkpoint(
                    adapter, persona_id, idx
                )
                checkpoints_done.append(checkpoint_result)

        # Also run checkpoint at final episode if not already done
        if len(episodes) not in self.config.checkpoints:
            checkpoint_result = self._run_checkpoint(
                adapter, persona_id, len(episodes)
            )
            checkpoints_done.append(checkpoint_result)

        return PersonaResult(persona_id=persona_id, checkpoints=checkpoints_done)

    def _run_checkpoint(
        self,
        adapter: MemoryAdapter,
        persona_id: str,
        checkpoint: int,
    ) -> CheckpointResult:
        """Execute refresh + core + search at a checkpoint."""
        self.logger.info(f"  Checkpoint {checkpoint} for {persona_id}")
        errors: list[str] = []
        timing: dict[str, float] = {}

        # Phase A: refresh (offline, metered)
        self.budgeted_llm.set_context("refresh", "refresh")
        t0 = time.monotonic()
        adapter.refresh(persona_id, checkpoint)
        timing["refresh_ms"] = (time.monotonic() - t0) * 1000

        # Phase B: core (online, budgeted)
        self.budgeted_llm.set_context("core", "core")
        t0 = time.monotonic()
        insights = adapter.core(persona_id, self.config.core_k, checkpoint)
        core_ms = (time.monotonic() - t0) * 1000
        timing["core_ms"] = core_ms

        if core_ms > self.config.budget.core.max_latency_ms:
            errors.append(
                f"core latency {core_ms:.0f}ms exceeds "
                f"{self.config.budget.core.max_latency_ms}ms"
            )

        # Validate insights
        validation_errors = self.validator.validate_insights(insights)
        errors.extend(validation_errors)

        # Phase B: search (online, budgeted)
        search_results: dict[str, list] = {}
        for query in self.config.search_queries:
            self.budgeted_llm.set_context("search", "search")
            t0 = time.monotonic()
            hits = adapter.search(persona_id, query, self.config.search_k, checkpoint)
            search_ms = (time.monotonic() - t0) * 1000
            timing[f"search_{query}_ms"] = search_ms

            if search_ms > self.config.budget.search.max_latency_ms:
                errors.append(
                    f"search latency {search_ms:.0f}ms exceeds "
                    f"{self.config.budget.search.max_latency_ms}ms"
                )

            hit_errors = self.validator.validate_hits(hits)
            errors.extend(hit_errors)
            search_results[query] = hits

        return CheckpointResult(
            persona_id=persona_id,
            checkpoint=checkpoint,
            insights=insights,
            search_results=search_results,
            validation_errors=errors,
            budget_used=self.tracker.summary(),
            timing=timing,
        )

    def save_artifacts(self, result: RunResult, output_dir: str | Path) -> Path:
        """Write run artifacts to disk."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        # Run manifest
        manifest = {
            "run_id": result.run_id,
            "adapter": result.adapter,
            "dataset_version": result.dataset_version,
            "budget_preset": result.budget_preset,
        }
        with atomic_write(out / "run_manifest.json") as tmp:
            tmp.write_text(json.dumps(manifest, indent=2))

        # Config
        with atomic_write(out / "config.json") as tmp:
            tmp.write_text(json.dumps(self.config.to_dict(), indent=2))

        # Per-persona checkpoints
        for persona in result.personas:
            persona_dir = out / "personas" / persona.persona_id
            for cp in persona.checkpoints:
                cp_dir = persona_dir / f"checkpoint_{cp.checkpoint}"
                cp_dir.mkdir(parents=True, exist_ok=True)

                with atomic_write(cp_dir / "insights.json") as tmp:
                    tmp.write_text(json.dumps(
                        [i.to_dict() for i in cp.insights], indent=2
                    ))

                for query, hits in cp.search_results.items():
                    safe_query = query.replace(" ", "_")[:50]
                    with atomic_write(cp_dir / f"search_{safe_query}.json") as tmp:
                        tmp.write_text(json.dumps(
                            [h.to_dict() for h in hits], indent=2
                        ))

                if cp.validation_errors:
                    with atomic_write(cp_dir / "validation.json") as tmp:
                        tmp.write_text(json.dumps(cp.validation_errors, indent=2))

        # Budget report
        with atomic_write(out / "budget_report.json") as tmp:
            tmp.write_text(json.dumps(self.tracker.summary(), indent=2))

        self.logger.success(f"Artifacts saved to {out}")
        return out

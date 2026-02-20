from __future__ import annotations

import concurrent.futures
import hashlib
import json
import logging
import os
import time
import uuid
from pathlib import Path

from lens.adapters.base import MemoryAdapter
from lens.adapters.registry import get_adapter
from lens.agent.budget_enforcer import QuestionBudget
from lens.agent.harness import AgentHarness
from lens.agent.llm_client import BaseLLMClient, MockLLMClient
from lens.core.config import RunConfig
from lens.core.errors import ConfigError, atomic_write
from lens.core.logging import LensLogger, Verbosity
from lens.core.models import (
    CheckpointResult,
    Episode,
    ScopeResult,
    Question,
    QuestionResult,
    RunResult,
)
from lens.metering.manager import MeteringManager
from lens.runner.anticheat import EpisodeVault

log = logging.getLogger(__name__)


def _cache_key(adapter_name: str, scope_id: str, episode_ids: list[str]) -> str:
    """Deterministic hash for adapter + scope + episode set."""
    h = hashlib.sha256()
    h.update(adapter_name.encode())
    h.update(scope_id.encode())
    for eid in sorted(episode_ids):
        h.update(eid.encode())
    return h.hexdigest()[:12]


class RunEngine:
    """Orchestrates a benchmark run: episodes -> checkpoints -> agent questions -> artifacts."""

    def __init__(
        self,
        config: RunConfig,
        logger: LensLogger | None = None,
        llm_client: BaseLLMClient | None = None,
    ) -> None:
        self.config = config
        self.logger = logger or LensLogger(Verbosity.NORMAL)
        self.vault = EpisodeVault()
        self.run_id = uuid.uuid4().hex[:12]
        self.llm_client = llm_client or MockLLMClient()
        self._metering: MeteringManager | None = None

    def run(
        self,
        scopes: dict[str, list[Episode]],
        questions: list[Question] | None = None,
    ) -> RunResult:
        """Execute the full benchmark run across all scopes."""
        self.logger.info(
            f"Starting run [bold]{self.run_id}[/bold] "
            f"adapter={self.config.adapter} budget={self.config.agent_budget.preset}"
        )

        adapter_cls = get_adapter(self.config.adapter)

        # Start metering proxy if adapter needs it
        needs_metering = getattr(adapter_cls, "requires_metering", False)
        original_base_url = os.environ.get("OPENAI_BASE_URL")

        if needs_metering:
            self._metering = MeteringManager()
            proxy_url = self._metering.start()
            os.environ["OPENAI_BASE_URL"] = proxy_url
            self.logger.info(f"Metering proxy started at {proxy_url}")

        try:
            adapter = adapter_cls()

            # Group questions by scope_id and checkpoint
            questions = questions or []
            q_index: dict[str, dict[int, list[Question]]] = {}
            for q in questions:
                q_index.setdefault(q.scope_id, {}).setdefault(q.checkpoint_after, []).append(q)

            # Validate: every question's checkpoint_after must be reachable for its scope
            scope_reachable: dict[str, set[int]] = {}
            for scope_id, episodes in scopes.items():
                reachable = set(self.config.checkpoints)
                reachable.add(len(episodes))  # final episode is always a checkpoint
                scope_reachable[scope_id] = reachable
            for q in questions:
                reachable = scope_reachable.get(q.scope_id, set())
                if q.checkpoint_after not in reachable:
                    raise ConfigError(
                        f"Question {q.question_id!r} targets checkpoint_after={q.checkpoint_after} "
                        f"for scope {q.scope_id!r}, but that checkpoint is not reachable. "
                        f"Reachable checkpoints: {sorted(reachable)}. "
                        f"This question would be silently skipped."
                    )

            scope_results: list[ScopeResult] = []

            for scope_id, episodes in scopes.items():
                self.logger.info(f"Scope [bold]{scope_id}[/bold]: {len(episodes)} episodes")
                scope_questions = q_index.get(scope_id, {})
                result = self._run_scope(adapter, scope_id, episodes, scope_questions)
                scope_results.append(result)

            run_result = RunResult(
                run_id=self.run_id,
                adapter=self.config.adapter,
                dataset_version="",  # Set by caller
                budget_preset=self.config.agent_budget.preset,
                scopes=scope_results,
            )

            self.logger.success(f"Run {self.run_id} complete")
            return run_result
        finally:
            if self._metering is not None:
                self._metering.stop()
                if original_base_url is not None:
                    os.environ["OPENAI_BASE_URL"] = original_base_url
                elif "OPENAI_BASE_URL" in os.environ:
                    del os.environ["OPENAI_BASE_URL"]
                self._metering = None

    def _run_scope(
        self,
        adapter: MemoryAdapter,
        scope_id: str,
        episodes: list[Episode],
        questions_by_checkpoint: dict[int, list[Question]],
    ) -> ScopeResult:
        """Run the benchmark for a single scope."""
        episodes = sorted(episodes, key=lambda e: e.timestamp)

        # Always populate vault (needed for scoring regardless of cache)
        for ep in episodes:
            self.vault.store(ep.episode_id, ep.text)

        # Check adapter state cache
        cache_hit = False
        manifest_path: Path | None = None
        ck: str | None = None

        if self.config.cache_dir:
            adapter_name = type(adapter).__name__
            ck = _cache_key(adapter_name, scope_id, [e.episode_id for e in episodes])
            manifest_path = Path(self.config.cache_dir) / f"{adapter_name}_{ck}.json"
            if manifest_path.exists():
                try:
                    state = json.loads(manifest_path.read_text())
                    if adapter.restore_cache_state(state):
                        cache_hit = True
                        self.logger.info(
                            f"  Cache hit for {adapter_name} / {scope_id} — skipping ingest/prepare"
                        )
                        # Register synthetic refs from cached adapter
                        for ref_id, text in adapter.get_synthetic_refs():
                            self.vault.store(ref_id, text)
                except Exception as e:
                    log.warning("Cache restore failed for %s/%s: %s", adapter_name, scope_id, e)

        if cache_hit:
            # Jump straight to questions — run only the final checkpoint
            # (cache stores state after all episodes ingested + prepared)
            all_questions: list[Question] = []
            for qs in questions_by_checkpoint.values():
                all_questions.extend(qs)
            final = len(episodes)
            checkpoint_result = self._run_checkpoint(
                adapter, scope_id, final, all_questions, skip_prepare=True
            )
            return ScopeResult(scope_id=scope_id, checkpoints=[checkpoint_result])

        # Normal path: reset + ingest + prepare
        adapter.reset(scope_id)
        checkpoints_done: list[CheckpointResult] = []

        for idx, episode in enumerate(episodes, start=1):
            self.logger.start_step("ingest")
            t0 = time.monotonic()
            adapter.ingest(
                episode_id=episode.episode_id,
                scope_id=scope_id,
                timestamp=episode.timestamp.isoformat(),
                text=episode.text,
                meta=episode.meta,
            )
            elapsed_ms = (time.monotonic() - t0) * 1000
            self.logger.end_step(
                message=f"episode {episode.episode_id}",
                scope_id=scope_id,
                elapsed=elapsed_ms,
            )

            if elapsed_ms > self.config.agent_budget.ingest_max_latency_ms:
                self.logger.warn(
                    f"Ingest latency {elapsed_ms:.0f}ms exceeds "
                    f"{self.config.agent_budget.ingest_max_latency_ms}ms cap"
                )

            # Check if this is a checkpoint
            if idx in self.config.checkpoints:
                checkpoint_result = self._run_checkpoint(
                    adapter, scope_id, idx, questions_by_checkpoint.get(idx, [])
                )
                if self._metering is not None:
                    usage = self._metering.get_usage()
                    checkpoint_result.adapter_internal_tokens = usage.total_tokens
                    self._metering.reset()
                checkpoints_done.append(checkpoint_result)

        # Also run checkpoint at final episode if not already done
        final = len(episodes)
        if final not in self.config.checkpoints:
            checkpoint_result = self._run_checkpoint(
                adapter, scope_id, final, questions_by_checkpoint.get(final, [])
            )
            if self._metering is not None:
                usage = self._metering.get_usage()
                checkpoint_result.adapter_internal_tokens = usage.total_tokens
                self._metering.reset()
            checkpoints_done.append(checkpoint_result)

        # Save cache after successful run
        if manifest_path is not None:
            state = adapter.get_cache_state()
            if state is not None:
                try:
                    manifest_path.parent.mkdir(parents=True, exist_ok=True)
                    manifest_path.write_text(json.dumps(state))
                    self.logger.info(f"  Saved adapter cache to {manifest_path}")
                except Exception as e:
                    log.warning("Failed to save adapter cache: %s", e)

        return ScopeResult(scope_id=scope_id, checkpoints=checkpoints_done)

    def _run_checkpoint(
        self,
        adapter: MemoryAdapter,
        scope_id: str,
        checkpoint: int,
        questions: list[Question],
        skip_prepare: bool = False,
    ) -> CheckpointResult:
        """Execute prepare + agent questions at a checkpoint."""
        self.logger.info(f"  Checkpoint {checkpoint} for {scope_id}")
        errors: list[str] = []
        timing: dict[str, float] = {}

        if not skip_prepare:
            # Optional prepare hook
            t0 = time.monotonic()
            adapter.prepare(scope_id, checkpoint)
            timing["prepare_ms"] = (time.monotonic() - t0) * 1000

            # Register adapter-generated synthetic documents in vault
            for ref_id, text in adapter.get_synthetic_refs():
                self.vault.store(ref_id, text)

        # Build agent budget from config
        budget = QuestionBudget(
            max_turns=self.config.agent_budget.max_turns,
            max_payload_bytes=self.config.agent_budget.max_payload_bytes,
            max_latency_per_call_ms=self.config.agent_budget.max_latency_per_call_ms,
            max_total_tool_calls=self.config.agent_budget.max_tool_calls,
            max_agent_tokens=self.config.agent_budget.max_agent_tokens,
            max_cumulative_result_tokens=self.config.agent_budget.max_cumulative_result_tokens,
        )
        harness = AgentHarness(self.llm_client, budget)

        question_results: list[QuestionResult] = []

        def _answer_one(question: Question) -> tuple[QuestionResult, float]:
            self.logger.verbose(f"    Question: {question.question_id}")
            t0 = time.monotonic()
            answer = harness.answer(
                question_prompt=question.prompt,
                adapter=adapter,
                question_id=question.question_id,
            )
            q_ms = (time.monotonic() - t0) * 1000
            retrieved = answer.refs_cited
            valid = [r for r in retrieved if self.vault.has(r)]
            return QuestionResult(
                question=question,
                answer=answer,
                retrieved_ref_ids=retrieved,
                valid_ref_ids=valid,
            ), q_ms

        max_workers = min(self.config.parallel_questions, len(questions)) if questions else 1
        if max_workers > 1:
            self.logger.info(f"    Answering {len(questions)} questions with {max_workers} workers")
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = {pool.submit(_answer_one, q): i for i, q in enumerate(questions)}
                indexed_results: list[tuple[int, QuestionResult]] = []
                for future in concurrent.futures.as_completed(futures):
                    idx = futures[future]
                    qr, q_ms = future.result()
                    timing[f"question_{qr.question.question_id}_ms"] = q_ms
                    indexed_results.append((idx, qr))
            question_results = [qr for _, qr in sorted(indexed_results)]
        else:
            for question in questions:
                qr, q_ms = _answer_one(question)
                timing[f"question_{qr.question.question_id}_ms"] = q_ms
                question_results.append(qr)

        return CheckpointResult(
            scope_id=scope_id,
            checkpoint=checkpoint,
            question_results=question_results,
            validation_errors=errors,
            budget_used={},
            timing=timing,
        )

    def save_artifacts(self, result: RunResult, output_dir: str | Path) -> Path:
        """Write run artifacts to disk under a per-run subdirectory."""
        out = Path(output_dir) / self.run_id
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

        # Per-scope checkpoints
        for scope in result.scopes:
            scope_dir = out / "scopes" / scope.scope_id
            for cp in scope.checkpoints:
                cp_dir = scope_dir / f"checkpoint_{cp.checkpoint}"
                cp_dir.mkdir(parents=True, exist_ok=True)

                with atomic_write(cp_dir / "question_results.json") as tmp:
                    tmp.write_text(json.dumps(
                        [qr.to_dict() for qr in cp.question_results], indent=2
                    ))

                if cp.validation_errors:
                    with atomic_write(cp_dir / "validation.json") as tmp:
                        tmp.write_text(json.dumps(cp.validation_errors, indent=2))

        # Write structured log
        self.logger.save_log(out / "log.jsonl")

        self.logger.success(f"Artifacts saved to {out}")
        return out

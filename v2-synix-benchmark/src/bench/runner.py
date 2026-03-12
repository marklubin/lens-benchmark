"""StudyRunner — top-level orchestrator for benchmark execution.

Iterates scopes x policies x checkpoints x questions. Handles resume,
bank reuse across policies, and event emission. This is the entry point
for running a complete benchmark study.
"""
from __future__ import annotations

import hashlib
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from bench.agent import AgentHarness
from bench.bank import BankBuilder
from bench.broker import ModalBroker
from bench.dataset import ScopeData, load_scope
from bench.policy import create_policy
from bench.runtime import BenchmarkRuntime
from bench.schemas import (
    EventType,
    RunCost,
    RunManifest,
    RunStatus,
    StudyManifest,
)
from bench.state import EventWriter, StateStore

logger = logging.getLogger(__name__)


def _config_hash(policy_id: str, scope_id: str, study_id: str) -> str:
    """Deterministic hash for a (study, scope, policy) triple."""
    h = hashlib.sha256()
    h.update(f"{study_id}:{scope_id}:{policy_id}".encode())
    return h.hexdigest()[:16]


class StudyRunner:
    """Orchestrates a complete benchmark study.

    Usage:
        runner = StudyRunner(study, store, broker, work_dir, scope_dirs)
        runner.run_study()
    """

    def __init__(
        self,
        study: StudyManifest,
        store: StateStore,
        broker: ModalBroker,
        work_dir: Path,
        scope_dirs: dict[str, Path],
        *,
        families: list[str] | None = None,
        max_turns: int = 10,
        max_tool_calls: int = 20,
        temperature: float = 0.0,
        replicate_id: str = "r01",
    ) -> None:
        self._study = study
        self._store = store
        self._broker = broker
        self._work_dir = Path(work_dir)
        self._scope_dirs = scope_dirs
        self._families = families or ["chunks", "core_memory", "summary"]
        self._max_turns = max_turns
        self._max_tool_calls = max_tool_calls
        self._temperature = temperature
        self._replicate_id = replicate_id
        self._event_writer = EventWriter(store, study.study_id)
        self._bank_builder = BankBuilder(store, broker, work_dir / "banks")

    def run_study(self) -> list[RunManifest]:
        """Execute all scope x policy cells.

        Banks are built once per scope (shared across policies).
        Returns all RunManifests created/completed.
        """
        all_runs: list[RunManifest] = []

        for scope_id in self._study.scope_ids:
            scope_dir = self._scope_dirs.get(scope_id)
            if scope_dir is None:
                logger.error("No scope directory for %s, skipping", scope_id)
                continue

            scope = load_scope(scope_dir)
            logger.info("Loaded scope %s: %d episodes, %d questions, %d checkpoints",
                        scope.scope_id, len(scope.episodes), len(scope.questions),
                        len(scope.checkpoints))

            # Build banks once for this scope (reused across policies)
            bank_manifests = self._bank_builder.build_scope_banks(
                study_id=self._study.study_id,
                scope=scope,
                families=self._families,
                event_writer=self._event_writer,
            )
            bank_ids = [b.bank_manifest_id for b in bank_manifests]
            bank_by_checkpoint = {b.checkpoint_id: b for b in bank_manifests}

            for policy_id in self._study.policy_ids:
                run = self.run_cell(
                    scope=scope,
                    policy_id=policy_id,
                    bank_ids=bank_ids,
                    bank_by_checkpoint=bank_by_checkpoint,
                )
                all_runs.append(run)

        return all_runs

    def run_cell(
        self,
        scope: ScopeData,
        policy_id: str,
        bank_ids: list[str],
        bank_by_checkpoint: dict[str, Any],
    ) -> RunManifest:
        """Execute a single scope x policy cell.

        Iterates checkpoints, opens releases, runs agent for each question.
        Supports resume: skips already-completed questions.
        """
        policy = create_policy(policy_id, self._study.prompt_set_version)
        self._store.save_policy(policy)

        config = _config_hash(policy_id, scope.scope_id, self._study.study_id)
        run_id = f"run-{scope.scope_id}-{policy_id}-{self._replicate_id}-{self._study.study_id[:8]}"

        # Check for existing run (resume)
        existing = self._store.get_run(run_id)
        if existing is not None and existing.status == RunStatus.completed:
            logger.info("Run %s already completed, skipping", run_id)
            return existing

        run = RunManifest(
            run_id=run_id,
            study_id=self._study.study_id,
            scope_id=scope.scope_id,
            policy_id=policy_id,
            replicate_id=self._replicate_id,
            policy_manifest_id=policy.policy_manifest_id,
            bank_manifest_ids=bank_ids,
            config_hash=config,
            dataset_hash=scope.dataset_hash,
            status=RunStatus.running,
            started_at=datetime.now(timezone.utc),
        )
        self._store.save_run(run)

        self._event_writer.emit(
            EventType.run_started,
            run_id=run_id,
            scope_id=scope.scope_id,
            policy_id=policy_id,
        )

        completed_questions = self._store.get_completed_questions(run_id)
        total_prompt_tokens = 0
        total_completion_tokens = 0

        for checkpoint in scope.checkpoints:
            checkpoint_id = f"cp{checkpoint:02d}"
            bank_manifest = bank_by_checkpoint.get(checkpoint_id)
            if bank_manifest is None:
                logger.error("No bank manifest for checkpoint %s", checkpoint_id)
                continue

            self._event_writer.emit(
                EventType.checkpoint_started,
                run_id=run_id,
                scope_id=scope.scope_id,
                policy_id=policy_id,
                bank_manifest_id=bank_manifest.bank_manifest_id,
                payload={"checkpoint_id": checkpoint_id},
            )

            # Open release for this checkpoint
            release = self._bank_builder.open_release(bank_manifest)
            runtime = BenchmarkRuntime(release, policy)
            harness = AgentHarness(
                self._broker,
                runtime,
                model=self._study.agent_model,
                max_turns=self._max_turns,
                max_tool_calls=self._max_tool_calls,
                temperature=self._temperature,
            )

            questions = scope.questions_at(checkpoint)

            for question in questions:
                if question.question_id in completed_questions:
                    logger.info("Question %s already completed, skipping", question.question_id)
                    continue

                self._event_writer.emit(
                    EventType.question_started,
                    run_id=run_id,
                    scope_id=scope.scope_id,
                    policy_id=policy_id,
                    payload={"question_id": question.question_id, "checkpoint_id": checkpoint_id},
                )

                agent_answer = harness.answer(
                    question.prompt,
                    question_id=question.question_id,
                )

                # Save answer
                answer_data = {
                    "question_id": question.question_id,
                    "answer_text": agent_answer.answer_text,
                    "cited_refs": agent_answer.cited_refs,
                    "tool_calls_made": agent_answer.tool_calls_made,
                    "prompt_tokens": agent_answer.total_prompt_tokens,
                    "completion_tokens": agent_answer.total_completion_tokens,
                    "wall_time_ms": agent_answer.wall_time_ms,
                    "turns": agent_answer.turns,
                }
                self._store.save_answer(run_id, question.question_id, checkpoint_id, answer_data)

                total_prompt_tokens += agent_answer.total_prompt_tokens
                total_completion_tokens += agent_answer.total_completion_tokens

                self._event_writer.emit(
                    EventType.question_completed,
                    run_id=run_id,
                    scope_id=scope.scope_id,
                    policy_id=policy_id,
                    payload={
                        "question_id": question.question_id,
                        "tool_calls": agent_answer.tool_calls_made,
                        "wall_time_ms": agent_answer.wall_time_ms,
                    },
                )

                logger.info(
                    "Question %s answered: %d tool calls, %.0fms",
                    question.question_id,
                    agent_answer.tool_calls_made,
                    agent_answer.wall_time_ms,
                )

            self._event_writer.emit(
                EventType.checkpoint_completed,
                run_id=run_id,
                scope_id=scope.scope_id,
                policy_id=policy_id,
                payload={"checkpoint_id": checkpoint_id},
            )

            # Update resume cursor
            run.last_completed_checkpoint = checkpoint_id
            self._store.save_run(run)

        # Mark completed
        run.status = RunStatus.completed
        run.completed_at = datetime.now(timezone.utc)
        run.cost = RunCost(
            prompt_tokens=total_prompt_tokens,
            completion_tokens=total_completion_tokens,
        )
        self._store.save_run(run)

        self._event_writer.emit(
            EventType.run_completed,
            run_id=run_id,
            scope_id=scope.scope_id,
            policy_id=policy_id,
            payload={
                "prompt_tokens": total_prompt_tokens,
                "completion_tokens": total_completion_tokens,
            },
        )

        logger.info("Run %s completed", run_id)
        return run

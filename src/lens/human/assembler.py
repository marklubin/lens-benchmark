from __future__ import annotations

import json
from pathlib import Path

from lens.core.errors import atomic_write
from lens.core.models import (
    AgentAnswer,
    CheckpointResult,
    Question,
    QuestionResult,
    RunResult,
    ScopeResult,
)
from lens.human.state import HumanAnswerRecord, HumanBenchmarkState


def build_run_result(
    state: HumanBenchmarkState,
    questions: list[Question],
) -> RunResult:
    """Build a RunResult from completed human benchmark state."""
    q_by_id: dict[str, Question] = {q.question_id: q for q in questions}

    scope_results: list[ScopeResult] = []
    for scope_id in state.scope_order:
        progress = state.scope_progress[scope_id]

        # Group answers by checkpoint
        answers_by_cp: dict[int, list[HumanAnswerRecord]] = {}
        for ans in progress.answers:
            answers_by_cp.setdefault(ans.checkpoint, []).append(ans)

        checkpoints: list[CheckpointResult] = []
        for cp_num in sorted(answers_by_cp.keys()):
            question_results: list[QuestionResult] = []
            for ans in answers_by_cp[cp_num]:
                question = q_by_id[ans.question_id]
                agent_answer = AgentAnswer(
                    question_id=ans.question_id,
                    answer_text=ans.answer_text,
                    turns=[],
                    tool_calls_made=0,
                    total_tokens=0,
                    wall_time_ms=ans.wall_time_ms,
                    budget_violations=[],
                    refs_cited=ans.refs_cited,
                )
                question_results.append(QuestionResult(
                    question=question,
                    answer=agent_answer,
                    retrieved_ref_ids=ans.refs_cited,
                    valid_ref_ids=ans.refs_cited,
                ))

            checkpoints.append(CheckpointResult(
                scope_id=scope_id,
                checkpoint=cp_num,
                question_results=question_results,
            ))

        scope_results.append(ScopeResult(
            scope_id=scope_id,
            checkpoints=checkpoints,
        ))

    return RunResult(
        run_id=state.run_id,
        adapter="human",
        dataset_version=state.dataset_version,
        budget_preset="human",
        scopes=scope_results,
    )


def write_artifacts(result: RunResult, state: HumanBenchmarkState, output_dir: Path) -> Path:
    """Write RunResult artifacts to disk in the standard format."""
    out = output_dir / result.run_id
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

    # Human state (bonus, for reference)
    state.save(out / "human_state.json")

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

    return out

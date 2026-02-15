from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from lens.core.errors import atomic_write
from lens.core.models import Question
from lens.datasets.loader import load_episodes, load_questions, get_dataset_version


@dataclass
class HumanAnswerRecord:
    question_id: str
    scope_id: str
    checkpoint: int
    answer_text: str
    refs_cited: list[str]
    wall_time_ms: float
    answered_at: str

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> HumanAnswerRecord:
        return cls(**d)


@dataclass
class ScopeProgress:
    scope_id: str
    episodes_revealed: int
    total_episodes: int
    checkpoints_completed: list[int] = field(default_factory=list)
    answers: list[HumanAnswerRecord] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "scope_id": self.scope_id,
            "episodes_revealed": self.episodes_revealed,
            "total_episodes": self.total_episodes,
            "checkpoints_completed": self.checkpoints_completed,
            "answers": [a.to_dict() for a in self.answers],
        }

    @classmethod
    def from_dict(cls, d: dict) -> ScopeProgress:
        return cls(
            scope_id=d["scope_id"],
            episodes_revealed=d["episodes_revealed"],
            total_episodes=d["total_episodes"],
            checkpoints_completed=d.get("checkpoints_completed", []),
            answers=[HumanAnswerRecord.from_dict(a) for a in d.get("answers", [])],
        )


@dataclass
class HumanBenchmarkState:
    run_id: str
    dataset_path: str
    dataset_version: str
    current_scope_index: int
    scope_order: list[str]
    scope_progress: dict[str, ScopeProgress]
    is_complete: bool = False

    def save(self, path: Path) -> None:
        data = {
            "run_id": self.run_id,
            "dataset_path": self.dataset_path,
            "dataset_version": self.dataset_version,
            "current_scope_index": self.current_scope_index,
            "scope_order": self.scope_order,
            "scope_progress": {k: v.to_dict() for k, v in self.scope_progress.items()},
            "is_complete": self.is_complete,
        }
        with atomic_write(path) as tmp:
            tmp.write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: Path) -> HumanBenchmarkState:
        data = json.loads(path.read_text())
        return cls(
            run_id=data["run_id"],
            dataset_path=data["dataset_path"],
            dataset_version=data["dataset_version"],
            current_scope_index=data["current_scope_index"],
            scope_order=data["scope_order"],
            scope_progress={
                k: ScopeProgress.from_dict(v) for k, v in data["scope_progress"].items()
            },
            is_complete=data.get("is_complete", False),
        )

    @classmethod
    def initialize(cls, run_id: str, dataset_path: str, data: dict) -> HumanBenchmarkState:
        episodes_by_scope = load_episodes(data)
        version = get_dataset_version(data)

        scope_order = list(episodes_by_scope.keys())
        scope_progress: dict[str, ScopeProgress] = {}
        for scope_id, episodes in episodes_by_scope.items():
            scope_progress[scope_id] = ScopeProgress(
                scope_id=scope_id,
                episodes_revealed=0,
                total_episodes=len(episodes),
            )

        return cls(
            run_id=run_id,
            dataset_path=dataset_path,
            dataset_version=version,
            current_scope_index=0,
            scope_order=scope_order,
            scope_progress=scope_progress,
        )

    @property
    def current_scope_id(self) -> str | None:
        if self.current_scope_index >= len(self.scope_order):
            return None
        return self.scope_order[self.current_scope_index]

    def current_scope_progress(self) -> ScopeProgress | None:
        sid = self.current_scope_id
        if sid is None:
            return None
        return self.scope_progress[sid]


def discover_checkpoints(questions: list[Question], scope_id: str) -> list[int]:
    """Derive checkpoint numbers from dataset questions for a given scope."""
    cps = sorted({q.checkpoint_after for q in questions if q.scope_id == scope_id})
    return cps


def questions_at_checkpoint(
    questions: list[Question], scope_id: str, checkpoint: int
) -> list[Question]:
    """Return questions that trigger at a given checkpoint for a scope."""
    return [
        q for q in questions
        if q.scope_id == scope_id and q.checkpoint_after == checkpoint
    ]


def pending_questions_at_checkpoint(
    questions: list[Question],
    scope_id: str,
    checkpoint: int,
    answered_ids: set[str],
) -> list[Question]:
    """Return unanswered questions at a checkpoint."""
    return [
        q for q in questions_at_checkpoint(questions, scope_id, checkpoint)
        if q.question_id not in answered_ids
    ]

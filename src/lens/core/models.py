from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


# ---------------------------------------------------------------------------
# Ground truth & question models (dataset)
# ---------------------------------------------------------------------------


@dataclass
class GroundTruth:
    """Ground truth for a benchmark question."""

    canonical_answer: str
    required_evidence_refs: list[str]  # episode_ids the answer should draw from
    key_facts: list[str]  # factual claims that must appear

    def to_dict(self) -> dict:
        return {
            "canonical_answer": self.canonical_answer,
            "required_evidence_refs": self.required_evidence_refs,
            "key_facts": self.key_facts,
        }

    @classmethod
    def from_dict(cls, d: dict) -> GroundTruth:
        return cls(
            canonical_answer=d["canonical_answer"],
            required_evidence_refs=d["required_evidence_refs"],
            key_facts=d["key_facts"],
        )


@dataclass
class Question:
    """A benchmark question to be answered by the agent."""

    question_id: str
    scope_id: str
    checkpoint_after: int
    question_type: str  # "longitudinal" | "null_hypothesis" | "action_recommendation"
    prompt: str
    ground_truth: GroundTruth

    def to_dict(self) -> dict:
        return {
            "question_id": self.question_id,
            "scope_id": self.scope_id,
            "checkpoint_after": self.checkpoint_after,
            "question_type": self.question_type,
            "prompt": self.prompt,
            "ground_truth": self.ground_truth.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> Question:
        return cls(
            question_id=d["question_id"],
            scope_id=d["scope_id"],
            checkpoint_after=d["checkpoint_after"],
            question_type=d["question_type"],
            prompt=d["prompt"],
            ground_truth=GroundTruth.from_dict(d["ground_truth"]),
        )


# ---------------------------------------------------------------------------
# Agent answer models (agent output)
# ---------------------------------------------------------------------------


@dataclass
class AgentAnswer:
    """The agent's answer to a benchmark question."""

    question_id: str
    answer_text: str
    turns: list[dict] = field(default_factory=list)
    tool_calls_made: int = 0
    total_tokens: int = 0
    wall_time_ms: float = 0.0
    budget_violations: list[str] = field(default_factory=list)
    refs_cited: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "question_id": self.question_id,
            "answer_text": self.answer_text,
            "turns": self.turns,
            "tool_calls_made": self.tool_calls_made,
            "total_tokens": self.total_tokens,
            "wall_time_ms": self.wall_time_ms,
            "budget_violations": self.budget_violations,
            "refs_cited": self.refs_cited,
        }

    @classmethod
    def from_dict(cls, d: dict) -> AgentAnswer:
        return cls(
            question_id=d["question_id"],
            answer_text=d["answer_text"],
            turns=d.get("turns", []),
            tool_calls_made=d.get("tool_calls_made", 0),
            total_tokens=d.get("total_tokens", 0),
            wall_time_ms=d.get("wall_time_ms", 0.0),
            budget_violations=d.get("budget_violations", []),
            refs_cited=d.get("refs_cited", []),
        )


@dataclass
class QuestionResult:
    """Result of a single question answered by the agent."""

    question: Question
    answer: AgentAnswer
    retrieved_ref_ids: list[str] = field(default_factory=list)
    valid_ref_ids: list[str] = field(default_factory=list)  # subset that exist in vault

    def to_dict(self) -> dict:
        return {
            "question": self.question.to_dict(),
            "answer": self.answer.to_dict(),
            "retrieved_ref_ids": self.retrieved_ref_ids,
            "valid_ref_ids": self.valid_ref_ids,
        }

    @classmethod
    def from_dict(cls, d: dict) -> QuestionResult:
        return cls(
            question=Question.from_dict(d["question"]),
            answer=AgentAnswer.from_dict(d["answer"]),
            retrieved_ref_ids=d.get("retrieved_ref_ids", []),
            valid_ref_ids=d.get("valid_ref_ids", []),
        )


# ---------------------------------------------------------------------------
# Episode model (input)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Episode:
    """A single episode in a scope's longitudinal stream."""

    episode_id: str
    scope_id: str
    timestamp: datetime
    text: str
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "episode_id": self.episode_id,
            "scope_id": self.scope_id,
            "timestamp": self.timestamp.isoformat(),
            "text": self.text,
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Episode:
        ts = d["timestamp"]
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts)
        return cls(
            episode_id=d["episode_id"],
            scope_id=d["scope_id"],
            timestamp=ts,
            text=d["text"],
            meta=d.get("meta", {}),
        )


# ---------------------------------------------------------------------------
# Scoring models
# ---------------------------------------------------------------------------


@dataclass
class MetricResult:
    """Result of a single metric computation."""

    name: str
    tier: int
    value: float  # [0, 1]
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "tier": self.tier,
            "value": self.value,
            "details": self.details,
        }

    @classmethod
    def from_dict(cls, d: dict) -> MetricResult:
        return cls(
            name=d["name"],
            tier=d["tier"],
            value=d["value"],
            details=d.get("details", {}),
        )


@dataclass
class ScoreCard:
    """Aggregate scoring results for a run."""

    run_id: str
    adapter: str
    dataset_version: str
    budget_preset: str
    metrics: list[MetricResult] = field(default_factory=list)
    composite_score: float = 0.0

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "adapter": self.adapter,
            "dataset_version": self.dataset_version,
            "budget_preset": self.budget_preset,
            "metrics": [m.to_dict() for m in self.metrics],
            "composite_score": self.composite_score,
        }

    @classmethod
    def from_dict(cls, d: dict) -> ScoreCard:
        return cls(
            run_id=d["run_id"],
            adapter=d["adapter"],
            dataset_version=d["dataset_version"],
            budget_preset=d["budget_preset"],
            metrics=[MetricResult.from_dict(m) for m in d.get("metrics", [])],
            composite_score=d.get("composite_score", 0.0),
        )


# ---------------------------------------------------------------------------
# Checkpoint models (runner output)
# ---------------------------------------------------------------------------


@dataclass
class CheckpointResult:
    """Results captured at a single checkpoint for a scope."""

    scope_id: str
    checkpoint: int
    question_results: list[QuestionResult] = field(default_factory=list)
    validation_errors: list[str] = field(default_factory=list)
    budget_used: dict = field(default_factory=dict)
    timing: dict = field(default_factory=dict)
    adapter_internal_tokens: int = 0

    def to_dict(self) -> dict:
        return {
            "scope_id": self.scope_id,
            "checkpoint": self.checkpoint,
            "question_results": [qr.to_dict() for qr in self.question_results],
            "validation_errors": self.validation_errors,
            "budget_used": self.budget_used,
            "timing": self.timing,
            "adapter_internal_tokens": self.adapter_internal_tokens,
        }

    @classmethod
    def from_dict(cls, d: dict) -> CheckpointResult:
        return cls(
            scope_id=d["scope_id"],
            checkpoint=d["checkpoint"],
            question_results=[
                QuestionResult.from_dict(qr) for qr in d.get("question_results", [])
            ],
            validation_errors=d.get("validation_errors", []),
            budget_used=d.get("budget_used", {}),
            timing=d.get("timing", {}),
            adapter_internal_tokens=d.get("adapter_internal_tokens", 0),
        )


@dataclass
class ScopeResult:
    """All checkpoint results for a single scope."""

    scope_id: str
    checkpoints: list[CheckpointResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "scope_id": self.scope_id,
            "checkpoints": [c.to_dict() for c in self.checkpoints],
        }

    @classmethod
    def from_dict(cls, d: dict) -> ScopeResult:
        return cls(
            scope_id=d["scope_id"],
            checkpoints=[CheckpointResult.from_dict(c) for c in d.get("checkpoints", [])],
        )


@dataclass
class RunResult:
    """Complete results for a benchmark run."""

    run_id: str
    adapter: str
    dataset_version: str
    budget_preset: str
    scopes: list[ScopeResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "adapter": self.adapter,
            "dataset_version": self.dataset_version,
            "budget_preset": self.budget_preset,
            "scopes": [s.to_dict() for s in self.scopes],
        }

    @classmethod
    def from_dict(cls, d: dict) -> RunResult:
        return cls(
            run_id=d["run_id"],
            adapter=d["adapter"],
            dataset_version=d["dataset_version"],
            budget_preset=d["budget_preset"],
            scopes=[ScopeResult.from_dict(s) for s in d.get("scopes", [])],
        )

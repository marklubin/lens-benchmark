from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


# ---------------------------------------------------------------------------
# Evidence & Insight models (adapter output)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvidenceRef:
    """Reference to a specific quote within an episode."""

    episode_id: str
    quote: str  # Must be exact substring of episode text

    def to_dict(self) -> dict:
        return {"episode_id": self.episode_id, "quote": self.quote}

    @classmethod
    def from_dict(cls, d: dict) -> EvidenceRef:
        return cls(episode_id=d["episode_id"], quote=d["quote"])


@dataclass
class Insight:
    """A longitudinal insight produced by a memory system."""

    text: str
    confidence: float  # [0, 1]
    evidence: list[EvidenceRef]  # min 3 recommended
    falsifier: str  # What would disprove this insight

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            msg = f"confidence must be in [0, 1], got {self.confidence}"
            raise ValueError(msg)

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "confidence": self.confidence,
            "evidence": [e.to_dict() for e in self.evidence],
            "falsifier": self.falsifier,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Insight:
        return cls(
            text=d["text"],
            confidence=d["confidence"],
            evidence=[EvidenceRef.from_dict(e) for e in d["evidence"]],
            falsifier=d["falsifier"],
        )


@dataclass(frozen=True)
class Hit:
    """A search result from the memory system."""

    id: str
    kind: str  # e.g. "episode", "insight", "fragment"
    text: str
    score: float

    def to_dict(self) -> dict:
        return {"id": self.id, "kind": self.kind, "text": self.text, "score": self.score}

    @classmethod
    def from_dict(cls, d: dict) -> Hit:
        return cls(id=d["id"], kind=d["kind"], text=d["text"], score=d["score"])


# ---------------------------------------------------------------------------
# Episode model (input)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Episode:
    """A single episode in a persona's longitudinal stream."""

    episode_id: str
    persona_id: str
    timestamp: datetime
    text: str
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "episode_id": self.episode_id,
            "persona_id": self.persona_id,
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
            persona_id=d["persona_id"],
            timestamp=ts,
            text=d["text"],
            meta=d.get("meta", {}),
        )


# ---------------------------------------------------------------------------
# Dataset ground-truth models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvidenceFragment:
    """An exact substring expected to appear in a generated episode."""

    episode_id: str
    fragment: str  # exact substring

    def to_dict(self) -> dict:
        return {"episode_id": self.episode_id, "fragment": self.fragment}

    @classmethod
    def from_dict(cls, d: dict) -> EvidenceFragment:
        return cls(episode_id=d["episode_id"], fragment=d["fragment"])


@dataclass
class TruthPattern:
    """A planted longitudinal insight with known ground truth."""

    pattern_id: str
    persona_id: str
    canonical_insight: str
    insight_category: str  # trend, correlation, preference_evolution, etc.
    evidence_episode_ids: list[str]
    evidence_fragments: list[EvidenceFragment]
    min_episodes_required: int
    first_signal_episode: int
    difficulty: str  # easy, medium, hard
    expected_confidence: float
    supersedes: str | None = None

    def to_dict(self) -> dict:
        return {
            "pattern_id": self.pattern_id,
            "persona_id": self.persona_id,
            "canonical_insight": self.canonical_insight,
            "insight_category": self.insight_category,
            "evidence_episode_ids": self.evidence_episode_ids,
            "evidence_fragments": [f.to_dict() for f in self.evidence_fragments],
            "min_episodes_required": self.min_episodes_required,
            "first_signal_episode": self.first_signal_episode,
            "difficulty": self.difficulty,
            "expected_confidence": self.expected_confidence,
            "supersedes": self.supersedes,
        }

    @classmethod
    def from_dict(cls, d: dict) -> TruthPattern:
        return cls(
            pattern_id=d["pattern_id"],
            persona_id=d["persona_id"],
            canonical_insight=d["canonical_insight"],
            insight_category=d["insight_category"],
            evidence_episode_ids=d["evidence_episode_ids"],
            evidence_fragments=[EvidenceFragment.from_dict(f) for f in d["evidence_fragments"]],
            min_episodes_required=d["min_episodes_required"],
            first_signal_episode=d["first_signal_episode"],
            difficulty=d["difficulty"],
            expected_confidence=d["expected_confidence"],
            supersedes=d.get("supersedes"),
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
    """Results captured at a single checkpoint for a persona."""

    persona_id: str
    checkpoint: int
    insights: list[Insight] = field(default_factory=list)
    search_results: dict[str, list[Hit]] = field(default_factory=dict)  # query -> hits
    validation_errors: list[str] = field(default_factory=list)
    budget_used: dict = field(default_factory=dict)
    timing: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "persona_id": self.persona_id,
            "checkpoint": self.checkpoint,
            "insights": [i.to_dict() for i in self.insights],
            "search_results": {
                q: [h.to_dict() for h in hits] for q, hits in self.search_results.items()
            },
            "validation_errors": self.validation_errors,
            "budget_used": self.budget_used,
            "timing": self.timing,
        }

    @classmethod
    def from_dict(cls, d: dict) -> CheckpointResult:
        return cls(
            persona_id=d["persona_id"],
            checkpoint=d["checkpoint"],
            insights=[Insight.from_dict(i) for i in d.get("insights", [])],
            search_results={
                q: [Hit.from_dict(h) for h in hits]
                for q, hits in d.get("search_results", {}).items()
            },
            validation_errors=d.get("validation_errors", []),
            budget_used=d.get("budget_used", {}),
            timing=d.get("timing", {}),
        )


@dataclass
class PersonaResult:
    """All checkpoint results for a single persona."""

    persona_id: str
    checkpoints: list[CheckpointResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "persona_id": self.persona_id,
            "checkpoints": [c.to_dict() for c in self.checkpoints],
        }

    @classmethod
    def from_dict(cls, d: dict) -> PersonaResult:
        return cls(
            persona_id=d["persona_id"],
            checkpoints=[CheckpointResult.from_dict(c) for c in d.get("checkpoints", [])],
        )


@dataclass
class RunResult:
    """Complete results for a benchmark run."""

    run_id: str
    adapter: str
    dataset_version: str
    budget_preset: str
    personas: list[PersonaResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "adapter": self.adapter,
            "dataset_version": self.dataset_version,
            "budget_preset": self.budget_preset,
            "personas": [p.to_dict() for p in self.personas],
        }

    @classmethod
    def from_dict(cls, d: dict) -> RunResult:
        return cls(
            run_id=d["run_id"],
            adapter=d["adapter"],
            dataset_version=d["dataset_version"],
            budget_preset=d["budget_preset"],
            personas=[PersonaResult.from_dict(p) for p in d.get("personas", [])],
        )

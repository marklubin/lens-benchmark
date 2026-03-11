"""Canonical schemas for the V2 benchmark.

Six manifest/record types that together make every benchmark result
attributable and reproducible:

    StudyManifest      — top-level study definition
    PolicyManifest     — runtime policy configuration
    BankManifest       — per scope x checkpoint artifact bank
    RunManifest        — per scope x policy x replicate execution
    Event              — append-only event log entry
    ScoreRecord        — per-question score
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class BankStatus(str, Enum):
    planned = "planned"
    building = "building"
    built = "built"
    released = "released"
    failed = "failed"


class RunStatus(str, Enum):
    planned = "planned"
    running = "running"
    completed = "completed"
    failed = "failed"
    resumed = "resumed"


class EventType(str, Enum):
    # Bank lifecycle
    bank_build_started = "bank_build_started"
    bank_build_completed = "bank_build_completed"
    bank_build_failed = "bank_build_failed"
    artifact_compiled = "artifact_compiled"

    # Run lifecycle
    run_started = "run_started"
    run_completed = "run_completed"
    run_failed = "run_failed"
    run_resumed = "run_resumed"

    # Checkpoint lifecycle
    checkpoint_started = "checkpoint_started"
    checkpoint_completed = "checkpoint_completed"

    # Question lifecycle
    question_started = "question_started"
    question_completed = "question_completed"

    # Inference
    model_call_completed = "model_call_completed"

    # Scoring
    scoring_started = "scoring_started"
    scoring_completed = "scoring_completed"


# ---------------------------------------------------------------------------
# Study Manifest
# ---------------------------------------------------------------------------


class StudyManifest(BaseModel):
    """Top-level study definition.

    Fixes the scope set, policy set, model choices, and code version
    for a benchmark study. All runs and scores are tied back to this.
    """

    study_id: str
    benchmark_version: str = "v2"
    scope_ids: list[str]
    policy_ids: list[str]
    agent_model: str
    judge_model: str
    embedding_model: str
    prompt_set_version: str
    scoring_version: str
    code_sha: str
    synix_build_graph_hash: str | None = None
    artifact_family_set_version: str
    random_seed_policy: str = "deterministic"
    created_at: datetime
    notes: str | None = None


# ---------------------------------------------------------------------------
# Policy Manifest
# ---------------------------------------------------------------------------


class FusionConfig(BaseModel):
    method: str = "rrf"
    parameters: dict[str, Any] = Field(default_factory=dict)


class RetrievalCaps(BaseModel):
    max_results: int
    max_context_tokens: int


class PolicyManifest(BaseModel):
    """Runtime policy configuration.

    Defines which artifact families are visible, which search surfaces
    are queried, how results are fused, and retrieval caps. Policies
    consume a compiled bank — they never trigger rebuilds.
    """

    policy_manifest_id: str
    policy_id: str
    visible_artifact_families: list[str]
    query_surfaces: list[str]
    fusion: FusionConfig
    retrieval_caps: RetrievalCaps
    citation_policy: str = "source-backed-only"
    version: str

    # Optional configs for derived families
    fold_config: dict[str, Any] | None = None
    group_config: dict[str, Any] | None = None
    reduce_config: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Artifact Bank Manifest
# ---------------------------------------------------------------------------


class BuildCost(BaseModel):
    """Cost metrics captured during bank compilation."""

    wall_time_s: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    embed_tokens: int = 0
    estimated_cost_usd: float = 0.0


class BankManifest(BaseModel):
    """Per scope x checkpoint compiled artifact bank.

    Invariant: no artifact in this bank may contain information from
    episodes beyond max_episode_ordinal. This is enforced at build
    time via prefix-valid snapshot projections (D010, D014).
    """

    bank_manifest_id: str
    study_id: str
    scope_id: str
    checkpoint_id: str
    max_episode_ordinal: int
    source_episode_ids: list[str]
    artifact_families: dict[str, str | None]
    dataset_hash: str
    synix_build_ref: str | None = None
    synix_release_ref: str | None = None
    synix_build_graph_hash: str | None = None
    status: BankStatus = BankStatus.planned
    built_at: datetime | None = None
    build_cost: BuildCost | None = None


# ---------------------------------------------------------------------------
# Run Manifest
# ---------------------------------------------------------------------------


class RunCost(BaseModel):
    """Cost metrics captured during a policy run."""

    wall_time_s: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    estimated_cost_usd: float = 0.0


class RunManifest(BaseModel):
    """Per scope x policy x replicate execution.

    Links to the study, policy manifest, and the bank manifests
    used at each checkpoint. Tracks execution status and resume
    progress.
    """

    run_id: str
    study_id: str
    scope_id: str
    policy_id: str
    replicate_id: str
    policy_manifest_id: str
    bank_manifest_ids: list[str]
    config_hash: str
    dataset_hash: str
    status: RunStatus = RunStatus.planned
    started_at: datetime | None = None
    completed_at: datetime | None = None
    last_completed_checkpoint: str | None = None
    last_completed_question: str | None = None
    cost: RunCost | None = None


# ---------------------------------------------------------------------------
# Event
# ---------------------------------------------------------------------------


class Event(BaseModel):
    """Append-only event log entry.

    Every significant action during bank compilation or policy
    execution is recorded as an event. Events are the audit trail
    that ties results back to manifests.
    """

    event_id: str
    timestamp: datetime
    study_id: str
    run_id: str | None = None
    scope_id: str | None = None
    policy_id: str | None = None
    bank_manifest_id: str | None = None
    event_type: EventType
    attempt: int | None = None
    config_hash: str | None = None
    input_refs: list[str] = Field(default_factory=list)
    output_refs: list[str] = Field(default_factory=list)
    payload: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


# ---------------------------------------------------------------------------
# Score Record
# ---------------------------------------------------------------------------


class Diagnostics(BaseModel):
    """Secondary diagnostics tracked alongside the primary score."""

    latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    retrieval_count: int = 0
    tool_count: int = 0
    estimated_cost_usd: float = 0.0
    budget_overrun: bool = False


class ScoreRecord(BaseModel):
    """Per-question score record.

    Primary composite: 0.5 * fact_f1 + 0.3 * evidence_support + 0.2 * citation_validity
    No hard gates (D002).
    """

    score_id: str
    study_id: str
    run_id: str
    question_id: str
    scope_id: str
    policy_id: str
    checkpoint_id: str
    bank_manifest_id: str

    # Primary metrics
    fact_precision: float
    fact_recall: float
    fact_f1: float
    evidence_support: float
    citation_validity: float
    primary_score: float

    # Diagnostics (tracked, not in composite)
    diagnostics: Diagnostics = Field(default_factory=Diagnostics)

    # Provenance
    scored_at: datetime
    scorer_version: str
    judge_model: str

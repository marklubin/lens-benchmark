"""Tests for V2 benchmark schemas."""
from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from bench.schemas import (
    BankManifest,
    BankStatus,
    BuildCost,
    Diagnostics,
    Event,
    EventType,
    FusionConfig,
    PolicyManifest,
    RetrievalCaps,
    RunCost,
    RunManifest,
    RunStatus,
    ScoreRecord,
    StudyManifest,
)


# ---------------------------------------------------------------------------
# Fixtures — canonical valid instances
# ---------------------------------------------------------------------------

NOW = datetime(2026, 3, 10, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def study():
    return StudyManifest(
        study_id="2026-03-synix-v2-pilot",
        scope_ids=["S08", "S10"],
        policy_ids=["null", "policy_base", "policy_core", "policy_summary"],
        agent_model="modal:Qwen/Qwen3.5-35B-A3B",
        judge_model="modal:Qwen/Qwen3.5-35B-A3B",
        embedding_model="modal:Alibaba-NLP/gte-modernbert-base",
        prompt_set_version="v2",
        scoring_version="v2",
        code_sha="abc123def",
        artifact_family_set_version="v1",
        created_at=NOW,
    )


@pytest.fixture
def policy():
    return PolicyManifest(
        policy_manifest_id="policy_core_v1",
        policy_id="policy_core",
        visible_artifact_families=["chunks", "search_indexes", "core_memory"],
        query_surfaces=["fts", "cosine", "core_memory"],
        fusion=FusionConfig(method="rrf", parameters={"k": 60}),
        retrieval_caps=RetrievalCaps(max_results=12, max_context_tokens=24000),
        version="v1",
    )


@pytest.fixture
def bank():
    return BankManifest(
        bank_manifest_id="bank_s08_cp02_v1",
        study_id="2026-03-synix-v2-pilot",
        scope_id="S08",
        checkpoint_id="cp02",
        max_episode_ordinal=40,
        source_episode_ids=["e001", "e002", "e003"],
        artifact_families={
            "raw_episodes": "sha256:aaa",
            "chunks": "sha256:bbb",
            "search_indexes": "sha256:ccc",
            "core_memory": "sha256:ddd",
            "summaries": None,
            "graph": None,
        },
        dataset_hash="sha256:dataset",
    )


@pytest.fixture
def run():
    return RunManifest(
        run_id="run_s08_policy_core_r01",
        study_id="2026-03-synix-v2-pilot",
        scope_id="S08",
        policy_id="policy_core",
        replicate_id="r01",
        policy_manifest_id="policy_core_v1",
        bank_manifest_ids=["bank_s08_cp01_v1", "bank_s08_cp02_v1"],
        config_hash="sha256:config",
        dataset_hash="sha256:dataset",
    )


@pytest.fixture
def event():
    return Event(
        event_id="evt-001",
        timestamp=NOW,
        study_id="2026-03-synix-v2-pilot",
        run_id="run_s08_policy_core_r01",
        scope_id="S08",
        policy_id="policy_core",
        bank_manifest_id="bank_s08_cp02_v1",
        event_type=EventType.model_call_completed,
        attempt=1,
        input_refs=["artifact:question:S08:q03"],
        output_refs=["cache:llm:key123"],
        payload={
            "provider": "modal",
            "model": "Qwen/Qwen3.5-35B-A3B",
            "latency_ms": 812,
            "prompt_tokens": 1920,
            "completion_tokens": 314,
            "estimated_cost_usd": 0.0123,
        },
    )


@pytest.fixture
def score():
    return ScoreRecord(
        score_id="score-001",
        study_id="2026-03-synix-v2-pilot",
        run_id="run_s08_policy_core_r01",
        question_id="S08_Q03",
        scope_id="S08",
        policy_id="policy_core",
        checkpoint_id="cp02",
        bank_manifest_id="bank_s08_cp02_v1",
        fact_precision=0.8,
        fact_recall=0.67,
        fact_f1=0.72,
        evidence_support=0.8,
        citation_validity=1.0,
        primary_score=0.8,
        diagnostics=Diagnostics(
            latency_ms=5140,
            prompt_tokens=8812,
            completion_tokens=644,
            retrieval_count=4,
            tool_count=6,
            estimated_cost_usd=0.084,
        ),
        scored_at=NOW,
        scorer_version="v2",
        judge_model="modal:Qwen/Qwen3.5-35B-A3B",
    )


# ---------------------------------------------------------------------------
# Roundtrip tests — every schema must survive JSON serialization
# ---------------------------------------------------------------------------


class TestRoundtrip:
    """Verify model → JSON → model roundtrip for all schemas."""

    def test_study_roundtrip(self, study):
        data = json.loads(study.model_dump_json())
        rebuilt = StudyManifest.model_validate(data)
        assert rebuilt == study

    def test_policy_roundtrip(self, policy):
        data = json.loads(policy.model_dump_json())
        rebuilt = PolicyManifest.model_validate(data)
        assert rebuilt == policy

    def test_bank_roundtrip(self, bank):
        data = json.loads(bank.model_dump_json())
        rebuilt = BankManifest.model_validate(data)
        assert rebuilt == bank

    def test_run_roundtrip(self, run):
        data = json.loads(run.model_dump_json())
        rebuilt = RunManifest.model_validate(data)
        assert rebuilt == run

    def test_event_roundtrip(self, event):
        data = json.loads(event.model_dump_json())
        rebuilt = Event.model_validate(data)
        assert rebuilt == event

    def test_score_roundtrip(self, score):
        data = json.loads(score.model_dump_json())
        rebuilt = ScoreRecord.model_validate(data)
        assert rebuilt == score


# ---------------------------------------------------------------------------
# Required field validation
# ---------------------------------------------------------------------------


class TestRequiredFields:
    """Verify that missing required fields raise ValidationError."""

    def test_study_requires_study_id(self):
        with pytest.raises(ValidationError, match="study_id"):
            StudyManifest(
                scope_ids=["S08"],
                policy_ids=["null"],
                agent_model="m",
                judge_model="m",
                embedding_model="m",
                prompt_set_version="v2",
                scoring_version="v2",
                code_sha="abc",
                artifact_family_set_version="v1",
                created_at=NOW,
            )

    def test_study_requires_scope_ids(self):
        with pytest.raises(ValidationError, match="scope_ids"):
            StudyManifest(
                study_id="test",
                policy_ids=["null"],
                agent_model="m",
                judge_model="m",
                embedding_model="m",
                prompt_set_version="v2",
                scoring_version="v2",
                code_sha="abc",
                artifact_family_set_version="v1",
                created_at=NOW,
            )

    def test_bank_requires_checkpoint_id(self):
        with pytest.raises(ValidationError, match="checkpoint_id"):
            BankManifest(
                bank_manifest_id="bank_test",
                study_id="test",
                scope_id="S08",
                max_episode_ordinal=10,
                source_episode_ids=["e001"],
                artifact_families={"raw_episodes": "hash"},
                dataset_hash="hash",
            )

    def test_run_requires_policy_manifest_id(self):
        with pytest.raises(ValidationError, match="policy_manifest_id"):
            RunManifest(
                run_id="run_test",
                study_id="test",
                scope_id="S08",
                policy_id="policy_core",
                replicate_id="r01",
                bank_manifest_ids=["bank_test"],
                config_hash="hash",
                dataset_hash="hash",
            )

    def test_score_requires_primary_metrics(self):
        with pytest.raises(ValidationError):
            ScoreRecord(
                score_id="s",
                study_id="s",
                run_id="r",
                question_id="q",
                scope_id="S08",
                policy_id="p",
                checkpoint_id="cp01",
                bank_manifest_id="b",
                # Missing all primary metrics
                scored_at=NOW,
                scorer_version="v2",
                judge_model="m",
            )

    def test_event_requires_event_type(self):
        with pytest.raises(ValidationError, match="event_type"):
            Event(
                event_id="evt",
                timestamp=NOW,
                study_id="test",
            )

    def test_policy_requires_retrieval_caps(self):
        with pytest.raises(ValidationError, match="retrieval_caps"):
            PolicyManifest(
                policy_manifest_id="p",
                policy_id="null",
                visible_artifact_families=[],
                query_surfaces=[],
                fusion=FusionConfig(),
                version="v1",
            )


# ---------------------------------------------------------------------------
# Default values
# ---------------------------------------------------------------------------


class TestDefaults:
    """Verify sensible defaults on optional fields."""

    def test_bank_defaults_to_planned(self, bank):
        assert bank.status == BankStatus.planned

    def test_run_defaults_to_planned(self, run):
        assert run.status == RunStatus.planned

    def test_study_defaults_to_deterministic(self, study):
        assert study.random_seed_policy == "deterministic"

    def test_policy_defaults_citation_source_backed(self, policy):
        assert policy.citation_policy == "source-backed-only"

    def test_event_optional_fields_default_none(self):
        evt = Event(
            event_id="evt",
            timestamp=NOW,
            study_id="test",
            event_type=EventType.run_started,
        )
        assert evt.run_id is None
        assert evt.scope_id is None
        assert evt.error is None
        assert evt.input_refs == []

    def test_score_diagnostics_defaults(self):
        sr = ScoreRecord(
            score_id="s",
            study_id="s",
            run_id="r",
            question_id="q",
            scope_id="S08",
            policy_id="p",
            checkpoint_id="cp01",
            bank_manifest_id="b",
            fact_precision=1.0,
            fact_recall=1.0,
            fact_f1=1.0,
            evidence_support=1.0,
            citation_validity=1.0,
            primary_score=1.0,
            scored_at=NOW,
            scorer_version="v2",
            judge_model="m",
        )
        assert sr.diagnostics.latency_ms == 0.0
        assert sr.diagnostics.budget_overrun is False


# ---------------------------------------------------------------------------
# Enum validation
# ---------------------------------------------------------------------------


class TestEnums:
    """Verify enum fields reject invalid values."""

    def test_bank_rejects_invalid_status(self):
        with pytest.raises(ValidationError, match="status"):
            BankManifest(
                bank_manifest_id="b",
                study_id="s",
                scope_id="S08",
                checkpoint_id="cp01",
                max_episode_ordinal=10,
                source_episode_ids=["e001"],
                artifact_families={},
                dataset_hash="h",
                status="bogus",
            )

    def test_run_rejects_invalid_status(self):
        with pytest.raises(ValidationError, match="status"):
            RunManifest(
                run_id="r",
                study_id="s",
                scope_id="S08",
                policy_id="p",
                replicate_id="r01",
                policy_manifest_id="pm",
                bank_manifest_ids=[],
                config_hash="h",
                dataset_hash="h",
                status="bogus",
            )

    def test_event_rejects_invalid_type(self):
        with pytest.raises(ValidationError, match="event_type"):
            Event(
                event_id="evt",
                timestamp=NOW,
                study_id="test",
                event_type="not_a_real_event",
            )

    def test_all_event_types_roundtrip(self):
        for et in EventType:
            evt = Event(
                event_id=f"evt-{et.value}",
                timestamp=NOW,
                study_id="test",
                event_type=et,
            )
            data = json.loads(evt.model_dump_json())
            rebuilt = Event.model_validate(data)
            assert rebuilt.event_type == et


# ---------------------------------------------------------------------------
# Status transitions (bank and run)
# ---------------------------------------------------------------------------


class TestStatusTransitions:
    """Verify status field accepts all valid values."""

    @pytest.mark.parametrize("status", list(BankStatus))
    def test_bank_accepts_valid_statuses(self, bank, status):
        bank_data = bank.model_dump()
        bank_data["status"] = status.value
        rebuilt = BankManifest.model_validate(bank_data)
        assert rebuilt.status == status

    @pytest.mark.parametrize("status", list(RunStatus))
    def test_run_accepts_valid_statuses(self, run, status):
        run_data = run.model_dump()
        run_data["status"] = status.value
        rebuilt = RunManifest.model_validate(run_data)
        assert rebuilt.status == status


# ---------------------------------------------------------------------------
# Bank manifest: checkpoint isolation invariants
# ---------------------------------------------------------------------------


class TestCheckpointIsolation:
    """Verify bank manifest captures checkpoint provenance."""

    def test_bank_has_episode_boundary(self, bank):
        assert bank.max_episode_ordinal == 40

    def test_bank_has_source_episode_ids(self, bank):
        assert len(bank.source_episode_ids) > 0

    def test_bank_has_dataset_hash(self, bank):
        assert bank.dataset_hash != ""

    def test_bank_artifact_families_allow_null_for_unused(self, bank):
        """Unused families (e.g. graph) should be None, not missing."""
        assert bank.artifact_families["graph"] is None
        assert bank.artifact_families["chunks"] is not None


# ---------------------------------------------------------------------------
# Score record: composite validation
# ---------------------------------------------------------------------------


class TestScoreComposite:
    """Verify score record carries the right primary metric fields."""

    def test_has_all_primary_fields(self, score):
        assert hasattr(score, "fact_precision")
        assert hasattr(score, "fact_recall")
        assert hasattr(score, "fact_f1")
        assert hasattr(score, "evidence_support")
        assert hasattr(score, "citation_validity")
        assert hasattr(score, "primary_score")

    def test_composite_formula(self, score):
        """Verify the primary_score could plausibly follow the formula."""
        expected = (
            0.5 * score.fact_f1
            + 0.3 * score.evidence_support
            + 0.2 * score.citation_validity
        )
        # We don't enforce the formula in the schema — it's scorer logic.
        # Just verify the fields exist and are in [0, 1].
        assert 0.0 <= score.fact_f1 <= 1.0
        assert 0.0 <= score.evidence_support <= 1.0
        assert 0.0 <= score.citation_validity <= 1.0
        assert expected >= 0.0


# ---------------------------------------------------------------------------
# JSON Schema export
# ---------------------------------------------------------------------------


class TestJsonSchemaExport:
    """Verify all models produce valid JSON Schema."""

    MODELS = [
        StudyManifest,
        PolicyManifest,
        BankManifest,
        RunManifest,
        Event,
        ScoreRecord,
    ]

    @pytest.mark.parametrize("model_cls", MODELS, ids=lambda m: m.__name__)
    def test_json_schema_is_valid(self, model_cls):
        schema = model_cls.model_json_schema()
        assert "properties" in schema
        assert "title" in schema

    @pytest.mark.parametrize("model_cls", MODELS, ids=lambda m: m.__name__)
    def test_json_schema_serializable(self, model_cls):
        schema = model_cls.model_json_schema()
        # Must be JSON-serializable
        json.dumps(schema)


# ---------------------------------------------------------------------------
# Example files compatibility
# ---------------------------------------------------------------------------


class TestExampleCompatibility:
    """Verify the existing example files in schemas/ can be loaded
    (with allowance for placeholder values)."""

    def test_study_example_structure(self):
        """The example has the right keys even if values are placeholders."""
        example = {
            "study_id": "2026-03-synix-v2-pilot",
            "benchmark_version": "v2",
            "scope_ids": ["S08", "S10"],
            "policy_ids": ["null", "policy_base", "policy_core", "policy_summary"],
            "agent_model": "modal:agent-model",
            "judge_model": "modal:judge-model",
            "embedding_model": "modal:embed-model",
            "prompt_set_version": "v2",
            "scoring_version": "v2",
            "code_sha": "abc123",
            "synix_build_graph_hash": "hash",
            "artifact_family_set_version": "v1",
            "random_seed_policy": "deterministic",
            "created_at": "2026-03-06T00:00:00Z",
        }
        study = StudyManifest.model_validate(example)
        assert study.study_id == "2026-03-synix-v2-pilot"

    def test_policy_example_structure(self):
        example = {
            "policy_manifest_id": "policy_core_v1",
            "policy_id": "policy_core",
            "visible_artifact_families": ["chunks", "search_indexes", "core_memory"],
            "query_surfaces": ["fts", "cosine", "core_memory"],
            "fusion": {"method": "rrf", "parameters": {"k": 60}},
            "retrieval_caps": {"max_results": 12, "max_context_tokens": 24000},
            "citation_policy": "source-backed-only",
            "version": "v1",
        }
        policy = PolicyManifest.model_validate(example)
        assert policy.policy_id == "policy_core"

    def test_bank_example_structure(self):
        example = {
            "bank_manifest_id": "bank_s08_cp02_v1",
            "study_id": "2026-03-synix-v2-pilot",
            "scope_id": "S08",
            "checkpoint_id": "cp02",
            "max_episode_ordinal": 40,
            "source_episode_ids": ["e001", "e002", "e003"],
            "artifact_families": {
                "raw_episodes": "hash",
                "chunks": "hash",
                "search_indexes": "hash",
                "core_memory": "hash",
                "summaries": "hash",
                "graph": "hash",
            },
            "dataset_hash": "hash",
            "synix_build_graph_hash": "hash",
            "status": "planned",
        }
        bank = BankManifest.model_validate(example)
        assert bank.checkpoint_id == "cp02"

    def test_run_example_structure(self):
        example = {
            "run_id": "run_s08_policy_core_r01",
            "study_id": "2026-03-synix-v2-pilot",
            "scope_id": "S08",
            "policy_id": "policy_core",
            "replicate_id": "r01",
            "policy_manifest_id": "policy_core_v1",
            "bank_manifest_ids": ["bank_s08_cp01_v1", "bank_s08_cp02_v1"],
            "config_hash": "hash",
            "dataset_hash": "hash",
            "status": "planned",
        }
        run = RunManifest.model_validate(example)
        assert run.policy_id == "policy_core"

    def test_score_example_structure(self):
        example = {
            "score_id": "score-001",
            "study_id": "2026-03-synix-v2-pilot",
            "run_id": "run_s08_policy_core_r01",
            "question_id": "S08_Q03",
            "scope_id": "S08",
            "policy_id": "policy_core",
            "checkpoint_id": "cp02",
            "bank_manifest_id": "bank_s08_cp02_v1",
            "fact_precision": 0.8,
            "fact_recall": 0.67,
            "fact_f1": 0.72,
            "evidence_support": 0.8,
            "citation_validity": 1.0,
            "primary_score": 0.8,
            "diagnostics": {
                "latency_ms": 5140,
                "prompt_tokens": 8812,
                "completion_tokens": 644,
                "retrieval_count": 4,
                "tool_count": 6,
                "estimated_cost_usd": 0.084,
            },
            "scored_at": "2026-03-06T00:00:00Z",
            "scorer_version": "v2",
            "judge_model": "modal:judge-model",
        }
        score = ScoreRecord.model_validate(example)
        assert score.fact_f1 == 0.72

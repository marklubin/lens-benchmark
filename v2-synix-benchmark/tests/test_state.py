"""Tests for the V2 state store, resume, and replay primitives."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from bench.schemas import (
    BankManifest,
    BankStatus,
    Diagnostics,
    Event,
    EventType,
    FusionConfig,
    PolicyManifest,
    RetrievalCaps,
    RunManifest,
    RunStatus,
    ScoreRecord,
    StudyManifest,
)
from bench.state import EventWriter, StateStore

NOW = datetime(2026, 3, 10, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def store(tmp_path):
    s = StateStore(tmp_path / "state.sqlite")
    yield s
    s.close()


@pytest.fixture
def study():
    return StudyManifest(
        study_id="test-study",
        scope_ids=["S08", "S10"],
        policy_ids=["null", "policy_base"],
        agent_model="modal:test",
        judge_model="modal:test",
        embedding_model="modal:test",
        prompt_set_version="v2",
        scoring_version="v2",
        code_sha="abc123",
        artifact_family_set_version="v1",
        created_at=NOW,
    )


@pytest.fixture
def policy():
    return PolicyManifest(
        policy_manifest_id="policy_base_v1",
        policy_id="policy_base",
        visible_artifact_families=["chunks", "search_indexes"],
        query_surfaces=["fts", "cosine"],
        fusion=FusionConfig(method="rrf", parameters={"k": 60}),
        retrieval_caps=RetrievalCaps(max_results=12, max_context_tokens=8000),
        version="v1",
    )


@pytest.fixture
def bank():
    return BankManifest(
        bank_manifest_id="bank_S08_cp01_v1",
        study_id="test-study",
        scope_id="S08",
        checkpoint_id="cp01",
        max_episode_ordinal=6,
        source_episode_ids=["S08_ep_001", "S08_ep_002"],
        artifact_families={
            "raw_episodes": None,
            "chunks": None,
            "search_indexes": None,
            "core_memory": None,
        },
        dataset_hash="sha256:dataset",
    )


@pytest.fixture
def run():
    return RunManifest(
        run_id="run_S08_base_r01",
        study_id="test-study",
        scope_id="S08",
        policy_id="policy_base",
        replicate_id="r01",
        policy_manifest_id="policy_base_v1",
        bank_manifest_ids=["bank_S08_cp01_v1"],
        config_hash="sha256:config",
        dataset_hash="sha256:dataset",
    )


@pytest.fixture
def score():
    return ScoreRecord(
        score_id="score-001",
        study_id="test-study",
        run_id="run_S08_base_r01",
        question_id="S08_Q01",
        scope_id="S08",
        policy_id="policy_base",
        checkpoint_id="cp01",
        bank_manifest_id="bank_S08_cp01_v1",
        fact_precision=0.8,
        fact_recall=0.67,
        fact_f1=0.72,
        evidence_support=0.8,
        citation_validity=1.0,
        primary_score=0.8,
        diagnostics=Diagnostics(latency_ms=1200),
        scored_at=NOW,
        scorer_version="v2",
        judge_model="modal:test",
    )


# ---------------------------------------------------------------------------
# Schema initialization
# ---------------------------------------------------------------------------


class TestInit:
    def test_creates_db(self, tmp_path):
        db_path = tmp_path / "sub" / "state.sqlite"
        store = StateStore(db_path)
        assert db_path.exists()
        store.close()

    def test_wal_mode(self, store):
        row = store._conn.execute("PRAGMA journal_mode").fetchone()
        assert row[0] == "wal"

    def test_idempotent_init(self, tmp_path):
        db_path = tmp_path / "state.sqlite"
        s1 = StateStore(db_path)
        s1.close()
        s2 = StateStore(db_path)
        s2.close()


# ---------------------------------------------------------------------------
# Study CRUD
# ---------------------------------------------------------------------------


class TestStudy:
    def test_save_and_get(self, store, study):
        store.save_study(study)
        loaded = store.get_study("test-study")
        assert loaded is not None
        assert loaded.study_id == study.study_id
        assert loaded.scope_ids == study.scope_ids

    def test_get_missing(self, store):
        assert store.get_study("nonexistent") is None

    def test_overwrite(self, store, study):
        store.save_study(study)
        study.notes = "updated"
        store.save_study(study)
        loaded = store.get_study("test-study")
        assert loaded.notes == "updated"


# ---------------------------------------------------------------------------
# Policy CRUD
# ---------------------------------------------------------------------------


class TestPolicy:
    def test_save_and_get(self, store, policy):
        store.save_policy(policy)
        loaded = store.get_policy("policy_base_v1")
        assert loaded is not None
        assert loaded.policy_id == "policy_base"

    def test_get_missing(self, store):
        assert store.get_policy("nonexistent") is None


# ---------------------------------------------------------------------------
# Bank CRUD
# ---------------------------------------------------------------------------


class TestBank:
    def test_save_and_get(self, store, bank):
        store.save_bank(bank)
        loaded = store.get_bank("bank_S08_cp01_v1")
        assert loaded is not None
        assert loaded.checkpoint_id == "cp01"

    def test_get_missing(self, store):
        assert store.get_bank("nonexistent") is None

    def test_get_banks_for_scope(self, store, bank):
        store.save_bank(bank)

        bank2 = bank.model_copy(update={
            "bank_manifest_id": "bank_S08_cp02_v1",
            "checkpoint_id": "cp02",
            "max_episode_ordinal": 12,
        })
        store.save_bank(bank2)

        banks = store.get_banks_for_scope("test-study", "S08")
        assert len(banks) == 2
        assert banks[0].checkpoint_id == "cp01"
        assert banks[1].checkpoint_id == "cp02"

    def test_status_update(self, store, bank):
        store.save_bank(bank)
        assert store.get_bank(bank.bank_manifest_id).status == BankStatus.planned

        bank.status = BankStatus.building
        store.save_bank(bank)
        assert store.get_bank(bank.bank_manifest_id).status == BankStatus.building

        bank.status = BankStatus.released
        bank.artifact_families["chunks"] = "sha256:chunks"
        store.save_bank(bank)
        loaded = store.get_bank(bank.bank_manifest_id)
        assert loaded.status == BankStatus.released
        assert loaded.artifact_families["chunks"] == "sha256:chunks"


# ---------------------------------------------------------------------------
# Run CRUD
# ---------------------------------------------------------------------------


class TestRun:
    def test_save_and_get(self, store, run):
        store.save_run(run)
        loaded = store.get_run("run_S08_base_r01")
        assert loaded is not None
        assert loaded.policy_id == "policy_base"

    def test_get_missing(self, store):
        assert store.get_run("nonexistent") is None

    def test_get_runs_for_study(self, store, run):
        store.save_run(run)
        run2 = run.model_copy(update={
            "run_id": "run_S08_core_r01",
            "policy_id": "policy_core",
        })
        store.save_run(run2)

        runs = store.get_runs_for_study("test-study")
        assert len(runs) == 2

    def test_resume_cursor_update(self, store, run):
        store.save_run(run)

        run.status = RunStatus.running
        run.last_completed_checkpoint = "cp01"
        run.last_completed_question = "S08_Q02"
        store.save_run(run)

        loaded = store.get_run(run.run_id)
        assert loaded.status == RunStatus.running
        assert loaded.last_completed_checkpoint == "cp01"
        assert loaded.last_completed_question == "S08_Q02"


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------


class TestEvents:
    def test_append_and_get(self, store):
        event = Event(
            event_id="evt-001",
            timestamp=NOW,
            study_id="test-study",
            event_type=EventType.run_started,
            payload={"test": True},
        )
        store.append_event(event)

        events = store.get_events(study_id="test-study")
        assert len(events) == 1
        assert events[0].event_id == "evt-001"
        assert events[0].payload == {"test": True}

    def test_append_order_preserved(self, store):
        for i in range(5):
            store.append_event(Event(
                event_id=f"evt-{i:03d}",
                timestamp=NOW,
                study_id="test-study",
                event_type=EventType.question_completed,
                payload={"index": i},
            ))

        events = store.get_events(study_id="test-study")
        assert len(events) == 5
        assert [e.payload["index"] for e in events] == [0, 1, 2, 3, 4]

    def test_filter_by_run_id(self, store):
        store.append_event(Event(
            event_id="evt-run1",
            timestamp=NOW,
            study_id="test-study",
            run_id="run-1",
            event_type=EventType.run_started,
        ))
        store.append_event(Event(
            event_id="evt-run2",
            timestamp=NOW,
            study_id="test-study",
            run_id="run-2",
            event_type=EventType.run_started,
        ))

        events = store.get_events(run_id="run-1")
        assert len(events) == 1
        assert events[0].run_id == "run-1"

    def test_filter_by_event_type(self, store):
        store.append_event(Event(
            event_id="evt-a",
            timestamp=NOW,
            study_id="test-study",
            event_type=EventType.run_started,
        ))
        store.append_event(Event(
            event_id="evt-b",
            timestamp=NOW,
            study_id="test-study",
            event_type=EventType.run_completed,
        ))

        events = store.get_events(
            study_id="test-study",
            event_type=EventType.run_completed,
        )
        assert len(events) == 1
        assert events[0].event_type == EventType.run_completed

    def test_duplicate_event_id_raises(self, store):
        event = Event(
            event_id="evt-dup",
            timestamp=NOW,
            study_id="test-study",
            event_type=EventType.run_started,
        )
        store.append_event(event)
        with pytest.raises(Exception):
            store.append_event(event)


# ---------------------------------------------------------------------------
# Answers
# ---------------------------------------------------------------------------


class TestAnswers:
    def test_save_and_get(self, store):
        answer = {"answer_text": "The cause is...", "refs_cited": ["chunk_001"]}
        store.save_answer("run-1", "Q01", "cp01", answer)

        loaded = store.get_answer("run-1", "Q01")
        assert loaded is not None
        assert loaded["answer_text"] == "The cause is..."

    def test_get_missing(self, store):
        assert store.get_answer("run-1", "Q99") is None

    def test_get_all_answers(self, store):
        store.save_answer("run-1", "Q01", "cp01", {"answer_text": "a1"})
        store.save_answer("run-1", "Q02", "cp01", {"answer_text": "a2"})
        store.save_answer("run-1", "Q03", "cp02", {"answer_text": "a3"})

        answers = store.get_answers("run-1")
        assert len(answers) == 3

    def test_overwrite_answer(self, store):
        store.save_answer("run-1", "Q01", "cp01", {"answer_text": "first"})
        store.save_answer("run-1", "Q01", "cp01", {"answer_text": "second"})

        loaded = store.get_answer("run-1", "Q01")
        assert loaded["answer_text"] == "second"


# ---------------------------------------------------------------------------
# Scores
# ---------------------------------------------------------------------------


class TestScores:
    def test_save_and_get(self, store, score):
        store.save_score(score)
        loaded = store.get_score("run_S08_base_r01", "S08_Q01")
        assert loaded is not None
        assert loaded.fact_f1 == 0.72

    def test_get_missing(self, store):
        assert store.get_score("run-1", "Q99") is None

    def test_get_all_scores(self, store, score):
        store.save_score(score)

        score2 = score.model_copy(update={
            "score_id": "score-002",
            "question_id": "S08_Q02",
            "fact_f1": 0.50,
            "primary_score": 0.55,
        })
        store.save_score(score2)

        scores = store.get_scores("run_S08_base_r01")
        assert len(scores) == 2


# ---------------------------------------------------------------------------
# Resume: completed questions
# ---------------------------------------------------------------------------


class TestResumeQuestions:
    def test_no_answers_means_empty(self, store):
        assert store.get_completed_questions("run-1") == set()

    def test_tracks_completed(self, store):
        store.save_answer("run-1", "Q01", "cp01", {"answer_text": "a"})
        store.save_answer("run-1", "Q02", "cp01", {"answer_text": "b"})

        completed = store.get_completed_questions("run-1")
        assert completed == {"Q01", "Q02"}

    def test_is_question_completed(self, store):
        assert not store.is_question_completed("run-1", "Q01")
        store.save_answer("run-1", "Q01", "cp01", {"answer_text": "a"})
        assert store.is_question_completed("run-1", "Q01")

    def test_different_runs_isolated(self, store):
        store.save_answer("run-1", "Q01", "cp01", {"answer_text": "a"})
        store.save_answer("run-2", "Q01", "cp01", {"answer_text": "b"})
        store.save_answer("run-2", "Q02", "cp01", {"answer_text": "c"})

        assert store.get_completed_questions("run-1") == {"Q01"}
        assert store.get_completed_questions("run-2") == {"Q01", "Q02"}


# ---------------------------------------------------------------------------
# Resume: completed bank families
# ---------------------------------------------------------------------------


class TestResumeBankFamilies:
    def test_no_bank_means_empty(self, store):
        assert store.get_completed_families("nonexistent") == set()

    def test_all_none_means_empty(self, store, bank):
        store.save_bank(bank)
        assert store.get_completed_families(bank.bank_manifest_id) == set()

    def test_partial_build(self, store, bank):
        bank.artifact_families["raw_episodes"] = "sha256:raw"
        bank.artifact_families["chunks"] = "sha256:chunks"
        store.save_bank(bank)

        completed = store.get_completed_families(bank.bank_manifest_id)
        assert completed == {"raw_episodes", "chunks"}

    def test_full_build(self, store, bank):
        for family in bank.artifact_families:
            bank.artifact_families[family] = f"sha256:{family}"
        store.save_bank(bank)

        completed = store.get_completed_families(bank.bank_manifest_id)
        assert completed == set(bank.artifact_families.keys())


# ---------------------------------------------------------------------------
# Resume: simulate failure and recovery
# ---------------------------------------------------------------------------


class TestFailureAndResume:
    def test_run_failure_mid_checkpoint(self, store, run):
        """Simulate: run crashes after Q01 but before Q02 at cp01."""
        run.status = RunStatus.running
        store.save_run(run)

        # Q01 completed
        store.save_answer(run.run_id, "Q01", "cp01", {"answer_text": "a1"})
        run.last_completed_question = "Q01"
        store.save_run(run)

        # CRASH HERE — Q02 never saved

        # Resume: load run state
        loaded = store.get_run(run.run_id)
        assert loaded.status == RunStatus.running
        assert loaded.last_completed_question == "Q01"

        # Check what's done
        completed = store.get_completed_questions(run.run_id)
        assert "Q01" in completed
        assert "Q02" not in completed

        # Skip Q01, do Q02
        assert store.is_question_completed(run.run_id, "Q01")
        assert not store.is_question_completed(run.run_id, "Q02")

        # Complete Q02
        store.save_answer(run.run_id, "Q02", "cp01", {"answer_text": "a2"})
        run.last_completed_question = "Q02"
        run.status = RunStatus.completed
        store.save_run(run)

        assert store.get_run(run.run_id).status == RunStatus.completed

    def test_bank_failure_mid_build(self, store, bank):
        """Simulate: bank build crashes after chunks but before search_indexes."""
        bank.status = BankStatus.building
        bank.artifact_families["raw_episodes"] = "sha256:raw"
        bank.artifact_families["chunks"] = "sha256:chunks"
        store.save_bank(bank)

        # CRASH HERE — search_indexes never completed

        # Resume: load bank state
        loaded = store.get_bank(bank.bank_manifest_id)
        assert loaded.status == BankStatus.building

        completed = store.get_completed_families(bank.bank_manifest_id)
        assert "raw_episodes" in completed
        assert "chunks" in completed
        assert "search_indexes" not in completed

        # Complete remaining families
        bank.artifact_families["search_indexes"] = "sha256:search"
        bank.artifact_families["core_memory"] = "sha256:core"
        bank.status = BankStatus.released
        store.save_bank(bank)

        assert store.get_completed_families(bank.bank_manifest_id) == {
            "raw_episodes", "chunks", "search_indexes", "core_memory"
        }


# ---------------------------------------------------------------------------
# Replay: re-score from saved answers
# ---------------------------------------------------------------------------


class TestReplay:
    def test_rescore_from_saved_answers(self, store, run, score):
        """Verify that a completed run can be rescored from saved answers only."""
        run.status = RunStatus.completed
        store.save_run(run)

        # Save original answer
        store.save_answer(
            run.run_id, "S08_Q01", "cp01",
            {"answer_text": "The cause is...", "refs_cited": ["chunk_001"]},
        )

        # Save original score
        store.save_score(score)

        # Replay: load answer, re-score, compare
        answer = store.get_answer(run.run_id, "S08_Q01")
        assert answer is not None

        original_score = store.get_score(run.run_id, "S08_Q01")
        assert original_score is not None
        assert original_score.fact_f1 == 0.72

        # In real replay: re-run scoring pipeline against saved answer
        # Verify: same inputs → same score (deterministic judge via cache)

    def test_all_answers_available_for_replay(self, store, run):
        """All answers from a completed run must be loadable."""
        run.status = RunStatus.completed
        store.save_run(run)

        questions = ["Q01", "Q02", "Q03"]
        for q in questions:
            store.save_answer(run.run_id, q, "cp01", {"answer_text": f"answer_{q}"})

        answers = store.get_answers(run.run_id)
        assert len(answers) == 3
        assert all("answer_text" in a for a in answers)


# ---------------------------------------------------------------------------
# EventWriter
# ---------------------------------------------------------------------------


class TestEventWriter:
    def test_emit(self, store):
        writer = EventWriter(store, study_id="test-study")
        event = writer.emit(
            EventType.run_started,
            run_id="run-1",
            scope_id="S08",
            policy_id="policy_base",
            payload={"test": True},
        )

        assert event.study_id == "test-study"
        assert event.run_id == "run-1"
        assert event.event_type == EventType.run_started

        # Verify persisted
        events = store.get_events(run_id="run-1")
        assert len(events) == 1
        assert events[0].event_id == event.event_id

    def test_auto_generates_id_and_timestamp(self, store):
        writer = EventWriter(store, study_id="test-study")
        event = writer.emit(EventType.bank_build_started)

        assert event.event_id  # non-empty
        assert event.timestamp is not None

    def test_error_field(self, store):
        writer = EventWriter(store, study_id="test-study")
        event = writer.emit(
            EventType.run_failed,
            run_id="run-1",
            error="Connection timeout to Modal endpoint",
        )

        loaded = store.get_events(run_id="run-1")[0]
        assert loaded.error == "Connection timeout to Modal endpoint"

    def test_multiple_events_append(self, store):
        writer = EventWriter(store, study_id="test-study")
        writer.emit(EventType.run_started, run_id="run-1")
        writer.emit(EventType.checkpoint_started, run_id="run-1")
        writer.emit(EventType.question_started, run_id="run-1")

        events = store.get_events(run_id="run-1")
        assert len(events) == 3
        types = [e.event_type for e in events]
        assert types == [
            EventType.run_started,
            EventType.checkpoint_started,
            EventType.question_started,
        ]

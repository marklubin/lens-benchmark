"""Tests for dataset loader and bank builder (T005)."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from bench.bank import BankBuilder, _make_pipeline
from bench.dataset import (
    EpisodeData,
    QuestionData,
    ScopeData,
    load_scope,
)
from bench.schemas import BankManifest, BankStatus, EventType, StudyManifest
from bench.state import EventWriter, StateStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def store(tmp_dir):
    s = StateStore(tmp_dir / "state.db")
    yield s
    s.close()


@pytest.fixture
def mock_broker():
    broker = MagicMock()
    broker._llm_base_url = "http://localhost:8000/v1"
    broker._llm_api_key = "test-key"
    broker._embed_base_url = "http://localhost:8001"
    return broker


@pytest.fixture
def event_writer(store):
    return EventWriter(store, "test-study")


def _make_scope_dir(base: Path, *, n_signal: int = 6, n_distractor: int = 6) -> Path:
    """Create a minimal scope directory with spec + episodes + questions."""
    scope_dir = base / "test_scope"
    ep_dir = scope_dir / "generated" / "episodes"
    ep_dir.mkdir(parents=True)

    # spec.yaml
    spec = {
        "scope_id": "test_scope_01",
        "domain": "test",
        "description": "Test scope",
        "episodes": {"count": n_signal},
    }
    with open(scope_dir / "spec.yaml", "w") as f:
        yaml.dump(spec, f)

    # Signal episodes
    for i in range(1, n_signal + 1):
        (ep_dir / f"signal_{i:03d}.txt").write_text(f"Signal episode {i} content. Important metric: {i * 10}.")

    # Distractor episodes
    themes = ["alpha", "beta"]
    for i in range(1, n_distractor + 1):
        theme = themes[(i - 1) % len(themes)]
        (ep_dir / f"distractor_{theme}_{i:03d}.txt").write_text(f"Distractor {theme} episode {i}.")

    # questions.json
    questions = [
        {
            "question_id": "q01",
            "scope_id": "test_scope_01",
            "checkpoint_after": 3,
            "question_type": "longitudinal",
            "prompt": "What happened in the first three episodes?",
            "ground_truth": {
                "canonical_answer": "Metrics increased from 10 to 30.",
                "key_facts": ["metric was 10", "metric was 20", "metric was 30"],
                "required_evidence_refs": ["test_scope_01_ep_001", "test_scope_01_ep_003"],
            },
        },
        {
            "question_id": "q02",
            "scope_id": "test_scope_01",
            "checkpoint_after": 3,
            "question_type": "null_hypothesis",
            "prompt": "Was there anything unusual?",
            "ground_truth": {
                "canonical_answer": "No unusual activity.",
                "key_facts": ["no anomalies detected"],
                "required_evidence_refs": [],
            },
        },
        {
            "question_id": "q03",
            "scope_id": "test_scope_01",
            "checkpoint_after": 6,
            "question_type": "longitudinal",
            "prompt": "What is the trend over all episodes?",
            "ground_truth": {
                "canonical_answer": "Metrics increased from 10 to 60.",
                "key_facts": ["metric started at 10", "metric ended at 60", "steady increase"],
                "required_evidence_refs": ["test_scope_01_ep_001", "test_scope_01_ep_006"],
            },
        },
    ]
    with open(scope_dir / "generated" / "questions.json", "w") as f:
        json.dump(questions, f)

    return scope_dir


# ---------------------------------------------------------------------------
# Dataset Loader Tests
# ---------------------------------------------------------------------------


class TestDatasetLoader:
    def test_load_scope_basic(self, tmp_dir):
        scope_dir = _make_scope_dir(tmp_dir)
        scope = load_scope(scope_dir)

        assert scope.scope_id == "test_scope_01"
        assert len(scope.episodes) == 12  # 6 signal + 6 distractor
        assert len(scope.questions) == 3
        assert scope.checkpoints == [3, 6]

    def test_episodes_signal_vs_distractor(self, tmp_dir):
        scope_dir = _make_scope_dir(tmp_dir)
        scope = load_scope(scope_dir)

        signal = [e for e in scope.episodes if not e.is_distractor]
        distractors = [e for e in scope.episodes if e.is_distractor]

        assert len(signal) == 6
        assert len(distractors) == 6
        assert all(e.ordinal >= 1 for e in signal)
        assert signal[0].content.startswith("Signal episode 1")

    def test_episode_ids_format(self, tmp_dir):
        scope_dir = _make_scope_dir(tmp_dir)
        scope = load_scope(scope_dir)

        signal_ids = {e.episode_id for e in scope.episodes if not e.is_distractor}
        assert "test_scope_01_ep_001" in signal_ids
        assert "test_scope_01_ep_006" in signal_ids

    def test_questions_parsed(self, tmp_dir):
        scope_dir = _make_scope_dir(tmp_dir)
        scope = load_scope(scope_dir)

        q1 = scope.questions[0]
        assert q1.question_id == "q01"
        assert q1.checkpoint_after == 3
        assert q1.question_type == "longitudinal"
        assert "key_facts" in q1.ground_truth

    def test_episodes_up_to_checkpoint(self, tmp_dir):
        scope_dir = _make_scope_dir(tmp_dir)
        scope = load_scope(scope_dir)

        # At checkpoint 3: 3 signal + 3 distractors (proportional)
        eps_cp3 = scope.episodes_up_to(3)
        signal_cp3 = [e for e in eps_cp3 if not e.is_distractor]
        distractor_cp3 = [e for e in eps_cp3 if e.is_distractor]

        assert len(signal_cp3) == 3
        assert len(distractor_cp3) == 3
        assert all(e.ordinal <= 3 for e in signal_cp3)

    def test_episodes_up_to_full(self, tmp_dir):
        scope_dir = _make_scope_dir(tmp_dir)
        scope = load_scope(scope_dir)

        # At checkpoint 6: all episodes
        eps_cp6 = scope.episodes_up_to(6)
        assert len(eps_cp6) == 12  # 6 signal + 6 distractor

    def test_questions_at_checkpoint(self, tmp_dir):
        scope_dir = _make_scope_dir(tmp_dir)
        scope = load_scope(scope_dir)

        qs_cp3 = scope.questions_at(3)
        assert len(qs_cp3) == 2
        assert all(q.checkpoint_after == 3 for q in qs_cp3)

        qs_cp6 = scope.questions_at(6)
        assert len(qs_cp6) == 1

    def test_dataset_hash_stable(self, tmp_dir):
        scope_dir = _make_scope_dir(tmp_dir)
        scope1 = load_scope(scope_dir)
        scope2 = load_scope(scope_dir)

        assert scope1.dataset_hash == scope2.dataset_hash
        assert len(scope1.dataset_hash) == 16


# ---------------------------------------------------------------------------
# Pipeline Construction Tests
# ---------------------------------------------------------------------------


class TestPipelineConstruction:
    def test_base_pipeline_layers(self):
        pipeline = _make_pipeline(
            "test_scope",
            6,
            families=["chunks"],
            llm_config={"provider": "openai-compatible", "model": "test"},
            embedding_config={"provider": "fastembed", "model": "test"},
        )

        layer_names = [l.name for l in pipeline.layers]
        assert "episodes" in layer_names
        assert "chunks" in layer_names

        surface_names = [s.name for s in pipeline.surfaces]
        assert "search" in surface_names

        projection_names = [p.name for p in pipeline.projections]
        assert "search_output" in projection_names

    def test_pipeline_with_core_memory(self):
        pipeline = _make_pipeline(
            "test_scope",
            6,
            families=["chunks", "core_memory"],
            llm_config={"provider": "openai-compatible", "model": "test"},
            embedding_config={"provider": "fastembed", "model": "test"},
        )

        layer_names = [l.name for l in pipeline.layers]
        assert "core-memory" in layer_names

    def test_pipeline_with_summary(self):
        pipeline = _make_pipeline(
            "test_scope",
            6,
            families=["chunks", "summary"],
            llm_config={"provider": "openai-compatible", "model": "test"},
            embedding_config={"provider": "fastembed", "model": "test"},
        )

        layer_names = [l.name for l in pipeline.layers]
        assert "episode-groups" in layer_names
        assert "summary" in layer_names

    def test_pipeline_with_all_families(self):
        pipeline = _make_pipeline(
            "test_scope",
            6,
            families=["chunks", "core_memory", "summary"],
            llm_config={"provider": "openai-compatible", "model": "test"},
            embedding_config={"provider": "fastembed", "model": "test"},
        )

        layer_names = [l.name for l in pipeline.layers]
        assert "episodes" in layer_names
        assert "chunks" in layer_names
        assert "core-memory" in layer_names
        assert "episode-groups" in layer_names
        assert "summary" in layer_names


# ---------------------------------------------------------------------------
# Bank Builder Tests
# ---------------------------------------------------------------------------


class TestBankBuilder:
    def test_resume_skips_released_bank(self, tmp_dir, store, mock_broker, event_writer):
        """Already-released bank should be returned without rebuilding."""
        scope_dir = _make_scope_dir(tmp_dir)
        scope = load_scope(scope_dir)

        # Pre-save a released bank
        bank = BankManifest(
            bank_manifest_id=f"bank-{scope.scope_id}-cp03-test-stu",
            study_id="test-study",
            scope_id=scope.scope_id,
            checkpoint_id="cp03",
            max_episode_ordinal=3,
            source_episode_ids=["ep_001", "ep_002", "ep_003"],
            artifact_families={"chunks": "abc123"},
            dataset_hash=scope.dataset_hash,
            status=BankStatus.released,
        )
        store.save_bank(bank)

        builder = BankBuilder(store, mock_broker, tmp_dir / "work")
        result = builder._build_checkpoint_bank(
            study_id="test-study",
            scope=scope,
            checkpoint=3,
            families=["chunks"],
            event_writer=event_writer,
        )

        assert result.status == BankStatus.released
        assert result.bank_manifest_id == bank.bank_manifest_id

    def test_bank_manifest_saved_to_store(self, tmp_dir, store, mock_broker, event_writer):
        """After build, bank manifest should be persisted in state store."""
        scope_dir = _make_scope_dir(tmp_dir)
        scope = load_scope(scope_dir)

        # Mock Synix SDK calls
        mock_project = MagicMock()
        mock_project.build.return_value = MagicMock(
            built=10,
            cached=0,
            skipped=0,
            total_time=1.5,
            snapshot_oid="snap-123",
            manifest_oid="manifest-456",
        )
        mock_source = MagicMock()
        mock_project.source.return_value = mock_source

        with patch("bench.bank.synix") as mock_synix:
            mock_synix.init.return_value = mock_project
            mock_synix.open_project.return_value = mock_project

            builder = BankBuilder(store, mock_broker, tmp_dir / "work")
            result = builder._build_checkpoint_bank(
                study_id="test-study",
                scope=scope,
                checkpoint=3,
                families=["chunks"],
                event_writer=event_writer,
            )

        assert result.status == BankStatus.released
        assert result.synix_build_ref == "snap-123"
        assert result.max_episode_ordinal == 3

        # Verify persisted
        saved = store.get_bank(result.bank_manifest_id)
        assert saved is not None
        assert saved.status == BankStatus.released

    def test_checkpoint_isolation_episode_count(self, tmp_dir, store, mock_broker, event_writer):
        """Bank for checkpoint 3 should only ingest 3 signal + 3 distractor episodes."""
        scope_dir = _make_scope_dir(tmp_dir)
        scope = load_scope(scope_dir)

        mock_project = MagicMock()
        mock_project.build.return_value = MagicMock(
            built=6, cached=0, skipped=0, total_time=1.0,
            snapshot_oid="snap-cp3", manifest_oid="m-cp3",
        )
        mock_source = MagicMock()
        mock_project.source.return_value = mock_source

        with patch("bench.bank.synix") as mock_synix:
            mock_synix.init.return_value = mock_project

            builder = BankBuilder(store, mock_broker, tmp_dir / "work")
            result = builder._build_checkpoint_bank(
                study_id="test-study",
                scope=scope,
                checkpoint=3,
                families=["chunks"],
                event_writer=event_writer,
            )

        # Check that source.add was called exactly 6 times (3 signal + 3 distractor)
        # bank.py uses source.add(filepath) when the original .txt file exists
        assert mock_source.add.call_count == 6
        assert result.max_episode_ordinal == 3
        assert len(result.source_episode_ids) == 6

    def test_build_failure_persists_failed_manifest(self, tmp_dir, store, mock_broker, event_writer):
        """Build failure should save a failed manifest and re-raise."""
        scope_dir = _make_scope_dir(tmp_dir)
        scope = load_scope(scope_dir)

        mock_project = MagicMock()
        mock_project.build.side_effect = RuntimeError("LLM timeout")
        mock_project.source.return_value = MagicMock()

        with patch("bench.bank.synix") as mock_synix:
            mock_synix.init.return_value = mock_project

            builder = BankBuilder(store, mock_broker, tmp_dir / "work")
            with pytest.raises(RuntimeError, match="LLM timeout"):
                builder._build_checkpoint_bank(
                    study_id="test-study",
                    scope=scope,
                    checkpoint=3,
                    families=["chunks"],
                    event_writer=event_writer,
                )

        # Failed manifest should be persisted
        bank_id = f"bank-{scope.scope_id}-cp03-test-stu"
        saved = store.get_bank(bank_id)
        assert saved is not None
        assert saved.status == BankStatus.failed

    def test_build_events_emitted(self, tmp_dir, store, mock_broker, event_writer):
        """Build should emit bank_build_started and bank_build_completed events."""
        scope_dir = _make_scope_dir(tmp_dir)
        scope = load_scope(scope_dir)

        mock_project = MagicMock()
        mock_project.build.return_value = MagicMock(
            built=6, cached=0, skipped=0, total_time=1.0,
            snapshot_oid="snap-123", manifest_oid="m-123",
        )
        mock_project.source.return_value = MagicMock()

        with patch("bench.bank.synix") as mock_synix:
            mock_synix.init.return_value = mock_project

            builder = BankBuilder(store, mock_broker, tmp_dir / "work")
            builder._build_checkpoint_bank(
                study_id="test-study",
                scope=scope,
                checkpoint=3,
                families=["chunks"],
                event_writer=event_writer,
            )

        events = store.get_events(study_id="test-study")
        event_types = [e.event_type for e in events]
        assert EventType.bank_build_started in event_types
        assert EventType.bank_build_completed in event_types

    def test_build_scope_banks_all_checkpoints(self, tmp_dir, store, mock_broker, event_writer):
        """build_scope_banks should produce one manifest per checkpoint."""
        scope_dir = _make_scope_dir(tmp_dir)
        scope = load_scope(scope_dir)

        mock_project = MagicMock()
        mock_project.build.return_value = MagicMock(
            built=6, cached=0, skipped=0, total_time=1.0,
            snapshot_oid="snap-all", manifest_oid="m-all",
        )
        mock_project.source.return_value = MagicMock()

        with patch("bench.bank.synix") as mock_synix:
            mock_synix.init.return_value = mock_project

            builder = BankBuilder(store, mock_broker, tmp_dir / "work")
            manifests = builder.build_scope_banks(
                study_id="test-study",
                scope=scope,
                families=["chunks"],
                event_writer=event_writer,
            )

        assert len(manifests) == 2  # checkpoints [3, 6]
        assert manifests[0].checkpoint_id == "cp03"
        assert manifests[1].checkpoint_id == "cp06"

    def test_source_cleared_before_ingest(self, tmp_dir, store, mock_broker, event_writer):
        """Source should be cleared before adding episodes to avoid stale data."""
        scope_dir = _make_scope_dir(tmp_dir)
        scope = load_scope(scope_dir)

        mock_project = MagicMock()
        mock_project.build.return_value = MagicMock(
            built=6, cached=0, skipped=0, total_time=1.0,
            snapshot_oid="snap-123", manifest_oid="m-123",
        )
        mock_source = MagicMock()
        mock_project.source.return_value = mock_source

        with patch("bench.bank.synix") as mock_synix:
            mock_synix.init.return_value = mock_project

            builder = BankBuilder(store, mock_broker, tmp_dir / "work")
            builder._build_checkpoint_bank(
                study_id="test-study",
                scope=scope,
                checkpoint=3,
                families=["chunks"],
                event_writer=event_writer,
            )

        mock_source.clear.assert_called_once()


# ---------------------------------------------------------------------------
# Real V1 Dataset Tests (if available)
# ---------------------------------------------------------------------------


_V1_SCOPE_DIR = Path("/home/mark/lens-benchmark/datasets/scopes/01_cascading_failure")


@pytest.mark.skipif(
    not (_V1_SCOPE_DIR / "generated" / "episodes").is_dir(),
    reason="V1 S01 dataset does not have individual episode files",
)
class TestRealV1Dataset:
    def test_load_s01(self):
        scope = load_scope(_V1_SCOPE_DIR)
        assert scope.scope_id  # should have a scope_id
        assert len(scope.episodes) > 0
        assert len(scope.questions) > 0
        assert len(scope.checkpoints) > 0

    def test_s01_checkpoint_isolation(self):
        scope = load_scope(_V1_SCOPE_DIR)
        first_cp = scope.checkpoints[0]
        eps = scope.episodes_up_to(first_cp)

        signal_eps = [e for e in eps if not e.is_distractor]
        for ep in signal_eps:
            assert ep.ordinal <= first_cp, f"Episode {ep.episode_id} ordinal {ep.ordinal} > checkpoint {first_cp}"


_V1_S08_DIR = Path("/home/mark/lens-benchmark/datasets/scopes/08_corporate_acquisition")


@pytest.mark.skipif(
    not _V1_S08_DIR.exists(),
    reason="V1 S08 dataset not available",
)
class TestRealS08Dataset:
    def test_load_s08(self):
        scope = load_scope(_V1_S08_DIR)
        assert scope.scope_id == "corporate_acquisition_08"
        assert len(scope.episodes) == 40  # 20 signal + 20 distractor
        assert len(scope.questions) == 10
        assert scope.checkpoints == [6, 12, 16, 20]

    def test_s08_checkpoint_episode_counts(self):
        scope = load_scope(_V1_S08_DIR)

        eps_cp6 = scope.episodes_up_to(6)
        signal_cp6 = [e for e in eps_cp6 if not e.is_distractor]
        assert len(signal_cp6) == 6

        eps_cp20 = scope.episodes_up_to(20)
        assert len(eps_cp20) == 40  # all episodes

    def test_s08_questions_at_checkpoints(self):
        scope = load_scope(_V1_S08_DIR)
        assert len(scope.questions_at(6)) == 2
        assert len(scope.questions_at(12)) == 2
        assert len(scope.questions_at(16)) == 2
        assert len(scope.questions_at(20)) == 4

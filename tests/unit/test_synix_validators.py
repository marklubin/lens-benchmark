"""Tests for lens.datagen.synix.validators — synix validator classes.

Uses mock synix types to test validator logic without requiring synix installed.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Add synix directory to path
SYNIX_DIR = str(Path(__file__).resolve().parents[2] / "src" / "lens" / "datagen" / "synix")
sys.path.insert(0, SYNIX_DIR)

import spec_utils  # noqa: E402


# ---------------------------------------------------------------------------
# Mock synix types (plain classes, no dataclass to avoid field() conflicts)
# ---------------------------------------------------------------------------

class MockArtifact:
    def __init__(self, label="", artifact_type="", content="", artifact_id="",
                 input_ids=None, metadata=None):
        self.label = label
        self.artifact_type = artifact_type
        self.content = content
        self.input_ids = input_ids or []
        self.metadata = metadata or {}
        if not artifact_id:
            self.artifact_id = f"sha256:{hashlib.sha256(content.encode()).hexdigest()}"
        else:
            self.artifact_id = artifact_id


class MockViolation:
    def __init__(self, violation_type="", severity="error", message="",
                 label="", field="", metadata=None, provenance_trace=None,
                 violation_id=""):
        self.violation_type = violation_type
        self.severity = severity
        self.message = message
        self.label = label
        self.field = field
        self.metadata = metadata or {}
        self.provenance_trace = provenance_trace or []
        self.violation_id = violation_id


class MockLLMResponse:
    def __init__(self, content="", model="test", input_tokens=0,
                 output_tokens=0, total_tokens=0):
        self.content = content
        self.model = model
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.total_tokens = total_tokens


class MockValidationContext:
    def __init__(self, build_dir: str | None = None):
        self.store = MagicMock()
        self.provenance = MagicMock()
        self.pipeline = MagicMock()
        if build_dir:
            self.pipeline.build_dir = build_dir


# Install synix mocks — v0.11 API: BaseValidator accepts layers in __init__
mock_synix_validators = MagicMock()
mock_synix_validators.BaseValidator = type("BaseValidator", (), {
    "validate": lambda self, artifacts, ctx: [],
    "to_config_dict": lambda self: {},
})
mock_synix_validators.Violation = MockViolation

mock_synix_llm = MagicMock()
mock_synix_llm._get_llm_client = MagicMock()
mock_synix_llm._logged_complete = MagicMock()

mock_synix_models = MagicMock()
mock_synix_models.Artifact = MockArtifact

sys.modules.setdefault("synix", MagicMock())
sys.modules["synix.build"] = MagicMock()
sys.modules["synix.build.validators"] = mock_synix_validators
sys.modules["synix.build.llm_transforms"] = mock_synix_llm
sys.modules["synix.core"] = MagicMock()
sys.modules["synix.core.models"] = mock_synix_models

# Force reimport if already cached
if "validators" in sys.modules:
    del sys.modules["validators"]

import validators as validators_mod  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

MINIMAL_SPEC_RAW = {
    "scope_id": "test_scope",
    "episodes": {
        "count": 10,
        "timeline": {"start": "2024-01-01", "interval": "1d"},
        "target_words": 200,
    },
    "arc": [
        {"id": "baseline", "episodes": "1-5", "signal_density": "none"},
        {"id": "signal", "episodes": "6-10", "signal_density": "low"},
    ],
    "key_facts": [
        {"id": "kf1", "fact": "latency increasing", "first_appears": "signal:1", "reinforced_in": []},
    ],
    "questions": [],
    "distractors": None,
    "scenario": {},
    "noise": {},
}


def _make_spec_artifact(raw=None):
    spec = spec_utils.parse_spec(raw or MINIMAL_SPEC_RAW)
    return MockArtifact(
        label="spec",
        artifact_type="scope_spec",
        content=json.dumps(spec),
    )


def _make_episode(label, text, artifact_type="signal_episode", theme="", metadata=None):
    return MockArtifact(
        label=label,
        artifact_type=artifact_type,
        content=text,
        metadata={
            "episode_type": "signal" if artifact_type == "signal_episode" else "distractor",
            "theme": theme,
            **(metadata or {}),
        },
    )


def _make_question(q_id, q_type, key_facts, checkpoint=8):
    q_data = {
        "question_id": q_id,
        "scope_id": "test_scope",
        "checkpoint_after": checkpoint,
        "question_type": q_type,
        "prompt": f"Question {q_id}?",
        "ground_truth": {
            "canonical_answer": "Answer",
            "required_evidence_refs": [],
            "key_facts": key_facts,
        },
    }
    return MockArtifact(
        label=f"q_{q_id}",
        artifact_type="question",
        content=json.dumps(q_data),
        metadata={"question_id": q_id, "question_type": q_type, "checkpoint_after": checkpoint},
    )


# ---------------------------------------------------------------------------
# WordCount
# ---------------------------------------------------------------------------


class TestWordCount:
    def test_passes_when_above_minimum(self):
        v = validators_mod.WordCount(layers=[], min_words=10)
        ep = _make_episode("ep_001", "word " * 20)
        violations = v.validate([ep], None)
        assert len(violations) == 0

    def test_fails_when_below_minimum(self):
        v = validators_mod.WordCount(layers=[], min_words=350)
        ep = _make_episode("ep_001", "only five words here now")
        violations = v.validate([ep], None)
        assert len(violations) == 1
        assert violations[0].violation_type == "word_count"
        assert violations[0].severity == "error"

    def test_checks_both_signal_and_distractor(self):
        v = validators_mod.WordCount(layers=[], min_words=100)
        signal = _make_episode("ep_001", "short", artifact_type="signal_episode")
        distractor = _make_episode("dx_001", "also short", artifact_type="distractor_episode")
        violations = v.validate([signal, distractor], None)
        assert len(violations) == 2

    def test_ignores_other_artifact_types(self):
        v = validators_mod.WordCount(layers=[], min_words=100)
        spec = MockArtifact(label="spec", artifact_type="scope_spec", content="short")
        violations = v.validate([spec], None)
        assert len(violations) == 0

    def test_default_min_words(self):
        v = validators_mod.WordCount(layers=[])
        ep = _make_episode("ep_001", "word " * 400)
        violations = v.validate([ep], None)
        assert len(violations) == 0

    def test_to_config_dict(self):
        v = validators_mod.WordCount(layers=[], min_words=500, severity="warning")
        cfg = v.to_config_dict()
        assert cfg == {"layers": [], "min_words": 500, "severity": "warning"}


# ---------------------------------------------------------------------------
# ContaminationCheck
# ---------------------------------------------------------------------------


class TestContaminationCheck:
    def test_passes_clean_questions(self, tmp_path):
        v = validators_mod.ContaminationCheck(
            layers=[],
            llm_config={"provider": "openai", "model": "test"},
        )

        spec_art = _make_spec_artifact()
        q = _make_question("q01", "longitudinal", ["latency increasing"])
        # Episodes 6, 7, 8 are within checkpoint (8)
        ep6 = _make_episode("test_scope_ep_006", "Normal looking episode with no relevant info")
        ep7 = _make_episode("test_scope_ep_007", "Another normal episode here")
        ep8 = _make_episode("test_scope_ep_008", "Still nothing relevant")

        mock_client = MagicMock()
        mock_synix_llm._get_llm_client.return_value = mock_client
        # LLM answers don't contain the key fact words
        mock_synix_llm._logged_complete.return_value = MockLLMResponse(
            content="I cannot determine any trends from this single record. The data looks normal."
        )

        ctx = MockValidationContext(str(tmp_path))
        violations = v.validate([spec_art, q, ep6, ep7, ep8], ctx)
        assert len(violations) == 0

    def test_flags_contaminated_question(self, tmp_path):
        v = validators_mod.ContaminationCheck(
            layers=[],
            llm_config={"provider": "openai", "model": "test"},
        )

        spec_art = _make_spec_artifact()
        q = _make_question("q01", "longitudinal", ["latency increasing"])
        ep6 = _make_episode("test_scope_ep_006", "text")

        mock_client = MagicMock()
        mock_synix_llm._get_llm_client.return_value = mock_client
        # LLM answer contains the key fact — contamination!
        mock_synix_llm._logged_complete.return_value = MockLLMResponse(
            content="Based on this record, latency is clearly increasing significantly."
        )

        ctx = MockValidationContext(str(tmp_path))
        violations = v.validate([spec_art, q, ep6], ctx)
        assert len(violations) == 1
        assert violations[0].violation_type == "contamination"

    def test_skips_non_longitudinal_questions(self, tmp_path):
        v = validators_mod.ContaminationCheck(
            layers=[],
            llm_config={"provider": "openai", "model": "test"},
        )

        spec_art = _make_spec_artifact()
        q = _make_question("q02", "null_hypothesis", [])

        ctx = MockValidationContext(str(tmp_path))
        violations = v.validate([spec_art, q], ctx)
        assert len(violations) == 0

    def test_writes_results_file(self, tmp_path):
        v = validators_mod.ContaminationCheck(
            layers=[],
            llm_config={"provider": "openai", "model": "test"},
        )

        spec_art = _make_spec_artifact()
        q = _make_question("q01", "longitudinal", ["latency increasing"])
        ep = _make_episode("test_scope_ep_006", "text")

        mock_synix_llm._get_llm_client.return_value = MagicMock()
        mock_synix_llm._logged_complete.return_value = MockLLMResponse(content="No info here.")

        ctx = MockValidationContext(str(tmp_path))
        v.validate([spec_art, q, ep], ctx)

        results_path = tmp_path / "contamination_results.json"
        assert results_path.exists()
        data = json.loads(results_path.read_text())
        assert "summary" in data
        assert "questions" in data

    def test_to_config_dict(self):
        v = validators_mod.ContaminationCheck(
            layers=[],
            llm_config={"provider": "openai", "model": "test"},
        )
        cfg = v.to_config_dict()
        assert cfg == {"layers": [], "llm_config": {"provider": "openai", "model": "test"}}


# ---------------------------------------------------------------------------
# NaiveBaseline
# ---------------------------------------------------------------------------


class TestNaiveBaseline:
    def test_no_warnings_normal_coverage(self, tmp_path):
        v = validators_mod.NaiveBaseline(
            layers=[],
            llm_config={"provider": "openai", "model": "test"},
            warn_threshold=0.95,
        )

        spec_art = _make_spec_artifact()
        q = _make_question("q01", "longitudinal", ["latency increasing"])
        ep = _make_episode("test_scope_ep_006", "text")

        mock_synix_llm._get_llm_client.return_value = MagicMock()
        # LLM answer partially covers key facts
        mock_synix_llm._logged_complete.return_value = MockLLMResponse(
            content="The system shows some latency changes."
        )

        ctx = MockValidationContext(str(tmp_path))
        violations = v.validate([spec_art, q, ep], ctx)
        # Should not warn since coverage is partial (< 95%)
        assert all(v.severity != "error" for v in violations)

    def test_warns_when_too_easy(self, tmp_path):
        v = validators_mod.NaiveBaseline(
            layers=[],
            llm_config={"provider": "openai", "model": "test"},
            warn_threshold=0.5,
        )

        spec_art = _make_spec_artifact()
        q = _make_question("q01", "longitudinal", ["latency increasing"])
        ep = _make_episode("test_scope_ep_006", "text")

        mock_synix_llm._get_llm_client.return_value = MagicMock()
        # LLM answer fully covers key facts
        mock_synix_llm._logged_complete.return_value = MockLLMResponse(
            content="The latency is clearly increasing across all services."
        )

        ctx = MockValidationContext(str(tmp_path))
        violations = v.validate([spec_art, q, ep], ctx)
        assert len(violations) == 1
        assert violations[0].severity == "warning"
        assert "too easy" in violations[0].message

    def test_writes_results_file(self, tmp_path):
        v = validators_mod.NaiveBaseline(
            layers=[],
            llm_config={"provider": "openai", "model": "test"},
            warn_threshold=0.95,
        )

        spec_art = _make_spec_artifact()
        q = _make_question("q01", "longitudinal", ["latency increasing"])
        ep = _make_episode("test_scope_ep_006", "text")

        mock_synix_llm._get_llm_client.return_value = MagicMock()
        mock_synix_llm._logged_complete.return_value = MockLLMResponse(content="answer")

        ctx = MockValidationContext(str(tmp_path))
        v.validate([spec_art, q, ep], ctx)

        results_path = tmp_path / "baseline_results.json"
        assert results_path.exists()
        data = json.loads(results_path.read_text())
        assert "summary" in data
        assert "questions" in data

    def test_to_config_dict(self):
        v = validators_mod.NaiveBaseline(
            layers=[],
            llm_config={"provider": "openai", "model": "test"},
            warn_threshold=0.9,
        )
        cfg = v.to_config_dict()
        assert cfg == {"layers": [], "llm_config": {"provider": "openai", "model": "test"}, "warn_threshold": 0.9}


# ---------------------------------------------------------------------------
# Helper tests
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_episode_index_from_label_signal(self):
        assert validators_mod._episode_index_from_label("scope_ep_009") == 9

    def test_episode_index_from_label_distractor(self):
        assert validators_mod._episode_index_from_label("scope_dx_dns_003") == 3

    def test_episode_index_from_label_unknown(self):
        assert validators_mod._episode_index_from_label("no_number") == 999999

    def test_get_build_dir_from_context(self, tmp_path):
        ctx = MockValidationContext(str(tmp_path))
        assert validators_mod._get_build_dir(ctx) == str(tmp_path)

    def test_get_build_dir_none_context(self):
        assert validators_mod._get_build_dir(None) is None

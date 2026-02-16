"""Tests for lens.datagen.synix.spec_utils — standalone spec parsing."""
from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

# spec_utils is designed to be imported standalone (from the synix directory),
# but for tests we import via the package path.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src" / "lens" / "datagen" / "synix"))

import spec_utils  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

MINIMAL_SPEC_RAW = {
    "scope_id": "test_scope",
    "domain": "test_domain",
    "description": "A test scope",
    "generation": {"temperature": 0.7, "seed": 42},
    "episodes": {
        "count": 10,
        "timeline": {"start": "2024-01-01", "interval": "1d"},
        "format": "Daily log",
        "target_words": 200,
    },
    "scenario": {"setting": "Test scenario", "voice": "Terse ops style"},
    "arc": [
        {"id": "baseline", "episodes": "1-5", "description": "Normal ops", "signal_density": "none"},
        {"id": "signal", "episodes": "6-10", "description": "Degradation", "signal_density": "low"},
    ],
    "noise": {"description": "Normal noise", "examples": ["CDN cache 94%"]},
    "key_facts": [
        {
            "id": "kf_latency",
            "fact": "API latency increasing",
            "first_appears": "signal:1",
            "reinforced_in": ["signal:3"],
        },
    ],
    "questions": [
        {
            "id": "q01",
            "checkpoint_after": 8,
            "type": "longitudinal",
            "prompt": "What patterns are emerging?",
            "ground_truth": {
                "canonical_answer": "Latency is increasing",
                "key_facts": ["kf_latency"],
                "evidence": ["signal:1"],
            },
        },
    ],
}


def _spec_with_distractors():
    raw = {**MINIMAL_SPEC_RAW}
    raw["distractors"] = {
        "count": 10,
        "target_words": 200,
        "themes": [
            {"id": "theme_a", "scenario": "Theme A scenario", "excluded_terms": ["latency"]},
        ],
        "seed": 99,
        "max_similarity": 0.3,
    }
    return raw


# ---------------------------------------------------------------------------
# parse_spec
# ---------------------------------------------------------------------------


class TestParseSpec:
    def test_basic_parse(self):
        spec = spec_utils.parse_spec(MINIMAL_SPEC_RAW)
        assert spec["scope_id"] == "test_scope"
        assert spec["domain"] == "test_domain"
        assert spec["episodes"]["count"] == 10
        assert len(spec["arc"]) == 2
        assert len(spec["key_facts"]) == 1
        assert len(spec["questions"]) == 1

    def test_missing_scope_id_raises(self):
        with pytest.raises(ValueError, match="scope_id"):
            spec_utils.parse_spec({})

    def test_distractors_parsed(self):
        spec = spec_utils.parse_spec(_spec_with_distractors())
        assert spec["distractors"] is not None
        assert spec["distractors"]["count"] == 10
        assert len(spec["distractors"]["themes"]) == 1

    def test_no_distractors_is_none(self):
        spec = spec_utils.parse_spec(MINIMAL_SPEC_RAW)
        assert spec["distractors"] is None

    def test_defaults(self):
        spec = spec_utils.parse_spec({"scope_id": "minimal"})
        assert spec["episodes"]["count"] == 30
        assert spec["generation"]["temperature"] == 0.7
        assert spec["scenario"]["setting"] == ""


# ---------------------------------------------------------------------------
# validate_spec
# ---------------------------------------------------------------------------


class TestValidateSpec:
    def test_valid_spec_no_errors(self):
        spec = spec_utils.parse_spec(MINIMAL_SPEC_RAW)
        errors = spec_utils.validate_spec(spec)
        assert errors == []

    def test_empty_scope_id(self):
        spec = spec_utils.parse_spec({**MINIMAL_SPEC_RAW, "scope_id": ""})
        errors = spec_utils.validate_spec(spec)
        assert any("scope_id" in e for e in errors)

    def test_episode_count_zero(self):
        raw = {**MINIMAL_SPEC_RAW}
        raw["episodes"] = {**raw["episodes"], "count": 0}
        spec = spec_utils.parse_spec(raw)
        errors = spec_utils.validate_spec(spec)
        assert any("episodes.count" in e for e in errors)

    def test_duplicate_phase_id(self):
        raw = {**MINIMAL_SPEC_RAW}
        raw["arc"] = [
            {"id": "dup", "episodes": "1-5"},
            {"id": "dup", "episodes": "6-10"},
        ]
        spec = spec_utils.parse_spec(raw)
        errors = spec_utils.validate_spec(spec)
        assert any("Duplicate phase id" in e for e in errors)

    def test_unknown_key_fact_in_question(self):
        raw = {**MINIMAL_SPEC_RAW}
        raw["questions"] = [{
            "id": "q_bad",
            "checkpoint_after": 5,
            "type": "longitudinal",
            "prompt": "test?",
            "ground_truth": {
                "canonical_answer": "x",
                "key_facts": ["nonexistent_fact"],
                "evidence": [],
            },
        }]
        spec = spec_utils.parse_spec(raw)
        errors = spec_utils.validate_spec(spec)
        assert any("nonexistent_fact" in e for e in errors)

    def test_validate_or_raise(self):
        spec = spec_utils.parse_spec({**MINIMAL_SPEC_RAW, "scope_id": ""})
        with pytest.raises(ValueError, match="Spec validation failed"):
            spec_utils.validate_spec_or_raise(spec)

    def test_valid_spec_with_distractors(self):
        spec = spec_utils.parse_spec(_spec_with_distractors())
        errors = spec_utils.validate_spec(spec)
        assert errors == []


# ---------------------------------------------------------------------------
# Episode helpers
# ---------------------------------------------------------------------------


class TestEpisodeHelpers:
    def test_episode_range(self):
        phase = {"id": "test", "episodes": "5-10"}
        assert spec_utils.episode_range(phase) == (5, 10)

    def test_episode_range_invalid(self):
        with pytest.raises(ValueError, match="Invalid episode range"):
            spec_utils.episode_range({"id": "bad", "episodes": "5"})

    def test_make_episode_id(self):
        assert spec_utils.make_episode_id("scope01", 7) == "scope01_ep_007"

    def test_make_episode_timestamp(self):
        spec = spec_utils.parse_spec(MINIMAL_SPEC_RAW)
        ts = spec_utils.make_episode_timestamp(spec, 1)
        assert ts == "2024-01-01T10:00:00"
        ts3 = spec_utils.make_episode_timestamp(spec, 3)
        assert ts3 == "2024-01-03T10:00:00"

    def test_get_phase_for_episode(self):
        spec = spec_utils.parse_spec(MINIMAL_SPEC_RAW)
        phase = spec_utils.get_phase_for_episode(spec, 3)
        assert phase is not None
        assert phase["id"] == "baseline"

        phase = spec_utils.get_phase_for_episode(spec, 7)
        assert phase is not None
        assert phase["id"] == "signal"

        assert spec_utils.get_phase_for_episode(spec, 99) is None


# ---------------------------------------------------------------------------
# Phase-relative references
# ---------------------------------------------------------------------------


class TestPhaseRefs:
    def test_resolve_phase_ref(self):
        spec = spec_utils.parse_spec(MINIMAL_SPEC_RAW)
        # signal phase starts at episode 6, so signal:1 = episode 6
        eid = spec_utils.resolve_phase_ref("signal:1", spec)
        assert eid == "test_scope_ep_006"

    def test_resolve_phase_ref_unknown_phase(self):
        spec = spec_utils.parse_spec(MINIMAL_SPEC_RAW)
        with pytest.raises(ValueError, match="Unknown phase"):
            spec_utils.resolve_phase_ref("nonexistent:1", spec)

    def test_resolve_phase_ref_invalid_format(self):
        spec = spec_utils.parse_spec(MINIMAL_SPEC_RAW)
        with pytest.raises(ValueError, match="Invalid phase ref"):
            spec_utils.resolve_phase_ref("bad_ref", spec)

    def test_get_key_facts_for_phase(self):
        spec = spec_utils.parse_spec(MINIMAL_SPEC_RAW)
        signal_phase = spec["arc"][1]
        kf_placements = spec_utils.get_key_facts_for_phase(spec, signal_phase)
        assert len(kf_placements) == 1
        local_idx, kf = kf_placements[0]
        assert local_idx == 1
        assert kf["id"] == "kf_latency"

    def test_get_key_facts_baseline_phase_empty(self):
        spec = spec_utils.parse_spec(MINIMAL_SPEC_RAW)
        baseline_phase = spec["arc"][0]
        kf_placements = spec_utils.get_key_facts_for_phase(spec, baseline_phase)
        assert kf_placements == []


# ---------------------------------------------------------------------------
# Spec hash & checkpoints
# ---------------------------------------------------------------------------


class TestSpecHash:
    def test_compute_spec_hash(self, tmp_path):
        spec_file = tmp_path / "spec.yaml"
        spec_file.write_text("scope_id: test\n")
        h = spec_utils.compute_spec_hash(spec_file)
        assert h.startswith("sha256:")
        assert len(h) > 10

    def test_get_checkpoints(self):
        spec = spec_utils.parse_spec(MINIMAL_SPEC_RAW)
        checkpoints = spec_utils.get_checkpoints(spec)
        assert checkpoints == [8]


# ---------------------------------------------------------------------------
# YAML loading
# ---------------------------------------------------------------------------


class TestLoadSpec:
    def test_load_spec_from_file(self, tmp_path):
        spec_file = tmp_path / "spec.yaml"
        import yaml
        spec_file.write_text(yaml.dump(MINIMAL_SPEC_RAW))
        spec = spec_utils.load_spec(spec_file)
        assert spec["scope_id"] == "test_scope"

    def test_load_spec_file_not_found(self, tmp_path):
        with pytest.raises(ValueError, match="not found"):
            spec_utils.load_spec(tmp_path / "nonexistent.yaml")

    def test_load_spec_not_mapping(self, tmp_path):
        spec_file = tmp_path / "spec.yaml"
        spec_file.write_text("- just\n- a\n- list\n")
        with pytest.raises(ValueError, match="YAML mapping"):
            spec_utils.load_spec(spec_file)

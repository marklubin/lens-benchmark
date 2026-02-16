"""Tests for lens.datagen.synix.release — release step logic."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from lens.datagen.spec import load_spec


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

MINIMAL_SPEC_YAML = """\
scope_id: test_scope
domain: test_domain
description: "Test scope"
generation:
  temperature: 0.7
  seed: 42
episodes:
  count: 4
  timeline:
    start: "2024-01-01"
    interval: "1d"
  format: "Daily log"
  target_words: 100
scenario:
  setting: "Test scenario"
  voice: "Terse"
arc:
  - id: baseline
    episodes: "1-2"
    description: "Normal"
    signal_density: none
  - id: signal
    episodes: "3-4"
    description: "Degradation"
    signal_density: low
noise:
  description: "Normal"
key_facts:
  - id: kf1
    fact: "latency increasing"
    first_appears: "signal:1"
    reinforced_in: []
questions:
  - id: q01
    checkpoint_after: 4
    type: longitudinal
    prompt: "What patterns?"
    ground_truth:
      canonical_answer: "Latency increasing"
      key_facts:
        - kf1
      evidence:
        - "signal:1"
distractors:
  count: 2
  target_words: 100
  themes:
    - id: dns
      scenario: "DNS migration"
      excluded_terms:
        - latency
  seed: 99
  max_similarity: 0.3
"""


def _setup_synix_build_dir(scope_dir: Path):
    """Create a mock synix build directory structure with artifacts."""
    gen_dir = scope_dir / "generated"

    # Write spec.yaml
    (scope_dir / "spec.yaml").write_text(MINIMAL_SPEC_YAML)

    # Create synix layer directories with artifact JSON files
    # Layer 0: spec
    spec_layer = gen_dir / "layer0-spec"
    spec_layer.mkdir(parents=True)
    spec_data = {
        "label": "spec",
        "artifact_type": "scope_spec",
        "content": json.dumps({
            "scope_id": "test_scope",
            "_spec_hash": "sha256:abc123",
        }),
        "metadata": {"scope_id": "test_scope"},
    }
    (spec_layer / "spec.json").write_text(json.dumps(spec_data))

    # Layer 1: signal_episodes
    signal_layer = gen_dir / "layer1-signal_episodes"
    signal_layer.mkdir(parents=True)
    for i in range(1, 5):
        ep = {
            "label": f"test_scope_ep_{i:03d}",
            "artifact_type": "signal_episode",
            "content": f"Signal episode {i} text with many words to fill the content. " * 10,
            "metadata": {
                "episode_id": f"test_scope_ep_{i:03d}",
                "scope_id": "test_scope",
                "timestamp": f"2024-01-{i:02d}T10:00:00",
                "phase": "baseline" if i <= 2 else "signal",
                "signal_density": "none" if i <= 2 else "low",
                "episode_type": "signal",
            },
        }
        (signal_layer / f"test_scope_ep_{i:03d}.json").write_text(json.dumps(ep))

    # Layer 1: distractor_episodes
    distractor_layer = gen_dir / "layer1-distractor_episodes"
    distractor_layer.mkdir(parents=True)
    for i in range(1, 3):
        dx = {
            "label": f"test_scope_dx_dns_{i:03d}",
            "artifact_type": "distractor_episode",
            "content": f"DNS distractor episode {i} text. " * 10,
            "metadata": {
                "theme": "dns",
                "episode_type": "distractor",
                "scope_id": "test_scope",
            },
        }
        (distractor_layer / f"test_scope_dx_dns_{i:03d}.json").write_text(json.dumps(dx))

    # Layer 2: questions
    questions_layer = gen_dir / "layer2-questions"
    questions_layer.mkdir(parents=True)
    q_data = {
        "label": "q_q01",
        "artifact_type": "question",
        "content": json.dumps({
            "question_id": "q01",
            "scope_id": "test_scope",
            "checkpoint_after": 4,
            "question_type": "longitudinal",
            "prompt": "What patterns?",
            "ground_truth": {
                "canonical_answer": "Latency increasing",
                "required_evidence_refs": ["test_scope_ep_003"],
                "key_facts": ["latency increasing"],
            },
        }),
        "metadata": {"question_id": "q01", "question_type": "longitudinal"},
    }
    (questions_layer / "q_q01.json").write_text(json.dumps(q_data))

    # Layer 2: key_fact_audit
    audit_layer = gen_dir / "layer2-key_fact_audit"
    audit_layer.mkdir(parents=True)
    audit_data = {
        "label": "key-fact-audit",
        "artifact_type": "audit",
        "content": json.dumps({
            "key_fact_coverage": {
                "kf1": {
                    "target_episodes": ["test_scope_ep_003"],
                    "found_in": ["test_scope_ep_003"],
                },
            },
            "hit_rate": 1.0,
            "total_targets": 1,
            "total_found": 1,
        }),
        "metadata": {"hit_rate": 1.0},
    }
    (audit_layer / "key-fact-audit.json").write_text(json.dumps(audit_data))

    return gen_dir


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRelease:
    def test_produces_all_output_files(self, tmp_path):
        scope_dir = tmp_path / "scope"
        scope_dir.mkdir()
        gen_dir = _setup_synix_build_dir(scope_dir)

        from lens.datagen.synix.release import run_release
        result = run_release(str(scope_dir))

        assert result == gen_dir
        assert (gen_dir / "episodes.json").exists()
        assert (gen_dir / "distractors.json").exists()
        assert (gen_dir / "questions.json").exists()
        assert (gen_dir / "release_manifest.json").exists()
        assert (gen_dir / "verification.json").exists()
        assert (gen_dir / "verification_report.html").exists()

    def test_episodes_json_format(self, tmp_path):
        scope_dir = tmp_path / "scope"
        scope_dir.mkdir()
        _setup_synix_build_dir(scope_dir)

        from lens.datagen.synix.release import run_release
        run_release(str(scope_dir))

        episodes = json.loads((scope_dir / "generated" / "episodes.json").read_text())
        assert len(episodes) == 4
        for ep in episodes:
            assert "episode_id" in ep
            assert "scope_id" in ep
            assert "timestamp" in ep
            assert "text" in ep
            assert "meta" in ep
            assert ep["meta"]["episode_type"] == "signal"

    def test_distractors_json_format(self, tmp_path):
        scope_dir = tmp_path / "scope"
        scope_dir.mkdir()
        _setup_synix_build_dir(scope_dir)

        from lens.datagen.synix.release import run_release
        run_release(str(scope_dir))

        distractors = json.loads((scope_dir / "generated" / "distractors.json").read_text())
        assert len(distractors) == 2
        for dx in distractors:
            assert dx["episode_id"].startswith("test_scope_dx_")
            assert dx["scope_id"] == "test_scope"
            assert "timestamp" in dx
            assert dx["meta"]["episode_type"] == "distractor"

    def test_distractor_ids_sequential(self, tmp_path):
        scope_dir = tmp_path / "scope"
        scope_dir.mkdir()
        _setup_synix_build_dir(scope_dir)

        from lens.datagen.synix.release import run_release
        run_release(str(scope_dir))

        distractors = json.loads((scope_dir / "generated" / "distractors.json").read_text())
        ids = [d["episode_id"] for d in distractors]
        assert ids == ["test_scope_dx_001", "test_scope_dx_002"]

    def test_questions_json_format(self, tmp_path):
        scope_dir = tmp_path / "scope"
        scope_dir.mkdir()
        _setup_synix_build_dir(scope_dir)

        from lens.datagen.synix.release import run_release
        run_release(str(scope_dir))

        questions = json.loads((scope_dir / "generated" / "questions.json").read_text())
        assert len(questions) == 1
        q = questions[0]
        assert q["question_id"] == "q01"
        assert q["question_type"] == "longitudinal"
        assert q["checkpoint_after"] == 4

    def test_manifest_contains_key_fact_coverage(self, tmp_path):
        scope_dir = tmp_path / "scope"
        scope_dir.mkdir()
        _setup_synix_build_dir(scope_dir)

        from lens.datagen.synix.release import run_release
        run_release(str(scope_dir))

        manifest = json.loads((scope_dir / "generated" / "release_manifest.json").read_text())
        assert manifest["scope_id"] == "test_scope"
        assert "key_fact_coverage" in manifest
        assert manifest["validation"]["key_fact_hit_rate"] == 1.0

    def test_html_report_generated(self, tmp_path):
        scope_dir = tmp_path / "scope"
        scope_dir.mkdir()
        _setup_synix_build_dir(scope_dir)

        from lens.datagen.synix.release import run_release
        run_release(str(scope_dir))

        html = (scope_dir / "generated" / "verification_report.html").read_text()
        assert "<!DOCTYPE html>" in html
        assert "LENS Verification Report" in html

    def test_reads_validator_results(self, tmp_path):
        scope_dir = tmp_path / "scope"
        scope_dir.mkdir()
        gen_dir = _setup_synix_build_dir(scope_dir)

        # Write contamination results
        contamination = {
            "summary": "pass",
            "questions": [{"question_id": "q01", "contaminated": False}],
        }
        (gen_dir / "contamination_results.json").write_text(json.dumps(contamination))

        # Write baseline results
        baseline = {
            "summary": {"longitudinal": 0.75},
            "questions": [{"question_id": "q01", "fact_coverage": 0.75}],
        }
        (gen_dir / "baseline_results.json").write_text(json.dumps(baseline))

        from lens.datagen.synix.release import run_release
        run_release(str(scope_dir))

        manifest = json.loads((gen_dir / "release_manifest.json").read_text())
        assert manifest["validation"]["contamination_check"] == "pass"

        verification = json.loads((gen_dir / "verification.json").read_text())
        assert "contamination" in verification
        assert "naive_baseline" in verification


class TestArtifactReader:
    def test_lists_layer_artifacts(self, tmp_path):
        gen_dir = tmp_path / "generated"
        layer_dir = gen_dir / "layer1-episodes"
        layer_dir.mkdir(parents=True)
        (layer_dir / "ep_001.json").write_text(json.dumps({
            "label": "ep_001",
            "content": "text",
            "metadata": {},
        }))

        from lens.datagen.synix.release import _ArtifactReader
        reader = _ArtifactReader(gen_dir)
        artifacts = reader.list_artifacts("episodes")
        assert len(artifacts) == 1
        assert artifacts[0]["label"] == "ep_001"

    def test_empty_layer(self, tmp_path):
        gen_dir = tmp_path / "generated"
        gen_dir.mkdir(parents=True)

        from lens.datagen.synix.release import _ArtifactReader
        reader = _ArtifactReader(gen_dir)
        artifacts = reader.list_artifacts("nonexistent")
        assert artifacts == []

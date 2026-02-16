from __future__ import annotations

import json
from pathlib import Path

import pytest

from lens.core.errors import DatasetError
from lens.datagen.compiler import compile_dataset


@pytest.fixture
def scope_dir_a(tmp_path: Path) -> Path:
    """Create a minimal generated scope directory."""
    scope = tmp_path / "scope_a"
    gen = scope / "generated"
    gen.mkdir(parents=True)

    episodes = [
        {
            "episode_id": "scope_a_ep_001",
            "scope_id": "scope_a",
            "timestamp": "2024-01-15T10:00:00",
            "text": "Episode 1 text.",
        },
        {
            "episode_id": "scope_a_ep_002",
            "scope_id": "scope_a",
            "timestamp": "2024-01-16T10:00:00",
            "text": "Episode 2 text.",
        },
    ]
    (gen / "episodes.json").write_text(json.dumps(episodes))

    questions = [
        {
            "question_id": "scope_a_q01",
            "scope_id": "scope_a",
            "checkpoint_after": 2,
            "question_type": "longitudinal",
            "prompt": "What happened?",
            "ground_truth": {
                "canonical_answer": "Things happened.",
                "required_evidence_refs": ["scope_a_ep_001"],
                "key_facts": ["something happened"],
            },
        },
    ]
    (gen / "questions.json").write_text(json.dumps(questions))

    return scope


@pytest.fixture
def scope_dir_b(tmp_path: Path) -> Path:
    """Create a second minimal generated scope directory."""
    scope = tmp_path / "scope_b"
    gen = scope / "generated"
    gen.mkdir(parents=True)

    episodes = [
        {
            "episode_id": "scope_b_ep_001",
            "scope_id": "scope_b",
            "timestamp": "2024-02-01T10:00:00",
            "text": "Scope B episode 1.",
        },
    ]
    (gen / "episodes.json").write_text(json.dumps(episodes))
    (gen / "questions.json").write_text(json.dumps([]))

    return scope


class TestCompileDataset:
    def test_single_scope(self, scope_dir_a: Path, tmp_path: Path) -> None:
        out = tmp_path / "output" / "dataset.json"
        result = compile_dataset([scope_dir_a], "0.1.0-test", out)
        assert result.exists()

        data = json.loads(result.read_text())
        assert data["version"] == "0.1.0-test"
        assert len(data["scopes"]) == 1
        assert data["scopes"][0]["scope_id"] == "scope_a"
        assert len(data["scopes"][0]["episodes"]) == 2
        assert len(data["questions"]) == 1

    def test_multiple_scopes(
        self, scope_dir_a: Path, scope_dir_b: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "output" / "dataset.json"
        result = compile_dataset([scope_dir_a, scope_dir_b], "0.2.0", out)
        data = json.loads(result.read_text())
        assert len(data["scopes"]) == 2
        scope_ids = {s["scope_id"] for s in data["scopes"]}
        assert scope_ids == {"scope_a", "scope_b"}

    def test_validates_output(self, scope_dir_a: Path, tmp_path: Path) -> None:
        """Output should pass validate_or_raise."""
        from lens.datasets.schema import validate_dataset

        out = tmp_path / "output" / "dataset.json"
        compile_dataset([scope_dir_a], "0.1.0", out)
        data = json.loads(out.read_text())
        errors = validate_dataset(data)
        assert errors == []

    def test_missing_episodes_file(self, tmp_path: Path) -> None:
        empty_scope = tmp_path / "empty_scope"
        (empty_scope / "generated").mkdir(parents=True)
        out = tmp_path / "out.json"
        with pytest.raises(DatasetError, match="No generated episodes"):
            compile_dataset([empty_scope], "0.1.0", out)

    def test_duplicate_episode_ids(self, scope_dir_a: Path, tmp_path: Path) -> None:
        # Create a second scope with same episode IDs
        scope_dupe = tmp_path / "scope_dupe"
        gen = scope_dupe / "generated"
        gen.mkdir(parents=True)
        episodes = [
            {
                "episode_id": "scope_a_ep_001",  # duplicate!
                "scope_id": "scope_dupe",
                "timestamp": "2024-03-01T10:00:00",
                "text": "Duplicate ep.",
            },
        ]
        (gen / "episodes.json").write_text(json.dumps(episodes))

        out = tmp_path / "out.json"
        with pytest.raises(DatasetError, match="Duplicate episode_id"):
            compile_dataset([scope_dir_a, scope_dupe], "0.1.0", out)

    def test_no_scopes(self, tmp_path: Path) -> None:
        out = tmp_path / "out.json"
        with pytest.raises(DatasetError, match="No scopes"):
            compile_dataset([], "0.1.0", out)

    def test_creates_parent_dirs(self, scope_dir_a: Path, tmp_path: Path) -> None:
        out = tmp_path / "deep" / "nested" / "dir" / "dataset.json"
        result = compile_dataset([scope_dir_a], "0.1.0", out)
        assert result.exists()


class TestCompileWithDistractors:
    @pytest.fixture
    def scope_with_distractors(self, tmp_path: Path) -> Path:
        scope = tmp_path / "scope_dist"
        gen = scope / "generated"
        gen.mkdir(parents=True)

        episodes = [
            {
                "episode_id": "scope_dist_ep_001",
                "scope_id": "scope_dist",
                "timestamp": "2024-01-15T10:00:00",
                "text": "Signal episode 1.",
                "meta": {"episode_type": "signal"},
            },
            {
                "episode_id": "scope_dist_ep_002",
                "scope_id": "scope_dist",
                "timestamp": "2024-01-17T10:00:00",
                "text": "Signal episode 2.",
                "meta": {"episode_type": "signal"},
            },
        ]
        (gen / "episodes.json").write_text(json.dumps(episodes))

        distractors = [
            {
                "episode_id": "scope_dist_dx_001",
                "scope_id": "scope_dist",
                "timestamp": "2024-01-16T10:30:00",
                "text": "Distractor episode 1.",
                "meta": {"episode_type": "distractor", "theme": "dns"},
            },
        ]
        (gen / "distractors.json").write_text(json.dumps(distractors))

        questions = [
            {
                "question_id": "scope_dist_q01",
                "scope_id": "scope_dist",
                "checkpoint_after": 3,
                "question_type": "longitudinal",
                "prompt": "What happened?",
                "ground_truth": {
                    "canonical_answer": "Things happened.",
                    "required_evidence_refs": [],
                    "key_facts": [],
                },
            },
        ]
        (gen / "questions.json").write_text(json.dumps(questions))

        return scope

    def test_merges_distractors(self, scope_with_distractors: Path, tmp_path: Path) -> None:
        out = tmp_path / "output" / "dataset.json"
        compile_dataset([scope_with_distractors], "0.2.0-test", out)
        data = json.loads(out.read_text())

        eps = data["scopes"][0]["episodes"]
        assert len(eps) == 3  # 2 signal + 1 distractor

        ep_ids = [e["episode_id"] for e in eps]
        assert "scope_dist_dx_001" in ep_ids

    def test_distractors_sorted_by_timestamp(self, scope_with_distractors: Path, tmp_path: Path) -> None:
        out = tmp_path / "output" / "dataset.json"
        compile_dataset([scope_with_distractors], "0.2.0-test", out)
        data = json.loads(out.read_text())

        eps = data["scopes"][0]["episodes"]
        timestamps = [e["timestamp"] for e in eps]
        assert timestamps == sorted(timestamps)

    def test_no_distractors_file_works(self, scope_dir_a: Path, tmp_path: Path) -> None:
        """Scope without distractors.json still compiles fine."""
        out = tmp_path / "output" / "dataset.json"
        result = compile_dataset([scope_dir_a], "0.1.0", out)
        data = json.loads(result.read_text())
        assert len(data["scopes"][0]["episodes"]) == 2

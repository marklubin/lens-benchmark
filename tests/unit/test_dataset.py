from __future__ import annotations

import pytest

from lens.core.errors import DatasetError
from lens.datasets.loader import (
    get_dataset_version,
    get_search_queries,
    load_dataset,
    load_episodes,
    load_smoke_dataset,
    load_truth_patterns,
)
from lens.datasets.schema import validate_dataset, validate_or_raise


class TestDatasetSchema:
    def test_valid_dataset(self, smoke_dataset):
        errors = validate_dataset(smoke_dataset)
        assert errors == []

    def test_missing_version(self):
        data = {"personas": []}
        errors = validate_dataset(data)
        assert any("version" in e for e in errors)

    def test_missing_personas(self):
        data = {"version": "1.0.0"}
        errors = validate_dataset(data)
        assert any("personas" in e for e in errors)

    def test_duplicate_episode_ids(self):
        data = {
            "version": "1.0.0",
            "personas": [
                {
                    "persona_id": "p1",
                    "episodes": [
                        {
                            "episode_id": "dup",
                            "persona_id": "p1",
                            "timestamp": "2024-01-01T00:00:00",
                            "text": "text1",
                        },
                        {
                            "episode_id": "dup",
                            "persona_id": "p1",
                            "timestamp": "2024-01-02T00:00:00",
                            "text": "text2",
                        },
                    ],
                }
            ],
        }
        errors = validate_dataset(data)
        assert any("duplicate" in e for e in errors)


class TestDatasetLoader:
    def test_load_smoke_dataset(self):
        data = load_smoke_dataset()
        assert data["version"] == "0.1.0-smoke"
        assert len(data["personas"]) == 2

    def test_load_episodes(self, smoke_dataset):
        episodes = load_episodes(smoke_dataset)
        assert "smoke_therapy_01" in episodes
        assert "smoke_product_01" in episodes
        assert len(episodes["smoke_therapy_01"]) == 10
        assert len(episodes["smoke_product_01"]) == 5

    def test_load_truth_patterns(self, smoke_dataset):
        patterns = load_truth_patterns(smoke_dataset)
        assert len(patterns) == 2
        assert patterns[0].pattern_id == "smoke_tp_01"

    def test_get_version(self, smoke_dataset):
        assert get_dataset_version(smoke_dataset) == "0.1.0-smoke"

    def test_get_search_queries(self, smoke_dataset):
        queries = get_search_queries(smoke_dataset)
        assert "anxiety patterns" in queries

    def test_load_nonexistent_raises(self, tmp_path):
        with pytest.raises(DatasetError, match="not found"):
            load_dataset(tmp_path / "nonexistent.json")

    def test_load_invalid_json_raises(self, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_text("not json")
        with pytest.raises(DatasetError, match="Invalid JSON"):
            load_dataset(bad)

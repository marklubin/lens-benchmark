from __future__ import annotations

import json
from pathlib import Path

from lens.core.errors import DatasetError
from lens.core.models import Episode, TruthPattern
from lens.datasets.schema import validate_or_raise


def load_dataset(path: str | Path) -> dict:
    """Load and validate a dataset from a JSON file.

    Returns the raw dataset dict with 'personas', 'episodes' parsed,
    and optional 'truth_patterns'.
    """
    path = Path(path)
    if not path.exists():
        raise DatasetError(f"Dataset not found: {path}")

    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        raise DatasetError(f"Invalid JSON in dataset: {e}") from e

    validate_or_raise(data)
    return data


def load_episodes(data: dict) -> dict[str, list[Episode]]:
    """Extract episodes grouped by persona_id from a dataset dict."""
    personas: dict[str, list[Episode]] = {}

    for persona in data["personas"]:
        pid = persona["persona_id"]
        episodes = [Episode.from_dict(ep) for ep in persona["episodes"]]
        personas[pid] = episodes

    return personas


def load_truth_patterns(data: dict) -> list[TruthPattern]:
    """Extract truth patterns from a dataset dict."""
    if "truth_patterns" not in data:
        return []
    return [TruthPattern.from_dict(tp) for tp in data["truth_patterns"]]


def load_smoke_dataset() -> dict:
    """Load the bundled smoke test dataset."""
    smoke_path = Path(__file__).parent / "smoke" / "smoke_dataset.json"
    if not smoke_path.exists():
        raise DatasetError("Smoke dataset not found. Package may be corrupted.")
    return load_dataset(smoke_path)


def get_dataset_version(data: dict) -> str:
    """Extract version string from dataset."""
    return data.get("version", "unknown")


def get_search_queries(data: dict) -> list[str]:
    """Extract search queries from dataset metadata."""
    return data.get("search_queries", [])

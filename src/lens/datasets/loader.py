from __future__ import annotations

import json
from pathlib import Path

from lens.core.errors import DatasetError
from lens.core.models import Episode, Question
from lens.datasets.schema import validate_or_raise


def load_dataset(path: str | Path) -> dict:
    """Load and validate a dataset from a JSON file.

    Returns the raw dataset dict with 'scopes' and 'episodes' parsed.
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
    """Extract episodes grouped by scope_id from a dataset dict."""
    scopes: dict[str, list[Episode]] = {}

    for scope in data["scopes"]:
        pid = scope["scope_id"]
        episodes = [Episode.from_dict(ep) for ep in scope["episodes"]]
        scopes[pid] = episodes

    return scopes


def load_questions(data: dict) -> list[Question]:
    """Extract questions from a dataset dict."""
    if "questions" not in data:
        return []
    return [Question.from_dict(q) for q in data["questions"]]


def load_smoke_dataset() -> dict:
    """Load the bundled smoke test dataset."""
    smoke_path = Path(__file__).parent / "smoke" / "smoke_dataset.json"
    if not smoke_path.exists():
        raise DatasetError("Smoke dataset not found. Package may be corrupted.")
    return load_dataset(smoke_path)


def get_dataset_version(data: dict) -> str:
    """Extract version string from dataset."""
    return data.get("version", "unknown")



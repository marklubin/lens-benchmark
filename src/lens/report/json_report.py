from __future__ import annotations

import json
from pathlib import Path

from lens.core.errors import atomic_write
from lens.core.models import ScoreCard


def save_scorecard(scorecard: ScoreCard, path: str | Path) -> None:
    """Save a ScoreCard to a JSON file."""
    path = Path(path)
    with atomic_write(path) as tmp:
        tmp.write_text(json.dumps(scorecard.to_dict(), indent=2))


def load_scorecard(path: str | Path) -> ScoreCard:
    """Load a ScoreCard from a JSON file."""
    path = Path(path)
    data = json.loads(path.read_text())
    return ScoreCard.from_dict(data)


def save_comparison(scorecards: list[ScoreCard], path: str | Path) -> None:
    """Save a comparison of multiple ScoreCards to a JSON file."""
    path = Path(path)
    with atomic_write(path) as tmp:
        tmp.write_text(json.dumps(
            [sc.to_dict() for sc in scorecards],
            indent=2,
        ))

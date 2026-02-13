from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from lens.core.models import Episode, EvidenceRef, Hit, Insight


@pytest.fixture
def sample_episodes() -> list[Episode]:
    """Create sample episodes for testing."""
    return [
        Episode(
            episode_id=f"test_ep_{i:03d}",
            persona_id="test_persona",
            timestamp=datetime(2024, 1, i + 1, 10, 0, 0),
            text=f"This is test episode {i} with some unique content about topic_{i}. "
            f"The episode discusses pattern_alpha and mentions evidence_fragment_{i} explicitly.",
            meta={"index": i},
        )
        for i in range(15)
    ]


@pytest.fixture
def sample_insights() -> list[Insight]:
    """Create sample insights with valid evidence refs."""
    return [
        Insight(
            text="Test pattern across multiple episodes",
            confidence=0.8,
            evidence=[
                EvidenceRef(episode_id="test_ep_000", quote="evidence_fragment_0"),
                EvidenceRef(episode_id="test_ep_001", quote="evidence_fragment_1"),
                EvidenceRef(episode_id="test_ep_002", quote="evidence_fragment_2"),
            ],
            falsifier="Pattern not found in future episodes",
        ),
        Insight(
            text="Another pattern with strong support",
            confidence=0.6,
            evidence=[
                EvidenceRef(episode_id="test_ep_003", quote="evidence_fragment_3"),
                EvidenceRef(episode_id="test_ep_004", quote="evidence_fragment_4"),
                EvidenceRef(episode_id="test_ep_005", quote="evidence_fragment_5"),
                EvidenceRef(episode_id="test_ep_006", quote="evidence_fragment_6"),
            ],
            falsifier="Contradicted by episodes after ep_010",
        ),
    ]


@pytest.fixture
def smoke_dataset_path() -> Path:
    """Path to the bundled smoke dataset."""
    return Path(__file__).parent.parent / "src" / "lens" / "datasets" / "smoke" / "smoke_dataset.json"


@pytest.fixture
def smoke_dataset(smoke_dataset_path: Path) -> dict:
    """Load the smoke dataset."""
    return json.loads(smoke_dataset_path.read_text())

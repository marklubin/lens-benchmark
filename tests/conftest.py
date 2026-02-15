from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from lens.core.models import (
    AgentAnswer,
    Episode,
    GroundTruth,
    Question,
    QuestionResult,
)


@pytest.fixture
def sample_episodes() -> list[Episode]:
    """Create sample episodes for testing."""
    return [
        Episode(
            episode_id=f"test_ep_{i:03d}",
            scope_id="test_scope",
            timestamp=datetime(2024, 1, i + 1, 10, 0, 0),
            text=f"This is test episode {i} with some unique content about topic_{i}. "
            f"The episode discusses pattern_alpha and mentions evidence_fragment_{i} explicitly.",
            meta={"index": i},
        )
        for i in range(15)
    ]


@pytest.fixture
def sample_question() -> Question:
    """A sample question for testing."""
    return Question(
        question_id="test_q01",
        scope_id="test_scope",
        checkpoint_after=10,
        question_type="longitudinal",
        prompt="What patterns have emerged?",
        ground_truth=GroundTruth(
            canonical_answer="A pattern of topic evolution was found.",
            required_evidence_refs=["test_ep_001", "test_ep_003", "test_ep_005"],
            key_facts=["pattern_alpha", "evidence_fragment"],
        ),
    )


@pytest.fixture
def sample_answer() -> AgentAnswer:
    """A sample agent answer for testing."""
    return AgentAnswer(
        question_id="test_q01",
        answer_text="Based on the data, pattern_alpha was found across multiple episodes with evidence_fragment references.",
        turns=[{"role": "assistant", "content": "answer"}],
        tool_calls_made=3,
        total_tokens=300,
        wall_time_ms=150.0,
        budget_violations=[],
        refs_cited=["test_ep_001", "test_ep_003"],
    )


@pytest.fixture
def sample_question_result(sample_question, sample_answer) -> QuestionResult:
    """A sample question result for testing."""
    return QuestionResult(
        question=sample_question,
        answer=sample_answer,
        retrieved_ref_ids=["test_ep_001", "test_ep_003"],
        valid_ref_ids=["test_ep_001", "test_ep_003"],
    )


@pytest.fixture
def smoke_dataset_path() -> Path:
    """Path to the bundled smoke dataset."""
    return Path(__file__).parent.parent / "src" / "lens" / "datasets" / "smoke" / "smoke_dataset.json"


@pytest.fixture
def smoke_dataset(smoke_dataset_path: Path) -> dict:
    """Load the smoke dataset."""
    return json.loads(smoke_dataset_path.read_text())

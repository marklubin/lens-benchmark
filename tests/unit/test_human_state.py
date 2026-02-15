from __future__ import annotations

import json
from pathlib import Path

import pytest

from lens.core.models import GroundTruth, Question
from lens.human.state import (
    HumanAnswerRecord,
    HumanBenchmarkState,
    ScopeProgress,
    discover_checkpoints,
    pending_questions_at_checkpoint,
    questions_at_checkpoint,
)


def _make_questions() -> list[Question]:
    return [
        Question(
            question_id="q1",
            scope_id="scope_a",
            checkpoint_after=5,
            question_type="longitudinal",
            prompt="What happened?",
            ground_truth=GroundTruth(
                canonical_answer="answer",
                required_evidence_refs=["ep1"],
                key_facts=["fact1"],
            ),
        ),
        Question(
            question_id="q2",
            scope_id="scope_a",
            checkpoint_after=5,
            question_type="null_hypothesis",
            prompt="Detail?",
            ground_truth=GroundTruth(
                canonical_answer="answer2",
                required_evidence_refs=["ep2"],
                key_facts=["fact2"],
            ),
        ),
        Question(
            question_id="q3",
            scope_id="scope_a",
            checkpoint_after=10,
            question_type="action_recommendation",
            prompt="What next?",
            ground_truth=GroundTruth(
                canonical_answer="answer3",
                required_evidence_refs=["ep3"],
                key_facts=["fact3"],
            ),
        ),
        Question(
            question_id="q4",
            scope_id="scope_b",
            checkpoint_after=3,
            question_type="longitudinal",
            prompt="Summary?",
            ground_truth=GroundTruth(
                canonical_answer="answer4",
                required_evidence_refs=["ep4"],
                key_facts=["fact4"],
            ),
        ),
    ]


def _make_dataset() -> dict:
    return {
        "version": "1.0-test",
        "scopes": [
            {
                "scope_id": "scope_a",
                "episodes": [
                    {
                        "episode_id": f"scope_a_ep{i:03d}",
                        "scope_id": "scope_a",
                        "timestamp": f"2024-01-{i+1:02d}T10:00:00",
                        "text": f"Episode {i} text",
                    }
                    for i in range(10)
                ],
            },
            {
                "scope_id": "scope_b",
                "episodes": [
                    {
                        "episode_id": f"scope_b_ep{i:03d}",
                        "scope_id": "scope_b",
                        "timestamp": f"2024-02-{i+1:02d}T10:00:00",
                        "text": f"Episode {i} text",
                    }
                    for i in range(5)
                ],
            },
        ],
        "questions": [q.to_dict() for q in _make_questions()],
    }


class TestCheckpointDiscovery:
    def test_discovers_checkpoints_for_scope(self):
        questions = _make_questions()
        cps = discover_checkpoints(questions, "scope_a")
        assert cps == [5, 10]

    def test_discovers_single_checkpoint(self):
        questions = _make_questions()
        cps = discover_checkpoints(questions, "scope_b")
        assert cps == [3]

    def test_empty_for_unknown_scope(self):
        questions = _make_questions()
        cps = discover_checkpoints(questions, "nonexistent")
        assert cps == []


class TestQuestionsAtCheckpoint:
    def test_returns_matching_questions(self):
        questions = _make_questions()
        result = questions_at_checkpoint(questions, "scope_a", 5)
        assert len(result) == 2
        assert {q.question_id for q in result} == {"q1", "q2"}

    def test_returns_empty_for_wrong_checkpoint(self):
        questions = _make_questions()
        result = questions_at_checkpoint(questions, "scope_a", 7)
        assert result == []


class TestPendingQuestions:
    def test_all_pending_initially(self):
        questions = _make_questions()
        result = pending_questions_at_checkpoint(questions, "scope_a", 5, set())
        assert len(result) == 2

    def test_excludes_answered(self):
        questions = _make_questions()
        result = pending_questions_at_checkpoint(questions, "scope_a", 5, {"q1"})
        assert len(result) == 1
        assert result[0].question_id == "q2"

    def test_none_pending_when_all_answered(self):
        questions = _make_questions()
        result = pending_questions_at_checkpoint(questions, "scope_a", 5, {"q1", "q2"})
        assert result == []


class TestHumanBenchmarkStateInit:
    def test_initialize_from_dataset(self):
        data = _make_dataset()
        state = HumanBenchmarkState.initialize("run123", "/path/to/ds.json", data)
        assert state.run_id == "run123"
        assert state.dataset_path == "/path/to/ds.json"
        assert state.dataset_version == "1.0-test"
        assert state.current_scope_index == 0
        assert len(state.scope_order) == 2
        assert "scope_a" in state.scope_progress
        assert "scope_b" in state.scope_progress
        assert state.scope_progress["scope_a"].total_episodes == 10
        assert state.scope_progress["scope_b"].total_episodes == 5
        assert state.scope_progress["scope_a"].episodes_revealed == 0
        assert not state.is_complete

    def test_current_scope_id(self):
        data = _make_dataset()
        state = HumanBenchmarkState.initialize("run123", "/path", data)
        assert state.current_scope_id == state.scope_order[0]

    def test_current_scope_id_none_when_complete(self):
        data = _make_dataset()
        state = HumanBenchmarkState.initialize("run123", "/path", data)
        state.current_scope_index = len(state.scope_order)
        assert state.current_scope_id is None


class TestHumanBenchmarkStatePersistence:
    def test_save_and_load(self, tmp_path: Path):
        data = _make_dataset()
        state = HumanBenchmarkState.initialize("run456", "/some/path.json", data)

        # Simulate some progress
        sp = state.scope_progress[state.scope_order[0]]
        sp.episodes_revealed = 3
        sp.answers.append(HumanAnswerRecord(
            question_id="q1",
            scope_id=state.scope_order[0],
            checkpoint=5,
            answer_text="My answer",
            refs_cited=["ep1", "ep2"],
            wall_time_ms=12345.0,
            answered_at="2024-06-01T12:00:00Z",
        ))

        path = tmp_path / "state.json"
        state.save(path)
        assert path.exists()

        loaded = HumanBenchmarkState.load(path)
        assert loaded.run_id == "run456"
        assert loaded.dataset_path == "/some/path.json"
        assert loaded.dataset_version == "1.0-test"
        assert loaded.current_scope_index == 0
        lp = loaded.scope_progress[loaded.scope_order[0]]
        assert lp.episodes_revealed == 3
        assert len(lp.answers) == 1
        assert lp.answers[0].question_id == "q1"
        assert lp.answers[0].answer_text == "My answer"
        assert lp.answers[0].refs_cited == ["ep1", "ep2"]
        assert lp.answers[0].wall_time_ms == 12345.0

    def test_save_creates_parent_dirs(self, tmp_path: Path):
        data = _make_dataset()
        state = HumanBenchmarkState.initialize("run789", "/p", data)
        path = tmp_path / "deep" / "nested" / "state.json"
        state.save(path)
        assert path.exists()

    def test_roundtrip_preserves_completion(self, tmp_path: Path):
        data = _make_dataset()
        state = HumanBenchmarkState.initialize("runXYZ", "/p", data)
        state.is_complete = True
        state.current_scope_index = 2

        path = tmp_path / "state.json"
        state.save(path)
        loaded = HumanBenchmarkState.load(path)
        assert loaded.is_complete is True
        assert loaded.current_scope_index == 2


class TestScopeProgress:
    def test_to_dict_from_dict_roundtrip(self):
        sp = ScopeProgress(
            scope_id="s1",
            episodes_revealed=3,
            total_episodes=10,
            checkpoints_completed=[5],
            answers=[
                HumanAnswerRecord(
                    question_id="q1",
                    scope_id="s1",
                    checkpoint=5,
                    answer_text="ans",
                    refs_cited=["e1"],
                    wall_time_ms=100.0,
                    answered_at="2024-01-01T00:00:00Z",
                )
            ],
        )
        d = sp.to_dict()
        restored = ScopeProgress.from_dict(d)
        assert restored.scope_id == "s1"
        assert restored.episodes_revealed == 3
        assert restored.total_episodes == 10
        assert restored.checkpoints_completed == [5]
        assert len(restored.answers) == 1
        assert restored.answers[0].question_id == "q1"

from __future__ import annotations

from datetime import datetime

import pytest

from lens.core.models import (
    AgentAnswer,
    CheckpointResult,
    Episode,
    GroundTruth,
    MetricResult,
    ScopeResult,
    Question,
    QuestionResult,
    RunResult,
    ScoreCard,
)


class TestGroundTruth:
    def test_round_trip(self):
        gt = GroundTruth(
            canonical_answer="The answer",
            required_evidence_refs=["ep_001", "ep_002"],
            key_facts=["fact1", "fact2"],
        )
        d = gt.to_dict()
        restored = GroundTruth.from_dict(d)
        assert restored.canonical_answer == gt.canonical_answer
        assert restored.required_evidence_refs == gt.required_evidence_refs
        assert restored.key_facts == gt.key_facts


class TestQuestion:
    def test_round_trip(self):
        q = Question(
            question_id="q01",
            scope_id="p1",
            checkpoint_after=10,
            question_type="longitudinal",
            prompt="What patterns emerged?",
            ground_truth=GroundTruth(
                canonical_answer="answer",
                required_evidence_refs=["ep_001"],
                key_facts=["fact1"],
            ),
        )
        d = q.to_dict()
        restored = Question.from_dict(d)
        assert restored.question_id == q.question_id
        assert restored.question_type == "longitudinal"
        assert restored.ground_truth.canonical_answer == "answer"


class TestAgentAnswer:
    def test_round_trip(self):
        a = AgentAnswer(
            question_id="q01",
            answer_text="The answer text",
            turns=[{"role": "assistant", "content": "hi"}],
            tool_calls_made=5,
            total_tokens=500,
            wall_time_ms=200.0,
            budget_violations=["turn limit exceeded"],
            refs_cited=["ep_001", "ep_002"],
        )
        d = a.to_dict()
        restored = AgentAnswer.from_dict(d)
        assert restored.question_id == a.question_id
        assert restored.answer_text == a.answer_text
        assert restored.tool_calls_made == 5
        assert restored.budget_violations == ["turn limit exceeded"]
        assert restored.refs_cited == ["ep_001", "ep_002"]

    def test_defaults(self):
        a = AgentAnswer(question_id="q", answer_text="a")
        assert a.turns == []
        assert a.tool_calls_made == 0
        assert a.budget_violations == []


class TestQuestionResult:
    def test_round_trip(self):
        qr = QuestionResult(
            question=Question(
                question_id="q01",
                scope_id="p1",
                checkpoint_after=10,
                question_type="null_hypothesis",
                prompt="What happened?",
                ground_truth=GroundTruth(
                    canonical_answer="X",
                    required_evidence_refs=["ep_001"],
                    key_facts=["X"],
                ),
            ),
            answer=AgentAnswer(question_id="q01", answer_text="Y"),
            retrieved_ref_ids=["ep_001", "ep_002"],
            valid_ref_ids=["ep_001"],
        )
        d = qr.to_dict()
        restored = QuestionResult.from_dict(d)
        assert restored.question.question_id == "q01"
        assert restored.answer.answer_text == "Y"
        assert restored.retrieved_ref_ids == ["ep_001", "ep_002"]
        assert restored.valid_ref_ids == ["ep_001"]


class TestCheckpointResult:
    def test_round_trip_with_question_results(self):
        cp = CheckpointResult(
            scope_id="p1",
            checkpoint=10,
            question_results=[
                QuestionResult(
                    question=Question(
                        question_id="q01",
                        scope_id="p1",
                        checkpoint_after=10,
                        question_type="longitudinal",
                        prompt="Q?",
                        ground_truth=GroundTruth(
                            canonical_answer="A",
                            required_evidence_refs=[],
                            key_facts=[],
                        ),
                    ),
                    answer=AgentAnswer(question_id="q01", answer_text="A"),
                ),
            ],
        )
        d = cp.to_dict()
        restored = CheckpointResult.from_dict(d)
        assert len(restored.question_results) == 1
        assert restored.question_results[0].question.question_id == "q01"


class TestEpisode:
    def test_round_trip(self):
        ep = Episode(
            episode_id="ep_001",
            scope_id="p1",
            timestamp=datetime(2024, 1, 15, 10, 0, 0),
            text="episode text",
            meta={"key": "value"},
        )
        d = ep.to_dict()
        restored = Episode.from_dict(d)
        assert restored.episode_id == ep.episode_id
        assert restored.timestamp == ep.timestamp
        assert restored.meta == ep.meta

    def test_from_dict_string_timestamp(self):
        d = {
            "episode_id": "ep_001",
            "scope_id": "p1",
            "timestamp": "2024-01-15T10:00:00",
            "text": "text",
        }
        ep = Episode.from_dict(d)
        assert ep.timestamp == datetime(2024, 1, 15, 10, 0, 0)


class TestScoreCard:
    def test_round_trip(self):
        sc = ScoreCard(
            run_id="abc123",
            adapter="null",
            dataset_version="0.1.0",
            budget_preset="standard",
            metrics=[
                MetricResult(name="ev", tier=1, value=0.95),
            ],
            composite_score=0.5,
        )
        d = sc.to_dict()
        restored = ScoreCard.from_dict(d)
        assert restored.run_id == sc.run_id
        assert len(restored.metrics) == 1
        assert restored.composite_score == 0.5

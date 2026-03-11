"""Tests for scoring V2 (T006)."""
from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from bench.dataset import QuestionData
from bench.scorer import ScorerV2
from bench.state import EventWriter, StateStore
from bench.schemas import EventType


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def store(tmp_dir):
    s = StateStore(tmp_dir / "state.db")
    yield s
    s.close()


@dataclass
class FakeArtifact:
    content: str
    label: str


_DEFAULT_KEY_FACTS = ["metric was 10", "metric was 20", "metric was 30"]


def _make_question(*, key_facts=_DEFAULT_KEY_FACTS, prompt="What happened?") -> QuestionData:
    return QuestionData(
        question_id="q01",
        scope_id="test_scope",
        checkpoint_after=3,
        question_type="longitudinal",
        prompt=prompt,
        ground_truth={
            "canonical_answer": "Metrics increased.",
            "key_facts": key_facts,
            "required_evidence_refs": [],
        },
    )


def _make_answer(**overrides) -> dict:
    base = {
        "answer_text": "The metric started at 10 and increased to 30.",
        "cited_refs": ["test_scope_ep_001", "test_scope_ep_003"],
        "tool_calls_made": 2,
        "prompt_tokens": 500,
        "completion_tokens": 100,
        "wall_time_ms": 1500.0,
    }
    base.update(overrides)
    return base


def _mock_broker_yes_no(yes_count: int, total: int):
    """Create a broker that returns YES for first `yes_count` calls, NO for rest."""
    broker = MagicMock()
    responses = []
    for i in range(total):
        response = MagicMock()
        choice = MagicMock()
        choice.message.content = "YES" if i < yes_count else "NO"
        response.choices = [choice]
        responses.append(response)
    broker.chat_completion.side_effect = responses
    return broker


def _mock_broker_score(score: float, *, prepend_yes_no: list[str] | None = None):
    """Create a broker that returns fact check verdicts then a numeric score."""
    broker = MagicMock()
    responses = []
    if prepend_yes_no:
        for verdict in prepend_yes_no:
            r = MagicMock()
            r.choices = [MagicMock()]
            r.choices[0].message.content = verdict
            responses.append(r)
    # Evidence support call
    r = MagicMock()
    r.choices = [MagicMock()]
    r.choices[0].message.content = str(score)
    responses.append(r)
    broker.chat_completion.side_effect = responses
    return broker


# ---------------------------------------------------------------------------
# Fact Score Tests
# ---------------------------------------------------------------------------


class TestFactScore:
    def test_all_facts_found(self):
        broker = _mock_broker_yes_no(3, 3)
        scorer = ScorerV2(broker)
        question = _make_question(key_facts=["fact1", "fact2", "fact3"])
        p, r, f1 = scorer._fact_score(question, "Answer with all facts.")
        assert r == 1.0
        assert f1 == 1.0

    def test_no_facts_found(self):
        broker = _mock_broker_yes_no(0, 3)
        scorer = ScorerV2(broker)
        question = _make_question(key_facts=["fact1", "fact2", "fact3"])
        p, r, f1 = scorer._fact_score(question, "Irrelevant answer.")
        assert r == 0.0
        assert f1 == 0.0

    def test_partial_facts(self):
        broker = _mock_broker_yes_no(2, 3)
        scorer = ScorerV2(broker)
        question = _make_question(key_facts=["fact1", "fact2", "fact3"])
        p, r, f1 = scorer._fact_score(question, "Some facts found.")
        assert r == pytest.approx(2 / 3, abs=0.01)
        assert f1 == pytest.approx(2 / 3, abs=0.01)

    def test_empty_answer(self):
        broker = MagicMock()
        scorer = ScorerV2(broker)
        question = _make_question()
        p, r, f1 = scorer._fact_score(question, "")
        assert f1 == 0.0
        broker.chat_completion.assert_not_called()

    def test_no_key_facts(self):
        broker = MagicMock()
        scorer = ScorerV2(broker)
        question = _make_question(key_facts=[])
        p, r, f1 = scorer._fact_score(question, "Any answer.")
        assert f1 == 1.0
        broker.chat_completion.assert_not_called()


# ---------------------------------------------------------------------------
# Citation Validity Tests
# ---------------------------------------------------------------------------


class TestCitationValidity:
    def test_all_valid_refs(self):
        release = MagicMock()
        release.artifact.return_value = FakeArtifact(content="Episode content", label="test")
        scorer = ScorerV2(MagicMock())
        score = scorer._citation_validity(["ref1", "ref2"], release)
        assert score == 1.0

    def test_mixed_valid_invalid(self):
        release = MagicMock()
        release.artifact.side_effect = [
            FakeArtifact(content="Content", label="ref1"),
            KeyError("not found"),
        ]
        scorer = ScorerV2(MagicMock())
        score = scorer._citation_validity(["ref1", "ref2"], release)
        assert score == 0.5

    def test_no_citations(self):
        scorer = ScorerV2(MagicMock())
        score = scorer._citation_validity([], None)
        assert score == 0.0

    def test_no_release_gives_benefit(self):
        scorer = ScorerV2(MagicMock())
        score = scorer._citation_validity(["ref1"], None)
        assert score == 1.0

    def test_all_invalid(self):
        release = MagicMock()
        release.artifact.side_effect = KeyError("not found")
        scorer = ScorerV2(MagicMock())
        score = scorer._citation_validity(["ref1", "ref2"], release)
        assert score == 0.0


# ---------------------------------------------------------------------------
# Evidence Support Tests
# ---------------------------------------------------------------------------


class TestEvidenceSupport:
    def test_high_support(self):
        release = MagicMock()
        release.artifact.return_value = FakeArtifact(
            content="The metric was 10 in episode 1 and 30 in episode 3.",
            label="ep_001",
        )
        broker = MagicMock()
        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message.content = "0.9"
        broker.chat_completion.return_value = response

        scorer = ScorerV2(broker)
        score = scorer._evidence_support(
            _make_question(), "Metrics went from 10 to 30.",
            ["ep_001"], release,
        )
        assert score == 0.9

    def test_empty_answer(self):
        scorer = ScorerV2(MagicMock())
        score = scorer._evidence_support(_make_question(), "", [], None)
        assert score == 0.0

    def test_no_evidence(self):
        scorer = ScorerV2(MagicMock())
        score = scorer._evidence_support(
            _make_question(), "Some answer.", [], None,
        )
        assert score == 0.0

    def test_clamped_to_1(self):
        release = MagicMock()
        release.artifact.return_value = FakeArtifact(content="Evidence", label="ep")
        broker = MagicMock()
        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message.content = "1.5"  # out of range
        broker.chat_completion.return_value = response

        scorer = ScorerV2(broker)
        score = scorer._evidence_support(
            _make_question(), "Answer.", ["ep"], release,
        )
        assert score == 1.0


# ---------------------------------------------------------------------------
# Composite Score Tests
# ---------------------------------------------------------------------------


class TestCompositeScore:
    def test_composite_formula(self):
        # fact_f1=0.8, evidence=0.6, citation=1.0
        # primary = 0.5*0.8 + 0.3*0.6 + 0.2*1.0 = 0.4 + 0.18 + 0.2 = 0.78
        broker = _mock_broker_yes_no(3, 3)  # all facts found → f1=1.0

        # Need to chain: 3 fact checks (YES), then 1 evidence support call
        responses = []
        for _ in range(3):
            r = MagicMock()
            r.choices = [MagicMock()]
            r.choices[0].message.content = "YES"
            responses.append(r)
        # Evidence support
        r = MagicMock()
        r.choices = [MagicMock()]
        r.choices[0].message.content = "0.8"
        responses.append(r)
        broker.chat_completion.side_effect = responses

        release = MagicMock()
        release.artifact.return_value = FakeArtifact(content="Evidence text", label="ref")

        scorer = ScorerV2(broker)
        answer = _make_answer(cited_refs=["ref1", "ref2"])
        score = scorer.score_answer(
            _make_question(), answer, release,
        )

        assert score.fact_f1 == 1.0
        assert score.citation_validity == 1.0
        assert score.evidence_support == 0.8
        expected = 0.5 * 1.0 + 0.3 * 0.8 + 0.2 * 1.0
        assert score.primary_score == pytest.approx(expected, abs=0.01)

    def test_zero_score(self):
        broker = _mock_broker_yes_no(0, 3)
        scorer = ScorerV2(broker)
        answer = _make_answer(cited_refs=[], answer_text="")
        score = scorer.score_answer(_make_question(), answer, None)
        assert score.primary_score == 0.0


# ---------------------------------------------------------------------------
# Score Run Tests
# ---------------------------------------------------------------------------


class TestScoreRun:
    def test_scores_all_answers(self, tmp_dir, store):
        # Save some answers
        store.save_answer("run-1", "q01", "cp03", {
            "answer_text": "Answer 1.",
            "cited_refs": [],
            "tool_calls_made": 1,
        })
        store.save_answer("run-1", "q02", "cp03", {
            "answer_text": "Answer 2.",
            "cited_refs": [],
            "tool_calls_made": 2,
        })

        broker = MagicMock()
        # Each question has 3 key facts → 3 LLM calls for fact check
        # fact check returns NO for everything
        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message.content = "NO"
        broker.chat_completion.return_value = response

        scorer = ScorerV2(broker)
        questions = [
            _make_question(),
            QuestionData(
                question_id="q02",
                scope_id="test_scope",
                checkpoint_after=3,
                question_type="longitudinal",
                prompt="What else?",
                ground_truth={"key_facts": ["fact_a"], "canonical_answer": "", "required_evidence_refs": []},
            ),
        ]

        scores = scorer.score_run("run-1", store, questions, study_id="s1", policy_id="null")
        assert len(scores) == 2
        assert all(s.run_id == "run-1" for s in scores)

        # Verify saved to store
        s1 = store.get_score("run-1", "q01")
        assert s1 is not None

    def test_rescore_from_saved_answers(self, tmp_dir, store):
        """Rescoring should work from saved answers only — no new agent inference."""
        store.save_answer("run-1", "q01", "cp03", {
            "answer_text": "Previous answer.",
            "cited_refs": [],
            "tool_calls_made": 0,
        })

        broker = MagicMock()
        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message.content = "NO"
        broker.chat_completion.return_value = response

        scorer = ScorerV2(broker)
        questions = [_make_question(key_facts=["one fact"])]

        # First score
        scores1 = scorer.score_run("run-1", store, questions)
        # Second score (rescore)
        scores2 = scorer.score_run("run-1", store, questions)

        # Both should have same fact_f1 (same answer, same judge)
        assert scores1[0].fact_f1 == scores2[0].fact_f1

    def test_scoring_events_emitted(self, store):
        store.save_answer("run-1", "q01", "cp03", {
            "answer_text": "Answer.",
            "cited_refs": [],
        })

        broker = MagicMock()
        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message.content = "YES"
        broker.chat_completion.return_value = response

        ew = EventWriter(store, "study-1")
        scorer = ScorerV2(broker)
        scorer.score_run(
            "run-1", store, [_make_question(key_facts=["f"])],
            event_writer=ew,
        )

        events = store.get_events(study_id="study-1")
        types = [e.event_type for e in events]
        assert EventType.scoring_started in types
        assert EventType.scoring_completed in types


# ---------------------------------------------------------------------------
# Schema Validation
# ---------------------------------------------------------------------------


class TestScoreSchema:
    def test_score_record_validates(self):
        broker = _mock_broker_yes_no(1, 1)
        # Evidence support
        r = MagicMock()
        r.choices = [MagicMock()]
        r.choices[0].message.content = "0.5"
        broker.chat_completion.side_effect = list(broker.chat_completion.side_effect) + [r]

        release = MagicMock()
        release.artifact.return_value = FakeArtifact(content="Evidence", label="ref")

        scorer = ScorerV2(broker)
        answer = _make_answer(cited_refs=["ref1"])
        score = scorer.score_answer(
            _make_question(key_facts=["single fact"]), answer, release,
        )

        # Should be a valid ScoreRecord
        assert score.score_id
        assert 0.0 <= score.primary_score <= 1.0
        assert 0.0 <= score.fact_f1 <= 1.0
        assert 0.0 <= score.evidence_support <= 1.0
        assert 0.0 <= score.citation_validity <= 1.0

        # Should serialize
        data = score.model_dump_json()
        from bench.schemas import ScoreRecord
        roundtripped = ScoreRecord.model_validate_json(data)
        assert roundtripped.score_id == score.score_id

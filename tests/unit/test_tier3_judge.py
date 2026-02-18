"""Tests for judge-based scoring in tier 3 metrics."""
from __future__ import annotations

from lens.core.models import (
    AgentAnswer,
    CheckpointResult,
    GroundTruth,
    Question,
    QuestionResult,
    RunResult,
    ScopeResult,
)
from lens.scorer.tier3 import ActionQuality, LongitudinalAdvantage, _judge_fact_score


def _make_qr(
    question_type: str = "longitudinal",
    answer_text: str = "",
    canonical_answer: str = "canonical answer",
    key_facts: list[str] | None = None,
    prompt: str = "Test question?",
) -> QuestionResult:
    return QuestionResult(
        question=Question(
            question_id="test_q",
            scope_id="p1",
            checkpoint_after=10,
            question_type=question_type,
            prompt=prompt,
            ground_truth=GroundTruth(
                canonical_answer=canonical_answer,
                required_evidence_refs=[],
                key_facts=key_facts or [],
            ),
        ),
        answer=AgentAnswer(
            question_id="test_q",
            answer_text=answer_text,
            tool_calls_made=2,
            total_tokens=100,
            wall_time_ms=50.0,
        ),
    )


def _make_run(question_results: list[QuestionResult]) -> RunResult:
    return RunResult(
        run_id="test",
        adapter="test",
        dataset_version="0.1.0",
        budget_preset="standard",
        scopes=[
            ScopeResult(
                scope_id="p1",
                checkpoints=[
                    CheckpointResult(
                        scope_id="p1",
                        checkpoint=10,
                        question_results=question_results,
                    )
                ],
            )
        ],
    )


class TestJudgeFactScore:
    """Tests for _judge_fact_score helper."""

    def test_candidate_wins(self):
        """Judge always picks candidate → score 1.0."""
        def judge_fn(prompt):
            if "good answer" in prompt.split("Response A:")[1].split("Response B:")[0]:
                return "A"
            return "B"

        score = _judge_fact_score(
            answer_text="good answer with detail",
            canonical_answer="bad answer",
            key_facts=["detail"],
            question_prompt="What?",
            judge_fn=judge_fn,
        )
        assert score == 1.0

    def test_tie_score(self):
        """Judge always says TIE → score 0.5."""
        score = _judge_fact_score(
            answer_text="answer",
            canonical_answer="other",
            key_facts=["fact1", "fact2"],
            question_prompt="What?",
            judge_fn=lambda p: "TIE",
        )
        assert score == 0.5

    def test_no_facts(self):
        """No facts → score 1.0."""
        score = _judge_fact_score(
            answer_text="answer",
            canonical_answer="other",
            key_facts=[],
            question_prompt="What?",
            judge_fn=lambda p: "A",
        )
        assert score == 1.0


class TestLongitudinalAdvantageWithJudge:
    """Tests for LongitudinalAdvantage with judge_fn configured."""

    def test_without_judge_uses_substring(self):
        """Without judge, falls back to substring matching."""
        long_qr = _make_qr(
            question_type="longitudinal",
            answer_text="Found pattern_alpha",
            key_facts=["pattern_alpha"],
        )
        null_qr = _make_qr(
            question_type="null_hypothesis",
            answer_text="No facts found",
            key_facts=["different_fact"],
        )
        result = _make_run([long_qr, null_qr])
        metric = LongitudinalAdvantage()
        mr = metric.compute(result)
        assert mr.value == 1.0  # substring: 1.0 - 0.0
        assert mr.details.get("method") == "substring"

    def test_with_judge_uses_pairwise(self):
        """With judge configured, uses pairwise scoring."""
        long_qr = _make_qr(
            question_type="longitudinal",
            answer_text="good answer",
            key_facts=["fact1"],
        )
        null_qr = _make_qr(
            question_type="null_hypothesis",
            answer_text="bad answer",
            key_facts=["fact2"],
        )
        result = _make_run([long_qr, null_qr])

        metric = LongitudinalAdvantage()
        # Judge always says TIE → both get 0.5, advantage = 0.0
        metric.configure(judge_fn=lambda p: "TIE")
        mr = metric.compute(result)
        assert mr.value == 0.0  # 0.5 - 0.5
        assert mr.details.get("method") == "pairwise"

    def test_configure_injects_judge(self):
        """configure(judge_fn=...) properly injects the judge."""
        metric = LongitudinalAdvantage()
        assert metric._judge_fn is None
        metric.configure(judge_fn=lambda p: "A")
        assert metric._judge_fn is not None

    def test_judge_changes_scores(self):
        """Judge-based scoring gives different results than substring."""
        # With substring: answer doesn't contain "obscure_technical_term" → 0.0
        # With judge that says TIE: → 0.5
        long_qr = _make_qr(
            question_type="longitudinal",
            answer_text="The system showed anomalous behavior",
            key_facts=["obscure_technical_term"],
        )
        null_qr = _make_qr(
            question_type="null_hypothesis",
            answer_text="Nothing found",
            key_facts=["other_term"],
        )
        result = _make_run([long_qr, null_qr])

        # Substring: synthesis=0.0 (no match), control=0.0 → advantage=0.0
        metric_sub = LongitudinalAdvantage()
        mr_sub = metric_sub.compute(result)
        assert mr_sub.value == 0.0

        # Judge (TIE): synthesis=0.5, control=0.5 → advantage=0.0
        metric_judge = LongitudinalAdvantage()
        metric_judge.configure(judge_fn=lambda p: "TIE")
        mr_judge = metric_judge.compute(result)
        assert mr_judge.value == 0.0
        assert mr_judge.details["synthesis_mean"] == 0.5
        assert mr_judge.details["control_mean"] == 0.5


class TestActionQualityWithJudge:
    """Tests for ActionQuality with judge_fn configured."""

    def test_without_judge_uses_substring(self):
        """Without judge, falls back to substring matching."""
        qr = _make_qr(
            question_type="action_recommendation",
            answer_text="Found pattern_alpha",
            key_facts=["pattern_alpha"],
        )
        result = _make_run([qr])
        metric = ActionQuality()
        mr = metric.compute(result)
        assert mr.value == 1.0
        assert mr.details.get("method") == "substring"

    def test_with_judge_uses_pairwise(self):
        """With judge configured, uses pairwise scoring."""
        qr = _make_qr(
            question_type="action_recommendation",
            answer_text="good recommendation",
            key_facts=["fact1", "fact2"],
        )
        result = _make_run([qr])

        metric = ActionQuality()
        metric.configure(judge_fn=lambda p: "TIE")
        mr = metric.compute(result)
        assert mr.value == 0.5  # TIE on all facts
        assert mr.details.get("method") == "pairwise"

    def test_configure_injects_judge(self):
        """configure(judge_fn=...) properly injects the judge."""
        metric = ActionQuality()
        assert metric._judge_fn is None
        metric.configure(judge_fn=lambda p: "A")
        assert metric._judge_fn is not None

    def test_no_action_questions_with_judge(self):
        """No action questions returns 0.0 even with judge."""
        qr = _make_qr(question_type="longitudinal")
        result = _make_run([qr])
        metric = ActionQuality()
        metric.configure(judge_fn=lambda p: "A")
        mr = metric.compute(result)
        assert mr.value == 0.0

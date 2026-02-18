from __future__ import annotations

from typing import Callable

from lens.core.models import MetricResult, RunResult
from lens.scorer.base import BaseMetric
from lens.scorer.registry import register_metric
from lens.scorer.tier1 import _all_question_results


def _fact_recall_score(answer_text: str, key_facts: list[str]) -> float:
    """Compute fact recall for a single question."""
    if not key_facts:
        return 1.0
    answer_lower = answer_text.lower()
    found = sum(1 for f in key_facts if f.lower() in answer_lower)
    return found / len(key_facts)


def _judge_fact_score(
    answer_text: str,
    canonical_answer: str,
    key_facts: list[str],
    question_prompt: str,
    judge_fn: Callable[[str], str],
    seed: int = 42,
) -> float:
    """Score a question using pairwise judge instead of substring matching."""
    from lens.scorer.judge import pairwise_fact_judge

    win_rate, _ = pairwise_fact_judge(
        candidate_answer=answer_text,
        reference_answer=canonical_answer,
        key_facts=key_facts,
        question=question_prompt,
        judge_fn=judge_fn,
        seed=seed,
    )
    return win_rate


# Question types that require cross-episode synthesis (numerator for advantage metric)
SYNTHESIS_QUESTION_TYPES = {
    "longitudinal", "negative", "temporal", "counterfactual", "paraphrase",
    "distractor_resistance", "severity_assessment", "evidence_sufficiency",
}

# Control question types (denominator for advantage metric)
CONTROL_QUESTION_TYPES = {"null_hypothesis"}


@register_metric("longitudinal_advantage")
class LongitudinalAdvantage(BaseMetric):
    """Differential: mean score for synthesis questions minus control questions.

    Synthesis types: longitudinal, negative, temporal, counterfactual, paraphrase.
    Control types: null_hypothesis.

    When a judge_fn is configured (via configure()), uses pairwise judging
    instead of substring matching for more accurate scoring.
    """

    def __init__(self) -> None:
        self._judge_fn: Callable[[str], str] | None = None

    def configure(self, *, judge_fn: Callable[[str], str] | None = None) -> None:
        """Inject the LLM judge callable after construction."""
        if judge_fn is not None:
            self._judge_fn = judge_fn

    @property
    def name(self) -> str:
        return "longitudinal_advantage"

    @property
    def tier(self) -> int:
        return 3

    @property
    def description(self) -> str:
        return "Mean score for synthesis questions minus control questions"

    def compute(self, result: RunResult) -> MetricResult:
        qrs = _all_question_results(result)

        synthesis_scores: list[float] = []
        control_scores: list[float] = []

        for qr in qrs:
            if self._judge_fn:
                score = _judge_fact_score(
                    qr.answer.answer_text,
                    qr.question.ground_truth.canonical_answer,
                    qr.question.ground_truth.key_facts,
                    qr.question.prompt,
                    self._judge_fn,
                )
            else:
                score = _fact_recall_score(
                    qr.answer.answer_text, qr.question.ground_truth.key_facts
                )
            if qr.question.question_type in SYNTHESIS_QUESTION_TYPES:
                synthesis_scores.append(score)
            elif qr.question.question_type in CONTROL_QUESTION_TYPES:
                control_scores.append(score)

        if not synthesis_scores or not control_scores:
            return MetricResult(
                name=self.name,
                tier=self.tier,
                value=0.0,
                details={
                    "synthesis_count": len(synthesis_scores),
                    "control_count": len(control_scores),
                },
            )

        synthesis_mean = sum(synthesis_scores) / len(synthesis_scores)
        control_mean = sum(control_scores) / len(control_scores)
        value = synthesis_mean - control_mean

        method = "pairwise" if self._judge_fn else "substring"
        return MetricResult(
            name=self.name,
            tier=self.tier,
            value=value,
            details={
                "synthesis_mean": synthesis_mean,
                "control_mean": control_mean,
                "synthesis_count": len(synthesis_scores),
                "control_count": len(control_scores),
                "method": method,
            },
        )


@register_metric("action_quality")
class ActionQuality(BaseMetric):
    """Mean score for action_recommendation questions.

    When a judge_fn is configured (via configure()), uses pairwise judging
    instead of substring matching for more accurate scoring.
    """

    def __init__(self) -> None:
        self._judge_fn: Callable[[str], str] | None = None

    def configure(self, *, judge_fn: Callable[[str], str] | None = None) -> None:
        """Inject the LLM judge callable after construction."""
        if judge_fn is not None:
            self._judge_fn = judge_fn

    @property
    def name(self) -> str:
        return "action_quality"

    @property
    def tier(self) -> int:
        return 3

    @property
    def description(self) -> str:
        return "Mean score for action_recommendation questions"

    def compute(self, result: RunResult) -> MetricResult:
        qrs = _all_question_results(result)

        scores: list[float] = []
        for qr in qrs:
            if qr.question.question_type == "action_recommendation":
                if self._judge_fn:
                    scores.append(
                        _judge_fact_score(
                            qr.answer.answer_text,
                            qr.question.ground_truth.canonical_answer,
                            qr.question.ground_truth.key_facts,
                            qr.question.prompt,
                            self._judge_fn,
                        )
                    )
                else:
                    scores.append(
                        _fact_recall_score(
                            qr.answer.answer_text, qr.question.ground_truth.key_facts
                        )
                    )

        if not scores:
            return MetricResult(
                name=self.name,
                tier=self.tier,
                value=0.0,
                details={"action_recommendation_count": 0},
            )

        value = sum(scores) / len(scores)
        method = "pairwise" if self._judge_fn else "substring"
        return MetricResult(
            name=self.name,
            tier=self.tier,
            value=value,
            details={
                "action_recommendation_count": len(scores),
                "method": method,
            },
        )

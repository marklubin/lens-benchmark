from __future__ import annotations

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
    """

    @property
    def name(self) -> str:
        return "longitudinal_advantage"

    @property
    def tier(self) -> int:
        return 3

    @property
    def description(self) -> str:
        return "Mean fact-recall for synthesis questions minus control questions"

    def compute(self, result: RunResult) -> MetricResult:
        qrs = _all_question_results(result)

        synthesis_scores: list[float] = []
        control_scores: list[float] = []

        for qr in qrs:
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

        return MetricResult(
            name=self.name,
            tier=self.tier,
            value=value,
            details={
                "synthesis_mean": synthesis_mean,
                "control_mean": control_mean,
                "synthesis_count": len(synthesis_scores),
                "control_count": len(control_scores),
            },
        )


@register_metric("action_quality")
class ActionQuality(BaseMetric):
    """Mean fact recall for action_recommendation questions only."""

    @property
    def name(self) -> str:
        return "action_quality"

    @property
    def tier(self) -> int:
        return 3

    @property
    def description(self) -> str:
        return "Mean fact-recall for action_recommendation questions"

    def compute(self, result: RunResult) -> MetricResult:
        qrs = _all_question_results(result)

        scores: list[float] = []
        for qr in qrs:
            if qr.question.question_type == "action_recommendation":
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
        return MetricResult(
            name=self.name,
            tier=self.tier,
            value=value,
            details={"action_recommendation_count": len(scores)},
        )

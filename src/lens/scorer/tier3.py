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


@register_metric("longitudinal_advantage")
class LongitudinalAdvantage(BaseMetric):
    """Differential: mean score for longitudinal minus mean score for null_hypothesis questions."""

    @property
    def name(self) -> str:
        return "longitudinal_advantage"

    @property
    def tier(self) -> int:
        return 3

    @property
    def description(self) -> str:
        return "Mean fact-recall for longitudinal questions minus null_hypothesis questions"

    def compute(self, result: RunResult) -> MetricResult:
        qrs = _all_question_results(result)

        longitudinal_scores: list[float] = []
        null_scores: list[float] = []

        for qr in qrs:
            score = _fact_recall_score(
                qr.answer.answer_text, qr.question.ground_truth.key_facts
            )
            if qr.question.question_type == "longitudinal":
                longitudinal_scores.append(score)
            elif qr.question.question_type == "null_hypothesis":
                null_scores.append(score)

        if not longitudinal_scores or not null_scores:
            return MetricResult(
                name=self.name,
                tier=self.tier,
                value=0.0,
                details={
                    "longitudinal_count": len(longitudinal_scores),
                    "null_hypothesis_count": len(null_scores),
                },
            )

        long_mean = sum(longitudinal_scores) / len(longitudinal_scores)
        null_mean = sum(null_scores) / len(null_scores)
        value = long_mean - null_mean

        return MetricResult(
            name=self.name,
            tier=self.tier,
            value=value,
            details={
                "longitudinal_mean": long_mean,
                "null_hypothesis_mean": null_mean,
                "longitudinal_count": len(longitudinal_scores),
                "null_hypothesis_count": len(null_scores),
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

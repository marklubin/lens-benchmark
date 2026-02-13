from __future__ import annotations

from lens.core.models import Insight, MetricResult, RunResult
from lens.matcher.base import BaseMatcher
from lens.matcher.embedding import EmbeddingMatcher
from lens.scorer.base import BaseMetric
from lens.scorer.registry import register_metric


def _get_matcher() -> BaseMatcher:
    """Get the default matcher for stability metrics."""
    return EmbeddingMatcher()


def _consecutive_checkpoint_pairs(
    result: RunResult,
) -> list[tuple[list[Insight], list[Insight]]]:
    """Extract consecutive checkpoint insight pairs across all personas."""
    pairs: list[tuple[list[Insight], list[Insight]]] = []
    for persona in result.personas:
        cps = sorted(persona.checkpoints, key=lambda c: c.checkpoint)
        for i in range(len(cps) - 1):
            pairs.append((cps[i].insights, cps[i + 1].insights))
    return pairs


@register_metric("survival_at_k")
class SurvivalAtK(BaseMetric):
    """Insight Survival@k: overlap of top-k insights between consecutive checkpoints.

    Uses a pluggable matcher to determine if two insights are "the same".
    Default: embedding cosine similarity >= 0.85 threshold.
    """

    def __init__(self, k: int = 10, matcher: BaseMatcher | None = None) -> None:
        self.k = k
        self.matcher = matcher or _get_matcher()

    @property
    def name(self) -> str:
        return "survival_at_k"

    @property
    def tier(self) -> int:
        return 2

    @property
    def description(self) -> str:
        return f"Overlap of top-{self.k} insights between consecutive checkpoints"

    def compute(self, result: RunResult) -> MetricResult:
        pairs = _consecutive_checkpoint_pairs(result)
        if not pairs:
            return MetricResult(name=self.name, tier=self.tier, value=0.0)

        survival_rates: list[float] = []

        for prev_insights, curr_insights in pairs:
            prev_top = prev_insights[: self.k]
            curr_top = curr_insights[: self.k]

            if not prev_top or not curr_top:
                survival_rates.append(0.0)
                continue

            # Count how many prev insights survived into curr
            survived = 0
            for prev_insight in prev_top:
                for curr_insight in curr_top:
                    if self.matcher.match(prev_insight.text, curr_insight.text):
                        survived += 1
                        break

            survival_rates.append(survived / len(prev_top))

        avg_survival = sum(survival_rates) / len(survival_rates)
        return MetricResult(
            name=self.name,
            tier=self.tier,
            value=avg_survival,
            details={
                "k": self.k,
                "num_pairs": len(pairs),
                "survival_rates": survival_rates,
            },
        )


@register_metric("churn_at_k")
class ChurnAtK(BaseMetric):
    """Churn@k = 1 - Survival@k. Measures instability."""

    def __init__(self, k: int = 10, matcher: BaseMatcher | None = None) -> None:
        self.k = k
        self.matcher = matcher or _get_matcher()
        self._survival = SurvivalAtK(k=k, matcher=self.matcher)

    @property
    def name(self) -> str:
        return "churn_at_k"

    @property
    def tier(self) -> int:
        return 2

    @property
    def description(self) -> str:
        return f"1 - Survival@{self.k} — measures insight instability"

    def compute(self, result: RunResult) -> MetricResult:
        survival = self._survival.compute(result)
        return MetricResult(
            name=self.name,
            tier=self.tier,
            value=1.0 - survival.value,
            details=survival.details,
        )

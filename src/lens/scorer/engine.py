from __future__ import annotations

from lens.core.logging import LensLogger, Verbosity
from lens.core.models import MetricResult, RunResult, ScoreCard
from lens.scorer.aggregate import build_scorecard
from lens.scorer.registry import list_metrics


class ScorerEngine:
    """Runs all registered metrics against a run result and produces a ScoreCard."""

    def __init__(
        self,
        tier_filter: int | None = None,
        logger: LensLogger | None = None,
    ) -> None:
        self.tier_filter = tier_filter
        self.logger = logger or LensLogger(Verbosity.NORMAL)

    def score(self, result: RunResult) -> ScoreCard:
        """Score a run result with all applicable metrics."""
        all_metrics = list_metrics()
        results: list[MetricResult] = []

        for name, metric_cls in sorted(all_metrics.items()):
            metric = metric_cls()

            if self.tier_filter is not None and metric.tier != self.tier_filter:
                continue

            self.logger.verbose(f"Computing {name} (tier {metric.tier})")
            metric_result = metric.compute(result)
            results.append(metric_result)
            self.logger.verbose(f"  {name} = {metric_result.value:.4f}")

        scorecard = build_scorecard(
            run_id=result.run_id,
            adapter=result.adapter,
            dataset_version=result.dataset_version,
            budget_preset=result.budget_preset,
            metrics=results,
        )

        self.logger.info(f"Composite score: [bold]{scorecard.composite_score:.4f}[/bold]")
        return scorecard

from __future__ import annotations

from typing import Callable

from lens.core.logging import LensLogger, Verbosity
from lens.core.models import MetricResult, RunResult, ScoreCard
from lens.scorer.aggregate import build_scorecard
from lens.scorer.registry import list_metrics


class ScorerEngine:
    """Runs all registered metrics against a run result and produces a ScoreCard.

    Supports optional judge_fn for metrics that require LLM-based evaluation
    (e.g., pairwise answer_quality). Metrics that implement configure() will
    receive the judge_fn and any extra kwargs automatically.
    """

    def __init__(
        self,
        tier_filter: int | None = None,
        logger: LensLogger | None = None,
        judge_fn: Callable[[str], str] | None = None,
        gate_thresholds: dict[str, float] | None = None,
        baseline_generator=None,
        max_judge_workers: int = 1,
    ) -> None:
        self.tier_filter = tier_filter
        self.logger = logger or LensLogger(Verbosity.NORMAL)
        self.judge_fn = judge_fn
        self.gate_thresholds = gate_thresholds
        self.baseline_generator = baseline_generator
        self.max_judge_workers = max_judge_workers

    def score(self, result: RunResult) -> ScoreCard:
        """Score a run result with all applicable metrics."""
        all_metrics = list_metrics()
        results: list[MetricResult] = []

        # Build configure kwargs
        configure_kwargs: dict = {}
        if self.judge_fn is not None:
            configure_kwargs["judge_fn"] = self.judge_fn
        if self.baseline_generator is not None:
            configure_kwargs["baseline_generator"] = self.baseline_generator
        if self.max_judge_workers > 1:
            configure_kwargs["max_judge_workers"] = self.max_judge_workers

        for name, metric_cls in sorted(all_metrics.items()):
            metric = metric_cls()

            # Inject configuration for metrics that support it
            if configure_kwargs and hasattr(metric, "configure"):
                metric.configure(**configure_kwargs)

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
            gate_thresholds=self.gate_thresholds,
        )

        self.logger.info(f"Composite score: [bold]{scorecard.composite_score:.4f}[/bold]")
        return scorecard

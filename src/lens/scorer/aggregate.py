from __future__ import annotations

from lens.core.models import MetricResult, ScoreCard

# Default v2 composite weights
DEFAULT_WEIGHTS: dict[str, float] = {
    "evidence_grounding": 0.10,
    "fact_recall": 0.10,
    "evidence_coverage": 0.10,
    "budget_compliance": 0.10,
    "answer_quality": 0.15,
    "insight_depth": 0.15,
    "reasoning_quality": 0.10,
    "longitudinal_advantage": 0.15,
    "action_quality": 0.05,
}


def compute_composite(
    metrics: list[MetricResult],
    weights: dict[str, float] | None = None,
) -> float:
    """Compute weighted composite score from individual metrics.

    Score = sum(weight_i * value_i) for metrics in the weight table.
    Metrics not in the weight table are excluded from the composite.
    """
    w = weights or DEFAULT_WEIGHTS
    total_weight = 0.0
    weighted_sum = 0.0

    metric_map = {m.name: m.value for m in metrics}

    for name, weight in w.items():
        if name in metric_map:
            weighted_sum += weight * metric_map[name]
            total_weight += weight

    if total_weight == 0:
        return 0.0

    # Normalize in case not all metrics are present
    return weighted_sum / total_weight


def build_scorecard(
    run_id: str,
    adapter: str,
    dataset_version: str,
    budget_preset: str,
    metrics: list[MetricResult],
    weights: dict[str, float] | None = None,
) -> ScoreCard:
    """Build a complete ScoreCard with composite score."""
    composite = compute_composite(metrics, weights)
    return ScoreCard(
        run_id=run_id,
        adapter=adapter,
        dataset_version=dataset_version,
        budget_preset=budget_preset,
        metrics=metrics,
        composite_score=composite,
    )

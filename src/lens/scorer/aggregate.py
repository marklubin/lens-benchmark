from __future__ import annotations

from lens.core.models import MetricResult, ScoreCard

# Default v0 composite weights
DEFAULT_WEIGHTS: dict[str, float] = {
    "evidence_validity": 0.35,
    "multi_episode_support": 0.15,
    "non_locality": 0.10,
    "confidence_discipline": 0.10,
    "survival_at_k": 0.20,
    "budget_compliance": 0.10,
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

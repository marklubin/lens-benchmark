from __future__ import annotations

from lens.core.models import MetricResult, ScoreCard

# Default v3.3 composite weights
# fact_recall is zero-weighted: it provides no discrimination (0.167 uniformly
# across all adapters due to naive substring matching). Kept as a reported
# metric for transparency but excluded from the composite.
DEFAULT_WEIGHTS: dict[str, float] = {
    "evidence_grounding": 0.08,
    "fact_recall": 0.0,
    "evidence_coverage": 0.08,
    "budget_compliance": 0.07,
    "citation_coverage": 0.10,
    "answer_quality": 0.17,
    "insight_depth": 0.15,
    "reasoning_quality": 0.10,
    "naive_baseline_advantage": 0.17,
    "action_quality": 0.08,
    # longitudinal_advantage kept registered (backward compat) but 0 weight
}

# Tier 1 hard gate — if ANY gated metric falls below its threshold,
# the composite score is zeroed out. This prevents higher-tier scores
# (LLM judge) from compensating for fundamental mechanical failures.
TIER1_GATE_THRESHOLDS: dict[str, float] = {
    "evidence_grounding": 0.5,
    # budget_compliance is observational only — not gated.
    # Token usage and wall time are recorded in the metric details
    # for analysis, but don't zero out the composite.
}


def compute_composite(
    metrics: list[MetricResult],
    weights: dict[str, float] | None = None,
    gate_thresholds: dict[str, float] | None = None,
) -> float:
    """Compute weighted composite score from individual metrics.

    Two-phase scoring:
    1. Tier 1 gate check — if any gated metric falls below its threshold,
       the composite is 0.0 regardless of other scores.
    2. Weighted sum — sum(weight_i * value_i) for metrics in the weight table.

    Pass gate_thresholds={} to disable gating.
    """
    w = weights or DEFAULT_WEIGHTS
    gates = gate_thresholds if gate_thresholds is not None else TIER1_GATE_THRESHOLDS

    metric_map = {m.name: m.value for m in metrics}

    # Tier 1 hard gate
    for gate_name, threshold in gates.items():
        if gate_name in metric_map and metric_map[gate_name] < threshold:
            return 0.0

    # Weighted sum
    total_weight = 0.0
    weighted_sum = 0.0

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
    gate_thresholds: dict[str, float] | None = None,
) -> ScoreCard:
    """Build a complete ScoreCard with composite score."""
    composite = compute_composite(metrics, weights, gate_thresholds=gate_thresholds)
    return ScoreCard(
        run_id=run_id,
        adapter=adapter,
        dataset_version=dataset_version,
        budget_preset=budget_preset,
        metrics=metrics,
        composite_score=composite,
    )

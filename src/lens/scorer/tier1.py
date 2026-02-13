from __future__ import annotations

from lens.core.models import Insight, MetricResult, RunResult
from lens.scorer.base import BaseMetric
from lens.scorer.registry import register_metric


def _all_insights(result: RunResult) -> list[Insight]:
    """Collect all insights across all personas and checkpoints."""
    insights: list[Insight] = []
    for persona in result.personas:
        for cp in persona.checkpoints:
            insights.extend(cp.insights)
    return insights


@register_metric("evidence_validity")
class EvidenceValidity(BaseMetric):
    """EV: % of EvidenceRefs where quote is exact substring of episode.

    Note: This metric relies on validation_errors recorded during the run.
    A ref that failed validation is counted as invalid.
    """

    @property
    def name(self) -> str:
        return "evidence_validity"

    @property
    def tier(self) -> int:
        return 1

    @property
    def description(self) -> str:
        return "Percentage of evidence refs with valid exact-substring quotes"

    def compute(self, result: RunResult) -> MetricResult:
        total_refs = 0
        valid_refs = 0

        for persona in result.personas:
            for cp in persona.checkpoints:
                for insight in cp.insights:
                    for ref in insight.evidence:
                        total_refs += 1
                        # Check if this ref was flagged in validation errors
                        ref_invalid = any(
                            ref.episode_id in err and "quote not found" in err
                            for err in cp.validation_errors
                        )
                        if not ref_invalid:
                            valid_refs += 1

        value = valid_refs / total_refs if total_refs > 0 else 0.0
        return MetricResult(
            name=self.name,
            tier=self.tier,
            value=value,
            details={"total_refs": total_refs, "valid_refs": valid_refs},
        )


@register_metric("evidence_sufficiency")
class EvidenceSufficiency(BaseMetric):
    """ES: % of insights with >= 3 evidence refs."""

    @property
    def name(self) -> str:
        return "evidence_sufficiency"

    @property
    def tier(self) -> int:
        return 1

    @property
    def description(self) -> str:
        return "Percentage of insights with at least 3 evidence references"

    def compute(self, result: RunResult) -> MetricResult:
        insights = _all_insights(result)
        if not insights:
            return MetricResult(name=self.name, tier=self.tier, value=0.0)

        sufficient = sum(1 for i in insights if len(i.evidence) >= 3)
        return MetricResult(
            name=self.name,
            tier=self.tier,
            value=sufficient / len(insights),
            details={"total_insights": len(insights), "sufficient": sufficient},
        )


@register_metric("multi_episode_support")
class MultiEpisodeSupport(BaseMetric):
    """MES: % of insights citing >= 2 distinct episodes."""

    @property
    def name(self) -> str:
        return "multi_episode_support"

    @property
    def tier(self) -> int:
        return 1

    @property
    def description(self) -> str:
        return "Percentage of insights citing 2+ distinct episodes"

    def compute(self, result: RunResult) -> MetricResult:
        insights = _all_insights(result)
        if not insights:
            return MetricResult(name=self.name, tier=self.tier, value=0.0)

        multi = sum(
            1
            for i in insights
            if len({ref.episode_id for ref in i.evidence}) >= 2
        )
        return MetricResult(
            name=self.name,
            tier=self.tier,
            value=multi / len(insights),
            details={"total_insights": len(insights), "multi_episode": multi},
        )


@register_metric("non_locality")
class NonLocality(BaseMetric):
    """NL: % of insights where no single episode has >= 80% of refs."""

    @property
    def name(self) -> str:
        return "non_locality"

    @property
    def tier(self) -> int:
        return 1

    @property
    def description(self) -> str:
        return "Percentage of insights not dominated by a single episode"

    def compute(self, result: RunResult) -> MetricResult:
        insights = _all_insights(result)
        if not insights:
            return MetricResult(name=self.name, tier=self.tier, value=0.0)

        non_local = 0
        for insight in insights:
            if not insight.evidence:
                continue
            # Count refs per episode
            counts: dict[str, int] = {}
            for ref in insight.evidence:
                counts[ref.episode_id] = counts.get(ref.episode_id, 0) + 1
            max_frac = max(counts.values()) / len(insight.evidence)
            if max_frac < 0.8:
                non_local += 1

        return MetricResult(
            name=self.name,
            tier=self.tier,
            value=non_local / len(insights),
            details={"total_insights": len(insights), "non_local": non_local},
        )


@register_metric("confidence_discipline")
class ConfidenceDiscipline(BaseMetric):
    """CD: penalize high-confidence + weak grounding.

    Score = 1 - mean(penalty) where penalty = max(0, confidence - grounding_score).
    Grounding score = min(1, evidence_count / 3) * distinct_episodes_fraction.
    """

    @property
    def name(self) -> str:
        return "confidence_discipline"

    @property
    def tier(self) -> int:
        return 1

    @property
    def description(self) -> str:
        return "Penalizes high confidence with weak evidence grounding"

    def compute(self, result: RunResult) -> MetricResult:
        insights = _all_insights(result)
        if not insights:
            return MetricResult(name=self.name, tier=self.tier, value=0.0)

        penalties: list[float] = []
        for insight in insights:
            n_refs = len(insight.evidence)
            n_episodes = len({ref.episode_id for ref in insight.evidence})

            # Grounding: how well-supported is this insight?
            ref_score = min(1.0, n_refs / 3)
            episode_score = min(1.0, n_episodes / 2) if n_refs > 0 else 0.0
            grounding = ref_score * episode_score

            # Penalty: confidence exceeding grounding
            penalty = max(0.0, insight.confidence - grounding)
            penalties.append(penalty)

        avg_penalty = sum(penalties) / len(penalties)
        return MetricResult(
            name=self.name,
            tier=self.tier,
            value=1.0 - avg_penalty,
            details={"avg_penalty": avg_penalty, "total_insights": len(insights)},
        )


@register_metric("budget_compliance")
class BudgetCompliance(BaseMetric):
    """BC: % of online calls within all caps.

    Uses validation_errors from checkpoint results to detect violations.
    """

    @property
    def name(self) -> str:
        return "budget_compliance"

    @property
    def tier(self) -> int:
        return 1

    @property
    def description(self) -> str:
        return "Percentage of method calls within budget and latency caps"

    def compute(self, result: RunResult) -> MetricResult:
        total_checks = 0
        violations = 0

        for persona in result.personas:
            for cp in persona.checkpoints:
                total_checks += 1  # core call
                total_checks += len(cp.search_results)  # search calls

                # Count budget/latency violations in validation errors
                for err in cp.validation_errors:
                    if "latency" in err.lower() or "budget" in err.lower():
                        violations += 1

        value = (total_checks - violations) / total_checks if total_checks > 0 else 1.0
        return MetricResult(
            name=self.name,
            tier=self.tier,
            value=max(0.0, value),
            details={"total_checks": total_checks, "violations": violations},
        )

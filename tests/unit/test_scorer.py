from __future__ import annotations

from lens.core.models import (
    CheckpointResult,
    EvidenceRef,
    Insight,
    MetricResult,
    PersonaResult,
    RunResult,
)
from lens.scorer.aggregate import compute_composite
from lens.scorer.registry import list_metrics
from lens.scorer.tier1 import (
    BudgetCompliance,
    ConfidenceDiscipline,
    EvidenceSufficiency,
    EvidenceValidity,
    MultiEpisodeSupport,
    NonLocality,
)


def _make_run(insights: list[Insight], validation_errors: list[str] | None = None) -> RunResult:
    """Helper to create a RunResult with given insights."""
    return RunResult(
        run_id="test",
        adapter="test",
        dataset_version="0.1.0",
        budget_preset="standard",
        personas=[
            PersonaResult(
                persona_id="p1",
                checkpoints=[
                    CheckpointResult(
                        persona_id="p1",
                        checkpoint=10,
                        insights=insights,
                        validation_errors=validation_errors or [],
                    )
                ],
            )
        ],
    )


class TestEvidenceValidity:
    def test_all_valid(self):
        insights = [
            Insight(
                text="test",
                confidence=0.5,
                evidence=[
                    EvidenceRef(episode_id="ep_001", quote="quote1"),
                    EvidenceRef(episode_id="ep_002", quote="quote2"),
                ],
                falsifier="f",
            )
        ]
        result = _make_run(insights, [])
        metric = EvidenceValidity()
        mr = metric.compute(result)
        assert mr.value == 1.0

    def test_one_invalid(self):
        insights = [
            Insight(
                text="test",
                confidence=0.5,
                evidence=[
                    EvidenceRef(episode_id="ep_001", quote="good"),
                    EvidenceRef(episode_id="ep_002", quote="bad"),
                ],
                falsifier="f",
            )
        ]
        result = _make_run(
            insights,
            ["insight[0].evidence[1]: quote not found in episode 'ep_002'"],
        )
        metric = EvidenceValidity()
        mr = metric.compute(result)
        assert mr.value == 0.5

    def test_empty(self):
        result = _make_run([])
        metric = EvidenceValidity()
        mr = metric.compute(result)
        assert mr.value == 0.0


class TestEvidenceSufficiency:
    def test_sufficient(self):
        insights = [
            Insight(
                text="t",
                confidence=0.5,
                evidence=[
                    EvidenceRef(episode_id="e1", quote="q1"),
                    EvidenceRef(episode_id="e2", quote="q2"),
                    EvidenceRef(episode_id="e3", quote="q3"),
                ],
                falsifier="f",
            )
        ]
        result = _make_run(insights)
        assert EvidenceSufficiency().compute(result).value == 1.0

    def test_insufficient(self):
        insights = [
            Insight(
                text="t",
                confidence=0.5,
                evidence=[EvidenceRef(episode_id="e1", quote="q1")],
                falsifier="f",
            )
        ]
        result = _make_run(insights)
        assert EvidenceSufficiency().compute(result).value == 0.0


class TestMultiEpisodeSupport:
    def test_multi_episode(self):
        insights = [
            Insight(
                text="t",
                confidence=0.5,
                evidence=[
                    EvidenceRef(episode_id="e1", quote="q1"),
                    EvidenceRef(episode_id="e2", quote="q2"),
                ],
                falsifier="f",
            )
        ]
        result = _make_run(insights)
        assert MultiEpisodeSupport().compute(result).value == 1.0

    def test_single_episode(self):
        insights = [
            Insight(
                text="t",
                confidence=0.5,
                evidence=[
                    EvidenceRef(episode_id="e1", quote="q1"),
                    EvidenceRef(episode_id="e1", quote="q2"),
                ],
                falsifier="f",
            )
        ]
        result = _make_run(insights)
        assert MultiEpisodeSupport().compute(result).value == 0.0


class TestNonLocality:
    def test_non_local(self):
        insights = [
            Insight(
                text="t",
                confidence=0.5,
                evidence=[
                    EvidenceRef(episode_id="e1", quote="q1"),
                    EvidenceRef(episode_id="e2", quote="q2"),
                    EvidenceRef(episode_id="e3", quote="q3"),
                    EvidenceRef(episode_id="e4", quote="q4"),
                    EvidenceRef(episode_id="e5", quote="q5"),
                ],
                falsifier="f",
            )
        ]
        result = _make_run(insights)
        assert NonLocality().compute(result).value == 1.0

    def test_local(self):
        insights = [
            Insight(
                text="t",
                confidence=0.5,
                evidence=[
                    EvidenceRef(episode_id="e1", quote="q1"),
                    EvidenceRef(episode_id="e1", quote="q2"),
                    EvidenceRef(episode_id="e1", quote="q3"),
                    EvidenceRef(episode_id="e1", quote="q4"),
                    EvidenceRef(episode_id="e2", quote="q5"),
                ],
                falsifier="f",
            )
        ]
        result = _make_run(insights)
        # e1 has 4/5 = 80%, so this is NOT non-local
        assert NonLocality().compute(result).value == 0.0


class TestConfidenceDiscipline:
    def test_well_grounded_high_confidence(self):
        insights = [
            Insight(
                text="t",
                confidence=0.8,
                evidence=[
                    EvidenceRef(episode_id="e1", quote="q1"),
                    EvidenceRef(episode_id="e2", quote="q2"),
                    EvidenceRef(episode_id="e3", quote="q3"),
                ],
                falsifier="f",
            )
        ]
        result = _make_run(insights)
        mr = ConfidenceDiscipline().compute(result)
        assert mr.value > 0.7  # Well-grounded, so low penalty

    def test_poorly_grounded_high_confidence(self):
        insights = [
            Insight(
                text="t",
                confidence=0.95,
                evidence=[],
                falsifier="f",
            )
        ]
        result = _make_run(insights)
        mr = ConfidenceDiscipline().compute(result)
        assert mr.value < 0.2  # Poorly grounded + high confidence = big penalty


class TestBudgetCompliance:
    def test_all_compliant(self):
        result = _make_run([])
        mr = BudgetCompliance().compute(result)
        assert mr.value == 1.0


class TestMetricRegistry:
    def test_all_tier1_registered(self):
        metrics = list_metrics()
        tier1_names = {
            "evidence_validity",
            "evidence_sufficiency",
            "multi_episode_support",
            "non_locality",
            "confidence_discipline",
            "budget_compliance",
        }
        assert tier1_names.issubset(set(metrics.keys()))

    def test_tier2_registered(self):
        metrics = list_metrics()
        assert "survival_at_k" in metrics
        assert "churn_at_k" in metrics


class TestCompositeScore:
    def test_composite(self):
        metrics = [
            MetricResult(name="evidence_validity", tier=1, value=1.0),
            MetricResult(name="multi_episode_support", tier=1, value=0.8),
            MetricResult(name="non_locality", tier=1, value=0.6),
            MetricResult(name="confidence_discipline", tier=1, value=0.9),
            MetricResult(name="survival_at_k", tier=2, value=0.7),
            MetricResult(name="budget_compliance", tier=1, value=1.0),
        ]
        score = compute_composite(metrics)
        assert 0.0 < score < 1.0

    def test_empty_metrics(self):
        assert compute_composite([]) == 0.0

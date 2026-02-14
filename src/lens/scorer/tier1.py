from __future__ import annotations

from lens.core.models import MetricResult, QuestionResult, RunResult
from lens.scorer.base import BaseMetric
from lens.scorer.registry import register_metric


def _all_question_results(result: RunResult) -> list[QuestionResult]:
    """Collect all QuestionResults across all personas and checkpoints."""
    qrs: list[QuestionResult] = []
    for persona in result.personas:
        for cp in persona.checkpoints:
            qrs.extend(cp.question_results)
    return qrs


@register_metric("evidence_grounding")
class EvidenceGrounding(BaseMetric):
    """Fraction of retrieved ref_ids that exist in the vault."""

    @property
    def name(self) -> str:
        return "evidence_grounding"

    @property
    def tier(self) -> int:
        return 1

    @property
    def description(self) -> str:
        return "Fraction of retrieved ref_ids that exist in the vault"

    def compute(self, result: RunResult) -> MetricResult:
        qrs = _all_question_results(result)
        total_retrieved = 0
        total_valid = 0
        for qr in qrs:
            total_retrieved += len(qr.retrieved_ref_ids)
            total_valid += len(qr.valid_ref_ids)

        value = total_valid / total_retrieved if total_retrieved > 0 else 0.0
        return MetricResult(
            name=self.name,
            tier=self.tier,
            value=value,
            details={"total_retrieved": total_retrieved, "total_valid": total_valid},
        )


@register_metric("fact_recall")
class FactRecall(BaseMetric):
    """Fraction of ground-truth key_facts found in the answer text."""

    @property
    def name(self) -> str:
        return "fact_recall"

    @property
    def tier(self) -> int:
        return 1

    @property
    def description(self) -> str:
        return "Fraction of ground-truth key_facts found in the answer text"

    def compute(self, result: RunResult) -> MetricResult:
        qrs = _all_question_results(result)
        if not qrs:
            return MetricResult(name=self.name, tier=self.tier, value=0.0)

        scores: list[float] = []
        for qr in qrs:
            key_facts = qr.question.ground_truth.key_facts
            if not key_facts:
                scores.append(1.0)
                continue
            answer_lower = qr.answer.answer_text.lower()
            found = sum(1 for f in key_facts if f.lower() in answer_lower)
            scores.append(found / len(key_facts))

        value = sum(scores) / len(scores)
        return MetricResult(
            name=self.name,
            tier=self.tier,
            value=value,
            details={"num_questions": len(scores)},
        )


@register_metric("evidence_coverage")
class EvidenceCoverage(BaseMetric):
    """Fraction of required evidence refs that the agent actually retrieved."""

    @property
    def name(self) -> str:
        return "evidence_coverage"

    @property
    def tier(self) -> int:
        return 1

    @property
    def description(self) -> str:
        return "Fraction of required evidence refs actually retrieved by the agent"

    def compute(self, result: RunResult) -> MetricResult:
        qrs = _all_question_results(result)
        if not qrs:
            return MetricResult(name=self.name, tier=self.tier, value=0.0)

        scores: list[float] = []
        for qr in qrs:
            required = qr.question.ground_truth.required_evidence_refs
            if not required:
                scores.append(1.0)
                continue
            retrieved_set = set(qr.retrieved_ref_ids)
            found = sum(1 for r in required if r in retrieved_set)
            scores.append(found / len(required))

        value = sum(scores) / len(scores)
        return MetricResult(
            name=self.name,
            tier=self.tier,
            value=value,
            details={"num_questions": len(scores)},
        )


@register_metric("budget_compliance")
class BudgetCompliance(BaseMetric):
    """1.0 if no budget violations, degrades by 0.1 per violation."""

    @property
    def name(self) -> str:
        return "budget_compliance"

    @property
    def tier(self) -> int:
        return 1

    @property
    def description(self) -> str:
        return "Budget compliance score — degrades by 0.1 per violation"

    def compute(self, result: RunResult) -> MetricResult:
        qrs = _all_question_results(result)
        total_violations = 0
        for qr in qrs:
            total_violations += len(qr.answer.budget_violations)

        value = max(0.0, 1.0 - 0.1 * total_violations)
        return MetricResult(
            name=self.name,
            tier=self.tier,
            value=value,
            details={"total_violations": total_violations},
        )

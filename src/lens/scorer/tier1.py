from __future__ import annotations

from lens.core.models import MetricResult, QuestionResult, RunResult
from lens.scorer.base import BaseMetric
from lens.scorer.registry import register_metric


def _all_question_results(result: RunResult) -> list[QuestionResult]:
    """Collect all QuestionResults across all scopes and checkpoints."""
    qrs: list[QuestionResult] = []
    for scope in result.scopes:
        for cp in scope.checkpoints:
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
    """Observational metric — records token usage and wall time per run.

    Not gated. The score is 1.0 - (violations / total_questions) which
    gives the fraction of questions that stayed within budget, but the
    primary value is the detailed stats in the ``details`` dict.
    """

    @property
    def name(self) -> str:
        return "budget_compliance"

    @property
    def tier(self) -> int:
        return 1

    @property
    def description(self) -> str:
        return "Budget compliance — observational (token/time stats)"

    def compute(self, result: RunResult) -> MetricResult:
        qrs = _all_question_results(result)
        total_violations = 0
        total_tokens = 0
        max_tokens = 0
        total_wall_ms = 0.0
        max_wall_ms = 0.0
        per_question_timing: list[dict] = []
        for qr in qrs:
            total_violations += len(qr.answer.budget_violations)
            tokens = qr.answer.total_tokens
            wall = qr.answer.wall_time_ms
            total_tokens += tokens
            if tokens > max_tokens:
                max_tokens = tokens
            total_wall_ms += wall
            if wall > max_wall_ms:
                max_wall_ms = wall
            per_question_timing.append({
                "question_id": qr.question.question_id,
                "question_type": qr.question.question_type,
                "checkpoint": qr.question.checkpoint_after,
                "wall_time_ms": round(wall, 1),
                "total_tokens": tokens,
                "tool_calls": qr.answer.tool_calls_made,
            })

        # Sort by wall_time descending (slowest first)
        per_question_timing.sort(key=lambda x: x["wall_time_ms"], reverse=True)

        n = len(qrs) if qrs else 1
        value = max(0.0, 1.0 - total_violations / n)
        return MetricResult(
            name=self.name,
            tier=self.tier,
            value=value,
            details={
                "total_questions": len(qrs),
                "total_violations": total_violations,
                "violation_rate": total_violations / n,
                "total_tokens": total_tokens,
                "avg_tokens_per_question": total_tokens / n,
                "max_tokens_single_question": max_tokens,
                "total_wall_time_minutes": round(total_wall_ms / 60_000, 2),
                "avg_wall_time_ms": round(total_wall_ms / n, 1),
                "max_wall_time_ms": round(max_wall_ms, 1),
                "per_question_timing": per_question_timing,
            },
        )


@register_metric("citation_coverage")
class CitationCoverage(BaseMetric):
    """Fraction of required_evidence_refs that appear in agent citations.

    Measures whether the agent cited the correct evidence episodes.
    Uses refs_cited (from tool calls + inline [ref_id] citations)
    and valid_ref_ids to check against required_evidence_refs.
    """

    @property
    def name(self) -> str:
        return "citation_coverage"

    @property
    def tier(self) -> int:
        return 1

    @property
    def description(self) -> str:
        return "Fraction of required evidence refs cited by the agent"

    def compute(self, result: RunResult) -> MetricResult:
        qrs = _all_question_results(result)
        if not qrs:
            return MetricResult(name=self.name, tier=self.tier, value=0.0)

        scores: list[float] = []
        for qr in qrs:
            required = set(qr.question.ground_truth.required_evidence_refs)
            if not required:
                scores.append(1.0)
                continue
            # Combine tool-call refs and inline citation refs
            cited = set(qr.retrieved_ref_ids) | set(qr.valid_ref_ids)
            overlap = required & cited
            scores.append(len(overlap) / len(required))

        value = sum(scores) / len(scores)
        return MetricResult(
            name=self.name,
            tier=self.tier,
            value=value,
            details={"num_questions": len(scores)},
        )

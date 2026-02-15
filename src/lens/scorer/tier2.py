from __future__ import annotations

from lens.core.models import MetricResult, RunResult
from lens.scorer.base import BaseMetric
from lens.scorer.registry import register_metric
from lens.scorer.tier1 import _all_question_results


@register_metric("answer_quality")
class AnswerQuality(BaseMetric):
    """Overall answer correctness — requires LLM judge (not yet implemented).

    This metric is intended to use an LLM judge to compare the agent's answer
    against the ground truth canonical_answer, assessing semantic correctness
    beyond simple substring matching. Currently returns 0.0 as a stub.

    Once implemented, this will NOT duplicate fact_recall — fact_recall checks
    for specific factual claims via substring matching, while answer_quality
    will assess holistic correctness, coherence, and completeness via LLM judge.
    """

    @property
    def name(self) -> str:
        return "answer_quality"

    @property
    def tier(self) -> int:
        return 2

    @property
    def description(self) -> str:
        return "Overall answer correctness via LLM judge (stub — not yet implemented)"

    def compute(self, result: RunResult) -> MetricResult:
        return MetricResult(
            name=self.name,
            tier=self.tier,
            value=0.0,
            details={"not_implemented": True},
        )


@register_metric("insight_depth")
class InsightDepth(BaseMetric):
    """Cross-episode reasoning — fraction of questions citing refs from 2+ distinct episodes."""

    @property
    def name(self) -> str:
        return "insight_depth"

    @property
    def tier(self) -> int:
        return 2

    @property
    def description(self) -> str:
        return "Fraction of questions where the agent cited refs from 2+ distinct episodes"

    def compute(self, result: RunResult) -> MetricResult:
        qrs = _all_question_results(result)
        if not qrs:
            return MetricResult(name=self.name, tier=self.tier, value=0.0)

        multi_episode_count = 0
        for qr in qrs:
            distinct_refs = set(qr.retrieved_ref_ids)
            if len(distinct_refs) >= 2:
                multi_episode_count += 1

        value = multi_episode_count / len(qrs)
        return MetricResult(
            name=self.name,
            tier=self.tier,
            value=value,
            details={
                "num_questions": len(qrs),
                "multi_episode_questions": multi_episode_count,
            },
        )


@register_metric("reasoning_quality")
class ReasoningQuality(BaseMetric):
    """Logical coherence proxy — fraction of questions with substantive answers and tool use."""

    @property
    def name(self) -> str:
        return "reasoning_quality"

    @property
    def tier(self) -> int:
        return 2

    @property
    def description(self) -> str:
        return "Fraction of questions with answer > 50 chars and tool_calls > 0"

    def compute(self, result: RunResult) -> MetricResult:
        qrs = _all_question_results(result)
        if not qrs:
            return MetricResult(name=self.name, tier=self.tier, value=0.0)

        qualified = 0
        for qr in qrs:
            if len(qr.answer.answer_text) > 50 and qr.answer.tool_calls_made > 0:
                qualified += 1

        value = qualified / len(qrs)
        return MetricResult(
            name=self.name,
            tier=self.tier,
            value=value,
            details={"num_questions": len(qrs), "qualified": qualified},
        )

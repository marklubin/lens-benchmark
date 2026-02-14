from __future__ import annotations

from lens.core.models import MetricResult, RunResult
from lens.scorer.base import BaseMetric
from lens.scorer.registry import register_metric
from lens.scorer.tier1 import _all_question_results


@register_metric("answer_quality")
class AnswerQuality(BaseMetric):
    """Overall correctness proxy — average fact recall per question."""

    @property
    def name(self) -> str:
        return "answer_quality"

    @property
    def tier(self) -> int:
        return 2

    @property
    def description(self) -> str:
        return "Overall answer correctness (fact-recall heuristic proxy)"

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

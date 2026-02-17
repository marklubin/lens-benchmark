from __future__ import annotations

from typing import Callable

from lens.core.models import MetricResult, RunResult
from lens.scorer.base import BaseMetric
from lens.scorer.registry import register_metric
from lens.scorer.tier1 import _all_question_results


@register_metric("answer_quality")
class AnswerQuality(BaseMetric):
    """Answer quality via pairwise LLM judge comparison.

    Compares each agent answer against the canonical ground-truth answer
    using pairwise judging. For each key fact, the judge picks which answer
    better demonstrates the finding. Position bias is controlled via
    random assignment.

    Requires a judge_fn to be set via configure(). Without it, returns
    0.0 as a stub (backward compatible).
    """

    def __init__(self, judge_fn: Callable[[str], str] | None = None) -> None:
        self._judge_fn = judge_fn

    def configure(self, *, judge_fn: Callable[[str], str] | None = None) -> None:
        """Inject the LLM judge callable after construction."""
        if judge_fn is not None:
            self._judge_fn = judge_fn

    @property
    def name(self) -> str:
        return "answer_quality"

    @property
    def tier(self) -> int:
        return 2

    @property
    def description(self) -> str:
        return "Pairwise answer quality — candidate vs canonical ground truth"

    def compute(self, result: RunResult) -> MetricResult:
        if self._judge_fn is None:
            return MetricResult(
                name=self.name,
                tier=self.tier,
                value=0.0,
                details={"not_implemented": True},
            )

        from lens.scorer.judge import pairwise_fact_judge

        qrs = _all_question_results(result)
        if not qrs:
            return MetricResult(name=self.name, tier=self.tier, value=0.0)

        scores: list[float] = []
        per_question: list[dict] = []

        for qr in qrs:
            key_facts = qr.question.ground_truth.key_facts
            if not key_facts:
                scores.append(1.0)
                continue

            win_rate, details = pairwise_fact_judge(
                candidate_answer=qr.answer.answer_text,
                reference_answer=qr.question.ground_truth.canonical_answer,
                key_facts=key_facts,
                question=qr.question.prompt,
                judge_fn=self._judge_fn,
            )
            scores.append(win_rate)
            per_question.append({
                "question_id": qr.question.question_id,
                "win_rate": win_rate,
                "fact_details": details,
            })

        value = sum(scores) / len(scores) if scores else 0.0
        return MetricResult(
            name=self.name,
            tier=self.tier,
            value=value,
            details={"per_question": per_question, "method": "pairwise"},
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

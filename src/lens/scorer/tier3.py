from __future__ import annotations

import concurrent.futures
import logging
from typing import Callable

from lens.core.models import MetricResult, RunResult
from lens.scorer.base import BaseMetric
from lens.scorer.registry import register_metric
from lens.scorer.tier1 import _all_question_results

logger = logging.getLogger(__name__)


def _fact_recall_score(answer_text: str, key_facts: list[str]) -> float:
    """Compute fact recall for a single question."""
    if not key_facts:
        return 0.5
    answer_lower = answer_text.lower()
    found = sum(1 for f in key_facts if f.lower() in answer_lower)
    return found / len(key_facts)


def _judge_fact_score(
    answer_text: str,
    canonical_answer: str,
    key_facts: list[str],
    question_prompt: str,
    judge_fn: Callable[[str], str],
    seed: int = 42,
    max_workers: int = 1,
) -> float:
    """Score a question using pairwise judge instead of substring matching."""
    from lens.scorer.judge import pairwise_fact_judge

    win_rate, _ = pairwise_fact_judge(
        candidate_answer=answer_text,
        reference_answer=canonical_answer,
        key_facts=key_facts,
        question=question_prompt,
        judge_fn=judge_fn,
        seed=seed,
        max_workers=max_workers,
    )
    return win_rate


# Question types that require cross-episode synthesis (numerator for advantage metric)
SYNTHESIS_QUESTION_TYPES = {
    "longitudinal", "negative", "temporal", "counterfactual", "paraphrase",
    "distractor_resistance", "severity_assessment", "evidence_sufficiency",
}

# Control question types (denominator for advantage metric)
CONTROL_QUESTION_TYPES = {"null_hypothesis"}


@register_metric("longitudinal_advantage")
class LongitudinalAdvantage(BaseMetric):
    """Differential: mean score for synthesis questions minus control questions.

    Synthesis types: longitudinal, negative, temporal, counterfactual, paraphrase.
    Control types: null_hypothesis.

    When a judge_fn is configured (via configure()), uses pairwise judging
    instead of substring matching for more accurate scoring.
    """

    def __init__(self) -> None:
        self._judge_fn: Callable[[str], str] | None = None
        self._max_workers: int = 1

    def configure(
        self,
        *,
        judge_fn: Callable[[str], str] | None = None,
        max_judge_workers: int | None = None,
        **kwargs,
    ) -> None:
        """Inject the LLM judge callable after construction."""
        if judge_fn is not None:
            self._judge_fn = judge_fn
        if max_judge_workers is not None:
            self._max_workers = max_judge_workers

    @property
    def name(self) -> str:
        return "longitudinal_advantage"

    @property
    def tier(self) -> int:
        return 3

    @property
    def description(self) -> str:
        return "Mean score for synthesis questions minus control questions"

    def compute(self, result: RunResult) -> MetricResult:
        qrs = _all_question_results(result)

        if self._judge_fn and self._max_workers > 1:
            # Parallel judge scoring
            def _score_qr(qr):
                return qr, _judge_fact_score(
                    qr.answer.answer_text,
                    qr.question.ground_truth.canonical_answer,
                    qr.question.ground_truth.key_facts,
                    qr.question.prompt,
                    self._judge_fn,
                    max_workers=self._max_workers,
                )

            with concurrent.futures.ThreadPoolExecutor(
                max_workers=min(self._max_workers, len(qrs) or 1)
            ) as pool:
                scored = list(pool.map(_score_qr, qrs))
        else:
            scored = []
            for qr in qrs:
                if self._judge_fn:
                    s = _judge_fact_score(
                        qr.answer.answer_text,
                        qr.question.ground_truth.canonical_answer,
                        qr.question.ground_truth.key_facts,
                        qr.question.prompt,
                        self._judge_fn,
                        max_workers=self._max_workers,
                    )
                else:
                    s = _fact_recall_score(
                        qr.answer.answer_text, qr.question.ground_truth.key_facts
                    )
                scored.append((qr, s))

        synthesis_scores: list[float] = []
        control_scores: list[float] = []
        for qr, score in scored:
            if qr.question.question_type in SYNTHESIS_QUESTION_TYPES:
                synthesis_scores.append(score)
            elif qr.question.question_type in CONTROL_QUESTION_TYPES:
                control_scores.append(score)

        if not synthesis_scores or not control_scores:
            return MetricResult(
                name=self.name,
                tier=self.tier,
                value=0.0,
                details={
                    "synthesis_count": len(synthesis_scores),
                    "control_count": len(control_scores),
                },
            )

        synthesis_mean = sum(synthesis_scores) / len(synthesis_scores)
        control_mean = sum(control_scores) / len(control_scores)
        value = synthesis_mean - control_mean

        method = "pairwise" if self._judge_fn else "substring"
        return MetricResult(
            name=self.name,
            tier=self.tier,
            value=value,
            details={
                "synthesis_mean": synthesis_mean,
                "control_mean": control_mean,
                "synthesis_count": len(synthesis_scores),
                "control_count": len(control_scores),
                "method": method,
            },
        )


@register_metric("action_quality")
class ActionQuality(BaseMetric):
    """Mean score for action_recommendation questions.

    When a judge_fn is configured (via configure()), uses pairwise judging
    instead of substring matching for more accurate scoring.
    """

    def __init__(self) -> None:
        self._judge_fn: Callable[[str], str] | None = None
        self._max_workers: int = 1

    def configure(
        self,
        *,
        judge_fn: Callable[[str], str] | None = None,
        max_judge_workers: int | None = None,
        **kwargs,
    ) -> None:
        """Inject the LLM judge callable after construction."""
        if judge_fn is not None:
            self._judge_fn = judge_fn
        if max_judge_workers is not None:
            self._max_workers = max_judge_workers

    @property
    def name(self) -> str:
        return "action_quality"

    @property
    def tier(self) -> int:
        return 3

    @property
    def description(self) -> str:
        return "Mean score for action_recommendation questions"

    def compute(self, result: RunResult) -> MetricResult:
        qrs = _all_question_results(result)

        action_qrs = [qr for qr in qrs if qr.question.question_type == "action_recommendation"]

        if not action_qrs:
            return MetricResult(
                name=self.name,
                tier=self.tier,
                value=0.0,
                details={"action_recommendation_count": 0},
            )

        def _score_one(qr):
            if self._judge_fn:
                return _judge_fact_score(
                    qr.answer.answer_text,
                    qr.question.ground_truth.canonical_answer,
                    qr.question.ground_truth.key_facts,
                    qr.question.prompt,
                    self._judge_fn,
                    max_workers=self._max_workers,
                )
            return _fact_recall_score(
                qr.answer.answer_text, qr.question.ground_truth.key_facts
            )

        if self._judge_fn and self._max_workers > 1 and len(action_qrs) > 1:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=min(self._max_workers, len(action_qrs))
            ) as pool:
                scores = list(pool.map(_score_one, action_qrs))
        else:
            scores = [_score_one(qr) for qr in action_qrs]

        value = sum(scores) / len(scores)
        method = "pairwise" if self._judge_fn else "substring"
        return MetricResult(
            name=self.name,
            tier=self.tier,
            value=value,
            details={
                "action_recommendation_count": len(scores),
                "method": method,
            },
        )


@register_metric("naive_baseline_advantage")
class NaiveBaselineAdvantage(BaseMetric):
    """Head-to-head: does the adapter beat context stuffing?

    For each question, a naive baseline answer is generated by concatenating
    all episodes up to the checkpoint and asking the same LLM. Then a pairwise
    judge compares the adapter's answer vs the naive answer per key fact.

    value = overall mean win rate (0.5 = parity, >0.5 = adapter beats naive).

    Requires both judge_fn and baseline_generator to be set via configure().
    Without them, returns 0.0 as a stub.
    """

    def __init__(self) -> None:
        self._judge_fn: Callable[[str], str] | None = None
        self._baseline_generator = None  # NaiveBaselineGenerator
        self._max_workers: int = 1

    def configure(
        self,
        *,
        judge_fn: Callable[[str], str] | None = None,
        baseline_generator=None,
        max_judge_workers: int | None = None,
        **kwargs,
    ) -> None:
        """Inject judge and baseline generator after construction."""
        if judge_fn is not None:
            self._judge_fn = judge_fn
        if baseline_generator is not None:
            self._baseline_generator = baseline_generator
        if max_judge_workers is not None:
            self._max_workers = max_judge_workers

    @property
    def name(self) -> str:
        return "naive_baseline_advantage"

    @property
    def tier(self) -> int:
        return 3

    @property
    def description(self) -> str:
        return "Adapter vs context-stuffed naive baseline — pairwise win rate"

    def compute(self, result: RunResult) -> MetricResult:
        if self._judge_fn is None or self._baseline_generator is None:
            return MetricResult(
                name=self.name,
                tier=self.tier,
                value=0.0,
                details={"not_configured": True},
            )

        from lens.scorer.judge import pairwise_fact_judge

        qrs = _all_question_results(result)
        if not qrs:
            return MetricResult(name=self.name, tier=self.tier, value=0.0)

        # Pre-generate all baselines in parallel via generate_all()
        questions_needing_baseline = [
            qr.question for qr in qrs if qr.question.ground_truth.key_facts
        ]
        baseline_answers = self._baseline_generator.generate_all(questions_needing_baseline)

        qrs_with_baselines = []
        no_fact_scores: list[float] = []
        for qr in qrs:
            key_facts = qr.question.ground_truth.key_facts
            if not key_facts:
                no_fact_scores.append(0.5)
                continue
            naive_answer = baseline_answers.get(
                qr.question.question_id,
                self._baseline_generator.get_answer(qr.question),
            )
            qrs_with_baselines.append((qr, key_facts, naive_answer))

        def _score_one(args):
            qr, key_facts, naive_answer = args
            win_rate, details = pairwise_fact_judge(
                candidate_answer=qr.answer.answer_text,
                reference_answer=naive_answer,
                key_facts=key_facts,
                question=qr.question.prompt,
                judge_fn=self._judge_fn,
                max_workers=self._max_workers,
            )
            return qr, win_rate, details

        if self._max_workers > 1 and len(qrs_with_baselines) > 1:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=min(self._max_workers, len(qrs_with_baselines))
            ) as pool:
                judged = list(pool.map(_score_one, qrs_with_baselines))
        else:
            judged = [_score_one(args) for args in qrs_with_baselines]

        scores: list[float] = list(no_fact_scores)
        per_question: list[dict] = []
        wins = 0
        losses = 0
        ties = 0
        per_type: dict[str, list[float]] = {}

        for qr, win_rate, details in judged:
            scores.append(win_rate)

            for d in details:
                if d["winner"] == "candidate":
                    wins += 1
                elif d["winner"] == "reference":
                    losses += 1
                else:
                    ties += 1

            qtype = qr.question.question_type
            per_type.setdefault(qtype, []).append(win_rate)

            per_question.append({
                "question_id": qr.question.question_id,
                "question_type": qtype,
                "win_rate": round(win_rate, 4),
                "fact_count": len(qr.question.ground_truth.key_facts),
            })

        value = sum(scores) / len(scores) if scores else 0.0
        total_facts = wins + losses + ties

        per_type_summary = {
            k: round(sum(v) / len(v), 4) for k, v in sorted(per_type.items())
        }

        return MetricResult(
            name=self.name,
            tier=self.tier,
            value=value,
            details={
                "win_rate": round(wins / total_facts, 4) if total_facts else 0.0,
                "loss_rate": round(losses / total_facts, 4) if total_facts else 0.0,
                "tie_rate": round(ties / total_facts, 4) if total_facts else 0.0,
                "total_fact_comparisons": total_facts,
                "per_type": per_type_summary,
                "per_question": per_question,
                "method": "pairwise_vs_naive",
            },
        )

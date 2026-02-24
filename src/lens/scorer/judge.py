"""Pairwise LLM judge for comparing agent answers.

Implements position-debiased pairwise comparison: for each key fact,
randomly assigns answers to positions A/B, asks the judge which
better demonstrates the finding, then maps back to original answers.

This approach is more robust than absolute scoring (0-5 or YES/NO)
because:
  - Relative comparisons are easier for LLM judges
  - Position bias is controlled via random assignment
  - The canonical answer provides a stable anchor
"""
from __future__ import annotations

import concurrent.futures
import logging
import random

logger = logging.getLogger(__name__)


def pairwise_fact_judge(
    candidate_answer: str,
    reference_answer: str,
    key_facts: list[str],
    question: str,
    judge_fn,
    seed: int = 42,
    max_workers: int = 1,
) -> tuple[float, list[dict]]:
    """Compare candidate vs reference answer on key facts via pairwise judging.

    For each key fact, the judge picks which answer better demonstrates
    awareness of the finding. Position assignment is randomized to control
    for position bias.

    Args:
        candidate_answer: The answer being evaluated (e.g., agent's answer).
        reference_answer: The anchor answer (e.g., canonical ground truth).
        key_facts: Factual claims to check for.
        question: The question that was asked.
        judge_fn: Callable(prompt: str) -> str that returns "A", "B", or "TIE".
        seed: Random seed for reproducible position assignment.
        max_workers: Number of concurrent judge calls (>1 for parallel).

    Returns:
        (win_rate, per_fact_details) where win_rate is the fraction of facts
        where candidate wins (1.0) or ties (0.5).
    """
    if not key_facts:
        return 0.5, []

    # Pre-compute all position assignments deterministically before threading
    rng = random.Random(seed)
    tasks: list[tuple[str, bool, str]] = []
    for fact in key_facts:
        candidate_is_a = rng.random() < 0.5
        if candidate_is_a:
            text_a, text_b = candidate_answer, reference_answer
        else:
            text_a, text_b = reference_answer, candidate_answer
        prompt = _build_pairwise_prompt(question, fact, text_a, text_b)
        tasks.append((fact, candidate_is_a, prompt))

    def _judge_one(args: tuple[str, bool, str]) -> dict:
        fact, candidate_is_a, prompt = args
        try:
            verdict_raw = judge_fn(prompt).strip().upper()
        except Exception:
            logger.warning("Judge call failed for fact %r, defaulting to TIE", fact, exc_info=True)
            verdict_raw = "TIE"

        # Map positional verdict back to candidate/reference
        if verdict_raw.startswith("A"):
            winner = "candidate" if candidate_is_a else "reference"
        elif verdict_raw.startswith("B"):
            winner = "reference" if candidate_is_a else "candidate"
        else:
            winner = "tie"

        # Score: candidate wins = 1.0, tie = 0.5, reference wins = 0.0
        if winner == "candidate":
            fact_score = 1.0
        elif winner == "tie":
            fact_score = 0.5
        else:
            fact_score = 0.0

        return {
            "fact": fact,
            "winner": winner,
            "verdict_raw": verdict_raw,
            "candidate_position": "A" if candidate_is_a else "B",
            "fact_score": fact_score,
        }

    if max_workers > 1 and len(tasks) > 1:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(max_workers, len(tasks))
        ) as pool:
            results = list(pool.map(_judge_one, tasks))
    else:
        results = [_judge_one(t) for t in tasks]

    total_score = sum(r["fact_score"] for r in results)
    win_rate = total_score / len(key_facts)
    return win_rate, results


def position_swap_audit(
    scored_run_dir: str,
    judge_fn,
    n_samples: int = 100,
    max_workers: int = 4,
) -> dict:
    """Re-run a random sample of judge calls with A↔B positions swapped.

    Measures inter-rater reliability of the pairwise judge by comparing
    verdicts when the same content is presented in swapped positions.

    Args:
        scored_run_dir: Path to a scored run directory containing results.json
                        and scores/scorecard.json.
        judge_fn: Callable(prompt: str) -> str that returns "A", "B", or "TIE".
        n_samples: Number of judgment calls to re-run (randomly selected).
        max_workers: Number of concurrent judge calls.

    Returns:
        Dict with agreement stats: total, agree, disagree, agreement_pct,
        cohens_kappa, position_bias (fraction of A wins).
    """
    import json
    from pathlib import Path

    run_path = Path(scored_run_dir)

    # Load run results — try results.json first, then traverse checkpoint dirs
    judgeable: list[tuple[str, str, str, str]] = []

    results_file = run_path / "results.json"
    if results_file.exists():
        results = json.loads(results_file.read_text())
        for scope in results.get("scopes", []):
            for cp in scope.get("checkpoints", []):
                for qr in cp.get("question_results", []):
                    q = qr["question"]
                    key_facts = q.get("ground_truth", {}).get("key_facts", [])
                    canonical = q.get("ground_truth", {}).get("canonical_answer", "")
                    answer_text = qr.get("answer", {}).get("answer_text", "")
                    question_text = q.get("prompt", "")
                    if not key_facts or not answer_text or not canonical:
                        continue
                    for fact in key_facts:
                        judgeable.append((question_text, answer_text, canonical, fact))
    else:
        # Traverse scopes/*/checkpoint_*/question_results.json
        scopes_dir = run_path / "scopes"
        if not scopes_dir.exists():
            raise FileNotFoundError(
                f"No results.json or scopes/ directory in {scored_run_dir}"
            )
        for scope_dir in sorted(scopes_dir.iterdir()):
            if not scope_dir.is_dir():
                continue
            for cp_dir in sorted(scope_dir.iterdir()):
                if not cp_dir.is_dir() or not cp_dir.name.startswith("checkpoint_"):
                    continue
                qr_file = cp_dir / "question_results.json"
                if not qr_file.exists():
                    continue
                question_results = json.loads(qr_file.read_text())
                if not isinstance(question_results, list):
                    question_results = [question_results]
                for qr in question_results:
                    q = qr.get("question", {})
                    key_facts = q.get("ground_truth", {}).get("key_facts", [])
                    canonical = q.get("ground_truth", {}).get("canonical_answer", "")
                    answer_text = qr.get("answer", {}).get("answer_text", "")
                    question_text = q.get("prompt", "")
                    if not key_facts or not answer_text or not canonical:
                        continue
                    for fact in key_facts:
                        judgeable.append((question_text, answer_text, canonical, fact))

    if not judgeable:
        return {"error": "No judgeable items found", "total": 0}

    # Sample
    rng = random.Random(42)
    sample = rng.sample(judgeable, min(n_samples, len(judgeable)))

    def _judge_pair(item: tuple[str, str, str, str]) -> dict:
        question_text, candidate, reference, fact = item

        # Original order: candidate as A
        prompt_ab = _build_pairwise_prompt(question_text, fact, candidate, reference)
        try:
            verdict_ab = judge_fn(prompt_ab).strip().upper()
        except Exception:
            logger.warning("Judge call failed (A=candidate), defaulting to TIE", exc_info=True)
            verdict_ab = "TIE"

        # Swapped order: candidate as B
        prompt_ba = _build_pairwise_prompt(question_text, fact, reference, candidate)
        try:
            verdict_ba = judge_fn(prompt_ba).strip().upper()
        except Exception:
            logger.warning("Judge call failed (B=candidate), defaulting to TIE", exc_info=True)
            verdict_ba = "TIE"

        # Map to semantic winners
        def _map_winner(verdict: str, candidate_is_a: bool) -> str:
            if verdict.startswith("A"):
                return "candidate" if candidate_is_a else "reference"
            elif verdict.startswith("B"):
                return "reference" if candidate_is_a else "candidate"
            return "tie"

        winner_original = _map_winner(verdict_ab, candidate_is_a=True)
        winner_swapped = _map_winner(verdict_ba, candidate_is_a=False)

        return {
            "fact": fact,
            "verdict_ab": verdict_ab,
            "verdict_ba": verdict_ba,
            "winner_original": winner_original,
            "winner_swapped": winner_swapped,
            "agree": winner_original == winner_swapped,
        }

    if max_workers > 1 and len(sample) > 1:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(max_workers, len(sample))
        ) as pool:
            audit_results = list(pool.map(_judge_pair, sample))
    else:
        audit_results = [_judge_pair(s) for s in sample]

    total = len(audit_results)
    agree = sum(1 for r in audit_results if r["agree"])
    disagree = total - agree
    agreement_pct = agree / total if total > 0 else 0.0

    # Position bias: fraction of time A wins across all calls
    a_wins = sum(
        1 for r in audit_results
        for v in [r["verdict_ab"], r["verdict_ba"]]
        if v.startswith("A")
    )
    total_verdicts = total * 2
    position_bias = a_wins / total_verdicts if total_verdicts > 0 else 0.5

    # Cohen's kappa
    # Map to categories for kappa: candidate, reference, tie
    cats = ["candidate", "reference", "tie"]
    cat_idx = {c: i for i, c in enumerate(cats)}
    confusion = [[0] * 3 for _ in range(3)]
    for r in audit_results:
        i = cat_idx.get(r["winner_original"], 2)
        j = cat_idx.get(r["winner_swapped"], 2)
        confusion[i][j] += 1

    observed_agreement = agreement_pct
    # Expected agreement by chance
    expected = 0.0
    for k in range(3):
        row_sum = sum(confusion[k]) / total if total > 0 else 0
        col_sum = sum(confusion[i][k] for i in range(3)) / total if total > 0 else 0
        expected += row_sum * col_sum
    kappa = (observed_agreement - expected) / (1 - expected) if expected < 1.0 else 1.0

    return {
        "total": total,
        "agree": agree,
        "disagree": disagree,
        "agreement_pct": round(agreement_pct, 4),
        "cohens_kappa": round(kappa, 4),
        "position_bias": round(position_bias, 4),
        "details": audit_results,
    }


def _build_pairwise_prompt(
    question: str,
    fact: str,
    text_a: str,
    text_b: str,
) -> str:
    """Build the pairwise comparison prompt for the judge."""
    return (
        "You are comparing two analyst responses to determine which better "
        "demonstrates knowledge of a specific finding from a longitudinal dataset.\n\n"
        f"Question asked: {question}\n\n"
        f"Finding to evaluate: {fact}\n\n"
        f"Response A:\n{text_a}\n\n"
        f"Response B:\n{text_b}\n\n"
        "Which response better demonstrates awareness of this finding? "
        "The response may use different terminology. Focus on whether the "
        "core insight is present, not exact wording.\n\n"
        "Respond with ONLY one word: A, B, or TIE"
    )

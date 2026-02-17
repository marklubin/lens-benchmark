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

import random


def pairwise_fact_judge(
    candidate_answer: str,
    reference_answer: str,
    key_facts: list[str],
    question: str,
    judge_fn,
    seed: int = 42,
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

    Returns:
        (win_rate, per_fact_details) where win_rate is the fraction of facts
        where candidate wins (1.0) or ties (0.5).
    """
    if not key_facts:
        return 1.0, []

    rng = random.Random(seed)
    results: list[dict] = []
    total_score = 0.0

    for fact in key_facts:
        # Randomly assign positions to debias
        candidate_is_a = rng.random() < 0.5

        if candidate_is_a:
            text_a, text_b = candidate_answer, reference_answer
        else:
            text_a, text_b = reference_answer, candidate_answer

        prompt = _build_pairwise_prompt(question, fact, text_a, text_b)
        verdict_raw = judge_fn(prompt).strip().upper()

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

        total_score += fact_score
        results.append({
            "fact": fact,
            "winner": winner,
            "verdict_raw": verdict_raw,
            "candidate_position": "A" if candidate_is_a else "B",
            "fact_score": fact_score,
        })

    win_rate = total_score / len(key_facts)
    return win_rate, results


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

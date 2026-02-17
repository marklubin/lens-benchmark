"""Fact coverage and similarity scoring functions for the synix pipeline.

Self-contained — no lens.* imports.
"""
from __future__ import annotations


def compute_fact_coverage(answer: str, key_facts: list[str]) -> float:
    """Compute what fraction of key facts appear in an answer (fuzzy matching).

    Uses word-overlap: a fact counts as matched if >= 50% of its words
    appear in the answer.
    """
    if not key_facts:
        return 1.0

    answer_words = set(answer.lower().split())
    hits = 0

    for fact in key_facts:
        fact_words = set(fact.lower().split())
        overlap = fact_words & answer_words
        if len(overlap) >= max(1, len(fact_words) * 0.5):
            hits += 1

    return hits / len(key_facts)


def compute_per_fact_matches(answer: str, key_facts: list[str]) -> list[dict]:
    """Return per-fact match details: fact, matched, overlap_ratio."""
    if not key_facts:
        return []

    answer_words = set(answer.lower().split())
    results: list[dict] = []

    for fact in key_facts:
        fact_words = set(fact.lower().split())
        if not fact_words:
            results.append({"fact": fact, "matched": False, "overlap_ratio": 0.0})
            continue
        overlap = fact_words & answer_words
        ratio = len(overlap) / len(fact_words)
        matched = len(overlap) >= max(1, len(fact_words) * 0.5)
        results.append({
            "fact": fact,
            "matched": matched,
            "overlap_ratio": round(ratio, 3),
        })

    return results


def compute_fact_coverage_llm(
    answer: str,
    key_facts: list[str],
    question: str,
    client,
    config: dict,
) -> tuple[float, list[dict]]:
    """Compute fact coverage using LLM-as-judge semantic matching.

    For each key fact, asks a judge LLM whether the answer demonstrates
    knowledge of that finding. Returns (coverage_score, per_fact_details).

    Uses a lazy import of ``_logged_complete`` so that scoring.py remains
    importable without synix installed (word-overlap functions still work).
    """
    from synix.build.llm_transforms import _logged_complete

    if not key_facts:
        return 1.0, []

    results: list[dict] = []
    hits = 0

    for i, fact in enumerate(key_facts):
        prompt = (
            "You are evaluating whether an analyst's answer demonstrates knowledge "
            "of a specific finding from a longitudinal dataset.\n\n"
            f"Question that was asked: {question}\n\n"
            f"Finding to check for: {fact}\n\n"
            "Analyst's answer:\n"
            f"{answer}\n\n"
            "Does the answer demonstrate awareness of this finding? The answer may "
            "use different terminology. Focus on whether the core insight is "
            "present, not exact wording.\n\n"
            "Respond with ONLY one word: YES or NO"
        )

        response = _logged_complete(
            client, config,
            messages=[{"role": "user", "content": prompt}],
            artifact_desc=f"judge-fact-{i}",
        )

        verdict = response.content.strip().upper()
        matched = verdict.startswith("YES")
        if matched:
            hits += 1

        results.append({
            "fact": fact,
            "matched": matched,
            "judge_verdict": verdict,
            "judge_raw": response.content,
        })

    coverage = hits / len(key_facts)
    return coverage, results


def compute_pairwise_fact_coverage_llm(
    answer_a: str,
    answer_b: str,
    key_facts: list[str],
    question: str,
    client,
    config: dict,
    seed: int = 42,
) -> tuple[float, list[dict]]:
    """Pairwise fact coverage: for each fact, judge picks which answer is better.

    Position-debiased: randomly assigns answers to A/B slots to control
    for position bias. Returns (win_rate_a, per_fact_details) where
    win_rate_a is the fraction of facts where answer_a wins (1.0) or
    ties (0.5).
    """
    import random

    from synix.build.llm_transforms import _logged_complete

    if not key_facts:
        return 1.0, []

    rng = random.Random(seed)
    results: list[dict] = []
    total_score = 0.0

    for i, fact in enumerate(key_facts):
        a_is_first = rng.random() < 0.5

        if a_is_first:
            text_first, text_second = answer_a, answer_b
        else:
            text_first, text_second = answer_b, answer_a

        prompt = (
            "You are comparing two analyst responses to determine which better "
            "demonstrates knowledge of a specific finding.\n\n"
            f"Question asked: {question}\n\n"
            f"Finding to evaluate: {fact}\n\n"
            f"Response A:\n{text_first}\n\n"
            f"Response B:\n{text_second}\n\n"
            "Which response better demonstrates awareness of this finding? "
            "Focus on the core insight, not exact wording.\n\n"
            "Respond with ONLY one word: A, B, or TIE"
        )

        response = _logged_complete(
            client, config,
            messages=[{"role": "user", "content": prompt}],
            artifact_desc=f"pairwise-judge-fact-{i}",
        )

        verdict = response.content.strip().upper()

        # Map positional verdict back to answer_a/answer_b
        if verdict.startswith("A"):
            winner = "a" if a_is_first else "b"
        elif verdict.startswith("B"):
            winner = "b" if a_is_first else "a"
        else:
            winner = "tie"

        if winner == "a":
            fact_score = 1.0
        elif winner == "tie":
            fact_score = 0.5
        else:
            fact_score = 0.0

        total_score += fact_score
        results.append({
            "fact": fact,
            "winner": winner,
            "verdict_raw": response.content,
            "a_position": "first" if a_is_first else "second",
            "fact_score": fact_score,
        })

    win_rate = total_score / len(key_facts)
    return win_rate, results


def compute_distractor_similarity(text: str, key_facts: list[str]) -> float:
    """Compute max word-overlap similarity between a text and any key fact.

    Returns 0.0–1.0 representing the highest overlap ratio.
    """
    if not key_facts:
        return 0.0

    text_words = set(text.lower().split())
    max_sim = 0.0

    for fact in key_facts:
        fact_words = set(fact.lower().split())
        if not fact_words:
            continue
        overlap = fact_words & text_words
        sim = len(overlap) / len(fact_words)
        if sim > max_sim:
            max_sim = sim

    return max_sim


def compute_word_count(text: str) -> int:
    """Return the word count of a text string."""
    return len(text.split())

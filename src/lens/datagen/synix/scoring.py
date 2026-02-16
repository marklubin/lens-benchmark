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

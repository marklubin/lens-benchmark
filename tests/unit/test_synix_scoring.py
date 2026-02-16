"""Tests for lens.datagen.synix.scoring — fact coverage + similarity scoring."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src" / "lens" / "datagen" / "synix"))

import scoring  # noqa: E402


# ---------------------------------------------------------------------------
# compute_fact_coverage
# ---------------------------------------------------------------------------


class TestComputeFactCoverage:
    def test_all_facts_present(self):
        answer = "The API latency is increasing and the connection pool is exhausted"
        facts = ["API latency increasing", "connection pool exhausted"]
        assert scoring.compute_fact_coverage(answer, facts) == 1.0

    def test_no_facts_present(self):
        answer = "Everything looks normal"
        facts = ["API latency increasing", "connection pool exhausted"]
        assert scoring.compute_fact_coverage(answer, facts) == 0.0

    def test_partial_match(self):
        answer = "The API latency seems to be increasing slightly"
        facts = ["API latency increasing", "connection pool exhausted"]
        cov = scoring.compute_fact_coverage(answer, facts)
        assert cov == 0.5

    def test_empty_facts(self):
        assert scoring.compute_fact_coverage("anything", []) == 1.0

    def test_case_insensitive(self):
        answer = "the api LATENCY is INCREASING"
        facts = ["API latency increasing"]
        assert scoring.compute_fact_coverage(answer, facts) == 1.0

    def test_fuzzy_50_percent_threshold(self):
        # "API latency increasing" has 3 words, need at least 2 (50%)
        answer = "latency increasing but not API related"
        facts = ["API latency increasing"]
        cov = scoring.compute_fact_coverage(answer, facts)
        assert cov == 1.0  # 3/3 words present

    def test_below_threshold(self):
        # Only 1 of 3 words present (33%) — below 50%
        answer = "The latency is fine"
        facts = ["API latency increasing"]
        # "latency" matches, "API" and "increasing" don't
        # 1/3 = 0.33, max(1, 3*0.5) = max(1, 1.5) = 1.5 → need 2 words
        # Only 1 match < 1.5 required → NOT matched
        cov = scoring.compute_fact_coverage(answer, facts)
        assert cov == 0.0


# ---------------------------------------------------------------------------
# compute_per_fact_matches
# ---------------------------------------------------------------------------


class TestComputePerFactMatches:
    def test_returns_all_facts(self):
        results = scoring.compute_per_fact_matches(
            "API latency increasing",
            ["API latency increasing", "pool exhaustion"],
        )
        assert len(results) == 2
        assert results[0]["fact"] == "API latency increasing"
        assert results[0]["matched"] is True
        assert results[0]["overlap_ratio"] == 1.0

    def test_empty_facts(self):
        assert scoring.compute_per_fact_matches("text", []) == []

    def test_overlap_ratio(self):
        results = scoring.compute_per_fact_matches(
            "The latency is growing",
            ["API latency increasing"],
        )
        # "latency" matches (1/3 = 0.333)
        assert results[0]["overlap_ratio"] == pytest.approx(0.333, abs=0.01)

    def test_empty_fact_string(self):
        results = scoring.compute_per_fact_matches("text", [""])
        assert len(results) == 1
        assert results[0]["matched"] is False
        assert results[0]["overlap_ratio"] == 0.0


# ---------------------------------------------------------------------------
# compute_distractor_similarity
# ---------------------------------------------------------------------------


class TestComputeDistractorSimilarity:
    def test_high_similarity(self):
        text = "The API latency is increasing due to geo-lookup degradation"
        facts = ["API latency increasing"]
        sim = scoring.compute_distractor_similarity(text, facts)
        assert sim == 1.0

    def test_zero_similarity(self):
        text = "DNS migration proceeding on schedule with zone transfers complete"
        facts = ["API latency increasing", "connection pool exhaustion"]
        sim = scoring.compute_distractor_similarity(text, facts)
        assert sim == 0.0

    def test_empty_facts(self):
        assert scoring.compute_distractor_similarity("any text", []) == 0.0

    def test_max_across_facts(self):
        text = "The connection pool is showing signs of exhaustion"
        facts = ["API latency increasing", "connection pool exhaustion"]
        sim = scoring.compute_distractor_similarity(text, facts)
        # "connection pool exhaustion" has 3 words, 2 match = 0.667
        assert sim > 0.5

    def test_case_insensitive(self):
        text = "THE API LATENCY IS INCREASING"
        facts = ["api latency increasing"]
        sim = scoring.compute_distractor_similarity(text, facts)
        assert sim == 1.0


# ---------------------------------------------------------------------------
# compute_word_count
# ---------------------------------------------------------------------------


class TestComputeWordCount:
    def test_basic(self):
        assert scoring.compute_word_count("one two three") == 3

    def test_empty(self):
        assert scoring.compute_word_count("") == 0

    def test_whitespace(self):
        assert scoring.compute_word_count("  one  two  three  ") == 3

    def test_multiline(self):
        assert scoring.compute_word_count("one\ntwo\nthree") == 3

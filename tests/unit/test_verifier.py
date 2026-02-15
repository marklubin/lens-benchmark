from __future__ import annotations

import pytest

from lens.datagen.verifier import _compute_fact_coverage


class TestComputeFactCoverage:
    def test_full_coverage(self) -> None:
        answer = "The API latency was increasing and connection pool was exhausted."
        facts = ["API latency increasing", "connection pool exhausted"]
        assert _compute_fact_coverage(answer, facts) == 1.0

    def test_partial_coverage(self) -> None:
        answer = "The API latency was increasing."
        facts = ["API latency increasing", "connection pool exhausted"]
        assert _compute_fact_coverage(answer, facts) == 0.5

    def test_no_coverage(self) -> None:
        answer = "Everything was fine."
        facts = ["API latency increasing", "connection pool exhausted"]
        assert _compute_fact_coverage(answer, facts) == 0.0

    def test_empty_facts(self) -> None:
        answer = "Some answer."
        assert _compute_fact_coverage(answer, []) == 1.0

    def test_case_insensitive(self) -> None:
        answer = "The API LATENCY was INCREASING rapidly."
        facts = ["api latency increasing"]
        assert _compute_fact_coverage(answer, facts) == 1.0

    def test_fuzzy_matching_partial_words(self) -> None:
        """At least 50% of fact words must appear."""
        answer = "The latency was high."
        facts = ["API latency increasing rapidly"]  # 4 words, need 2
        # "latency" is 1 of 4 words = 25%, below 50% threshold
        assert _compute_fact_coverage(answer, facts) == 0.0

    def test_fuzzy_matching_sufficient_overlap(self) -> None:
        answer = "API latency was observed in the system."
        facts = ["API latency increasing"]  # 3 words, need 2 (50% of 3 = 1.5, ceil = 2)
        # "API" and "latency" = 2/3 = 66%, above 50%
        assert _compute_fact_coverage(answer, facts) == 1.0

from __future__ import annotations

import pytest

from lens.datagen.verifier import (
    _check_distractor_purity,
    _compute_fact_coverage,
    _compute_per_fact_matches,
)


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


class TestComputePerFactMatches:
    def test_all_matched(self) -> None:
        answer = "API latency increasing and connection pool exhausted."
        facts = ["API latency increasing", "connection pool exhausted"]
        result = _compute_per_fact_matches(answer, facts)
        assert len(result) == 2
        assert all(r["matched"] for r in result)
        assert result[0]["fact"] == "API latency increasing"
        assert result[0]["overlap_ratio"] == 1.0

    def test_partial_match(self) -> None:
        answer = "The API latency was high."
        facts = ["API latency increasing", "connection pool exhausted"]
        result = _compute_per_fact_matches(answer, facts)
        assert len(result) == 2
        assert result[0]["matched"] is True  # 2/3 words
        assert result[1]["matched"] is False

    def test_empty_facts(self) -> None:
        result = _compute_per_fact_matches("some answer", [])
        assert result == []

    def test_overlap_ratio_correct(self) -> None:
        answer = "latency was observed"
        facts = ["API latency increasing rapidly"]  # 4 words
        result = _compute_per_fact_matches(answer, facts)
        assert len(result) == 1
        assert result[0]["overlap_ratio"] == 0.25  # 1/4
        assert result[0]["matched"] is False

    def test_case_insensitive(self) -> None:
        answer = "The API LATENCY was INCREASING rapidly"
        facts = ["api latency increasing"]
        result = _compute_per_fact_matches(answer, facts)
        assert result[0]["matched"] is True
        assert result[0]["overlap_ratio"] == 1.0


class TestCheckDistractorPurity:
    def test_clean_distractors_pass(self) -> None:
        from lens.datagen.spec import DistractorConfig, KeyFact

        distractor_eps = [
            {
                "episode_id": "test_dx_001",
                "text": "DNS zone transfer completed successfully today.",
                "meta": {"episode_type": "distractor", "theme": "dns"},
            },
            {
                "episode_id": "test_dx_002",
                "text": "Storage cluster rebalancing finished overnight.",
                "meta": {"episode_type": "distractor", "theme": "storage"},
            },
        ]
        key_facts = [
            KeyFact(id="kf1", fact="geo-lookup API latency increasing", first_appears="signal:1"),
            KeyFact(id="kf2", fact="connection pool exhaustion", first_appears="signal:2"),
        ]
        dc = DistractorConfig(count=2, max_similarity=0.3)

        result = _check_distractor_purity(distractor_eps, key_facts, dc)
        assert result["summary"] == "pass"
        assert result["flagged_count"] == 0
        assert result["total_distractors"] == 2
        assert len(result["episodes"]) == 2

    def test_contaminated_distractors_fail(self) -> None:
        from lens.datagen.spec import DistractorConfig, KeyFact

        distractor_eps = [
            {
                "episode_id": "test_dx_001",
                "text": "The geo-lookup API latency is increasing rapidly today.",
                "meta": {"episode_type": "distractor", "theme": "dns"},
            },
        ]
        key_facts = [
            KeyFact(id="kf1", fact="geo-lookup API latency increasing", first_appears="signal:1"),
        ]
        dc = DistractorConfig(count=1, max_similarity=0.3)

        result = _check_distractor_purity(distractor_eps, key_facts, dc)
        assert result["summary"] == "fail"
        assert result["flagged_count"] == 1
        assert result["episodes"][0]["flagged"] is True

    def test_respects_threshold(self) -> None:
        from lens.datagen.spec import DistractorConfig, KeyFact

        distractor_eps = [
            {
                "episode_id": "test_dx_001",
                "text": "Some latency was observed in the system.",
                "meta": {"episode_type": "distractor", "theme": "dns"},
            },
        ]
        key_facts = [
            KeyFact(id="kf1", fact="API latency increasing", first_appears="signal:1"),
        ]

        # With high threshold, should pass
        dc_high = DistractorConfig(count=1, max_similarity=0.9)
        result = _check_distractor_purity(distractor_eps, key_facts, dc_high)
        assert result["summary"] == "pass"

        # With very low threshold, should fail
        dc_low = DistractorConfig(count=1, max_similarity=0.1)
        result = _check_distractor_purity(distractor_eps, key_facts, dc_low)
        assert result["summary"] == "fail"

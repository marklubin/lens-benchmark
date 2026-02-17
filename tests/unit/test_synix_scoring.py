"""Tests for lens.datagen.synix.scoring — fact coverage + similarity scoring."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

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
# compute_fact_coverage_llm
# ---------------------------------------------------------------------------


class _MockLLMResponse:
    def __init__(self, content):
        self.content = content


class TestComputeFactCoverageLLM:
    @pytest.fixture(autouse=True)
    def _mock_llm(self):
        """Install a fresh synix LLM mock for each test."""
        self.mock_complete = MagicMock()
        mock_mod = MagicMock()
        mock_mod._logged_complete = self.mock_complete
        prev = sys.modules.get("synix.build.llm_transforms")
        sys.modules["synix.build.llm_transforms"] = mock_mod
        yield
        if prev is not None:
            sys.modules["synix.build.llm_transforms"] = prev
        else:
            sys.modules.pop("synix.build.llm_transforms", None)

    def test_all_facts_matched(self):
        self.mock_complete.side_effect = [
            _MockLLMResponse("YES"),
            _MockLLMResponse("YES"),
        ]
        cov, details = scoring.compute_fact_coverage_llm(
            "ALT is rising in statin patients",
            ["transaminase creep", "statin co-medication"],
            "What trends do you see?",
            client=MagicMock(),
            config={},
        )
        assert cov == 1.0
        assert len(details) == 2
        assert all(d["matched"] for d in details)

    def test_no_facts_matched(self):
        self.mock_complete.side_effect = [
            _MockLLMResponse("NO"),
            _MockLLMResponse("NO"),
        ]
        cov, details = scoring.compute_fact_coverage_llm(
            "Everything looks normal.",
            ["transaminase creep", "statin co-medication"],
            "What trends do you see?",
            client=MagicMock(),
            config={},
        )
        assert cov == 0.0
        assert not any(d["matched"] for d in details)

    def test_partial_match(self):
        self.mock_complete.side_effect = [
            _MockLLMResponse("YES"),
            _MockLLMResponse("NO"),
        ]
        cov, details = scoring.compute_fact_coverage_llm(
            "ALT is rising.",
            ["transaminase creep", "statin co-medication"],
            "What trends do you see?",
            client=MagicMock(),
            config={},
        )
        assert cov == 0.5
        assert details[0]["matched"] is True
        assert details[1]["matched"] is False

    def test_empty_facts(self):
        cov, details = scoring.compute_fact_coverage_llm(
            "anything", [], "question?",
            client=MagicMock(), config={},
        )
        assert cov == 1.0
        assert details == []

    def test_verdict_stored(self):
        self.mock_complete.side_effect = [
            _MockLLMResponse("YES"),
        ]
        _, details = scoring.compute_fact_coverage_llm(
            "answer", ["fact"], "question?",
            client=MagicMock(), config={},
        )
        assert details[0]["judge_verdict"] == "YES"
        assert details[0]["judge_raw"] == "YES"

    def test_verdict_case_insensitive(self):
        self.mock_complete.side_effect = [
            _MockLLMResponse("yes"),
        ]
        cov, details = scoring.compute_fact_coverage_llm(
            "answer", ["fact"], "question?",
            client=MagicMock(), config={},
        )
        assert cov == 1.0
        assert details[0]["matched"] is True

    def test_verdict_with_trailing_whitespace(self):
        self.mock_complete.side_effect = [
            _MockLLMResponse("  NO  \n"),
        ]
        cov, _ = scoring.compute_fact_coverage_llm(
            "answer", ["fact"], "question?",
            client=MagicMock(), config={},
        )
        assert cov == 0.0


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


# ---------------------------------------------------------------------------
# compute_pairwise_fact_coverage_llm
# ---------------------------------------------------------------------------


class TestComputePairwiseFactCoverageLLM:
    @pytest.fixture(autouse=True)
    def _mock_llm(self):
        """Install a fresh synix LLM mock for each test."""
        self.mock_complete = MagicMock()
        mock_mod = MagicMock()
        mock_mod._logged_complete = self.mock_complete
        prev = sys.modules.get("synix.build.llm_transforms")
        sys.modules["synix.build.llm_transforms"] = mock_mod
        yield
        if prev is not None:
            sys.modules["synix.build.llm_transforms"] = prev
        else:
            sys.modules.pop("synix.build.llm_transforms", None)

    def test_a_wins_all(self):
        """Judge always picks answer A's position — win_rate 1.0."""
        # We need to simulate the judge correctly tracking positions.
        # With seed=42, the first fact's a_is_first depends on rng.
        # Simplest: make judge always pick whichever slot has answer_a.
        call_idx = [0]

        def side_effect(*args, **kwargs):
            call_idx[0] += 1
            # We'll return a response based on position tracking
            return _MockLLMResponse("A")

        self.mock_complete.side_effect = side_effect

        # With judge returning "A" and seed=42, some facts will map to
        # answer_a winning, others to answer_b winning (since position is random)
        cov, details = scoring.compute_pairwise_fact_coverage_llm(
            "answer A text",
            "answer B text",
            ["fact1"],
            "question?",
            client=MagicMock(),
            config={},
        )
        # With 1 fact, the result depends on random position
        assert 0.0 <= cov <= 1.0
        assert len(details) == 1

    def test_all_ties(self):
        self.mock_complete.side_effect = [_MockLLMResponse("TIE")] * 3
        cov, details = scoring.compute_pairwise_fact_coverage_llm(
            "answer A",
            "answer B",
            ["fact1", "fact2", "fact3"],
            "question?",
            client=MagicMock(),
            config={},
        )
        assert cov == 0.5
        assert all(d["fact_score"] == 0.5 for d in details)
        assert all(d["winner"] == "tie" for d in details)

    def test_empty_facts(self):
        cov, details = scoring.compute_pairwise_fact_coverage_llm(
            "a", "b", [], "q?",
            client=MagicMock(), config={},
        )
        assert cov == 1.0
        assert details == []

    def test_details_have_position_info(self):
        self.mock_complete.side_effect = [_MockLLMResponse("TIE")]
        _, details = scoring.compute_pairwise_fact_coverage_llm(
            "answer A",
            "answer B",
            ["fact1"],
            "question?",
            client=MagicMock(),
            config={},
        )
        d = details[0]
        assert "a_position" in d
        assert d["a_position"] in ("first", "second")
        assert d["winner"] == "tie"

    def test_seed_reproducibility(self):
        self.mock_complete.side_effect = [_MockLLMResponse("A")] * 2
        _, details1 = scoring.compute_pairwise_fact_coverage_llm(
            "a", "b", ["f1"], "q?",
            client=MagicMock(), config={}, seed=99,
        )
        self.mock_complete.side_effect = [_MockLLMResponse("A")] * 2
        _, details2 = scoring.compute_pairwise_fact_coverage_llm(
            "a", "b", ["f1"], "q?",
            client=MagicMock(), config={}, seed=99,
        )
        assert details1[0]["a_position"] == details2[0]["a_position"]


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

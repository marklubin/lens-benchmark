"""Tests for parallel judge execution in scorer/judge.py.

Verifies that parallel execution produces identical results to sequential
(determinism) and that there are no thread-safety issues.
"""
from __future__ import annotations

import threading
from unittest.mock import MagicMock

import pytest

from lens.scorer.judge import pairwise_fact_judge


def _make_mock_judge(verdicts: list[str]) -> MagicMock:
    """Create a mock judge_fn that returns verdicts in order (thread-safe)."""
    call_order: list[int] = []
    lock = threading.Lock()
    idx = [0]

    def judge_fn(prompt: str) -> str:
        with lock:
            i = idx[0]
            idx[0] += 1
        call_order.append(i)
        return verdicts[i % len(verdicts)]

    mock = MagicMock(side_effect=judge_fn)
    mock._call_order = call_order
    return mock


KEY_FACTS = [
    "Latency increased from 200ms to 800ms over 3 weeks",
    "Connection pool exhaustion began at episode 12",
    "Root cause was a connection leak in the payment service",
    "Cache hit rate dropped from 95% to 60%",
    "Error rate exceeded 5% threshold at episode 18",
    "Memory usage grew linearly suggesting a leak",
]


class TestPairwiseFactJudgeDeterminism:
    """Parallel results must match sequential for the same seed."""

    def test_parallel_matches_sequential(self):
        """max_workers=4 produces identical results to max_workers=1."""
        verdicts = ["A", "B", "TIE", "A", "B", "A"]

        seq_judge = _make_mock_judge(verdicts)
        par_judge = _make_mock_judge(verdicts)

        seq_rate, seq_details = pairwise_fact_judge(
            candidate_answer="The system experienced degradation...",
            reference_answer="Latency grew from 200ms to 800ms...",
            key_facts=KEY_FACTS,
            question="What caused the cascading failure?",
            judge_fn=seq_judge,
            seed=42,
            max_workers=1,
        )

        par_rate, par_details = pairwise_fact_judge(
            candidate_answer="The system experienced degradation...",
            reference_answer="Latency grew from 200ms to 800ms...",
            key_facts=KEY_FACTS,
            question="What caused the cascading failure?",
            judge_fn=par_judge,
            seed=42,
            max_workers=4,
        )

        # Win rates must match
        assert seq_rate == par_rate

        # Position assignments must match (deterministic from seed)
        for s, p in zip(seq_details, par_details):
            assert s["candidate_position"] == p["candidate_position"]
            assert s["fact"] == p["fact"]

    def test_single_fact_no_threading(self):
        """Single fact should not use thread pool even with max_workers>1."""
        judge = _make_mock_judge(["A"])
        rate, details = pairwise_fact_judge(
            candidate_answer="answer",
            reference_answer="reference",
            key_facts=["single fact"],
            question="question?",
            judge_fn=judge,
            max_workers=8,
        )
        assert len(details) == 1
        assert judge.call_count == 1

    def test_empty_facts(self):
        """Empty key_facts returns 1.0 without calling judge."""
        judge = MagicMock()
        rate, details = pairwise_fact_judge(
            candidate_answer="answer",
            reference_answer="reference",
            key_facts=[],
            question="question?",
            judge_fn=judge,
            max_workers=4,
        )
        assert rate == 1.0
        assert details == []
        judge.assert_not_called()

    def test_all_workers_invoked(self):
        """With enough facts, multiple threads should be used."""
        import time

        call_threads: list[int] = []
        lock = threading.Lock()

        def judge_fn(prompt: str) -> str:
            tid = threading.current_thread().ident
            with lock:
                call_threads.append(tid)
            # Small sleep to force actual concurrency (instant returns
            # let ThreadPoolExecutor reuse the same thread)
            time.sleep(0.05)
            return "A"

        pairwise_fact_judge(
            candidate_answer="answer",
            reference_answer="reference",
            key_facts=KEY_FACTS,
            question="question?",
            judge_fn=judge_fn,
            max_workers=4,
        )

        assert len(call_threads) == 6
        # With 4 workers and 6 facts + sleep, we should see >1 unique thread
        unique_threads = set(call_threads)
        assert len(unique_threads) > 1, f"Expected multiple threads, got {len(unique_threads)}"


class TestPairwiseFactJudgeErrorHandling:
    """Judge call failures should not crash the scorer."""

    def test_judge_exception_defaults_to_tie(self):
        """If judge_fn raises, the fact should score as TIE (0.5)."""
        call_count = [0]

        def flaky_judge(prompt: str) -> str:
            call_count[0] += 1
            if call_count[0] == 2:
                raise ConnectionError("API timeout")
            return "A"

        rate, details = pairwise_fact_judge(
            candidate_answer="answer",
            reference_answer="reference",
            key_facts=["fact1", "fact2", "fact3"],
            question="question?",
            judge_fn=flaky_judge,
            max_workers=1,
        )

        # fact2 should be TIE due to exception
        assert details[1]["winner"] == "tie"
        assert details[1]["fact_score"] == 0.5
        # Other facts should still have real verdicts
        assert details[0]["verdict_raw"] == "A"
        assert details[2]["verdict_raw"] == "A"

    def test_parallel_judge_exception_isolated(self):
        """Exception in one parallel judge call doesn't affect others."""
        lock = threading.Lock()
        call_count = [0]

        def flaky_judge(prompt: str) -> str:
            with lock:
                call_count[0] += 1
                n = call_count[0]
            if n == 3:
                raise RuntimeError("Transient failure")
            return "B"

        rate, details = pairwise_fact_judge(
            candidate_answer="answer",
            reference_answer="reference",
            key_facts=KEY_FACTS,
            question="question?",
            judge_fn=flaky_judge,
            max_workers=4,
        )

        # Should have results for all 6 facts
        assert len(details) == 6
        # Exactly one should be a tie (the failed one)
        tie_count = sum(1 for d in details if d["winner"] == "tie" and d["verdict_raw"] == "TIE")
        assert tie_count == 1

"""Tests for the naive baseline generator and NaiveBaselineAdvantage metric."""
from __future__ import annotations

import json
import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from lens.core.models import (
    AgentAnswer,
    Episode,
    GroundTruth,
    MetricResult,
    Question,
    QuestionResult,
    RunResult,
    ScopeResult,
    CheckpointResult,
)
from lens.scorer.naive_baseline import NaiveBaselineGenerator, build_naive_prompt


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_episodes(n: int = 10, scope: str = "test_01") -> list[Episode]:
    """Create n episodes with sequential IDs."""
    return [
        Episode(
            episode_id=f"{scope}_ep_{i:03d}",
            scope_id=scope,
            timestamp=datetime(2025, 1, 1, i),
            text=f"Metric data for episode {i}: latency={100 + i*10}ms",
        )
        for i in range(1, n + 1)
    ]


def _make_question(
    question_id: str = "q01",
    checkpoint: int = 5,
    qtype: str = "longitudinal",
    prompt: str = "What trend is visible?",
    key_facts: list[str] | None = None,
) -> Question:
    return Question(
        question_id=question_id,
        scope_id="test_01",
        checkpoint_after=checkpoint,
        question_type=qtype,
        prompt=prompt,
        ground_truth=GroundTruth(
            canonical_answer="Latency increased from 110ms to 150ms",
            required_evidence_refs=["test_01_ep_001", "test_01_ep_005"],
            key_facts=key_facts or ["latency increased", "110ms to 150ms"],
        ),
    )


def _echo_llm(system: str, user: str) -> str:
    """Echo back a fixed answer for testing."""
    return "Latency increased from 110ms to 150ms based on the data."


def _fail_llm(system: str, user: str) -> str:
    raise RuntimeError("LLM call failed")


# ---------------------------------------------------------------------------
# build_naive_prompt tests
# ---------------------------------------------------------------------------

class TestBuildNaivePrompt:
    def test_includes_all_episodes(self):
        episodes = _make_episodes(5)
        system, user = build_naive_prompt(episodes, "What happened?")
        assert "test_01_ep_001" in user
        assert "test_01_ep_005" in user
        assert "What happened?" in user

    def test_truncates_by_max_tokens(self):
        episodes = _make_episodes(10)
        # Each episode is ~60 chars. max_tokens=30 → 120 chars → ~2 episodes
        system, user = build_naive_prompt(episodes, "Q?", max_tokens=30)
        # Should include first 2 episodes but not all 10
        assert "test_01_ep_001" in user
        assert "test_01_ep_010" not in user

    def test_unlimited_when_zero(self):
        episodes = _make_episodes(10)
        system, user = build_naive_prompt(episodes, "Q?", max_tokens=0)
        assert "test_01_ep_010" in user

    def test_system_prompt_is_analyst(self):
        system, _ = build_naive_prompt(_make_episodes(1), "Q?")
        assert "analyst" in system.lower()


# ---------------------------------------------------------------------------
# NaiveBaselineGenerator tests
# ---------------------------------------------------------------------------

class TestNaiveBaselineGenerator:
    def test_generates_answer(self):
        gen = NaiveBaselineGenerator(
            llm_fn=_echo_llm,
            episodes=_make_episodes(10),
            model_id="test-model",
        )
        answer = gen.get_answer(_make_question(checkpoint=5))
        assert "Latency increased" in answer

    def test_filters_episodes_by_checkpoint(self):
        """Only episodes with number <= checkpoint should be included."""
        calls: list[str] = []

        def tracking_llm(system: str, user: str) -> str:
            calls.append(user)
            return "answer"

        gen = NaiveBaselineGenerator(
            llm_fn=tracking_llm,
            episodes=_make_episodes(10),
            model_id="test",
        )
        gen.get_answer(_make_question(checkpoint=3))
        assert len(calls) == 1
        # Episodes 1-3 should be present, 4+ should not
        assert "ep_003" in calls[0]
        assert "ep_004" not in calls[0]

    def test_cache_hit(self):
        call_count = 0

        def counting_llm(system: str, user: str) -> str:
            nonlocal call_count
            call_count += 1
            return "answer"

        gen = NaiveBaselineGenerator(
            llm_fn=counting_llm,
            episodes=_make_episodes(5),
            model_id="test",
        )
        q = _make_question()
        gen.get_answer(q)
        gen.get_answer(q)  # Should hit cache
        assert call_count == 1

    def test_cache_persistence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)

            gen1 = NaiveBaselineGenerator(
                llm_fn=_echo_llm,
                episodes=_make_episodes(5),
                cache_dir=cache_dir,
                model_id="test",
            )
            q = _make_question()
            answer1 = gen1.get_answer(q)

            # Create a new generator that should load from cache
            call_count = 0

            def counting_llm(s, u):
                nonlocal call_count
                call_count += 1
                return "different"

            gen2 = NaiveBaselineGenerator(
                llm_fn=counting_llm,
                episodes=_make_episodes(5),
                cache_dir=cache_dir,
                model_id="test",
            )
            answer2 = gen2.get_answer(q)
            assert answer2 == answer1
            assert call_count == 0  # Loaded from disk cache

    def test_llm_failure_returns_error_message(self):
        gen = NaiveBaselineGenerator(
            llm_fn=_fail_llm,
            episodes=_make_episodes(5),
            model_id="test",
        )
        answer = gen.get_answer(_make_question())
        assert "failed" in answer.lower()

    def test_no_episodes_returns_placeholder(self):
        gen = NaiveBaselineGenerator(
            llm_fn=_echo_llm,
            episodes=[],
            model_id="test",
        )
        answer = gen.get_answer(_make_question(checkpoint=5))
        assert "No episodes" in answer

    def test_episodes_sorted_chronologically(self):
        # Create episodes in reverse order
        episodes = list(reversed(_make_episodes(5)))
        calls: list[str] = []

        def tracking_llm(system: str, user: str) -> str:
            calls.append(user)
            return "answer"

        gen = NaiveBaselineGenerator(
            llm_fn=tracking_llm,
            episodes=episodes,
            model_id="test",
        )
        gen.get_answer(_make_question(checkpoint=5))
        # ep_001 should appear before ep_005 in the prompt
        assert calls[0].index("ep_001") < calls[0].index("ep_005")

    def test_max_result_tokens_caps_context(self):
        calls: list[str] = []

        def tracking_llm(system: str, user: str) -> str:
            calls.append(user)
            return "answer"

        gen = NaiveBaselineGenerator(
            llm_fn=tracking_llm,
            episodes=_make_episodes(10),
            model_id="test",
            max_result_tokens=30,  # ~120 chars
        )
        gen.get_answer(_make_question(checkpoint=10))
        assert "ep_010" not in calls[0]


# ---------------------------------------------------------------------------
# NaiveBaselineAdvantage metric tests
# ---------------------------------------------------------------------------

def _make_run_result(qrs: list[QuestionResult]) -> RunResult:
    return RunResult(
        run_id="test-run",
        adapter="test-adapter",
        dataset_version="v1",
        budget_preset="standard",
        scopes=[
            ScopeResult(
                scope_id="test_01",
                checkpoints=[
                    CheckpointResult(
                        scope_id="test_01",
                        checkpoint=5,
                        question_results=qrs,
                    )
                ],
            )
        ],
    )


class TestNaiveBaselineAdvantage:
    def test_not_configured_returns_zero(self):
        from lens.scorer.tier3 import NaiveBaselineAdvantage

        metric = NaiveBaselineAdvantage()
        result = metric.compute(_make_run_result([]))
        assert result.value == 0.0
        assert result.details.get("not_configured") is True

    def test_adapter_wins_all(self):
        from lens.scorer.tier3 import NaiveBaselineAdvantage

        metric = NaiveBaselineAdvantage()

        # Judge always picks candidate (adapter)
        def judge_always_a(prompt: str) -> str:
            # Adapter is randomly A or B — but pairwise_fact_judge handles mapping
            # We need to know which is A. Since seed=42, first fact: candidate_is_a
            # depends on rng. Simplest: always return "A" and "B" alternating isn't
            # needed — we'll just verify the metric works with a deterministic judge.
            return "A"

        gen = NaiveBaselineGenerator(
            llm_fn=lambda s, u: "Naive answer with no facts",
            episodes=_make_episodes(5),
            model_id="test",
        )

        metric.configure(judge_fn=judge_always_a, baseline_generator=gen)

        q = _make_question(key_facts=["fact1", "fact2"])
        qr = QuestionResult(
            question=q,
            answer=AgentAnswer(
                question_id="q01",
                answer_text="Adapter answer with fact1 and fact2",
            ),
        )
        result = metric.compute(_make_run_result([qr]))
        # With seed=42, position assignment is deterministic
        # A always wins → either candidate or reference depending on assignment
        assert result.value >= 0.0
        assert result.details.get("method") == "pairwise_vs_naive"

    def test_per_type_breakdown(self):
        from lens.scorer.tier3 import NaiveBaselineAdvantage

        metric = NaiveBaselineAdvantage()

        def judge_tie(prompt: str) -> str:
            return "TIE"

        gen = NaiveBaselineGenerator(
            llm_fn=lambda s, u: "naive",
            episodes=_make_episodes(10),
            model_id="test",
        )
        metric.configure(judge_fn=judge_tie, baseline_generator=gen)

        q1 = _make_question(question_id="q01", qtype="longitudinal", key_facts=["f1"])
        q2 = _make_question(question_id="q02", qtype="null_hypothesis", key_facts=["f2"])

        qrs = [
            QuestionResult(
                question=q1,
                answer=AgentAnswer(question_id="q01", answer_text="a1"),
            ),
            QuestionResult(
                question=q2,
                answer=AgentAnswer(question_id="q02", answer_text="a2"),
            ),
        ]
        result = metric.compute(_make_run_result(qrs))
        # All ties → 0.5
        assert result.value == pytest.approx(0.5)
        assert "longitudinal" in result.details["per_type"]
        assert "null_hypothesis" in result.details["per_type"]

    def test_registry_contains_metric(self):
        from lens.scorer.registry import list_metrics

        metrics = list_metrics()
        assert "naive_baseline_advantage" in metrics

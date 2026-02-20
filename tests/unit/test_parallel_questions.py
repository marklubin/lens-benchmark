"""Tests for parallel question answering in the runner."""
from __future__ import annotations

import time
from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from lens.adapters.base import CapabilityManifest, Document, MemoryAdapter, SearchResult
from lens.agent.llm_client import MockLLMClient
from lens.core.config import RunConfig
from lens.core.models import Episode, GroundTruth, Question
from lens.runner.runner import RunEngine


# ---------------------------------------------------------------------------
# Minimal adapter stub for testing
# ---------------------------------------------------------------------------

class _StubAdapter(MemoryAdapter):
    """In-memory adapter that stores text and returns it on search."""

    def __init__(self):
        self._store: dict[str, str] = {}

    def reset(self, scope_id: str) -> None:
        self._store.clear()

    def ingest(self, episode_id, scope_id, timestamp, text, meta=None):
        self._store[episode_id] = text

    def search(self, query, filters=None, limit=None):
        results = []
        for ref_id, text in list(self._store.items())[:3]:
            results.append(SearchResult(ref_id=ref_id, text=text[:100], score=0.5))
        return results

    def retrieve(self, ref_id):
        text = self._store.get(ref_id)
        return Document(ref_id=ref_id, text=text) if text else None

    def get_capabilities(self):
        return CapabilityManifest()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_episodes(n=5, scope="test_scope"):
    return [
        Episode(
            episode_id=f"{scope}_ep_{i:03d}",
            scope_id=scope,
            timestamp=datetime(2025, 1, i, tzinfo=timezone.utc),
            text=f"Episode {i}: server latency p99={100 + i * 10}ms",
        )
        for i in range(1, n + 1)
    ]


def _make_questions(n=4, scope="test_scope", checkpoint=5):
    return [
        Question(
            question_id=f"q{i}",
            scope_id=scope,
            checkpoint_after=checkpoint,
            prompt=f"What happened in question {i}?",
            question_type="longitudinal",
            ground_truth=GroundTruth(
                canonical_answer=f"Answer {i}",
                required_evidence_refs=[f"{scope}_ep_001"],
                key_facts=[f"fact_{i}"],
            ),
        )
        for i in range(1, n + 1)
    ]


def _patch_adapter():
    """Patch get_adapter in the runner module to return _StubAdapter class."""
    return patch("lens.runner.runner.get_adapter", return_value=_StubAdapter)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestParallelQuestions:
    def test_sequential_answers_all_questions(self):
        """Default parallel_questions=1 answers all questions sequentially."""
        config = RunConfig(adapter="stub", checkpoints=[])
        config.parallel_questions = 1

        with _patch_adapter():
            engine = RunEngine(config, llm_client=MockLLMClient())
            episodes = _make_episodes(5)
            questions = _make_questions(4, checkpoint=5)

            result = engine.run({"test_scope": episodes}, questions=questions)
            assert len(result.scopes) == 1
            cp = result.scopes[0].checkpoints[0]
            assert len(cp.question_results) == 4

    def test_parallel_answers_all_questions(self):
        """parallel_questions=4 answers all 4 questions concurrently."""
        config = RunConfig(adapter="stub", checkpoints=[])
        config.parallel_questions = 4

        with _patch_adapter():
            engine = RunEngine(config, llm_client=MockLLMClient())
            episodes = _make_episodes(5)
            questions = _make_questions(4, checkpoint=5)

            result = engine.run({"test_scope": episodes}, questions=questions)
            assert len(result.scopes) == 1
            cp = result.scopes[0].checkpoints[0]
            assert len(cp.question_results) == 4

    def test_parallel_preserves_question_order(self):
        """Results must be in the same order as input questions."""
        config = RunConfig(adapter="stub", checkpoints=[])
        config.parallel_questions = 4

        with _patch_adapter():
            engine = RunEngine(config, llm_client=MockLLMClient())
            episodes = _make_episodes(5)
            questions = _make_questions(4, checkpoint=5)

            result = engine.run({"test_scope": episodes}, questions=questions)
            cp = result.scopes[0].checkpoints[0]
            result_qids = [qr.question.question_id for qr in cp.question_results]
            assert result_qids == ["q1", "q2", "q3", "q4"]

    def test_parallel_records_timing(self):
        """Each question should have a timing entry."""
        config = RunConfig(adapter="stub", checkpoints=[])
        config.parallel_questions = 4

        with _patch_adapter():
            engine = RunEngine(config, llm_client=MockLLMClient())
            episodes = _make_episodes(5)
            questions = _make_questions(4, checkpoint=5)

            result = engine.run({"test_scope": episodes}, questions=questions)
            cp = result.scopes[0].checkpoints[0]
            for i in range(1, 5):
                assert f"question_q{i}_ms" in cp.timing
                assert cp.timing[f"question_q{i}_ms"] > 0


class TestParallelQuestionsConfig:
    def test_from_dict_default(self):
        config = RunConfig.from_dict({"adapter": "null"})
        assert config.parallel_questions == 1

    def test_from_dict_explicit(self):
        config = RunConfig.from_dict({"adapter": "null", "parallel_questions": 8})
        assert config.parallel_questions == 8

    def test_to_dict_default_omits(self):
        config = RunConfig()
        d = config.to_dict()
        assert "parallel_questions" not in d

    def test_to_dict_nondefault_includes(self):
        config = RunConfig(parallel_questions=4)
        d = config.to_dict()
        assert d["parallel_questions"] == 4

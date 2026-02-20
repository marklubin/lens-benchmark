"""Tests for adapter state caching (Strategy 3)."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from lens.adapters.base import CapabilityManifest, Document, MemoryAdapter, SearchResult
from lens.adapters.compaction import CompactionAdapter
from lens.agent.llm_client import MockLLMClient
from lens.core.config import RunConfig
from lens.core.models import Episode, GroundTruth, Question
from lens.runner.runner import RunEngine, _cache_key


# ---------------------------------------------------------------------------
# Cache key tests
# ---------------------------------------------------------------------------


class TestCacheKey:
    def test_deterministic(self):
        """Same inputs produce the same cache key."""
        k1 = _cache_key("GraphitiAdapter", "scope01", ["ep_001", "ep_002"])
        k2 = _cache_key("GraphitiAdapter", "scope01", ["ep_001", "ep_002"])
        assert k1 == k2

    def test_order_independent(self):
        """Episode ID order shouldn't matter (sorted internally)."""
        k1 = _cache_key("Adapter", "scope01", ["ep_002", "ep_001"])
        k2 = _cache_key("Adapter", "scope01", ["ep_001", "ep_002"])
        assert k1 == k2

    def test_different_episodes_different_key(self):
        k1 = _cache_key("Adapter", "scope01", ["ep_001", "ep_002"])
        k2 = _cache_key("Adapter", "scope01", ["ep_001", "ep_003"])
        assert k1 != k2

    def test_different_adapter_different_key(self):
        k1 = _cache_key("GraphitiAdapter", "scope01", ["ep_001"])
        k2 = _cache_key("CogneeAdapter", "scope01", ["ep_001"])
        assert k1 != k2

    def test_different_scope_different_key(self):
        k1 = _cache_key("Adapter", "scope01", ["ep_001"])
        k2 = _cache_key("Adapter", "scope02", ["ep_001"])
        assert k1 != k2

    def test_key_length(self):
        k = _cache_key("Adapter", "scope01", ["ep_001"])
        assert len(k) == 12


# ---------------------------------------------------------------------------
# Compaction cache round-trip
# ---------------------------------------------------------------------------


class TestCompactionCache:
    def test_get_cache_state_returns_none_without_summary(self):
        adapter = CompactionAdapter()
        adapter.reset("test")
        assert adapter.get_cache_state() is None

    def test_get_cache_state_returns_state(self):
        adapter = CompactionAdapter()
        adapter.reset("test")
        adapter.ingest("ep1", "test", "2025-01-01", "data")
        adapter._summary = "Test summary citing [ep1]"
        adapter._cited_episode_ids = ["ep1"]

        state = adapter.get_cache_state()
        assert state is not None
        assert state["summary"] == "Test summary citing [ep1]"
        assert len(state["episodes"]) == 1
        assert state["cited_episode_ids"] == ["ep1"]

    def test_round_trip(self):
        """Save state, create a new adapter, restore state, verify."""
        adapter1 = CompactionAdapter()
        adapter1.reset("test")
        adapter1.ingest("ep1", "test", "2025-01-01", "latency p99=200ms")
        adapter1.ingest("ep2", "test", "2025-01-02", "latency p99=300ms")
        adapter1._summary = "Latency trend from [ep1] to [ep2]"
        adapter1._cited_episode_ids = ["ep1", "ep2"]

        state = adapter1.get_cache_state()

        adapter2 = CompactionAdapter()
        assert adapter2.restore_cache_state(state) is True
        assert adapter2._summary == adapter1._summary
        assert len(adapter2._episodes) == 2
        assert adapter2.retrieve("ep1") is not None
        assert adapter2.search("latency")[0].ref_id == "compaction_summary"

    def test_round_trip_serializable(self):
        """State must be JSON-serializable."""
        adapter = CompactionAdapter()
        adapter.reset("test")
        adapter.ingest("ep1", "test", "2025-01-01", "data")
        adapter._summary = "summary"
        adapter._cited_episode_ids = ["ep1"]

        state = adapter.get_cache_state()
        # Must survive JSON round-trip
        restored = json.loads(json.dumps(state))
        adapter2 = CompactionAdapter()
        assert adapter2.restore_cache_state(restored) is True
        assert adapter2._summary == "summary"


# ---------------------------------------------------------------------------
# Base adapter caching protocol
# ---------------------------------------------------------------------------


class TestBaseCachingProtocol:
    def test_default_get_cache_state_returns_none(self):
        """Base class returns None (not cacheable)."""

        class DummyAdapter(MemoryAdapter):
            def reset(self, scope_id): pass
            def ingest(self, **kw): pass
            def search(self, query, **kw): return []
            def retrieve(self, ref_id): return None
            def get_capabilities(self): return CapabilityManifest()

        adapter = DummyAdapter()
        assert adapter.get_cache_state() is None

    def test_default_restore_returns_false(self):
        class DummyAdapter(MemoryAdapter):
            def reset(self, scope_id): pass
            def ingest(self, **kw): pass
            def search(self, query, **kw): return []
            def retrieve(self, ref_id): return None
            def get_capabilities(self): return CapabilityManifest()

        adapter = DummyAdapter()
        assert adapter.restore_cache_state({"some": "data"}) is False


# ---------------------------------------------------------------------------
# Runner cache integration
# ---------------------------------------------------------------------------

class _CacheableAdapter(MemoryAdapter):
    """Adapter that tracks ingest/prepare calls and supports caching."""

    def __init__(self):
        self._store: dict[str, str] = {}
        self.ingest_count = 0
        self.prepare_count = 0
        self.reset_count = 0

    def reset(self, scope_id: str) -> None:
        self.reset_count += 1
        self._store.clear()

    def ingest(self, episode_id, scope_id, timestamp, text, meta=None):
        self.ingest_count += 1
        self._store[episode_id] = text

    def prepare(self, scope_id, checkpoint):
        self.prepare_count += 1

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

    def get_cache_state(self):
        return {"store": self._store.copy()}

    def restore_cache_state(self, state):
        self._store = state.get("store", {})
        return True


def _make_episodes(n=5, scope="test_scope"):
    return [
        Episode(
            episode_id=f"{scope}_ep_{i:03d}",
            scope_id=scope,
            timestamp=datetime(2025, 1, i, tzinfo=timezone.utc),
            text=f"Episode {i}: data",
        )
        for i in range(1, n + 1)
    ]


def _make_questions(n=2, scope="test_scope", checkpoint=5):
    return [
        Question(
            question_id=f"q{i}",
            scope_id=scope,
            checkpoint_after=checkpoint,
            prompt=f"Question {i}?",
            question_type="longitudinal",
            ground_truth=GroundTruth(
                canonical_answer=f"Answer {i}",
                required_evidence_refs=[f"{scope}_ep_001"],
                key_facts=[f"fact_{i}"],
            ),
        )
        for i in range(1, n + 1)
    ]


def _make_adapter_class(adapter_instance):
    """Create a class that, when called, returns the given adapter instance."""
    class _Wrapper:
        def __new__(cls):
            return adapter_instance
    return _Wrapper


class TestRunnerCache:
    def test_cache_miss_then_hit(self, tmp_path):
        """First run writes cache; second run reads it and skips ingest/prepare."""
        cache_dir = str(tmp_path / "cache")

        adapter1 = _CacheableAdapter()
        episodes = _make_episodes(3)
        questions = _make_questions(2, checkpoint=3)

        with patch("lens.runner.runner.get_adapter", return_value=_make_adapter_class(adapter1)):
            config = RunConfig(adapter="test", checkpoints=[], cache_dir=cache_dir)
            engine = RunEngine(config, llm_client=MockLLMClient())
            engine.run({"test_scope": episodes}, questions=questions)

        assert adapter1.ingest_count == 3
        assert adapter1.prepare_count >= 1

        # Verify cache file written
        cache_files = list(Path(cache_dir).glob("*.json"))
        assert len(cache_files) == 1

        # Second run — cache hit, should skip ingest + prepare
        adapter2 = _CacheableAdapter()
        with patch("lens.runner.runner.get_adapter", return_value=_make_adapter_class(adapter2)):
            config2 = RunConfig(adapter="test", checkpoints=[], cache_dir=cache_dir)
            engine2 = RunEngine(config2, llm_client=MockLLMClient())
            result = engine2.run({"test_scope": episodes}, questions=questions)

        assert adapter2.ingest_count == 0
        assert adapter2.prepare_count == 0
        assert adapter2.reset_count == 0
        # Questions still answered
        assert len(result.scopes[0].checkpoints[0].question_results) == 2

    def test_cache_disabled_when_no_cache_dir(self, tmp_path):
        """Without cache_dir, no cache files are created."""
        adapter = _CacheableAdapter()
        episodes = _make_episodes(3)
        questions = _make_questions(2, checkpoint=3)

        with patch("lens.runner.runner.get_adapter", return_value=_make_adapter_class(adapter)):
            config = RunConfig(adapter="test", checkpoints=[])
            engine = RunEngine(config, llm_client=MockLLMClient())
            engine.run({"test_scope": episodes}, questions=questions)

        assert adapter.ingest_count == 3  # Normal flow
        # No cache files anywhere
        assert not list(tmp_path.glob("**/*.json"))


class TestCacheConfig:
    def test_from_dict_default(self):
        config = RunConfig.from_dict({"adapter": "null"})
        assert config.cache_dir is None

    def test_from_dict_explicit(self):
        config = RunConfig.from_dict({"adapter": "null", "cache_dir": "/tmp/cache"})
        assert config.cache_dir == "/tmp/cache"

    def test_to_dict_default_omits(self):
        config = RunConfig()
        d = config.to_dict()
        assert "cache_dir" not in d

    def test_to_dict_nondefault_includes(self):
        config = RunConfig(cache_dir="/tmp/cache")
        d = config.to_dict()
        assert d["cache_dir"] == "/tmp/cache"

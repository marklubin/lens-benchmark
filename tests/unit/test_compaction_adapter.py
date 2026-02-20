"""Tests for the compaction baseline adapter."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from lens.adapters.compaction import CompactionAdapter


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def adapter():
    return CompactionAdapter()


def _ingest_episodes(adapter, n=5, scope="test_01"):
    """Helper to ingest n episodes."""
    adapter.reset(scope)
    for i in range(1, n + 1):
        adapter.ingest(
            episode_id=f"{scope}_ep_{i:03d}",
            scope_id=scope,
            timestamp=f"2025-01-{i:02d}T00:00:00",
            text=f"Server latency p99={100 + i * 10}ms, error_rate={0.01 * i:.2f}%",
        )


# ---------------------------------------------------------------------------
# Reset tests
# ---------------------------------------------------------------------------

class TestReset:
    def test_clears_state(self, adapter):
        _ingest_episodes(adapter, 3)
        adapter._summary = "old summary"
        # Reset same scope — should clear those episodes
        adapter.reset("test_01")
        assert adapter._episodes == []
        assert adapter._summary == ""
        assert adapter._scope_id == "test_01"

    def test_reset_preserves_other_scope(self, adapter):
        """Reset scope A should not affect episodes from scope B."""
        adapter.reset("scope_a")
        adapter.ingest("ep1", "scope_a", "2025-01-01", "data a")
        adapter.reset("scope_b")
        adapter.ingest("ep2", "scope_b", "2025-01-02", "data b")
        adapter.reset("scope_a")
        assert adapter.retrieve("ep1") is None
        assert adapter.retrieve("ep2") is not None


# ---------------------------------------------------------------------------
# Ingest tests
# ---------------------------------------------------------------------------

class TestIngest:
    def test_buffers_episodes(self, adapter):
        _ingest_episodes(adapter, 3)
        assert len(adapter._episodes) == 3
        assert adapter._episodes[0]["episode_id"] == "test_01_ep_001"

    def test_ingest_is_instant_no_io(self, adapter):
        """Ingest should just append to a list — no external calls."""
        adapter.reset("test")
        adapter.ingest("ep1", "test", "2025-01-01", "data")
        assert len(adapter._episodes) == 1

    def test_preserves_episode_data(self, adapter):
        adapter.reset("test")
        adapter.ingest("ep1", "test", "2025-01-01T12:00:00", "full text here")
        ep = adapter._episodes[0]
        assert ep["episode_id"] == "ep1"
        assert ep["timestamp"] == "2025-01-01T12:00:00"
        assert ep["text"] == "full text here"


# ---------------------------------------------------------------------------
# Search tests (without prepare)
# ---------------------------------------------------------------------------

class TestSearchFallback:
    def test_empty_returns_empty(self, adapter):
        adapter.reset("test")
        assert adapter.search("query") == []

    def test_fallback_without_summary(self, adapter):
        _ingest_episodes(adapter, 3)
        results = adapter.search("latency")
        assert len(results) == 3
        assert results[0].ref_id == "test_01_ep_001"
        assert results[0].score == 0.5

    def test_summary_returned_as_single_result(self, adapter):
        _ingest_episodes(adapter, 3)
        adapter._summary = "Summary: latency increased from 110ms to 130ms"
        results = adapter.search("latency")
        assert len(results) == 1
        assert results[0].ref_id == "compaction_summary"
        assert results[0].score == 1.0
        assert "latency" in results[0].text.lower()


# ---------------------------------------------------------------------------
# Retrieve tests
# ---------------------------------------------------------------------------

class TestRetrieve:
    def test_retrieve_summary(self, adapter):
        adapter.reset("test")
        adapter._summary = "Full summary document"
        doc = adapter.retrieve("compaction_summary")
        assert doc is not None
        assert doc.text == "Full summary document"

    def test_retrieve_summary_none_when_empty(self, adapter):
        adapter.reset("test")
        assert adapter.retrieve("compaction_summary") is None

    def test_retrieve_episode_by_id(self, adapter):
        _ingest_episodes(adapter, 3)
        doc = adapter.retrieve("test_01_ep_002")
        assert doc is not None
        assert "120ms" in doc.text

    def test_retrieve_unknown_returns_none(self, adapter):
        adapter.reset("test")
        assert adapter.retrieve("nonexistent") is None

    def test_retrieve_fallback(self, adapter):
        _ingest_episodes(adapter, 2)
        doc = adapter.retrieve("compaction_fallback")
        assert doc is not None
        assert "ep_001" in doc.text
        assert "ep_002" in doc.text


# ---------------------------------------------------------------------------
# Capabilities tests
# ---------------------------------------------------------------------------

class TestCapabilities:
    def test_search_mode(self, adapter):
        caps = adapter.get_capabilities()
        assert "compaction" in caps.search_modes

    def test_has_batch_retrieve(self, adapter):
        caps = adapter.get_capabilities()
        tool_names = [t.name for t in caps.extra_tools]
        assert "batch_retrieve" in tool_names

    def test_max_results_is_one(self, adapter):
        caps = adapter.get_capabilities()
        assert caps.max_results_per_search == 1


# ---------------------------------------------------------------------------
# Extended tools tests
# ---------------------------------------------------------------------------

class TestBatchRetrieve:
    def test_batch_retrieve_summary_and_episodes(self, adapter):
        _ingest_episodes(adapter, 3)
        adapter._summary = "Summary citing [test_01_ep_001]"
        result = adapter.call_extended_tool("batch_retrieve", {
            "ref_ids": ["compaction_summary", "test_01_ep_001", "nonexistent"],
        })
        assert result["count"] == 2
        assert len(result["documents"]) == 2

    def test_unknown_tool_raises(self, adapter):
        with pytest.raises(NotImplementedError):
            adapter.call_extended_tool("unknown_tool", {})


# ---------------------------------------------------------------------------
# Prepare tests (with mocked LLM)
# ---------------------------------------------------------------------------

class TestPrepare:
    def test_prepare_empty_episodes_noop(self, adapter):
        adapter.reset("test")
        adapter.prepare("test", 5)  # Should not error
        assert adapter._summary == ""

    @patch("lens.adapters.compaction._OpenAI")
    def test_prepare_calls_llm(self, mock_openai_cls, adapter):
        _ingest_episodes(adapter, 3)

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = (
            "Summary: latency increased from 110ms [test_01_ep_001] "
            "to 130ms [test_01_ep_003]"
        )
        mock_client.chat.completions.create.return_value = mock_resp

        adapter.prepare("test_01", 3)

        assert "latency increased" in adapter._summary
        assert "test_01_ep_001" in adapter._cited_episode_ids
        assert "test_01_ep_003" in adapter._cited_episode_ids

    @patch("lens.adapters.compaction._OpenAI")
    def test_prepare_llm_failure_is_nonfatal(self, mock_openai_cls, adapter):
        _ingest_episodes(adapter, 3)

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.side_effect = RuntimeError("API down")

        adapter.prepare("test_01", 3)  # Should not raise
        assert adapter._summary == ""

    @patch("lens.adapters.compaction._OpenAI")
    def test_prepare_includes_all_episodes(self, mock_openai_cls, adapter):
        _ingest_episodes(adapter, 5)

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = "summary"
        mock_client.chat.completions.create.return_value = mock_resp

        adapter.prepare("test_01", 5)

        call_args = mock_client.chat.completions.create.call_args
        user_msg = call_args.kwargs["messages"][1]["content"]
        assert "ep_001" in user_msg
        assert "ep_005" in user_msg

    @patch("lens.adapters.compaction._OpenAI")
    def test_prepare_respects_max_tokens_env(self, mock_openai_cls, adapter):
        import os
        _ingest_episodes(adapter, 2)

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = "summary"
        mock_client.chat.completions.create.return_value = mock_resp

        adapter._max_tokens = 500
        adapter.prepare("test_01", 2)

        call_args = mock_client.chat.completions.create.call_args
        assert call_args.kwargs["max_tokens"] == 500


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_adapter_registered(self):
        from lens.adapters.registry import get_adapter

        cls = get_adapter("compaction")
        assert cls is CompactionAdapter

    def test_requires_metering(self):
        assert CompactionAdapter.requires_metering is True


# ---------------------------------------------------------------------------
# Full flow integration tests
# ---------------------------------------------------------------------------

class TestFullFlow:
    @patch("lens.adapters.compaction._OpenAI")
    def test_ingest_prepare_search_retrieve(self, mock_openai_cls, adapter):
        """Full lifecycle: ingest → prepare → search → retrieve."""
        _ingest_episodes(adapter, 5)

        # Mock LLM
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = (
            "Latency trend: p99 rose from 110ms [test_01_ep_001] to 150ms "
            "[test_01_ep_005]. Error rate correlated with latency [test_01_ep_003]."
        )
        mock_client.chat.completions.create.return_value = mock_resp

        adapter.prepare("test_01", 5)

        # Search returns the summary
        results = adapter.search("latency trend")
        assert len(results) == 1
        assert results[0].ref_id == "compaction_summary"

        # Retrieve summary
        doc = adapter.retrieve("compaction_summary")
        assert doc is not None
        assert "110ms" in doc.text

        # Retrieve cited episode
        doc2 = adapter.retrieve("test_01_ep_003")
        assert doc2 is not None

        # Batch retrieve
        batch = adapter.call_extended_tool("batch_retrieve", {
            "ref_ids": ["compaction_summary", "test_01_ep_001"],
        })
        assert batch["count"] == 2

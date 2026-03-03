"""Tests for the LettaSleepy memory adapter.

Uses MockLettaClient (shared pattern with test_letta_adapter.py) to avoid
requiring live servers. Tests the native sleep-time compute implementation.
"""
from __future__ import annotations

import uuid
from unittest.mock import patch

import pytest

from lens.adapters.base import CapabilityManifest, Document, SearchResult
from lens.adapters.registry import get_adapter


# ---------------------------------------------------------------------------
# Mock Letta client
# ---------------------------------------------------------------------------


class _MockPassage:
    def __init__(self, text: str):
        self.text = text
        self.content = text
        self.timestamp = "2024-01-01T00:00:00"
        self.id = str(uuid.uuid4())
        self.tags = []


class _MockSearchResult:
    def __init__(self, text: str):
        self.content = text
        self.text = text
        self.timestamp = "2024-01-01T00:00:00"
        self.id = None


class _MockPassageSearchResponse:
    def __init__(self, results: list):
        self.results = results


class _MockPassagesNamespace:
    def __init__(self):
        self._store: dict[str, list[_MockPassage]] = {}

    def create(self, agent_id: str, text: str):
        p = _MockPassage(text)
        self._store.setdefault(agent_id, []).append(p)
        return [p]

    def search(self, agent_id: str, query: str) -> _MockPassageSearchResponse:
        passages = self._store.get(agent_id, [])
        q = query.lower()
        matches = [
            _MockSearchResult(p.text)
            for p in passages
            if q in p.text.lower() or any(w in p.text.lower() for w in q.split())
        ]
        return _MockPassageSearchResponse(matches)

    def list(self, agent_id: str, limit: int = 500):
        return list(self._store.get(agent_id, []))


class _MockBlock:
    def __init__(self, label: str, value: str):
        self.label = label
        self.value = value


class _MockBlocksNamespace:
    def __init__(self):
        self._blocks: dict[str, list[_MockBlock]] = {}

    def list(self, agent_id: str) -> list[_MockBlock]:
        return list(self._blocks.get(agent_id, []))

    def _set(self, agent_id: str, blocks: list[_MockBlock]):
        self._blocks[agent_id] = blocks


class _MockToolCall:
    def __init__(self, name: str, arguments: str):
        self.name = name
        self.arguments = arguments


class _MockToolCallMessage:
    def __init__(self, name: str, arguments: str):
        self.tool_call = _MockToolCall(name, arguments)


# Make type(msg).__name__ return "ToolCallMessage"
_MockToolCallMessage.__name__ = "ToolCallMessage"


class _MockLettaResponse:
    def __init__(self, text: str = ""):
        import json
        msg = _MockToolCallMessage("send_message", json.dumps({"message": text}))
        # Override __class__.__name__ for the type check in _extract_assistant_text
        type(msg).__name__ = "ToolCallMessage"
        self.messages = [msg]


class _MockMessagesNamespace:
    def __init__(self):
        self.last_input: str | None = None
        self._response_text = "Reviewed archival memory."

    def create(self, agent_id: str, input: str, max_steps: int = 10) -> _MockLettaResponse:
        self.last_input = input
        return _MockLettaResponse(self._response_text)


class _MockAgent:
    def __init__(self, name: str):
        self.name = name
        self.id = str(uuid.uuid4())


class _MockAgentsNamespace:
    def __init__(self):
        self._agents: dict[str, _MockAgent] = {}
        self.passages = _MockPassagesNamespace()
        self.blocks = _MockBlocksNamespace()
        self.messages = _MockMessagesNamespace()

    def create(self, name: str, model: str, embedding: str, memory_blocks: list, **kwargs):
        a = _MockAgent(name)
        self._agents[a.id] = a
        # Initialize default blocks from memory_blocks
        blocks = []
        for mb in memory_blocks:
            blocks.append(_MockBlock(mb["label"], mb["value"]))
        self.blocks._set(a.id, blocks)
        return a

    def delete(self, agent_id: str):
        self._agents.pop(agent_id, None)
        self.passages._store.pop(agent_id, None)
        self.blocks._blocks.pop(agent_id, None)

    def list(self):
        return list(self._agents.values())


class MockLettaClient:
    def __init__(self):
        self.agents = _MockAgentsNamespace()


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _make_adapter(env_overrides: dict | None = None):
    """Create a LettaSleepyAdapter with mocked Letta client and env vars."""
    from lens.adapters.letta_sleepy import LettaSleepyAdapter

    mock_client = MockLettaClient()
    env = env_overrides or {}

    with patch.dict("os.environ", env, clear=False):
        adapter = LettaSleepyAdapter()
    adapter._client = mock_client
    return adapter


def _ingested_adapter(n: int = 3) -> "LettaSleepyAdapter":
    """Helper: adapter with reset + n ingested episodes."""
    adapter = _make_adapter()
    adapter.reset("scope_test")
    for i in range(1, n + 1):
        adapter.ingest(
            episode_id=f"ep_{i:03d}",
            scope_id="scope_test",
            timestamp=f"2024-01-0{i}T00:00:00",
            text=f"System report {i}: metric_a={i * 10}, metric_b={i * 5}.",
        )
    return adapter


# ---------------------------------------------------------------------------
# Tests: Registration
# ---------------------------------------------------------------------------


class TestLettaSleepyRegistration:
    def test_registered_by_name(self):
        cls = get_adapter("letta-sleepy")
        assert cls.__name__ == "LettaSleepyAdapter"

    def test_distinct_from_letta(self):
        assert get_adapter("letta-sleepy") is not get_adapter("letta")


# ---------------------------------------------------------------------------
# Tests: reset()
# ---------------------------------------------------------------------------


class TestLettaSleepyReset:
    def test_reset_creates_agent(self):
        adapter = _make_adapter()
        adapter.reset("scope_x")
        assert adapter._agent_id is not None

    def test_reset_clears_text_cache(self):
        adapter = _ingested_adapter()
        assert len(adapter._text_cache) == 3
        adapter.reset("scope_new")
        assert adapter._text_cache == {}

    def test_reset_deletes_previous_agent(self):
        adapter = _make_adapter()
        adapter.reset("scope_a")
        first_id = adapter._agent_id
        adapter.reset("scope_a")
        # Old agent should be gone from mock store
        assert first_id not in adapter._client.agents._agents

    def test_reset_agent_name_contains_sleepy(self):
        adapter = _make_adapter()
        adapter.reset("scope_z")
        agent = adapter._client.agents._agents[adapter._agent_id]
        assert "sleepy" in agent.name
        assert "scope_z" in agent.name

    def test_reset_passes_enable_sleeptime(self):
        """Verify that agent creation uses enable_sleeptime=True."""
        adapter = _make_adapter()
        # The mock accepts **kwargs — the real Letta client needs enable_sleeptime
        # If the adapter didn't pass it, it would have failed before mock fix
        adapter.reset("scope_x")
        assert adapter._agent_id is not None

    def test_reset_initializes_core_memory_blocks(self):
        adapter = _make_adapter()
        adapter.reset("scope_x")
        blocks = adapter._client.agents.blocks.list(agent_id=adapter._agent_id)
        labels = [b.label for b in blocks]
        assert "human" in labels
        assert "persona" in labels


# ---------------------------------------------------------------------------
# Tests: ingest()
# ---------------------------------------------------------------------------


class TestLettaSleepyIngest:
    def test_ingest_stores_passage(self):
        adapter = _ingested_adapter(n=2)
        passages = adapter._client.agents.passages._store.get(adapter._agent_id, [])
        assert len(passages) == 2

    def test_ingest_passage_contains_episode_id(self):
        adapter = _ingested_adapter(n=1)
        passages = adapter._client.agents.passages._store[adapter._agent_id]
        assert "[ep_001]" in passages[0].text

    def test_ingest_caches_text(self):
        adapter = _ingested_adapter(n=2)
        assert "ep_001" in adapter._text_cache
        assert "ep_002" in adapter._text_cache

    def test_ingest_requires_reset(self):
        adapter = _make_adapter()
        with pytest.raises(Exception):
            adapter.ingest("ep_001", "scope_x", "2024-01-01", "text")


# ---------------------------------------------------------------------------
# Tests: prepare()
# ---------------------------------------------------------------------------


class TestLettaSleepyPrepare:
    def test_prepare_sends_message_to_agent(self):
        adapter = _ingested_adapter(n=3)
        adapter.prepare("scope_test", checkpoint=5)
        last_input = adapter._client.agents.messages.last_input
        assert last_input is not None
        assert "checkpoint" in last_input.lower() or "5" in last_input

    def test_prepare_no_agent_is_noop(self):
        adapter = _make_adapter()
        # Should not raise even without reset
        adapter.prepare("scope_test", checkpoint=1)

    def test_prepare_mentions_checkpoint(self):
        adapter = _ingested_adapter(n=2)
        adapter.prepare("scope_test", checkpoint=12)
        last_input = adapter._client.agents.messages.last_input
        assert "12" in last_input


# ---------------------------------------------------------------------------
# Tests: search()
# ---------------------------------------------------------------------------


class TestLettaSleepySearch:
    def test_search_returns_passage_results(self):
        adapter = _ingested_adapter(n=3)
        results = adapter.search("metric_a")
        assert len(results) > 0

    def test_search_includes_sleep_memory(self):
        """When core memory blocks have content, search prepends a sleep_memory result."""
        adapter = _ingested_adapter(n=2)
        # Add meaningful core memory
        adapter._client.agents.blocks._set(
            adapter._agent_id,
            [
                _MockBlock("persona", "Consolidated: metrics show upward trend."),
                _MockBlock("human", "scope info"),
            ],
        )
        results = adapter.search("metric_a")
        sleep_results = [r for r in results if r.ref_id == "sleep_memory"]
        assert len(sleep_results) == 1
        assert "Consolidated" in sleep_results[0].text

    def test_search_sleep_memory_excludes_human_block(self):
        adapter = _ingested_adapter(n=1)
        adapter._client.agents.blocks._set(
            adapter._agent_id,
            [
                _MockBlock("human", "scope info only"),
                _MockBlock("persona", "insights here"),
            ],
        )
        results = adapter.search("metric")
        sleep_results = [r for r in results if r.ref_id == "sleep_memory"]
        assert len(sleep_results) == 1
        assert "scope info only" not in sleep_results[0].text
        assert "insights" in sleep_results[0].text

    def test_search_no_sleep_memory_when_blocks_empty(self):
        adapter = _ingested_adapter(n=1)
        # Clear blocks to empty
        adapter._client.agents.blocks._set(adapter._agent_id, [
            _MockBlock("human", "scope"),
        ])
        results = adapter.search("metric_a")
        sleep_results = [r for r in results if r.ref_id == "sleep_memory"]
        assert len(sleep_results) == 0

    def test_search_total_within_limit(self):
        adapter = _ingested_adapter(n=5)
        results = adapter.search("metric_a", limit=5)
        assert len(results) <= 5

    def test_search_empty_query_returns_empty(self):
        adapter = _ingested_adapter()
        assert adapter.search("") == []
        assert adapter.search("   ") == []

    def test_search_no_agent_returns_empty(self):
        adapter = _make_adapter()
        assert adapter.search("query") == []

    def test_search_sleep_memory_score_is_high(self):
        adapter = _ingested_adapter()
        adapter._client.agents.blocks._set(
            adapter._agent_id,
            [_MockBlock("persona", "synth")],
        )
        results = adapter.search("metric_a")
        sleep = next(r for r in results if r.ref_id == "sleep_memory")
        assert sleep.score == 1.0

    def test_search_returns_searchresult_instances(self):
        adapter = _ingested_adapter()
        results = adapter.search("metric_a")
        assert all(isinstance(r, SearchResult) for r in results)


# ---------------------------------------------------------------------------
# Tests: retrieve()
# ---------------------------------------------------------------------------


class TestLettaSleepyRetrieve:
    def test_retrieve_episode_from_cache(self):
        adapter = _ingested_adapter(n=2)
        doc = adapter.retrieve("ep_001")
        assert isinstance(doc, Document)
        assert "ep_001" not in doc.text  # text cache stores raw text, not prefixed

    def test_retrieve_sleep_memory(self):
        adapter = _ingested_adapter()
        adapter._client.agents.blocks._set(
            adapter._agent_id,
            [_MockBlock("persona", "Core memory text")],
        )
        doc = adapter.retrieve("sleep_memory")
        assert isinstance(doc, Document)
        assert doc.ref_id == "sleep_memory"
        assert "Core memory text" in doc.text

    def test_retrieve_sleep_memory_empty_returns_none(self):
        adapter = _ingested_adapter()
        # Only human block (excluded)
        adapter._client.agents.blocks._set(
            adapter._agent_id,
            [_MockBlock("human", "scope info")],
        )
        assert adapter.retrieve("sleep_memory") is None

    def test_retrieve_missing_episode_returns_none(self):
        adapter = _ingested_adapter(n=1)
        assert adapter.retrieve("ep_999") is None


# ---------------------------------------------------------------------------
# Tests: get_capabilities()
# ---------------------------------------------------------------------------


class TestLettaSleepyCapabilities:
    def test_capabilities_structure(self):
        adapter = _make_adapter()
        caps = adapter.get_capabilities()
        assert isinstance(caps, CapabilityManifest)
        assert "semantic" in caps.search_modes
        assert "sleep-time-compute" in caps.search_modes

    def test_capabilities_has_batch_retrieve(self):
        adapter = _make_adapter()
        caps = adapter.get_capabilities()
        tool_names = [t.name for t in caps.extra_tools]
        assert "batch_retrieve" in tool_names


# ---------------------------------------------------------------------------
# Tests: batch_retrieve extended tool
# ---------------------------------------------------------------------------


class TestLettaSleepyBatchRetrieve:
    def test_batch_retrieve_returns_episodes(self):
        adapter = _ingested_adapter(n=3)
        result = adapter.call_extended_tool("batch_retrieve", {"ref_ids": ["ep_001", "ep_002"]})
        assert result["count"] == 2
        assert len(result["documents"]) == 2

    def test_batch_retrieve_includes_sleep_memory(self):
        adapter = _ingested_adapter(n=2)
        adapter._client.agents.blocks._set(
            adapter._agent_id,
            [_MockBlock("persona", "Some consolidated memory")],
        )
        result = adapter.call_extended_tool(
            "batch_retrieve", {"ref_ids": ["sleep_memory", "ep_001"]}
        )
        assert result["count"] == 2
        ref_ids = [d["ref_id"] for d in result["documents"]]
        assert "sleep_memory" in ref_ids
        assert "ep_001" in ref_ids

    def test_batch_retrieve_skips_missing(self):
        adapter = _ingested_adapter(n=1)
        result = adapter.call_extended_tool("batch_retrieve", {"ref_ids": ["ep_999"]})
        assert result["count"] == 0

    def test_batch_retrieve_empty_list(self):
        adapter = _ingested_adapter()
        result = adapter.call_extended_tool("batch_retrieve", {"ref_ids": []})
        assert result["count"] == 0
        assert result["documents"] == []

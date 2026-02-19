"""Tests for the Letta memory adapter.

Uses a MockLettaClient to avoid requiring a running Letta server.
"""
from __future__ import annotations

import uuid
from unittest.mock import MagicMock, patch

import pytest

from lens.adapters.base import CapabilityManifest, Document, SearchResult
from lens.adapters.registry import get_adapter


# ---------------------------------------------------------------------------
# Mock Letta client
# ---------------------------------------------------------------------------


class _MockPassage:
    def __init__(self, text: str, passage_id: str | None = None):
        self.content = text
        self.text = text  # fallback attribute
        self.timestamp = "2024-01-01T00:00:00"
        self.id = passage_id or str(uuid.uuid4())
        self.tags = []


class _MockSearchResult:
    def __init__(self, text: str):
        self.content = text
        self.timestamp = "2024-01-01T00:00:00"
        self.id = None
        self.tags = []


class _MockPassageSearchResponse:
    def __init__(self, results: list[_MockSearchResult]):
        self.results = results


class _MockPassagesNamespace:
    """Mock of client.agents.passages.*"""

    def __init__(self):
        # agent_id -> list of _MockPassage
        self._store: dict[str, list[_MockPassage]] = {}

    def create(self, agent_id: str, text: str):
        passage = _MockPassage(text)
        self._store.setdefault(agent_id, []).append(passage)
        return [passage]  # Letta returns a list

    def search(self, agent_id: str, query: str) -> _MockPassageSearchResponse:
        passages = self._store.get(agent_id, [])
        q = query.lower()
        matches = []
        for p in passages:
            text_lower = p.content.lower()
            if q in text_lower or any(w in text_lower for w in q.split()):
                matches.append(_MockSearchResult(p.content))
        return _MockPassageSearchResponse(matches)

    def list(self, agent_id: str) -> list[_MockPassage]:
        return list(self._store.get(agent_id, []))


class _MockAgent:
    def __init__(self, name: str, agent_id: str | None = None):
        self.name = name
        self.id = agent_id or str(uuid.uuid4())


class _MockAgentsNamespace:
    """Mock of client.agents.*"""

    def __init__(self):
        self._agents: dict[str, _MockAgent] = {}
        self.passages = _MockPassagesNamespace()

    def create(self, name: str, model: str, embedding: str, memory_blocks: list) -> _MockAgent:
        agent = _MockAgent(name)
        self._agents[agent.id] = agent
        return agent

    def delete(self, agent_id: str) -> None:
        self._agents.pop(agent_id, None)
        self.passages._store.pop(agent_id, None)

    def list(self) -> list[_MockAgent]:
        return list(self._agents.values())


class MockLettaClient:
    def __init__(self):
        self.agents = _MockAgentsNamespace()


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _make_adapter():
    """Create a LettaAdapter with a mocked Letta client."""
    from lens.adapters.letta import LettaAdapter

    mock_client = MockLettaClient()
    mock_letta_cls = MagicMock()
    mock_letta_cls.return_value = mock_client

    with patch.dict("sys.modules", {"letta_client": MagicMock(Letta=mock_letta_cls)}):
        adapter = LettaAdapter()
        # Force client initialization
        adapter._client = mock_client

    return adapter


# ---------------------------------------------------------------------------
# Tests: Registration
# ---------------------------------------------------------------------------


class TestLettaRegistration:
    def test_letta_registered(self):
        cls = get_adapter("letta")
        assert cls.__name__ == "LettaAdapter"


# ---------------------------------------------------------------------------
# Tests: reset()
# ---------------------------------------------------------------------------


class TestLettaReset:
    def test_reset_creates_agent(self):
        adapter = _make_adapter()
        adapter.reset("scope_01")
        assert adapter._agent_id is not None
        assert adapter._scope_id == "scope_01"

    def test_reset_clears_text_cache(self):
        adapter = _make_adapter()
        adapter.reset("scope_01")
        adapter._text_cache["ep_001"] = "old data"

        adapter.reset("scope_01")
        assert "ep_001" not in adapter._text_cache

    def test_reset_deletes_previous_agent(self):
        adapter = _make_adapter()
        adapter.reset("scope_01")
        first_id = adapter._agent_id

        adapter.reset("scope_01")
        second_id = adapter._agent_id

        # New agent should have a different ID
        assert first_id != second_id

    def test_reset_scans_stale_agents(self):
        """reset() scans agents.list() and deletes any with matching name."""
        adapter = _make_adapter()
        # Pre-seed a stale agent with the same name
        stale_agent = adapter._client.agents.create(
            name="lens-scope_01",
            model="x",
            embedding="x",
            memory_blocks=[],
        )
        assert stale_agent.id in adapter._client.agents._agents

        adapter.reset("scope_01")

        # stale_agent should be gone (only new agent remains)
        assert stale_agent.id not in adapter._client.agents._agents


# ---------------------------------------------------------------------------
# Tests: ingest()
# ---------------------------------------------------------------------------


class TestLettaIngest:
    def test_ingest_stores_passage(self):
        adapter = _make_adapter()
        adapter.reset("scope_01")
        adapter.ingest("ep_001", "scope_01", "2024-01-01T00:00:00", "CPU at 85%")

        passages = adapter._client.agents.passages.list(adapter._agent_id)
        assert len(passages) == 1
        assert "[ep_001]" in passages[0].content

    def test_ingest_caches_text(self):
        adapter = _make_adapter()
        adapter.reset("scope_01")
        adapter.ingest("ep_001", "scope_01", "2024-01-01T00:00:00", "CPU at 85%")

        assert "ep_001" in adapter._text_cache
        assert adapter._text_cache["ep_001"] == "CPU at 85%"

    def test_ingest_prepends_ep_id_and_timestamp(self):
        adapter = _make_adapter()
        adapter.reset("scope_01")
        adapter.ingest("ep_001", "scope_01", "2024-01-01T00:00:00", "CPU at 85%")

        passages = adapter._client.agents.passages.list(adapter._agent_id)
        content = passages[0].content
        assert content.startswith("[ep_001]")
        assert "2024-01-01T00:00:00" in content
        assert "CPU at 85%" in content

    def test_ingest_without_reset_raises(self):
        adapter = _make_adapter()
        from lens.core.errors import AdapterError

        with pytest.raises(AdapterError, match="reset\\(\\) must be called"):
            adapter.ingest("ep_001", "scope_01", "2024-01-01", "Data")

    def test_ingest_multiple_episodes(self):
        adapter = _make_adapter()
        adapter.reset("scope_01")
        for i in range(5):
            adapter.ingest(f"ep_{i:03d}", "scope_01", f"2024-01-0{i+1}", f"Episode {i}")

        passages = adapter._client.agents.passages.list(adapter._agent_id)
        assert len(passages) == 5
        assert len(adapter._text_cache) == 5


# ---------------------------------------------------------------------------
# Tests: search()
# ---------------------------------------------------------------------------


class TestLettaSearch:
    def test_search_returns_list_of_search_results(self):
        adapter = _make_adapter()
        adapter.reset("scope_01")
        adapter.ingest("ep_001", "scope_01", "2024-01-01", "CPU utilization spiked")
        adapter.ingest("ep_002", "scope_01", "2024-01-02", "Memory usage normal")

        results = adapter.search("CPU")
        assert isinstance(results, list)
        assert all(isinstance(r, SearchResult) for r in results)

    def test_search_extracts_episode_id_from_content(self):
        adapter = _make_adapter()
        adapter.reset("scope_01")
        adapter.ingest("ep_001", "scope_01", "2024-01-01", "CPU spike detected")

        results = adapter.search("CPU")
        assert len(results) >= 1
        # The ref_id should be the episode ID extracted from [ep_id] prefix
        assert results[0].ref_id == "ep_001"

    def test_search_empty_query_returns_empty(self):
        adapter = _make_adapter()
        adapter.reset("scope_01")
        adapter.ingest("ep_001", "scope_01", "2024-01-01", "Some data")

        assert adapter.search("") == []
        assert adapter.search("   ") == []

    def test_search_no_agent_returns_empty(self):
        adapter = _make_adapter()
        # No reset called
        assert adapter.search("anything") == []

    def test_search_respects_limit(self):
        adapter = _make_adapter()
        adapter.reset("scope_01")
        for i in range(15):
            adapter.ingest(f"ep_{i:03d}", "scope_01", "2024-01-01", f"data point {i}")

        results = adapter.search("data", limit=5)
        assert len(results) <= 5

    def test_search_default_limit(self):
        adapter = _make_adapter()
        adapter.reset("scope_01")
        for i in range(20):
            adapter.ingest(f"ep_{i:03d}", "scope_01", "2024-01-01", f"data point {i}")

        results = adapter.search("data")
        assert len(results) <= 10  # default limit

    def test_search_truncates_text_to_500_chars(self):
        adapter = _make_adapter()
        adapter.reset("scope_01")
        long_text = "x" * 1000
        adapter.ingest("ep_001", "scope_01", "2024-01-01", long_text)

        results = adapter.search("x")
        if results:
            assert len(results[0].text) <= 500


# ---------------------------------------------------------------------------
# Tests: retrieve()
# ---------------------------------------------------------------------------


class TestLettaRetrieve:
    def test_retrieve_returns_full_text(self):
        adapter = _make_adapter()
        adapter.reset("scope_01")
        adapter.ingest("ep_001", "scope_01", "2024-01-01", "Full episode text here")

        doc = adapter.retrieve("ep_001")
        assert doc is not None
        assert isinstance(doc, Document)
        assert doc.ref_id == "ep_001"
        assert doc.text == "Full episode text here"

    def test_retrieve_missing_returns_none(self):
        adapter = _make_adapter()
        assert adapter.retrieve("nonexistent") is None

    def test_retrieve_returns_original_text_not_passage_content(self):
        """retrieve() returns raw episode text, not the [ep_id] prefixed content."""
        adapter = _make_adapter()
        adapter.reset("scope_01")
        adapter.ingest("ep_001", "scope_01", "2024-01-01", "Raw episode text")

        doc = adapter.retrieve("ep_001")
        assert doc is not None
        # Should NOT include the [ep_001] prefix
        assert not doc.text.startswith("[ep_001]")
        assert doc.text == "Raw episode text"


# ---------------------------------------------------------------------------
# Tests: capabilities and extended tools
# ---------------------------------------------------------------------------


class TestLettaCapabilities:
    def test_capabilities_returns_manifest(self):
        adapter = _make_adapter()
        caps = adapter.get_capabilities()
        assert isinstance(caps, CapabilityManifest)
        assert "semantic" in caps.search_modes
        assert caps.max_results_per_search == 10

    def test_batch_retrieve_tool_registered(self):
        adapter = _make_adapter()
        caps = adapter.get_capabilities()
        tool_names = [t.name for t in caps.extra_tools]
        assert "batch_retrieve" in tool_names

    def test_batch_retrieve_returns_documents(self):
        adapter = _make_adapter()
        adapter.reset("scope_01")
        adapter.ingest("ep_001", "scope_01", "2024-01-01", "First episode")
        adapter.ingest("ep_002", "scope_01", "2024-01-02", "Second episode")

        result = adapter.call_extended_tool(
            "batch_retrieve", {"ref_ids": ["ep_001", "ep_002"]}
        )
        assert result["count"] == 2
        assert len(result["documents"]) == 2

    def test_batch_retrieve_skips_missing(self):
        adapter = _make_adapter()
        adapter.reset("scope_01")
        adapter.ingest("ep_001", "scope_01", "2024-01-01", "First episode")

        result = adapter.call_extended_tool(
            "batch_retrieve", {"ref_ids": ["ep_001", "nonexistent"]}
        )
        assert result["count"] == 1
        assert len(result["documents"]) == 1

    def test_requires_metering_false(self):
        adapter = _make_adapter()
        assert adapter.requires_metering is False

    def test_unknown_tool_delegates_to_super(self):
        adapter = _make_adapter()
        from lens.core.errors import AdapterError

        with pytest.raises((AdapterError, NotImplementedError)):
            adapter.call_extended_tool("nonexistent_tool", {})


# ---------------------------------------------------------------------------
# Tests: ep_id parsing
# ---------------------------------------------------------------------------


class TestEpIdParsing:
    def test_parse_ep_id_from_bracket_prefix(self):
        from lens.adapters.letta import _parse_ep_id

        assert _parse_ep_id("[ep_001] 2024-01-01: text") == "ep_001"
        assert _parse_ep_id("[scope_01-ep_025] timestamp: data") == "scope_01-ep_025"

    def test_parse_ep_id_no_bracket_returns_prefix(self):
        from lens.adapters.letta import _parse_ep_id

        result = _parse_ep_id("no brackets here, just plain text")
        assert len(result) <= 32

    def test_parse_ep_id_empty_bracket(self):
        from lens.adapters.letta import _parse_ep_id

        # Edge case: empty bracket — falls through to prefix
        result = _parse_ep_id("[] rest of text")
        # The regex requires at least one char inside brackets
        assert len(result) <= 32

"""Unit tests for the Graphiti adapter.

All tests mock graphiti-core and FalkorDB — no running services required.
"""
from __future__ import annotations

import asyncio
import threading
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from lens.adapters.base import CapabilityManifest, Document, SearchResult
from lens.adapters.graphiti_adapter import GraphitiAdapter, _AsyncRunner


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_mock_graphiti(ep_uuid="ep-uuid-1234"):
    """Create a mock Graphiti client with typical response shapes."""
    g = MagicMock()

    # build_indices_and_constraints — async void
    g.build_indices_and_constraints = AsyncMock(return_value=None)

    # add_episode — async → AddEpisodeResults
    mock_ep_node = MagicMock()
    mock_ep_node.uuid = ep_uuid
    mock_result = MagicMock()
    mock_result.episode = mock_ep_node
    g.add_episode = AsyncMock(return_value=mock_result)

    # _search — async → SearchResults
    mock_search_results = MagicMock()
    mock_search_results.edges = []
    g._search = AsyncMock(return_value=mock_search_results)

    return g


def _make_mock_edge(fact="p99 latency exceeded threshold", episode_uuids=None):
    edge = MagicMock()
    edge.fact = fact
    edge.uuid = "edge-uuid-abc"
    edge.episodes = episode_uuids or ["ep-uuid-1234"]
    return edge


@pytest.fixture
def adapter():
    return GraphitiAdapter()


@pytest.fixture
def mock_graphiti_modules():
    """Patch all graphiti imports used in _make_graphiti, prepare, and search."""
    mock_graphiti_cls = MagicMock()
    mock_falkor_cls = MagicMock()
    mock_llm_cls = MagicMock()
    mock_llm_config_cls = MagicMock()
    mock_embedder_cls = MagicMock()
    mock_embedder_config_cls = MagicMock()
    # Sentinel config value returned by EDGE_HYBRID_SEARCH_EPISODE_MENTIONS
    mock_search_config = MagicMock()

    with patch.dict(
        "sys.modules",
        {
            "graphiti_core": MagicMock(Graphiti=mock_graphiti_cls),
            "graphiti_core.driver": MagicMock(),
            "graphiti_core.driver.falkordb_driver": MagicMock(FalkorDriver=mock_falkor_cls),
            "graphiti_core.llm_client": MagicMock(),
            "graphiti_core.llm_client.openai_generic_client": MagicMock(OpenAIGenericClient=mock_llm_cls),
            "graphiti_core.llm_client.config": MagicMock(LLMConfig=mock_llm_config_cls),
            "graphiti_core.embedder": MagicMock(),
            "graphiti_core.embedder.openai": MagicMock(
                OpenAIEmbedder=mock_embedder_cls,
                OpenAIEmbedderConfig=mock_embedder_config_cls,
            ),
            "graphiti_core.embedder.client": MagicMock(),
            "graphiti_core.nodes": MagicMock(),
            "graphiti_core.search": MagicMock(),
            "graphiti_core.search.search_config_recipes": MagicMock(
                EDGE_HYBRID_SEARCH_EPISODE_MENTIONS=mock_search_config
            ),
        },
    ):
        yield {
            "Graphiti": mock_graphiti_cls,
            "FalkorDriver": mock_falkor_cls,
        }


# ---------------------------------------------------------------------------
# AsyncRunner unit tests
# ---------------------------------------------------------------------------


def test_async_runner_executes_coroutine():
    runner = _AsyncRunner()

    async def double(x):
        return x * 2

    result = runner.run(double(7))
    assert result == 14


def test_async_runner_singleton_reused(adapter):
    """_get_runner() returns the same instance across calls."""
    from lens.adapters.graphiti_adapter import _get_runner

    r1 = _get_runner()
    r2 = _get_runner()
    assert r1 is r2


def test_async_runner_timeout():
    runner = _AsyncRunner()

    async def slow():
        await asyncio.sleep(10)

    with pytest.raises(Exception):
        runner.run(slow(), timeout=0.05)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def test_adapter_registered():
    from lens.adapters.registry import get_adapter

    cls = get_adapter("graphiti")
    assert cls is GraphitiAdapter


# ---------------------------------------------------------------------------
# reset()
# ---------------------------------------------------------------------------


def test_reset_creates_unique_db_names(mock_graphiti_modules):
    """Each reset() call produces a unique FalkorDB database name."""
    mock_instance = _make_mock_graphiti()
    mock_graphiti_modules["Graphiti"].return_value = mock_instance

    adapter = GraphitiAdapter()
    adapter.reset("scope_01")
    db1 = adapter._db_name

    adapter.reset("scope_01")
    db2 = adapter._db_name

    assert db1 != db2
    assert "scope_01" in db1


def test_reset_clears_state(mock_graphiti_modules):
    mock_instance = _make_mock_graphiti()
    mock_graphiti_modules["Graphiti"].return_value = mock_instance

    adapter = GraphitiAdapter()
    adapter._text_cache = {"ep01": "old text"}
    adapter._ep_uuid_to_id = {"uuid1": "ep01"}
    adapter._pending_episodes = [{"episode_id": "ep01"}]

    adapter.reset("scope_01")

    assert adapter._text_cache == {}
    assert adapter._ep_uuid_to_id == {}
    assert adapter._pending_episodes == []


def test_reset_raises_adapter_error_on_falkordb_failure(mock_graphiti_modules):
    mock_graphiti_modules["Graphiti"].return_value.build_indices_and_constraints = AsyncMock(
        side_effect=ConnectionRefusedError("FalkorDB not running")
    )
    mock_graphiti_modules["Graphiti"].return_value = _make_mock_graphiti()
    mock_graphiti_modules["Graphiti"].return_value.build_indices_and_constraints = AsyncMock(
        side_effect=ConnectionRefusedError("FalkorDB not running")
    )

    from lens.core.errors import AdapterError

    adapter = GraphitiAdapter()
    with pytest.raises(AdapterError, match="FalkorDB"):
        adapter.reset("scope_01")


# ---------------------------------------------------------------------------
# ingest()
# ---------------------------------------------------------------------------


def test_ingest_without_reset_raises(adapter):
    from lens.core.errors import AdapterError

    with pytest.raises(AdapterError, match="reset"):
        adapter.ingest("ep01", "scope_01", "2024-01-01T00:00:00Z", "text")


def test_ingest_buffers_episode(mock_graphiti_modules):
    mock_graphiti_modules["Graphiti"].return_value = _make_mock_graphiti()

    adapter = GraphitiAdapter()
    adapter.reset("scope_01")
    adapter.ingest("ep01", "scope_01", "2024-01-01T00:00:00Z", "p99: 342ms")

    assert len(adapter._pending_episodes) == 1
    assert adapter._pending_episodes[0]["episode_id"] == "ep01"
    assert adapter._text_cache["ep01"] == "p99: 342ms"


def test_ingest_is_instant_no_io(mock_graphiti_modules):
    """ingest() only appends to buffer — no async calls."""
    mock_instance = _make_mock_graphiti()
    mock_graphiti_modules["Graphiti"].return_value = mock_instance

    adapter = GraphitiAdapter()
    adapter.reset("scope_01")
    adapter.ingest("ep01", "scope_01", "2024-01-01T00:00:00Z", "text")

    # add_episode should NOT have been called yet
    mock_instance.add_episode.assert_not_called()


# ---------------------------------------------------------------------------
# prepare()
# ---------------------------------------------------------------------------


def test_prepare_calls_add_episode_for_each_buffered(mock_graphiti_modules):
    mock_instance = _make_mock_graphiti("ep-uuid-aaa")
    mock_graphiti_modules["Graphiti"].return_value = mock_instance

    adapter = GraphitiAdapter()
    adapter.reset("scope_01")
    adapter.ingest("ep01", "scope_01", "2024-01-01T00:00:00Z", "text 1")
    adapter.ingest("ep02", "scope_01", "2024-01-02T00:00:00Z", "text 2")

    adapter.prepare("scope_01", checkpoint=5)

    assert mock_instance.add_episode.call_count == 2


def test_prepare_clears_buffer(mock_graphiti_modules):
    mock_graphiti_modules["Graphiti"].return_value = _make_mock_graphiti()

    adapter = GraphitiAdapter()
    adapter.reset("scope_01")
    adapter.ingest("ep01", "scope_01", "2024-01-01T00:00:00Z", "text")
    adapter.prepare("scope_01", checkpoint=5)

    assert adapter._pending_episodes == []


def test_prepare_tracks_ep_uuid_to_id(mock_graphiti_modules):
    mock_instance = _make_mock_graphiti("uuid-9999")
    mock_graphiti_modules["Graphiti"].return_value = mock_instance

    adapter = GraphitiAdapter()
    adapter.reset("scope_01")
    adapter.ingest("ep01", "scope_01", "2024-01-01T00:00:00Z", "text")
    adapter.prepare("scope_01", checkpoint=5)

    assert adapter._ep_uuid_to_id.get("uuid-9999") == "ep01"


def test_prepare_noop_when_no_pending(mock_graphiti_modules):
    mock_instance = _make_mock_graphiti()
    mock_graphiti_modules["Graphiti"].return_value = mock_instance

    adapter = GraphitiAdapter()
    adapter.reset("scope_01")
    adapter.prepare("scope_01", checkpoint=5)

    mock_instance.add_episode.assert_not_called()


# ---------------------------------------------------------------------------
# search()
# ---------------------------------------------------------------------------


def test_search_returns_empty_without_graphiti(adapter):
    results = adapter.search("latency issue")
    assert results == []


def test_search_maps_edges_to_episodes(mock_graphiti_modules):
    mock_instance = _make_mock_graphiti("ep-uuid-1234")
    edge = _make_mock_edge("p99 latency was elevated", ["ep-uuid-1234"])
    mock_instance._search.return_value = MagicMock(edges=[edge])
    mock_graphiti_modules["Graphiti"].return_value = mock_instance

    adapter = GraphitiAdapter()
    adapter.reset("scope_01")
    adapter.ingest("ep01", "scope_01", "2024-01-01T00:00:00Z", "p99: 342ms")
    adapter.prepare("scope_01", checkpoint=5)

    # Manually set ep_uuid_to_id (prepare would normally do this)
    adapter._ep_uuid_to_id["ep-uuid-1234"] = "ep01"
    adapter._pending_episodes = []

    results = adapter.search("latency")
    assert len(results) == 1
    assert results[0].ref_id == "ep01"
    assert "latency" in results[0].text.lower()


def test_search_deduplicates_by_episode(mock_graphiti_modules):
    """Multiple edges from the same episode produce one SearchResult."""
    mock_instance = _make_mock_graphiti()
    edges = [
        _make_mock_edge("fact A", ["ep-uuid-1234"]),
        _make_mock_edge("fact B", ["ep-uuid-1234"]),
    ]
    mock_instance._search.return_value = MagicMock(edges=edges)
    mock_graphiti_modules["Graphiti"].return_value = mock_instance

    adapter = GraphitiAdapter()
    adapter.reset("scope_01")
    adapter._ep_uuid_to_id["ep-uuid-1234"] = "ep01"
    adapter._text_cache["ep01"] = "full text"

    results = adapter.search("fact")
    assert len(results) == 1
    assert results[0].ref_id == "ep01"


def test_search_skips_edges_with_unknown_episodes(mock_graphiti_modules):
    mock_instance = _make_mock_graphiti()
    edge = _make_mock_edge("some fact", ["unknown-uuid"])
    mock_instance._search.return_value = MagicMock(edges=[edge])
    mock_graphiti_modules["Graphiti"].return_value = mock_instance

    adapter = GraphitiAdapter()
    adapter.reset("scope_01")
    # No mapping for unknown-uuid

    results = adapter.search("fact")
    assert results == []


def test_search_respects_limit(mock_graphiti_modules):
    mock_instance = _make_mock_graphiti()
    edges = [
        _make_mock_edge(f"fact {i}", [f"uuid-{i}"]) for i in range(10)
    ]
    mock_instance._search.return_value = MagicMock(edges=edges)
    mock_graphiti_modules["Graphiti"].return_value = mock_instance

    adapter = GraphitiAdapter()
    adapter.reset("scope_01")
    for i in range(10):
        adapter._ep_uuid_to_id[f"uuid-{i}"] = f"ep{i:02d}"
        adapter._text_cache[f"ep{i:02d}"] = f"text {i}"

    results = adapter.search("fact", limit=3)
    assert len(results) == 3


# ---------------------------------------------------------------------------
# retrieve()
# ---------------------------------------------------------------------------


def test_retrieve_returns_document_from_cache():
    adapter = GraphitiAdapter()
    adapter._text_cache["ep01"] = "p99: 342ms, error_rate: 0.3%"

    doc = adapter.retrieve("ep01")
    assert isinstance(doc, Document)
    assert doc.ref_id == "ep01"
    assert "342ms" in doc.text


def test_retrieve_returns_none_for_unknown_ref():
    adapter = GraphitiAdapter()
    assert adapter.retrieve("nonexistent") is None


# ---------------------------------------------------------------------------
# get_capabilities() / call_extended_tool()
# ---------------------------------------------------------------------------


def test_get_capabilities_returns_manifest():
    adapter = GraphitiAdapter()
    caps = adapter.get_capabilities()
    assert isinstance(caps, CapabilityManifest)
    assert "semantic" in caps.search_modes
    assert "graph" in caps.search_modes
    tool_names = [t.name for t in caps.extra_tools]
    assert "batch_retrieve" in tool_names


def test_batch_retrieve_returns_multiple_docs():
    adapter = GraphitiAdapter()
    adapter._text_cache = {"ep01": "text one", "ep02": "text two"}

    result = adapter.call_extended_tool(
        "batch_retrieve", {"ref_ids": ["ep01", "ep02", "ep_missing"]}
    )
    assert result["count"] == 2
    ids = {d["ref_id"] for d in result["documents"]}
    assert "ep01" in ids
    assert "ep02" in ids


def test_batch_retrieve_empty_input():
    adapter = GraphitiAdapter()
    result = adapter.call_extended_tool("batch_retrieve", {"ref_ids": []})
    assert result == {"documents": [], "count": 0}


def test_unknown_tool_raises():
    adapter = GraphitiAdapter()
    with pytest.raises(NotImplementedError):
        adapter.call_extended_tool("nonexistent_tool", {})

"""Unit tests for the Cognee adapter.

All tests mock cognee — no running services required.
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import pytest

from lens.adapters.base import CapabilityManifest, Document, SearchResult
from lens.adapters.cognee_adapter import CogneeAdapter, _AsyncRunner, _parse_ep_id


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_mock_cognee(search_results=None):
    """Create a mock cognee module with typical response shapes."""
    c = MagicMock()
    # cognee.prune is a CLASS with static async methods, not a callable.
    # prune.prune_data() and prune.prune_system() are the real methods.
    c.prune = MagicMock()
    c.prune.prune_data = AsyncMock(return_value=None)
    c.prune.prune_system = AsyncMock(return_value=None)
    c.add = AsyncMock(return_value=None)
    c.cognify = AsyncMock(return_value=None)
    c.search = AsyncMock(return_value=search_results or [])

    # SearchType enum — search() tries CHUNKS then SUMMARIES
    st = MagicMock()
    st.CHUNKS = "CHUNKS"
    st.SUMMARIES = "SUMMARIES"
    c.SearchType = st

    # config mock
    c.config = MagicMock()
    c.config.set_llm_config = MagicMock()

    # __file__ attribute — reset() uses os.path.dirname(cognee.__file__)
    # to locate the .cognee_system database directory for cleanup.
    c.__file__ = "/tmp/fake_cognee/__init__.py"

    return c


def _make_search_result(text: str):
    """Create a mock cognee SearchResult for CHUNKS search."""
    sr = MagicMock()
    sr.search_result = [{"text": text}]
    return sr


@pytest.fixture
def mock_cognee():
    """Patch cognee import in cognee_adapter module."""
    mc = _make_mock_cognee()
    with patch("lens.adapters.cognee_adapter.CogneeAdapter._get_cognee", return_value=mc):
        yield mc


@pytest.fixture
def adapter():
    return CogneeAdapter()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def test_parse_ep_id_extracts_id():
    assert _parse_ep_id("[ep_01_cascade] 2024-01-01: text") == "ep_01_cascade"


def test_parse_ep_id_returns_none_no_bracket():
    assert _parse_ep_id("no bracket prefix here") is None


def test_parse_ep_id_empty_string():
    assert _parse_ep_id("") is None


# ---------------------------------------------------------------------------
# AsyncRunner
# ---------------------------------------------------------------------------


def test_async_runner_executes_coroutine():
    runner = _AsyncRunner()

    async def triple(x):
        return x * 3

    assert runner.run(triple(4)) == 12


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

    cls = get_adapter("cognee")
    assert cls is CogneeAdapter


# ---------------------------------------------------------------------------
# reset()
# ---------------------------------------------------------------------------


def test_reset_creates_unique_dataset_ids(mock_cognee):
    adapter = CogneeAdapter()
    adapter.reset("scope_01")
    ds1 = adapter._dataset_id

    adapter.reset("scope_01")
    ds2 = adapter._dataset_id

    assert ds1 != ds2
    assert "scope_01" in ds1


def test_reset_clears_state(mock_cognee):
    adapter = CogneeAdapter()
    adapter._text_cache = {"ep01": "old"}
    adapter._pending_episodes = [{"episode_id": "ep01"}]

    adapter.reset("scope_01")

    assert adapter._text_cache == {}
    assert adapter._pending_episodes == []


def test_reset_calls_prune(mock_cognee):
    adapter = CogneeAdapter()
    adapter.reset("scope_01")
    mock_cognee.prune.prune_data.assert_called_once()
    mock_cognee.prune.prune_system.assert_called_once()


def test_reset_prune_failure_logs_warning_and_continues():
    """Prune failure is non-fatal — we log a warning and continue with a fresh dataset."""
    mc = _make_mock_cognee()
    mc.prune.prune_data = AsyncMock(side_effect=RuntimeError("DB connection failed"))

    adapter = CogneeAdapter()
    with patch(
        "lens.adapters.cognee_adapter.CogneeAdapter._get_cognee", return_value=mc
    ):
        # Should NOT raise — prune failure is handled gracefully
        adapter.reset("scope_01")
        assert adapter._dataset_id is not None


# ---------------------------------------------------------------------------
# ingest()
# ---------------------------------------------------------------------------


def test_ingest_without_reset_raises(adapter):
    from lens.core.errors import AdapterError

    with pytest.raises(AdapterError, match="reset"):
        adapter.ingest("ep01", "scope_01", "2024-01-01T00:00:00Z", "text")


def test_ingest_buffers_episode(mock_cognee):
    adapter = CogneeAdapter()
    adapter.reset("scope_01")
    adapter.ingest("ep01", "scope_01", "2024-01-01T00:00:00Z", "p99: 342ms")

    assert len(adapter._pending_episodes) == 1
    assert adapter._text_cache["ep01"] == "p99: 342ms"


def test_ingest_content_has_episode_prefix(mock_cognee):
    """Ingested content is prefixed with [episode_id] so chunks carry the ID."""
    adapter = CogneeAdapter()
    adapter.reset("scope_01")
    adapter.ingest("ep42", "scope_01", "2024-03-15T10:00:00Z", "latency: 500ms")

    content = adapter._pending_episodes[0]["content"]
    assert content.startswith("[ep42]")


def test_ingest_does_not_call_cognee_add(mock_cognee):
    """ingest() must be instant — no I/O calls."""
    adapter = CogneeAdapter()
    adapter.reset("scope_01")
    adapter.ingest("ep01", "scope_01", "2024-01-01T00:00:00Z", "text")

    mock_cognee.add.assert_not_called()


# ---------------------------------------------------------------------------
# prepare()
# ---------------------------------------------------------------------------


def test_prepare_calls_add_with_batched_episodes(mock_cognee):
    adapter = CogneeAdapter()
    adapter.reset("scope_01")
    adapter.ingest("ep01", "scope_01", "2024-01-01T00:00:00Z", "text 1")
    adapter.ingest("ep02", "scope_01", "2024-01-02T00:00:00Z", "text 2")

    adapter.prepare("scope_01", checkpoint=5)

    # Single batched call with list of all episode texts
    assert mock_cognee.add.call_count == 1
    call_kwargs = mock_cognee.add.call_args
    data_arg = call_kwargs.kwargs.get("data") or call_kwargs[1].get("data")
    assert isinstance(data_arg, list)
    assert len(data_arg) == 2


def test_prepare_calls_cognify(mock_cognee):
    adapter = CogneeAdapter()
    adapter.reset("scope_01")
    adapter.ingest("ep01", "scope_01", "2024-01-01T00:00:00Z", "text")

    adapter.prepare("scope_01", checkpoint=5)

    mock_cognee.cognify.assert_called_once()


def test_prepare_clears_buffer(mock_cognee):
    adapter = CogneeAdapter()
    adapter.reset("scope_01")
    adapter.ingest("ep01", "scope_01", "2024-01-01T00:00:00Z", "text")
    adapter.prepare("scope_01", checkpoint=5)

    assert adapter._pending_episodes == []


def test_prepare_noop_when_no_pending(mock_cognee):
    adapter = CogneeAdapter()
    adapter.reset("scope_01")
    adapter.prepare("scope_01", checkpoint=5)

    mock_cognee.add.assert_not_called()
    mock_cognee.cognify.assert_not_called()


def test_prepare_cognify_failure_is_nonfatal():
    """cognify() failure is non-fatal — logs warning, doesn't raise."""
    mc = _make_mock_cognee()
    mc.cognify = AsyncMock(side_effect=RuntimeError("Cognify failed"))

    adapter = CogneeAdapter()
    with patch(
        "lens.adapters.cognee_adapter.CogneeAdapter._get_cognee", return_value=mc
    ):
        adapter.reset("scope_01")
        adapter.ingest("ep01", "scope_01", "2024-01-01T00:00:00Z", "text")
        # Should NOT raise — cognify failure is handled gracefully
        adapter.prepare("scope_01", checkpoint=5)
        # add() should still have been called
        mc.add.assert_called_once()


# ---------------------------------------------------------------------------
# search()
# ---------------------------------------------------------------------------


def test_search_returns_empty_without_dataset(adapter):
    results = adapter.search("latency issue")
    assert results == []


def test_search_parses_episode_id_from_chunk(mock_cognee):
    sr = _make_search_result("[ep_cascade_01] 2024-01-01: p99: 342ms error_rate: 0.3%")
    mock_cognee.search.return_value = [sr]

    adapter = CogneeAdapter()
    adapter.reset("scope_01")
    adapter._text_cache["ep_cascade_01"] = "p99: 342ms error_rate: 0.3%"

    results = adapter.search("latency")
    assert len(results) == 1
    assert results[0].ref_id == "ep_cascade_01"


def test_search_deduplicates_same_episode(mock_cognee):
    """Two chunks from the same episode produce one SearchResult."""
    srs = [
        _make_search_result("[ep01] 2024-01-01: chunk A"),
        _make_search_result("[ep01] 2024-01-01: chunk B"),
    ]
    mock_cognee.search.return_value = srs

    adapter = CogneeAdapter()
    adapter.reset("scope_01")
    adapter._text_cache["ep01"] = "full episode text"

    results = adapter.search("chunk")
    assert len(results) == 1
    assert results[0].ref_id == "ep01"


def test_search_respects_limit(mock_cognee):
    srs = [
        _make_search_result(f"[ep{i:02d}] 2024-01-01: text {i}") for i in range(10)
    ]
    mock_cognee.search.return_value = srs

    adapter = CogneeAdapter()
    adapter.reset("scope_01")
    for i in range(10):
        adapter._text_cache[f"ep{i:02d}"] = f"text {i}"

    results = adapter.search("text", limit=4)
    assert len(results) == 4


def test_search_handles_string_search_result(mock_cognee):
    """search_result may be a plain string (only_context=True scenario)."""
    sr = MagicMock()
    sr.search_result = "[ep01] 2024-01-01: p99: 500ms"
    mock_cognee.search.return_value = [sr]

    adapter = CogneeAdapter()
    adapter.reset("scope_01")
    adapter._text_cache["ep01"] = "p99: 500ms"

    results = adapter.search("latency")
    assert len(results) == 1
    assert results[0].ref_id == "ep01"


def test_search_handles_dict_payload_results(mock_cognee):
    """cognee.search(CHUNKS) returns flat list of payload dicts, not SearchResult objects."""
    mock_cognee.search.return_value = [
        {"text": "[ep01] 2024-01-01: p99: 342ms error_rate: 0.3%", "id": "uuid1"},
        {"text": "[ep02] 2024-01-02: p99: 580ms error_rate: 0.8%", "id": "uuid2"},
    ]

    adapter = CogneeAdapter()
    adapter.reset("scope_01")
    adapter._text_cache = {"ep01": "p99: 342ms", "ep02": "p99: 580ms"}

    results = adapter.search("latency")
    assert len(results) == 2
    assert results[0].ref_id == "ep01"
    assert results[1].ref_id == "ep02"


def test_search_handles_acl_wrapped_results(mock_cognee):
    """cognee 0.5.2+ with ACL mode wraps results in dicts with 'search_result' key."""
    mock_cognee.search.return_value = [
        {
            "dataset_id": "some-uuid",
            "dataset_name": "lens_scope_01_abc123",
            "dataset_tenant_id": "tenant-uuid",
            "search_result": [
                {"text": "[ep01] 2024-01-01: p99: 342ms error_rate: 0.3%", "id": "uuid1"},
                {"text": "[ep02] 2024-01-02: p99: 580ms error_rate: 0.8%", "id": "uuid2"},
            ],
        }
    ]

    adapter = CogneeAdapter()
    adapter.reset("scope_01")
    adapter._text_cache = {"ep01": "p99: 342ms", "ep02": "p99: 580ms"}

    results = adapter.search("latency")
    assert len(results) == 2
    assert results[0].ref_id == "ep01"
    assert results[1].ref_id == "ep02"


def test_search_handles_empty_results(mock_cognee):
    mock_cognee.search.return_value = []
    adapter = CogneeAdapter()
    adapter.reset("scope_01")

    results = adapter.search("anything")
    assert results == []


def test_search_returns_empty_on_exception(mock_cognee):
    mock_cognee.search = AsyncMock(side_effect=RuntimeError("Search failed"))

    adapter = CogneeAdapter()
    adapter.reset("scope_01")

    # Should log warning and return [] rather than raising
    results = adapter.search("query")
    assert results == []


# ---------------------------------------------------------------------------
# retrieve()
# ---------------------------------------------------------------------------


def test_retrieve_returns_document_from_cache():
    adapter = CogneeAdapter()
    adapter._text_cache["ep01"] = "p99: 342ms, error_rate: 0.3%"

    doc = adapter.retrieve("ep01")
    assert isinstance(doc, Document)
    assert doc.ref_id == "ep01"
    assert "342ms" in doc.text


def test_retrieve_returns_none_for_unknown_ref():
    adapter = CogneeAdapter()
    assert adapter.retrieve("nonexistent") is None


# ---------------------------------------------------------------------------
# get_capabilities() / call_extended_tool()
# ---------------------------------------------------------------------------


def test_get_capabilities_returns_manifest():
    adapter = CogneeAdapter()
    caps = adapter.get_capabilities()
    assert isinstance(caps, CapabilityManifest)
    assert "semantic" in caps.search_modes
    tool_names = [t.name for t in caps.extra_tools]
    assert "batch_retrieve" in tool_names


def test_batch_retrieve_multiple_docs():
    adapter = CogneeAdapter()
    adapter._text_cache = {"ep01": "text one", "ep02": "text two"}

    result = adapter.call_extended_tool(
        "batch_retrieve", {"ref_ids": ["ep01", "ep02", "missing"]}
    )
    assert result["count"] == 2
    ids = {d["ref_id"] for d in result["documents"]}
    assert "ep01" in ids
    assert "ep02" in ids


def test_batch_retrieve_empty_input():
    adapter = CogneeAdapter()
    result = adapter.call_extended_tool("batch_retrieve", {"ref_ids": []})
    assert result == {"documents": [], "count": 0}


def test_unknown_tool_raises():
    adapter = CogneeAdapter()
    with pytest.raises(NotImplementedError):
        adapter.call_extended_tool("nonexistent_tool", {})

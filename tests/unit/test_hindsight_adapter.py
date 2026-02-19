"""Tests for the Hindsight memory adapter.

Uses a MockHindsightClient to avoid requiring a running Hindsight server.
"""
from __future__ import annotations

import uuid
from unittest.mock import MagicMock, patch

import pytest

from lens.adapters.base import CapabilityManifest, Document, SearchResult
from lens.adapters.registry import get_adapter


# ---------------------------------------------------------------------------
# Mock Hindsight response types
# ---------------------------------------------------------------------------


class _MockRecallResult:
    def __init__(self, text: str, memory_type: str = "observation"):
        self.text = text
        self.id = str(uuid.uuid4())
        self.type = memory_type
        self.entities = None
        self.context = None
        self.occurred_start = None
        self.occurred_end = None
        self.mentioned_at = None
        self.document_id = None
        self.metadata = None
        self.chunk_id = None
        self.tags = None


class _MockRecallResponse:
    def __init__(self, results: list[_MockRecallResult]):
        self.results = results
        self.trace = None
        self.entities = None
        self.chunks = None


class _MockReflectResponse:
    def __init__(self, text: str):
        self.text = text
        self.based_on = None
        self.structured_output = None
        self.usage = None
        self.trace = None


class _MockRetainResponse:
    def __init__(self):
        self.id = str(uuid.uuid4())


class _MockBankProfileResponse:
    def __init__(self, bank_id: str, name: str = ""):
        self.bank_id = bank_id
        self.name = name


# ---------------------------------------------------------------------------
# Mock Hindsight client
# ---------------------------------------------------------------------------


class MockHindsightClient:
    """In-memory mock of the Hindsight client."""

    def __init__(self):
        # bank_id -> list of (content, timestamp)
        self._store: dict[str, list[tuple[str, object]]] = {}
        self._banks: set[str] = set()

    def create_bank(self, bank_id: str, name: str | None = None, mission: str | None = None,
                    disposition: dict | None = None) -> _MockBankProfileResponse:
        self._banks.add(bank_id)
        self._store.setdefault(bank_id, [])
        return _MockBankProfileResponse(bank_id=bank_id, name=name or "")

    def delete_bank(self, bank_id: str) -> None:
        self._banks.discard(bank_id)
        self._store.pop(bank_id, None)

    def retain(self, bank_id: str, content: str, timestamp=None, context=None,
               document_id=None, metadata=None, entities=None, tags=None) -> _MockRetainResponse:
        self._store.setdefault(bank_id, []).append((content, timestamp, document_id))
        return _MockRetainResponse()

    def recall(self, bank_id: str, query: str, types=None, max_tokens: int = 4096,
               budget: str = "mid", trace: bool = False, query_timestamp=None,
               include_entities: bool = False, max_entity_tokens: int = 500,
               include_chunks: bool = False, max_chunk_tokens: int = 8192,
               tags=None, tags_match: str = "any") -> _MockRecallResponse:
        entries = self._store.get(bank_id, [])
        q = query.lower()
        matches = []
        for content, _ts, doc_id in entries:
            content_lower = content.lower()
            if q in content_lower or any(w in content_lower for w in q.split()):
                result = _MockRecallResult(content)
                result.document_id = doc_id
                matches.append(result)
        return _MockRecallResponse(matches)

    def reflect(self, bank_id: str, query: str, budget: str = "low", context=None,
                max_tokens=None, response_schema=None, tags=None,
                tags_match: str = "any") -> _MockReflectResponse:
        entries = self._store.get(bank_id, [])
        synthesis = f"Synthesis for '{query}': {len(entries)} episodes analyzed."
        return _MockReflectResponse(synthesis)


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _make_adapter():
    """Create a HindsightAdapter with a mocked Hindsight client."""
    from lens.adapters.hindsight import HindsightAdapter

    mock_client = MockHindsightClient()
    mock_hindsight_cls = MagicMock()
    mock_hindsight_cls.return_value = mock_client

    with patch.dict("sys.modules", {"hindsight_client": MagicMock(Hindsight=mock_hindsight_cls)}):
        adapter = HindsightAdapter()
        adapter._client = mock_client  # inject directly

    return adapter


# ---------------------------------------------------------------------------
# Tests: Registration
# ---------------------------------------------------------------------------


class TestHindsightRegistration:
    def test_hindsight_registered(self):
        cls = get_adapter("hindsight")
        assert cls.__name__ == "HindsightAdapter"

    def test_hindsight_is_memory_adapter(self):
        from lens.adapters.base import MemoryAdapter
        from lens.adapters.hindsight import HindsightAdapter

        assert issubclass(HindsightAdapter, MemoryAdapter)


# ---------------------------------------------------------------------------
# Tests: reset()
# ---------------------------------------------------------------------------


class TestHindsightReset:
    def test_reset_creates_bank(self):
        adapter = _make_adapter()
        adapter.reset("scope_01")
        assert adapter._bank_id is not None
        assert "scope_01" in adapter._bank_id

    def test_reset_bank_id_contains_scope(self):
        adapter = _make_adapter()
        adapter.reset("scope_01")
        assert adapter._bank_id.startswith("scope_01-")

    def test_reset_generates_unique_bank_ids(self):
        adapter = _make_adapter()
        adapter.reset("scope_01")
        first_id = adapter._bank_id

        adapter.reset("scope_01")
        second_id = adapter._bank_id

        assert first_id != second_id

    def test_reset_clears_text_cache(self):
        adapter = _make_adapter()
        adapter.reset("scope_01")
        adapter._text_cache["ep_001"] = "old data"

        adapter.reset("scope_01")
        assert "ep_001" not in adapter._text_cache

    def test_reset_bank_created_in_server(self):
        adapter = _make_adapter()
        adapter.reset("scope_01")
        assert adapter._bank_id in adapter._client._banks

    def test_reset_server_error_raises_adapter_error(self):
        adapter = _make_adapter()
        from lens.core.errors import AdapterError

        adapter._client.create_bank = MagicMock(side_effect=RuntimeError("connection refused"))
        with pytest.raises(AdapterError, match="Failed to create Hindsight bank"):
            adapter.reset("scope_01")


# ---------------------------------------------------------------------------
# Tests: ingest()
# ---------------------------------------------------------------------------


class TestHindsightIngest:
    def test_ingest_stores_content(self):
        adapter = _make_adapter()
        adapter.reset("scope_01")
        adapter.ingest("ep_001", "scope_01", "2024-01-01T00:00:00", "CPU at 85%")

        entries = adapter._client._store.get(adapter._bank_id, [])
        assert len(entries) == 1

    def test_ingest_prepends_ep_id_and_timestamp(self):
        adapter = _make_adapter()
        adapter.reset("scope_01")
        adapter.ingest("ep_001", "scope_01", "2024-01-01T00:00:00", "CPU at 85%")

        content, _, doc_id = adapter._client._store[adapter._bank_id][0]
        assert content.startswith("[ep_001]")
        assert "2024-01-01T00:00:00" in content
        assert "CPU at 85%" in content

    def test_ingest_passes_document_id(self):
        """retain() is called with document_id=episode_id for reliable episode mapping."""
        adapter = _make_adapter()
        adapter.reset("scope_01")
        adapter.ingest("ep_001", "scope_01", "2024-01-01T00:00:00", "CPU at 85%")

        _, _, doc_id = adapter._client._store[adapter._bank_id][0]
        assert doc_id == "ep_001"

    def test_ingest_caches_text(self):
        adapter = _make_adapter()
        adapter.reset("scope_01")
        adapter.ingest("ep_001", "scope_01", "2024-01-01T00:00:00", "CPU at 85%")

        assert "ep_001" in adapter._text_cache
        assert adapter._text_cache["ep_001"] == "CPU at 85%"

    def test_ingest_without_reset_raises(self):
        adapter = _make_adapter()
        from lens.core.errors import AdapterError

        with pytest.raises(AdapterError, match="reset\\(\\) must be called"):
            adapter.ingest("ep_001", "scope_01", "2024-01-01", "Data")

    def test_ingest_multiple_episodes(self):
        adapter = _make_adapter()
        adapter.reset("scope_01")
        for i in range(5):
            adapter.ingest(f"ep_{i:03d}", "scope_01", f"2024-01-0{i+1}T00:00:00", f"Episode {i}")

        entries = adapter._client._store[adapter._bank_id]
        assert len(entries) == 5
        assert len(adapter._text_cache) == 5

    def test_ingest_passes_timestamp_to_retain(self):
        """retain() should receive a datetime object parsed from the ISO string."""
        adapter = _make_adapter()
        adapter.reset("scope_01")

        retain_calls = []
        original_retain = adapter._client.retain

        def capture_retain(**kwargs):
            retain_calls.append(kwargs)
            return original_retain(**kwargs)

        adapter._client.retain = capture_retain
        adapter.ingest("ep_001", "scope_01", "2024-01-15T12:00:00", "Data")

        assert len(retain_calls) == 1
        # timestamp should be converted to a datetime (or None for invalid)
        ts = retain_calls[0].get("timestamp")
        if ts is not None:
            from datetime import datetime
            assert isinstance(ts, datetime)



# ---------------------------------------------------------------------------
# Tests: search()
# ---------------------------------------------------------------------------


class TestHindsightSearch:
    def test_search_returns_list_of_search_results(self):
        adapter = _make_adapter()
        adapter.reset("scope_01")
        adapter.ingest("ep_001", "scope_01", "2024-01-01T00:00:00", "CPU utilization spiked")
        adapter.ingest("ep_002", "scope_01", "2024-01-02T00:00:00", "Memory usage normal")

        results = adapter.search("CPU")
        assert isinstance(results, list)
        assert all(isinstance(r, SearchResult) for r in results)

    def test_search_uses_document_id_as_ref_id(self):
        """search() uses document_id from recall results as ref_id for episode mapping."""
        adapter = _make_adapter()
        adapter.reset("scope_01")
        adapter.ingest("ep_001", "scope_01", "2024-01-01T00:00:00", "CPU spike detected")

        results = adapter.search("CPU")
        assert len(results) >= 1
        # ref_id should come from document_id set during retain
        assert results[0].ref_id == "ep_001"

    def test_search_empty_query_returns_empty(self):
        adapter = _make_adapter()
        adapter.reset("scope_01")
        adapter.ingest("ep_001", "scope_01", "2024-01-01T00:00:00", "Some data")

        assert adapter.search("") == []
        assert adapter.search("   ") == []

    def test_search_no_bank_returns_empty(self):
        adapter = _make_adapter()
        # No reset called
        assert adapter.search("anything") == []

    def test_search_respects_limit(self):
        adapter = _make_adapter()
        adapter.reset("scope_01")
        for i in range(15):
            adapter.ingest(f"ep_{i:03d}", "scope_01", "2024-01-01T00:00:00", f"data point {i}")

        results = adapter.search("data", limit=5)
        assert len(results) <= 5

    def test_search_default_limit(self):
        adapter = _make_adapter()
        adapter.reset("scope_01")
        for i in range(20):
            adapter.ingest(f"ep_{i:03d}", "scope_01", "2024-01-01T00:00:00", f"data point {i}")

        results = adapter.search("data")
        assert len(results) <= 10

    def test_search_truncates_text_to_500_chars(self):
        adapter = _make_adapter()
        adapter.reset("scope_01")
        long_text = "x" * 1000
        adapter.ingest("ep_001", "scope_01", "2024-01-01T00:00:00", long_text)

        results = adapter.search("x")
        if results:
            assert len(results[0].text) <= 500

    def test_search_exception_returns_empty(self):
        adapter = _make_adapter()
        adapter.reset("scope_01")
        adapter._client.recall = MagicMock(side_effect=RuntimeError("server error"))

        results = adapter.search("query")
        assert results == []

    def test_search_result_has_metadata(self):
        adapter = _make_adapter()
        adapter.reset("scope_01")
        adapter.ingest("ep_001", "scope_01", "2024-01-01T00:00:00", "latency 320ms")

        results = adapter.search("latency")
        if results:
            assert "memory_type" in results[0].metadata


# ---------------------------------------------------------------------------
# Tests: retrieve()
# ---------------------------------------------------------------------------


class TestHindsightRetrieve:
    def test_retrieve_returns_full_text(self):
        adapter = _make_adapter()
        adapter.reset("scope_01")
        adapter.ingest("ep_001", "scope_01", "2024-01-01T00:00:00", "Full episode text here")

        doc = adapter.retrieve("ep_001")
        assert doc is not None
        assert isinstance(doc, Document)
        assert doc.ref_id == "ep_001"
        assert doc.text == "Full episode text here"

    def test_retrieve_missing_returns_none(self):
        adapter = _make_adapter()
        assert adapter.retrieve("nonexistent") is None

    def test_retrieve_returns_original_text_not_prefixed_content(self):
        """retrieve() returns raw episode text, not the [ep_id] prefixed content."""
        adapter = _make_adapter()
        adapter.reset("scope_01")
        adapter.ingest("ep_001", "scope_01", "2024-01-01T00:00:00", "Raw episode text")

        doc = adapter.retrieve("ep_001")
        assert doc is not None
        assert not doc.text.startswith("[ep_001]")
        assert doc.text == "Raw episode text"


# ---------------------------------------------------------------------------
# Tests: capabilities
# ---------------------------------------------------------------------------


class TestHindsightCapabilities:
    def test_capabilities_returns_manifest(self):
        adapter = _make_adapter()
        caps = adapter.get_capabilities()
        assert isinstance(caps, CapabilityManifest)

    def test_search_modes_include_tempr(self):
        adapter = _make_adapter()
        caps = adapter.get_capabilities()
        for mode in ["semantic", "keyword", "graph", "temporal"]:
            assert mode in caps.search_modes

    def test_max_results_per_search(self):
        adapter = _make_adapter()
        caps = adapter.get_capabilities()
        assert caps.max_results_per_search == 10

    def test_batch_retrieve_tool_registered(self):
        adapter = _make_adapter()
        caps = adapter.get_capabilities()
        tool_names = [t.name for t in caps.extra_tools]
        assert "batch_retrieve" in tool_names

    def test_memory_reflect_tool_registered(self):
        adapter = _make_adapter()
        caps = adapter.get_capabilities()
        tool_names = [t.name for t in caps.extra_tools]
        assert "memory_reflect" in tool_names

    def test_requires_metering_false(self):
        adapter = _make_adapter()
        assert adapter.requires_metering is False

    def test_unknown_tool_raises(self):
        adapter = _make_adapter()
        with pytest.raises((NotImplementedError,)):
            adapter.call_extended_tool("nonexistent_tool", {})


# ---------------------------------------------------------------------------
# Tests: batch_retrieve
# ---------------------------------------------------------------------------


class TestHindsightBatchRetrieve:
    def test_batch_retrieve_returns_documents(self):
        adapter = _make_adapter()
        adapter.reset("scope_01")
        adapter.ingest("ep_001", "scope_01", "2024-01-01T00:00:00", "First episode")
        adapter.ingest("ep_002", "scope_01", "2024-01-02T00:00:00", "Second episode")

        result = adapter.call_extended_tool("batch_retrieve", {"ref_ids": ["ep_001", "ep_002"]})
        assert result["count"] == 2
        assert len(result["documents"]) == 2

    def test_batch_retrieve_skips_missing(self):
        adapter = _make_adapter()
        adapter.reset("scope_01")
        adapter.ingest("ep_001", "scope_01", "2024-01-01T00:00:00", "First episode")

        result = adapter.call_extended_tool("batch_retrieve", {"ref_ids": ["ep_001", "nonexistent"]})
        assert result["count"] == 1
        assert len(result["documents"]) == 1

    def test_batch_retrieve_empty_list(self):
        adapter = _make_adapter()
        result = adapter.call_extended_tool("batch_retrieve", {"ref_ids": []})
        assert result["count"] == 0
        assert result["documents"] == []

    def test_batch_retrieve_document_has_ref_id_and_text(self):
        adapter = _make_adapter()
        adapter.reset("scope_01")
        adapter.ingest("ep_001", "scope_01", "2024-01-01T00:00:00", "Episode content")

        result = adapter.call_extended_tool("batch_retrieve", {"ref_ids": ["ep_001"]})
        doc = result["documents"][0]
        assert doc["ref_id"] == "ep_001"
        assert doc["text"] == "Episode content"


# ---------------------------------------------------------------------------
# Tests: memory_reflect
# ---------------------------------------------------------------------------


class TestHindsightReflect:
    def test_reflect_returns_synthesis(self):
        adapter = _make_adapter()
        adapter.reset("scope_01")
        adapter.ingest("ep_001", "scope_01", "2024-01-01T00:00:00", "Latency 320ms")
        adapter.ingest("ep_002", "scope_01", "2024-01-02T00:00:00", "Latency 450ms")

        result = adapter.call_extended_tool("memory_reflect", {"query": "latency trends"})
        assert "synthesis" in result
        assert isinstance(result["synthesis"], str)
        assert len(result["synthesis"]) > 0

    def test_reflect_no_query_returns_error(self):
        adapter = _make_adapter()
        adapter.reset("scope_01")

        result = adapter.call_extended_tool("memory_reflect", {"query": ""})
        assert "synthesis" in result
        assert "error" in result

    def test_reflect_no_bank_returns_error(self):
        adapter = _make_adapter()
        # No reset called
        result = adapter.call_extended_tool("memory_reflect", {"query": "trends"})
        assert "synthesis" in result
        assert "error" in result

    def test_reflect_server_error_returns_error_dict(self):
        adapter = _make_adapter()
        adapter.reset("scope_01")
        adapter._client.reflect = MagicMock(side_effect=RuntimeError("timeout"))

        result = adapter.call_extended_tool("memory_reflect", {"query": "trends"})
        assert "synthesis" in result
        assert result.get("error") is True

    def test_reflect_uses_high_budget(self):
        """reflect() should use budget='high' for better quality."""
        adapter = _make_adapter()
        adapter.reset("scope_01")

        reflect_calls = []
        original_reflect = adapter._client.reflect

        def capture_reflect(**kwargs):
            reflect_calls.append(kwargs)
            return original_reflect(**kwargs)

        adapter._client.reflect = capture_reflect
        adapter.call_extended_tool("memory_reflect", {"query": "test"})

        assert len(reflect_calls) == 1
        assert reflect_calls[0].get("budget") == "high"


# ---------------------------------------------------------------------------
# Tests: ep_id parsing
# ---------------------------------------------------------------------------


class TestEpIdParsing:
    def test_parse_ep_id_from_bracket_prefix(self):
        from lens.adapters.hindsight import _parse_ep_id

        assert _parse_ep_id("[ep_001] 2024-01-01: text") == "ep_001"
        assert _parse_ep_id("[scope_01-ep_025] timestamp: data") == "scope_01-ep_025"

    def test_parse_ep_id_no_bracket_returns_prefix(self):
        from lens.adapters.hindsight import _parse_ep_id

        result = _parse_ep_id("no brackets here, just plain text")
        assert len(result) <= 32

    def test_parse_ep_id_empty_bracket(self):
        from lens.adapters.hindsight import _parse_ep_id

        # Edge case: empty bracket — falls through to prefix
        result = _parse_ep_id("[] rest of text")
        # The regex requires at least one char inside brackets
        assert len(result) <= 32

    def test_parse_ep_id_various_formats(self):
        from lens.adapters.hindsight import _parse_ep_id

        assert _parse_ep_id("[ep_030] last episode") == "ep_030"
        assert _parse_ep_id("[abc-123-xyz] data") == "abc-123-xyz"

"""Tests for Mem0 memory adapters (mem0-raw and mem0-extract).

Uses a MockMem0Client to avoid requiring Qdrant or real Mem0 infrastructure.
"""
from __future__ import annotations

import uuid
from unittest.mock import MagicMock, patch

import pytest

from lens.adapters.base import CapabilityManifest, Document, SearchResult
from lens.adapters.registry import get_adapter
from lens.core.errors import AdapterError


# ---------------------------------------------------------------------------
# Mock Mem0 client (in-memory dict + substring search)
# ---------------------------------------------------------------------------


class MockMem0Client:
    """In-memory mock of mem0.Memory that stores memories in a dict."""

    def __init__(self):
        # user_id -> list of {id, memory, metadata}
        self._store: dict[str, list[dict]] = {}

    def add(
        self,
        text: str,
        user_id: str = "",
        metadata: dict | None = None,
        infer: bool = True,
    ) -> dict:
        mem_id = str(uuid.uuid4())
        entry = {
            "id": mem_id,
            "memory": text,
            "metadata": dict(metadata or {}),
        }
        self._store.setdefault(user_id, []).append(entry)
        return {"results": [{"id": mem_id, "memory": text, "event": "ADD"}]}

    def search(self, query: str, limit: int = 10, user_id: str = "") -> dict:
        """Return results in mem0 v1 format: {"results": [...]}."""
        if not user_id:
            raise ValueError("At least one of 'user_id', 'agent_id', or 'run_id' must be provided")
        entries = self._store.get(user_id, [])
        # Simple substring match scoring
        results = []
        for entry in entries:
            text = entry["memory"].lower()
            q = query.lower()
            if q in text:
                score = 0.9
            elif any(word in text for word in q.split()):
                score = 0.5
            else:
                score = 0.1
            results.append({
                "id": entry["id"],
                "memory": entry["memory"],
                "score": score,
                "metadata": entry["metadata"],
            })
        results.sort(key=lambda r: r["score"], reverse=True)
        return {"results": results[:limit]}

    def get(self, mem_id: str) -> dict | None:
        for entries in self._store.values():
            for entry in entries:
                if entry["id"] == mem_id:
                    return entry
        return None

    def delete_all(self, user_id: str = "") -> None:
        self._store.pop(user_id, None)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_raw_adapter():
    """Create a Mem0RawAdapter with a mock client."""
    from lens.adapters.mem0 import Mem0RawAdapter

    mock_client = MockMem0Client()
    mock_memory_cls = MagicMock()
    mock_memory_cls.from_config.return_value = mock_client

    with patch("lens.adapters.mem0._check_mem0_available"):
        with patch.dict("sys.modules", {"mem0": MagicMock(Memory=mock_memory_cls)}):
            adapter = Mem0RawAdapter()
    return adapter


def _make_extract_adapter():
    """Create a Mem0ExtractAdapter with a mock client."""
    from lens.adapters.mem0 import Mem0ExtractAdapter

    mock_client = MockMem0Client()
    mock_memory_cls = MagicMock()
    mock_memory_cls.from_config.return_value = mock_client

    with patch("lens.adapters.mem0._check_mem0_available"):
        with patch.dict("sys.modules", {"mem0": MagicMock(Memory=mock_memory_cls)}):
            adapter = Mem0ExtractAdapter()
    return adapter


# ---------------------------------------------------------------------------
# Tests: Registration
# ---------------------------------------------------------------------------


class TestMem0Registration:
    def test_mem0_raw_registered(self):
        cls = get_adapter("mem0-raw")
        assert cls.__name__ == "Mem0RawAdapter"

    def test_mem0_extract_registered(self):
        cls = get_adapter("mem0-extract")
        assert cls.__name__ == "Mem0ExtractAdapter"


# ---------------------------------------------------------------------------
# Tests: Mem0RawAdapter
# ---------------------------------------------------------------------------


class TestMem0RawAdapter:
    def test_ingest_calls_add_with_infer_false(self):
        adapter = _make_raw_adapter()
        original_add = adapter._client.add
        calls = []

        def tracking_add(*args, **kwargs):
            calls.append(kwargs)
            return original_add(*args, **kwargs)

        adapter._client.add = tracking_add
        adapter.ingest("ep_001", "scope_1", "2024-01-01T00:00:00", "Server CPU at 85%")

        assert len(calls) == 1
        assert calls[0]["infer"] is False
        assert calls[0]["user_id"] == "scope_1"
        assert calls[0]["metadata"]["episode_id"] == "ep_001"

    def test_search_maps_results(self):
        adapter = _make_raw_adapter()
        adapter.ingest("ep_001", "s1", "2024-01-01", "CPU usage spiked to 95%")
        adapter.ingest("ep_002", "s1", "2024-01-02", "Memory allocation normal")

        results = adapter.search("CPU", filters={"scope_id": "s1"})
        assert len(results) > 0
        assert all(isinstance(r, SearchResult) for r in results)
        # The CPU episode should score higher
        cpu_results = [r for r in results if "CPU" in r.text]
        assert len(cpu_results) >= 1

    def test_search_uses_stored_scope_id(self):
        """Search works without explicit filters after reset/ingest sets scope."""
        adapter = _make_raw_adapter()
        adapter.reset("s1")
        adapter.ingest("ep_001", "s1", "2024-01-01", "CPU usage spiked to 95%")

        # Search without filters -- should use _current_scope_id
        results = adapter.search("CPU")
        assert len(results) > 0
        cpu_results = [r for r in results if "CPU" in r.text]
        assert len(cpu_results) >= 1

    def test_search_without_scope_returns_empty(self):
        """Search with no scope_id set and no filters returns empty."""
        adapter = _make_raw_adapter()
        # Don't call reset() or ingest(), so no scope_id is set
        results = adapter.search("CPU")
        assert results == []

    def test_search_empty_query_returns_empty(self):
        adapter = _make_raw_adapter()
        adapter.ingest("ep_001", "s1", "2024-01-01", "Some text")
        assert adapter.search("") == []
        assert adapter.search("   ") == []

    def test_search_respects_limit(self):
        adapter = _make_raw_adapter()
        for i in range(20):
            adapter.ingest(f"ep_{i:03d}", "s1", "2024-01-01", f"Episode {i} data")

        results = adapter.search("Episode", filters={"scope_id": "s1"}, limit=5)
        assert len(results) <= 5

    def test_retrieve_uses_episode_index(self):
        adapter = _make_raw_adapter()
        adapter.ingest("ep_001", "s1", "2024-01-01", "Server log entry")

        doc = adapter.retrieve("ep_001")
        assert doc is not None
        assert isinstance(doc, Document)
        assert doc.ref_id == "ep_001"
        assert "Server log entry" in doc.text

    def test_retrieve_missing_returns_none(self):
        adapter = _make_raw_adapter()
        assert adapter.retrieve("nonexistent") is None

    def test_reset_clears_memories(self):
        adapter = _make_raw_adapter()
        adapter.ingest("ep_001", "s1", "2024-01-01", "Data for scope 1")

        adapter.reset("s1")

        # Search should return empty after reset
        results = adapter.search("Data", filters={"scope_id": "s1"})
        assert len(results) == 0

    def test_reset_only_clears_target_scope(self):
        adapter = _make_raw_adapter()
        adapter.ingest("ep_001", "s1", "2024-01-01", "Scope 1 data")
        adapter.ingest("ep_002", "s2", "2024-01-01", "Scope 2 data")

        adapter.reset("s1")

        # Scope 2 should still have data
        results = adapter.search("Scope 2", filters={"scope_id": "s2"})
        assert len(results) > 0

    def test_capabilities(self):
        adapter = _make_raw_adapter()
        caps = adapter.get_capabilities()
        assert isinstance(caps, CapabilityManifest)
        assert "semantic" in caps.search_modes
        assert caps.max_results_per_search == 10

    def test_requires_metering_false(self):
        adapter = _make_raw_adapter()
        assert adapter.requires_metering is False


# ---------------------------------------------------------------------------
# Tests: Mem0ExtractAdapter
# ---------------------------------------------------------------------------


class TestMem0ExtractAdapter:
    def test_ingest_buffers_locally(self):
        adapter = _make_extract_adapter()

        # Should NOT call client.add during ingest
        original_add = adapter._client.add
        add_called = []
        adapter._client.add = lambda *a, **kw: add_called.append(1) or original_add(*a, **kw)

        adapter.ingest("ep_001", "s1", "2024-01-01", "Some data")

        assert len(add_called) == 0
        assert "s1" in adapter._staged
        assert len(adapter._staged["s1"]) == 1

    def test_prepare_processes_staged(self):
        adapter = _make_extract_adapter()

        calls = []
        original_add = adapter._client.add

        def tracking_add(*args, **kwargs):
            calls.append(kwargs)
            return original_add(*args, **kwargs)

        adapter._client.add = tracking_add

        adapter.ingest("ep_001", "s1", "2024-01-01", "First episode")
        adapter.ingest("ep_002", "s1", "2024-01-02", "Second episode")
        adapter.prepare("s1", checkpoint=5)

        assert len(calls) == 2
        assert calls[0]["infer"] is True
        assert calls[1]["infer"] is True
        # Staged should be cleared
        assert "s1" not in adapter._staged

    def test_prepare_empty_scope_noop(self):
        adapter = _make_extract_adapter()
        adapter.prepare("nonexistent", checkpoint=1)  # Should not raise

    def test_search_after_prepare(self):
        adapter = _make_extract_adapter()
        adapter.ingest("ep_001", "s1", "2024-01-01", "CPU utilization at 95%")
        adapter.prepare("s1", checkpoint=5)

        results = adapter.search("CPU", filters={"scope_id": "s1"})
        assert len(results) > 0

    def test_requires_metering_true(self):
        adapter = _make_extract_adapter()
        assert adapter.requires_metering is True


# ---------------------------------------------------------------------------
# Tests: Error handling
# ---------------------------------------------------------------------------


class TestMem0ErrorHandling:
    def test_missing_mem0_raises_adapter_error(self):
        """Verify _check_mem0_available raises AdapterError when mem0ai missing."""
        from lens.adapters.mem0 import _check_mem0_available

        with patch.dict("sys.modules", {"mem0": None}):
            with pytest.raises(AdapterError, match="mem0ai package not installed"):
                _check_mem0_available()

    def test_qdrant_connection_failure_raises_adapter_error(self):
        """Verify __init__ raises AdapterError when Qdrant is unreachable."""
        from lens.adapters.mem0 import Mem0RawAdapter

        mock_memory_cls = MagicMock()
        mock_memory_cls.from_config.side_effect = ConnectionError("Connection refused")

        with patch("lens.adapters.mem0._check_mem0_available"):
            with patch.dict("sys.modules", {"mem0": MagicMock(Memory=mock_memory_cls)}):
                with pytest.raises(AdapterError, match="Failed to initialize Mem0 client"):
                    Mem0RawAdapter()

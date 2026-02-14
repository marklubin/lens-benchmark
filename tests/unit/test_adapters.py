from __future__ import annotations

import pytest

from lens.adapters.base import CapabilityManifest, SearchResult
from lens.adapters.null import NullAdapter
from lens.adapters.registry import get_adapter, list_adapters


class TestNullAdapter:
    def test_reset(self):
        adapter = NullAdapter()
        adapter.reset("persona_1")  # Should not raise

    def test_ingest(self):
        adapter = NullAdapter()
        adapter.ingest("ep_001", "p1", "2024-01-01T00:00:00", "text")

    def test_search_returns_empty(self):
        adapter = NullAdapter()
        result = adapter.search("query")
        assert result == []

    def test_search_with_filters(self):
        adapter = NullAdapter()
        result = adapter.search("query", filters={"type": "episode"}, limit=5)
        assert result == []

    def test_retrieve_returns_none(self):
        adapter = NullAdapter()
        result = adapter.retrieve("ref_001")
        assert result is None

    def test_get_capabilities(self):
        adapter = NullAdapter()
        caps = adapter.get_capabilities()
        assert isinstance(caps, CapabilityManifest)
        assert "semantic" in caps.search_modes
        assert caps.max_results_per_search == 10

    def test_prepare_noop(self):
        adapter = NullAdapter()
        adapter.prepare("p1", 10)  # Should not raise


class TestAdapterRegistry:
    def test_get_null_adapter(self):
        cls = get_adapter("null")
        assert cls is NullAdapter

    def test_list_includes_null(self):
        adapters = list_adapters()
        assert "null" in adapters

    def test_unknown_adapter_raises(self):
        with pytest.raises(Exception, match="Unknown adapter"):
            get_adapter("nonexistent_adapter_xyz")

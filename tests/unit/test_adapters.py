from __future__ import annotations

import pytest

from lens.adapters.null import NullAdapter
from lens.adapters.registry import get_adapter, list_adapters


class TestNullAdapter:
    def test_reset(self):
        adapter = NullAdapter()
        adapter.reset("persona_1")  # Should not raise

    def test_ingest(self):
        adapter = NullAdapter()
        adapter.ingest("ep_001", "p1", "2024-01-01T00:00:00", "text")

    def test_refresh(self):
        adapter = NullAdapter()
        adapter.refresh("p1", 10)

    def test_core_returns_empty(self):
        adapter = NullAdapter()
        result = adapter.core("p1", 10, 10)
        assert result == []

    def test_search_returns_empty(self):
        adapter = NullAdapter()
        result = adapter.search("p1", "query", 10, 10)
        assert result == []


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

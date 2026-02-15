from __future__ import annotations

import pytest

from lens.adapters.base import CapabilityManifest, Document, SearchResult
from lens.adapters.registry import get_adapter, list_adapters
from lens.adapters.sqlite import SQLiteAdapter


class TestSQLiteAdapter:
    def test_registered(self):
        cls = get_adapter("sqlite")
        assert cls is SQLiteAdapter

    def test_listed(self):
        adapters = list_adapters()
        assert "sqlite" in adapters

    def test_reset_and_ingest(self):
        adapter = SQLiteAdapter()
        adapter.reset("s1")
        adapter.ingest("ep1", "s1", "2024-01-01T00:00:00", "hello world")
        result = adapter.retrieve("ep1")
        assert result is not None
        assert result.text == "hello world"

    def test_reset_clears_scope(self):
        adapter = SQLiteAdapter()
        adapter.ingest("ep1", "s1", "2024-01-01T00:00:00", "scope one text")
        adapter.ingest("ep2", "s2", "2024-01-01T00:00:00", "scope two text")
        adapter.reset("s1")
        assert adapter.retrieve("ep1") is None
        assert adapter.retrieve("ep2") is not None

    def test_search_returns_results(self):
        adapter = SQLiteAdapter()
        adapter.ingest("ep1", "s1", "2024-01-01T00:00:00", "the cat sat on the mat")
        adapter.ingest("ep2", "s1", "2024-01-02T00:00:00", "the dog played in the park")
        results = adapter.search("cat mat")
        assert len(results) >= 1
        assert results[0].ref_id == "ep1"
        assert isinstance(results[0], SearchResult)
        assert results[0].score > 0

    def test_search_empty_query(self):
        adapter = SQLiteAdapter()
        adapter.ingest("ep1", "s1", "2024-01-01T00:00:00", "some text")
        results = adapter.search("")
        assert results == []

    def test_search_with_limit(self):
        adapter = SQLiteAdapter()
        for i in range(20):
            adapter.ingest(f"ep{i}", "s1", f"2024-01-{i+1:02d}T00:00:00", f"document number {i} about testing")
        results = adapter.search("testing", limit=5)
        assert len(results) <= 5

    def test_search_with_scope_filter(self):
        adapter = SQLiteAdapter()
        adapter.ingest("ep1", "s1", "2024-01-01T00:00:00", "alpha beta gamma")
        adapter.ingest("ep2", "s2", "2024-01-01T00:00:00", "alpha beta delta")
        results = adapter.search("alpha", filters={"scope_id": "s1"})
        assert len(results) == 1
        assert results[0].ref_id == "ep1"

    def test_search_with_date_filter(self):
        adapter = SQLiteAdapter()
        adapter.ingest("ep1", "s1", "2024-01-01T00:00:00", "keyword early")
        adapter.ingest("ep2", "s1", "2024-06-15T00:00:00", "keyword late")
        results = adapter.search("keyword", filters={"start_date": "2024-06-01"})
        assert len(results) == 1
        assert results[0].ref_id == "ep2"

    def test_retrieve_existing(self):
        adapter = SQLiteAdapter()
        adapter.ingest("ep1", "s1", "2024-01-01T00:00:00", "document text", {"key": "val"})
        doc = adapter.retrieve("ep1")
        assert isinstance(doc, Document)
        assert doc.ref_id == "ep1"
        assert doc.text == "document text"
        assert doc.metadata == {"key": "val"}

    def test_retrieve_missing(self):
        adapter = SQLiteAdapter()
        assert adapter.retrieve("nonexistent") is None

    def test_get_capabilities(self):
        adapter = SQLiteAdapter()
        caps = adapter.get_capabilities()
        assert isinstance(caps, CapabilityManifest)
        assert "keyword" in caps.search_modes
        assert caps.supports_date_range is True
        filter_names = [f.name for f in caps.filter_fields]
        assert "scope_id" in filter_names

    def test_ingest_replace(self):
        adapter = SQLiteAdapter()
        adapter.ingest("ep1", "s1", "2024-01-01T00:00:00", "original text")
        adapter.ingest("ep1", "s1", "2024-01-01T00:00:00", "updated text")
        doc = adapter.retrieve("ep1")
        assert doc is not None
        assert doc.text == "updated text"

    def test_search_no_match(self):
        adapter = SQLiteAdapter()
        adapter.ingest("ep1", "s1", "2024-01-01T00:00:00", "the quick brown fox")
        results = adapter.search("zyxwvut")
        assert results == []

    def test_search_special_characters(self):
        adapter = SQLiteAdapter()
        adapter.ingest("ep1", "s1", "2024-01-01T00:00:00", "test with special chars: AND OR NOT")
        # Should not raise — FTS5 special chars are escaped
        results = adapter.search("AND OR NOT")
        assert isinstance(results, list)

    def test_prepare_noop(self):
        adapter = SQLiteAdapter()
        adapter.prepare("s1", 10)  # Should not raise

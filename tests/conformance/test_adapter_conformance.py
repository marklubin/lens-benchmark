"""Adapter conformance tests — contract tests for the MemoryAdapter ABC.

Every adapter registered in the LENS benchmark must pass these tests.
Tests are grouped into four categories:
  1. Type contracts — return types match the ABC specification
  2. Semantic contracts — behavioral guarantees hold
  3. Edge cases — unusual inputs don't crash
  4. Capability checks — capability manifest is well-formed
"""
from __future__ import annotations

import pytest

from lens.adapters.base import (
    CapabilityManifest,
    Document,
    MemoryAdapter,
    SearchResult,
)
from lens.adapters.null import NullAdapter

SCOPE = "conformance_test"


def _is_null(adapter: MemoryAdapter) -> bool:
    return isinstance(adapter, NullAdapter)


def _ingest_sample(adapter: MemoryAdapter, episode_id: str = "ep-001", text: str = "server latency p99 hit 450ms at 14:32 UTC") -> None:
    """Helper to reset scope and ingest a single episode."""
    adapter.reset(SCOPE)
    adapter.ingest(
        episode_id=episode_id,
        scope_id=SCOPE,
        timestamp="2025-01-15T14:32:00Z",
        text=text,
        meta={"source": "test"},
    )


# =========================================================================
# 1. Type contracts
# =========================================================================


class TestTypeContracts:
    """Return types match the ABC specification."""

    def test_search_returns_list_of_search_result(self, adapter_instance):
        results = adapter_instance.search("anything")
        assert isinstance(results, list)
        for r in results:
            assert isinstance(r, SearchResult)

    def test_retrieve_returns_document_or_none(self, adapter_instance):
        result = adapter_instance.retrieve("nonexistent-id")
        assert result is None or isinstance(result, Document)

    def test_get_capabilities_returns_manifest(self, adapter_instance):
        caps = adapter_instance.get_capabilities()
        assert isinstance(caps, CapabilityManifest)

    def test_search_result_has_required_fields(self, adapter_instance):
        if _is_null(adapter_instance):
            pytest.skip("NullAdapter returns no results to inspect")
        _ingest_sample(adapter_instance)
        results = adapter_instance.search("latency")
        assert len(results) > 0, "Expected at least one search result"
        r = results[0]
        assert hasattr(r, "ref_id")
        assert hasattr(r, "text")
        assert hasattr(r, "score")
        assert hasattr(r, "metadata")
        assert isinstance(r.ref_id, str)
        assert isinstance(r.text, str)
        assert isinstance(r.score, (int, float))
        assert isinstance(r.metadata, dict)

    def test_document_has_required_fields(self, adapter_instance):
        if _is_null(adapter_instance):
            pytest.skip("NullAdapter has no storage")
        _ingest_sample(adapter_instance)
        doc = adapter_instance.retrieve("ep-001")
        assert doc is not None
        assert hasattr(doc, "ref_id")
        assert hasattr(doc, "text")
        assert hasattr(doc, "metadata")
        assert isinstance(doc.ref_id, str)
        assert isinstance(doc.text, str)
        assert isinstance(doc.metadata, dict)

    def test_ingest_accepts_none_meta(self, adapter_instance):
        adapter_instance.reset(SCOPE)
        # Should not raise
        adapter_instance.ingest(
            episode_id="ep-none-meta",
            scope_id=SCOPE,
            timestamp="2025-01-15T00:00:00Z",
            text="test with no metadata",
            meta=None,
        )


# =========================================================================
# 2. Semantic contracts
# =========================================================================


class TestSemanticContracts:
    """Behavioral guarantees for adapters with storage."""

    def test_ingest_then_retrieve(self, adapter_instance):
        if _is_null(adapter_instance):
            pytest.skip("NullAdapter has no storage")
        _ingest_sample(adapter_instance)
        doc = adapter_instance.retrieve("ep-001")
        assert doc is not None
        assert doc.ref_id == "ep-001"
        assert "latency" in doc.text or "450ms" in doc.text

    def test_retrieve_missing_returns_none(self, adapter_instance):
        result = adapter_instance.retrieve("definitely-does-not-exist-xyz")
        assert result is None

    def test_ingest_then_search_finds_it(self, adapter_instance):
        if _is_null(adapter_instance):
            pytest.skip("NullAdapter has no storage")
        _ingest_sample(adapter_instance)
        results = adapter_instance.search("latency")
        assert len(results) >= 1
        ref_ids = [r.ref_id for r in results]
        assert "ep-001" in ref_ids

    def test_search_limit_respected(self, adapter_instance):
        if _is_null(adapter_instance):
            pytest.skip("NullAdapter has no storage")
        adapter_instance.reset(SCOPE)
        for i in range(5):
            adapter_instance.ingest(
                episode_id=f"ep-limit-{i}",
                scope_id=SCOPE,
                timestamp=f"2025-01-15T{10+i}:00:00Z",
                text=f"memory usage report number {i} with heap allocation data",
            )
        results = adapter_instance.search("memory usage report", limit=2)
        assert len(results) <= 2

    def test_reset_clears_scope(self, adapter_instance):
        if _is_null(adapter_instance):
            pytest.skip("NullAdapter has no storage")
        _ingest_sample(adapter_instance)
        doc = adapter_instance.retrieve("ep-001")
        assert doc is not None
        adapter_instance.reset(SCOPE)
        doc = adapter_instance.retrieve("ep-001")
        assert doc is None

    def test_reset_preserves_other_scope(self, adapter_instance):
        if _is_null(adapter_instance):
            pytest.skip("NullAdapter has no storage")
        other_scope = "conformance_other"
        adapter_instance.reset(SCOPE)
        adapter_instance.reset(other_scope)
        adapter_instance.ingest(
            episode_id="ep-scope1",
            scope_id=SCOPE,
            timestamp="2025-01-15T10:00:00Z",
            text="scope one data",
        )
        adapter_instance.ingest(
            episode_id="ep-scope2",
            scope_id=other_scope,
            timestamp="2025-01-15T11:00:00Z",
            text="scope two data",
        )
        adapter_instance.reset(SCOPE)
        assert adapter_instance.retrieve("ep-scope1") is None
        assert adapter_instance.retrieve("ep-scope2") is not None

    def test_ingest_idempotent(self, adapter_instance):
        if _is_null(adapter_instance):
            pytest.skip("NullAdapter has no storage")
        _ingest_sample(adapter_instance)
        # Ingest the same episode again — should not crash
        adapter_instance.ingest(
            episode_id="ep-001",
            scope_id=SCOPE,
            timestamp="2025-01-15T14:32:00Z",
            text="server latency p99 hit 450ms at 14:32 UTC",
            meta={"source": "test"},
        )
        doc = adapter_instance.retrieve("ep-001")
        assert doc is not None

    def test_prepare_does_not_crash(self, adapter_instance):
        adapter_instance.reset(SCOPE)
        # prepare() is optional; should complete without error
        adapter_instance.prepare(SCOPE, checkpoint=5)


# =========================================================================
# 3. Edge cases
# =========================================================================


class TestEdgeCases:
    """Unusual inputs should not crash the adapter."""

    def test_search_empty_query(self, adapter_instance):
        results = adapter_instance.search("")
        assert isinstance(results, list)

    def test_ingest_empty_text(self, adapter_instance):
        adapter_instance.reset(SCOPE)
        # Should not crash
        adapter_instance.ingest(
            episode_id="ep-empty",
            scope_id=SCOPE,
            timestamp="2025-01-15T00:00:00Z",
            text="",
        )

    def test_ingest_large_text(self, adapter_instance):
        if _is_null(adapter_instance):
            pytest.skip("NullAdapter has no storage to verify")
        adapter_instance.reset(SCOPE)
        large_text = "word " * 2000  # ~10KB
        adapter_instance.ingest(
            episode_id="ep-large",
            scope_id=SCOPE,
            timestamp="2025-01-15T00:00:00Z",
            text=large_text,
        )
        doc = adapter_instance.retrieve("ep-large")
        assert doc is not None
        assert len(doc.text) > 0

    def test_search_special_characters(self, adapter_instance):
        results = adapter_instance.search("@#$%")
        assert isinstance(results, list)

    def test_unicode_roundtrip(self, adapter_instance):
        if _is_null(adapter_instance):
            pytest.skip("NullAdapter has no storage")
        adapter_instance.reset(SCOPE)
        unicode_text = "CPU \u6e29\u5ea6 85\u00b0C \u2014 \u30ec\u30a4\u30c6\u30f3\u30b7 p99: 600ms \u2192 \u0430\u043b\u0435\u0440\u0442"
        adapter_instance.ingest(
            episode_id="ep-unicode",
            scope_id=SCOPE,
            timestamp="2025-01-15T00:00:00Z",
            text=unicode_text,
        )
        doc = adapter_instance.retrieve("ep-unicode")
        assert doc is not None
        assert doc.text == unicode_text

    def test_metadata_dict_roundtrip(self, adapter_instance):
        if _is_null(adapter_instance):
            pytest.skip("NullAdapter has no storage")
        adapter_instance.reset(SCOPE)
        meta = {"k": "v", "number": 42, "nested": {"a": 1}}
        adapter_instance.ingest(
            episode_id="ep-meta",
            scope_id=SCOPE,
            timestamp="2025-01-15T00:00:00Z",
            text="metadata test episode",
            meta=meta,
        )
        doc = adapter_instance.retrieve("ep-meta")
        assert doc is not None
        assert doc.metadata.get("k") == "v"
        assert doc.metadata.get("number") == 42
        assert doc.metadata.get("nested") == {"a": 1}


# =========================================================================
# 4. Capability checks
# =========================================================================


class TestCapabilityChecks:
    """Capability manifest is well-formed and consistent."""

    def test_search_modes_non_empty(self, adapter_instance):
        caps = adapter_instance.get_capabilities()
        assert isinstance(caps.search_modes, list)
        assert len(caps.search_modes) > 0

    def test_scores_non_negative(self, adapter_instance):
        if _is_null(adapter_instance):
            pytest.skip("NullAdapter returns no results")
        _ingest_sample(adapter_instance)
        results = adapter_instance.search("latency")
        for r in results:
            assert r.score >= 0, f"Negative score {r.score} for {r.ref_id}"

    def test_unknown_extended_tool_raises(self, adapter_instance):
        with pytest.raises(NotImplementedError):
            adapter_instance.call_extended_tool("fake_tool_name", {})

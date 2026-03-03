"""Tests for GraphRAG-light adapter.

Tests entity extraction parsing, RRF fusion, graph construction,
deduplication, neighborhood expansion, and full search round-trips.
LLM and embedding calls are mocked.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from lens.adapters.base import CapabilityManifest, SearchResult
from lens.adapters.graphrag_light import (
    GraphRAGLightAdapter,
    _cosine_similarity,
    _parse_extraction,
    _rrf_merge,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

MOCK_EXTRACTION = {
    "entities": [
        {"name": "Ming Chen", "type": "PERSON", "description": "Graduate student"},
        {"name": "AI Tutor", "type": "SYSTEM", "description": "Automated tutoring system"},
        {"name": "CS Department", "type": "ORGANIZATION", "description": "Computer science department"},
    ],
    "relationships": [
        {"source": "Ming Chen", "target": "AI Tutor", "type": "INTERACTS_WITH", "description": "Uses for homework"},
        {"source": "Ming Chen", "target": "CS Department", "type": "PART_OF", "description": "Enrolled student"},
    ],
}

MOCK_EMBEDDING = [0.1, 0.2, 0.3, 0.4, 0.5]  # Short for testing


@pytest.fixture
def adapter():
    """Create adapter with in-memory SQLite."""
    with patch("lens.adapters.graphrag_light._embed_texts_openai") as mock_embed:
        mock_embed.return_value = [MOCK_EMBEDDING]
        a = GraphRAGLightAdapter(db_path=":memory:")
    return a


@pytest.fixture
def populated_adapter():
    """Create adapter with episodes ingested and graph built."""
    a = GraphRAGLightAdapter(db_path=":memory:")

    # Ingest episodes
    a.ingest("ep_001", "s07", "2024-01-01", "Ming Chen submitted homework via AI Tutor system")
    a.ingest("ep_002", "s07", "2024-01-02", "Ming Chen used AI Tutor to generate essay content")
    a.ingest("ep_003", "s07", "2024-01-03", "Regular class schedule and assignments at CS Department")

    # Mock LLM extraction and embedding for prepare()
    extraction_results = [
        {
            "entities": [
                {"name": "Ming Chen", "type": "PERSON", "description": "Student"},
                {"name": "AI Tutor", "type": "SYSTEM", "description": "Tutoring system"},
            ],
            "relationships": [
                {"source": "Ming Chen", "target": "AI Tutor", "type": "INTERACTS_WITH", "description": "homework"},
            ],
        },
        {
            "entities": [
                {"name": "Ming Chen", "type": "PERSON", "description": "Graduate student in CS"},
                {"name": "AI Tutor", "type": "SYSTEM", "description": "Automated tutoring system"},
            ],
            "relationships": [
                {"source": "Ming Chen", "target": "AI Tutor", "type": "INTERACTS_WITH", "description": "essay generation"},
            ],
        },
        {
            "entities": [
                {"name": "CS Department", "type": "ORGANIZATION", "description": "Computer science department"},
            ],
            "relationships": [],
        },
    ]

    call_count = {"n": 0}

    def mock_extract(text, **kwargs):
        idx = call_count["n"]
        call_count["n"] += 1
        if idx < len(extraction_results):
            return extraction_results[idx]
        return {"entities": [], "relationships": []}

    with (
        patch("lens.adapters.graphrag_light._llm_extract_entities", side_effect=mock_extract),
        patch("lens.adapters.graphrag_light._embed_texts_openai", return_value=[MOCK_EMBEDDING]),
    ):
        a.prepare("s07", 14)

    return a


# ---------------------------------------------------------------------------
# Unit tests: _parse_extraction
# ---------------------------------------------------------------------------


class TestParseExtraction:
    def test_valid_json(self):
        result = _parse_extraction('{"entities": [{"name": "A", "type": "PERSON"}], "relationships": []}')
        assert len(result["entities"]) == 1
        assert result["entities"][0]["name"] == "A"

    def test_markdown_code_block(self):
        content = '```json\n{"entities": [{"name": "B", "type": "SYSTEM"}], "relationships": []}\n```'
        result = _parse_extraction(content)
        assert len(result["entities"]) == 1
        assert result["entities"][0]["name"] == "B"

    def test_json_embedded_in_text(self):
        content = 'Here are the entities:\n{"entities": [{"name": "C", "type": "CONCEPT"}], "relationships": []}\nDone.'
        result = _parse_extraction(content)
        assert len(result["entities"]) == 1

    def test_malformed_json(self):
        result = _parse_extraction("this is not json at all")
        assert result == {"entities": [], "relationships": []}

    def test_missing_name_skipped(self):
        result = _parse_extraction('{"entities": [{"type": "PERSON"}], "relationships": []}')
        assert len(result["entities"]) == 0

    def test_relationship_validation(self):
        content = '{"entities": [], "relationships": [{"source": "A", "target": "B", "type": "CAUSES"}]}'
        result = _parse_extraction(content)
        assert len(result["relationships"]) == 1
        assert result["relationships"][0]["source"] == "A"

    def test_relationship_missing_source(self):
        content = '{"entities": [], "relationships": [{"target": "B"}]}'
        result = _parse_extraction(content)
        assert len(result["relationships"]) == 0

    def test_empty_extraction(self):
        result = _parse_extraction('{"entities": [], "relationships": []}')
        assert result == {"entities": [], "relationships": []}


# ---------------------------------------------------------------------------
# Unit tests: cosine similarity
# ---------------------------------------------------------------------------


class TestCosineSimilarity:
    def test_identical_vectors(self):
        v = [1.0, 2.0, 3.0]
        assert abs(_cosine_similarity(v, v) - 1.0) < 1e-6

    def test_orthogonal_vectors(self):
        assert abs(_cosine_similarity([1, 0, 0], [0, 1, 0])) < 1e-6

    def test_zero_vector(self):
        assert _cosine_similarity([0, 0], [1, 2]) == 0.0


# ---------------------------------------------------------------------------
# Unit tests: RRF merge
# ---------------------------------------------------------------------------


class TestRRFMerge:
    def test_basic_merge(self):
        a = [SearchResult(ref_id="ep_001", text="a", score=1.0)]
        b = [SearchResult(ref_id="ep_002", text="b", score=1.0)]
        merged = _rrf_merge(a, b, k=60, limit=10)
        assert len(merged) == 2
        # Both should have RRF scores
        for r in merged:
            assert r.score > 0

    def test_shared_document_boosted(self):
        """Document appearing in both lists should rank higher."""
        shared = SearchResult(ref_id="ep_001", text="shared", score=1.0)
        only_a = SearchResult(ref_id="ep_002", text="only_a", score=1.0)
        only_b = SearchResult(ref_id="ep_003", text="only_b", score=1.0)

        a = [shared, only_a]
        b = [shared, only_b]
        merged = _rrf_merge(a, b, k=60, limit=10)

        # ep_001 (shared) should be first
        assert merged[0].ref_id == "ep_001"

    def test_limit_respected(self):
        a = [SearchResult(ref_id=f"ep_{i:03d}", text="t", score=1.0) for i in range(20)]
        merged = _rrf_merge(a, [], k=60, limit=5)
        assert len(merged) == 5

    def test_empty_lists(self):
        assert _rrf_merge([], [], k=60, limit=10) == []


# ---------------------------------------------------------------------------
# Integration tests: adapter lifecycle
# ---------------------------------------------------------------------------


class TestGraphRAGLightAdapter:
    def test_registry(self):
        from lens.adapters.registry import get_adapter
        cls = get_adapter("graphrag-light")
        assert cls is GraphRAGLightAdapter

    def test_reset(self, adapter):
        adapter.ingest("ep_001", "s07", "2024-01-01", "test text")
        adapter.reset("s07")
        assert adapter.retrieve("ep_001") is None

    def test_ingest_and_retrieve(self, adapter):
        adapter.ingest("ep_001", "s07", "2024-01-01", "Some episode text")
        doc = adapter.retrieve("ep_001")
        assert doc is not None
        assert doc.ref_id == "ep_001"
        assert "Some episode text" in doc.text

    def test_retrieve_nonexistent(self, adapter):
        assert adapter.retrieve("nonexistent") is None

    def test_ingest_appends_pending(self, adapter):
        adapter.ingest("ep_001", "s07", "2024-01-01", "text1")
        adapter.ingest("ep_002", "s07", "2024-01-02", "text2")
        assert len(adapter._pending_episodes) == 2

    def test_capabilities(self, adapter):
        caps = adapter.get_capabilities()
        assert isinstance(caps, CapabilityManifest)
        assert "keyword" in caps.search_modes
        assert "semantic" in caps.search_modes
        assert "graph" in caps.search_modes
        # Has batch_retrieve tool
        assert any(t.name == "batch_retrieve" for t in caps.extra_tools)

    def test_fts_search_before_prepare(self, adapter):
        adapter.ingest("ep_001", "s07", "2024-01-01", "Ming Chen used the AI Tutor")
        results = adapter._fts_search("Ming Chen", None, 10)
        assert len(results) >= 1
        assert results[0].ref_id == "ep_001"

    def test_fts_search_empty_query(self, adapter):
        assert adapter._fts_search("", None, 10) == []


class TestGraphRAGPrepare:
    def test_prepare_builds_graph(self, populated_adapter):
        a = populated_adapter
        # Should have entities in graph
        assert a._graph.number_of_nodes() > 0
        # Ming Chen should exist (normalized)
        assert a._graph.has_node("ming chen")
        assert a._graph.has_node("ai tutor")

    def test_prepare_deduplicates_entities(self, populated_adapter):
        a = populated_adapter
        # Ming Chen appears in ep_001 and ep_002 — should be one node with 2 source episodes
        node = a._graph.nodes["ming chen"]
        assert len(node["source_episodes"]) == 2

    def test_prepare_builds_edges(self, populated_adapter):
        a = populated_adapter
        assert a._graph.has_edge("ming chen", "ai tutor")
        edge = a._graph.edges["ming chen", "ai tutor"]
        assert edge["weight"] == 2  # appears in 2 episodes

    def test_prepare_description_merge(self, populated_adapter):
        a = populated_adapter
        # Should keep longer description
        node = a._graph.nodes["ming chen"]
        assert "Graduate" in node["description"] or "CS" in node["description"]

    def test_prepare_clears_pending(self, populated_adapter):
        assert len(populated_adapter._pending_episodes) == 0

    def test_prepare_no_pending_is_noop(self, populated_adapter):
        a = populated_adapter
        node_count = a._graph.number_of_nodes()
        with patch("lens.adapters.graphrag_light._llm_extract_entities") as mock:
            a.prepare("s07", 22)
            mock.assert_not_called()
        assert a._graph.number_of_nodes() == node_count


class TestGraphNeighborhood:
    def test_direct_entity_episodes(self, populated_adapter):
        a = populated_adapter
        # Ming Chen is in ep_001 and ep_002
        scores = a._graph_neighborhood_episodes([("ming chen", 0.9)])
        assert "ep_001" in scores
        assert "ep_002" in scores

    def test_1hop_expansion(self, populated_adapter):
        a = populated_adapter
        # CS Department is only in ep_003, connected to nothing in mock
        # But Ming Chen -> AI Tutor edge exists
        # Querying for "ai tutor" should also get Ming Chen's episodes via 1-hop
        scores = a._graph_neighborhood_episodes([("ai tutor", 0.8)])
        assert "ep_001" in scores or "ep_002" in scores

    def test_empty_matches(self, populated_adapter):
        scores = populated_adapter._graph_neighborhood_episodes([])
        assert scores == {}

    def test_nonexistent_entity(self, populated_adapter):
        scores = populated_adapter._graph_neighborhood_episodes([("nonexistent", 0.5)])
        assert scores == {}


class TestBatchRetrieve:
    def test_batch_retrieve(self, adapter):
        adapter.ingest("ep_001", "s07", "2024-01-01", "Text one")
        adapter.ingest("ep_002", "s07", "2024-01-02", "Text two")
        results = adapter.call_extended_tool("batch_retrieve", {"ref_ids": ["ep_001", "ep_002"]})
        assert len(results) == 2

    def test_batch_retrieve_missing(self, adapter):
        results = adapter.call_extended_tool("batch_retrieve", {"ref_ids": ["nonexistent"]})
        assert len(results) == 0

    def test_unknown_tool_raises(self, adapter):
        with pytest.raises(NotImplementedError):
            adapter.call_extended_tool("nonexistent_tool", {})


class TestFullSearch:
    def test_search_with_graph(self, populated_adapter):
        """Full search should work after prepare() builds graph."""
        with patch("lens.adapters.graphrag_light._embed_texts_openai", return_value=[MOCK_EMBEDDING]):
            results = populated_adapter.search("Ming Chen homework", limit=5)
        # Should return results (from FTS at minimum)
        assert len(results) > 0

    def test_search_empty_query(self, populated_adapter):
        assert populated_adapter.search("") == []
        assert populated_adapter.search("   ") == []

    def test_search_before_prepare(self, adapter):
        """Search should still work (FTS only) before prepare()."""
        adapter.ingest("ep_001", "s07", "2024-01-01", "homework assignment grades")
        with patch("lens.adapters.graphrag_light._embed_texts_openai", return_value=[MOCK_EMBEDDING]):
            results = adapter.search("homework", limit=5)
        assert len(results) >= 1

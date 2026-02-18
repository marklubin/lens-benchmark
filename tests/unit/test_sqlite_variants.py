from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from lens.adapters.base import CapabilityManifest, Document, SearchResult
from lens.adapters.registry import get_adapter, list_adapters
from lens.adapters.sqlite_variants import (
    SQLiteEmbeddingAdapter,
    SQLiteFTSAdapter,
    SQLiteHybridAdapter,
    _rrf_merge,
    cosine_similarity,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_embed(texts, model=None, ollama_url=None):
    """Return deterministic fake embeddings based on text length."""
    vecs = []
    for t in texts:
        # Create a simple 4-dim vector from text characteristics
        v = [len(t) / 100.0, sum(ord(c) for c in t[:10]) / 1000.0, 0.5, 0.3]
        norm = sum(x * x for x in v) ** 0.5
        vecs.append([x / norm for x in v])
    return vecs


EMBED_PATCH = "lens.adapters.sqlite_variants._embed_texts"


# ---------------------------------------------------------------------------
# SQLiteFTSAdapter tests
# ---------------------------------------------------------------------------

class TestSQLiteFTSAdapter:
    def test_registered(self):
        cls = get_adapter("sqlite-fts")
        assert cls is SQLiteFTSAdapter

    def test_listed(self):
        adapters = list_adapters()
        assert "sqlite-fts" in adapters

    def test_ingest_and_search(self):
        adapter = SQLiteFTSAdapter()
        adapter.ingest("ep1", "s1", "2024-01-01T00:00:00", "the cat sat on the mat")
        adapter.ingest("ep2", "s1", "2024-01-02T00:00:00", "the dog played in the park")
        results = adapter.search("cat mat")
        assert len(results) >= 1
        assert results[0].ref_id == "ep1"

    def test_get_capabilities(self):
        adapter = SQLiteFTSAdapter()
        caps = adapter.get_capabilities()
        assert isinstance(caps, CapabilityManifest)
        assert "keyword" in caps.search_modes

    def test_reset(self):
        adapter = SQLiteFTSAdapter()
        adapter.ingest("ep1", "s1", "2024-01-01T00:00:00", "hello world")
        adapter.reset("s1")
        assert adapter.retrieve("ep1") is None


# ---------------------------------------------------------------------------
# SQLiteEmbeddingAdapter tests
# ---------------------------------------------------------------------------

class TestSQLiteEmbeddingAdapter:
    def test_registered(self):
        cls = get_adapter("sqlite-embedding")
        assert cls is SQLiteEmbeddingAdapter

    @patch(EMBED_PATCH, side_effect=_fake_embed)
    def test_ingest_stores_embedding(self, mock_embed):
        adapter = SQLiteEmbeddingAdapter()
        adapter.ingest("ep1", "s1", "2024-01-01T00:00:00", "hello world")
        # Verify embedding was stored
        cur = adapter._conn.cursor()
        cur.execute("SELECT embedding FROM episode_embeddings WHERE episode_id = ?", ("ep1",))
        row = cur.fetchone()
        assert row is not None
        emb = json.loads(row["embedding"])
        assert isinstance(emb, list)
        assert len(emb) > 0

    @patch(EMBED_PATCH, side_effect=_fake_embed)
    def test_search_returns_results(self, mock_embed):
        adapter = SQLiteEmbeddingAdapter()
        adapter.ingest("ep1", "s1", "2024-01-01T00:00:00", "machine learning neural networks")
        adapter.ingest("ep2", "s1", "2024-01-02T00:00:00", "database query optimization sql")
        results = adapter.search("neural network deep learning")
        assert len(results) >= 1
        assert isinstance(results[0], SearchResult)
        assert results[0].score > 0

    @patch(EMBED_PATCH, side_effect=_fake_embed)
    def test_search_empty_query(self, mock_embed):
        adapter = SQLiteEmbeddingAdapter()
        adapter.ingest("ep1", "s1", "2024-01-01T00:00:00", "some text")
        results = adapter.search("")
        assert results == []

    @patch(EMBED_PATCH, side_effect=_fake_embed)
    def test_search_with_scope_filter(self, mock_embed):
        adapter = SQLiteEmbeddingAdapter()
        adapter.ingest("ep1", "s1", "2024-01-01T00:00:00", "alpha beta gamma")
        adapter.ingest("ep2", "s2", "2024-01-01T00:00:00", "alpha beta delta")
        results = adapter.search("alpha", filters={"scope_id": "s1"})
        assert len(results) == 1
        assert results[0].ref_id == "ep1"

    @patch(EMBED_PATCH, side_effect=_fake_embed)
    def test_search_with_limit(self, mock_embed):
        adapter = SQLiteEmbeddingAdapter()
        for i in range(20):
            adapter.ingest(f"ep{i}", "s1", f"2024-01-{i+1:02d}T00:00:00", f"document {i} testing")
        results = adapter.search("testing", limit=5)
        assert len(results) <= 5

    @patch(EMBED_PATCH, side_effect=_fake_embed)
    def test_retrieve(self, mock_embed):
        adapter = SQLiteEmbeddingAdapter()
        adapter.ingest("ep1", "s1", "2024-01-01T00:00:00", "test text", {"key": "val"})
        doc = adapter.retrieve("ep1")
        assert isinstance(doc, Document)
        assert doc.ref_id == "ep1"
        assert doc.text == "test text"
        assert doc.metadata == {"key": "val"}

    def test_retrieve_missing(self):
        adapter = SQLiteEmbeddingAdapter()
        assert adapter.retrieve("nonexistent") is None

    @patch(EMBED_PATCH, side_effect=_fake_embed)
    def test_reset_clears_both_tables(self, mock_embed):
        adapter = SQLiteEmbeddingAdapter()
        adapter.ingest("ep1", "s1", "2024-01-01T00:00:00", "scope one text")
        adapter.ingest("ep2", "s2", "2024-01-01T00:00:00", "scope two text")
        adapter.reset("s1")
        assert adapter.retrieve("ep1") is None
        assert adapter.retrieve("ep2") is not None
        # Check embedding was also deleted
        cur = adapter._conn.cursor()
        cur.execute("SELECT COUNT(*) FROM episode_embeddings WHERE episode_id = ?", ("ep1",))
        assert cur.fetchone()[0] == 0
        cur.execute("SELECT COUNT(*) FROM episode_embeddings WHERE episode_id = ?", ("ep2",))
        assert cur.fetchone()[0] == 1

    def test_get_capabilities(self):
        adapter = SQLiteEmbeddingAdapter()
        caps = adapter.get_capabilities()
        assert isinstance(caps, CapabilityManifest)
        assert "semantic" in caps.search_modes
        assert caps.supports_date_range is True


# ---------------------------------------------------------------------------
# SQLiteHybridAdapter tests
# ---------------------------------------------------------------------------

class TestSQLiteHybridAdapter:
    def test_registered(self):
        cls = get_adapter("sqlite-hybrid")
        assert cls is SQLiteHybridAdapter

    @patch(EMBED_PATCH, side_effect=_fake_embed)
    def test_ingest_and_search(self, mock_embed):
        adapter = SQLiteHybridAdapter()
        adapter.ingest("ep1", "s1", "2024-01-01T00:00:00", "the cat sat on the mat")
        adapter.ingest("ep2", "s1", "2024-01-02T00:00:00", "the dog played in the park")
        results = adapter.search("cat mat")
        assert len(results) >= 1
        assert isinstance(results[0], SearchResult)
        assert results[0].score > 0

    @patch(EMBED_PATCH, side_effect=_fake_embed)
    def test_search_empty_query(self, mock_embed):
        adapter = SQLiteHybridAdapter()
        results = adapter.search("")
        assert results == []

    @patch(EMBED_PATCH, side_effect=_fake_embed)
    def test_reset_clears_all_tables(self, mock_embed):
        adapter = SQLiteHybridAdapter()
        adapter.ingest("ep1", "s1", "2024-01-01T00:00:00", "hello world")
        adapter.reset("s1")
        assert adapter.retrieve("ep1") is None
        cur = adapter._conn.cursor()
        cur.execute("SELECT COUNT(*) FROM episode_embeddings")
        assert cur.fetchone()[0] == 0

    def test_get_capabilities(self):
        adapter = SQLiteHybridAdapter()
        caps = adapter.get_capabilities()
        assert isinstance(caps, CapabilityManifest)
        assert "keyword" in caps.search_modes
        assert "semantic" in caps.search_modes


# ---------------------------------------------------------------------------
# Cosine similarity tests
# ---------------------------------------------------------------------------

class TestCosineSimilarity:
    def test_identical_vectors(self):
        v = [1.0, 2.0, 3.0]
        assert cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert cosine_similarity(a, b) == pytest.approx(0.0)

    def test_opposite_vectors(self):
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        assert cosine_similarity(a, b) == pytest.approx(-1.0)

    def test_zero_vector(self):
        a = [0.0, 0.0]
        b = [1.0, 2.0]
        assert cosine_similarity(a, b) == 0.0


# ---------------------------------------------------------------------------
# RRF merge tests
# ---------------------------------------------------------------------------

class TestRRFMerge:
    def test_basic_merge(self):
        a = [
            SearchResult(ref_id="d1", text="doc1", score=10.0),
            SearchResult(ref_id="d2", text="doc2", score=5.0),
        ]
        b = [
            SearchResult(ref_id="d2", text="doc2", score=0.9),
            SearchResult(ref_id="d3", text="doc3", score=0.8),
        ]
        merged = _rrf_merge(a, b, k=60, limit=10)
        # d2 appears in both lists so should score highest
        assert merged[0].ref_id == "d2"
        assert len(merged) == 3

    def test_disjoint_lists(self):
        a = [SearchResult(ref_id="d1", text="doc1", score=1.0)]
        b = [SearchResult(ref_id="d2", text="doc2", score=1.0)]
        merged = _rrf_merge(a, b, k=60, limit=10)
        assert len(merged) == 2
        # Both have rank 1 in their list and default rank in the other
        # d1: 1/(60+1) + 1/(60+2) = ~0.01639 + ~0.01613 = ~0.03252
        # d2: 1/(60+2) + 1/(60+1) = ~0.01613 + ~0.01639 = ~0.03252
        # Scores should be equal (symmetric)
        assert merged[0].score == pytest.approx(merged[1].score, abs=1e-6)

    def test_limit_respected(self):
        a = [SearchResult(ref_id=f"d{i}", text=f"doc{i}", score=float(i)) for i in range(10)]
        b = [SearchResult(ref_id=f"d{i}", text=f"doc{i}", score=float(i)) for i in range(10)]
        merged = _rrf_merge(a, b, k=60, limit=3)
        assert len(merged) == 3

    def test_empty_lists(self):
        merged = _rrf_merge([], [], k=60, limit=10)
        assert merged == []

    def test_one_empty_list(self):
        a = [
            SearchResult(ref_id="d1", text="doc1", score=1.0),
            SearchResult(ref_id="d2", text="doc2", score=0.5),
        ]
        merged = _rrf_merge(a, [], k=60, limit=10)
        assert len(merged) == 2
        # d1 ranked higher in list a, should remain first
        assert merged[0].ref_id == "d1"

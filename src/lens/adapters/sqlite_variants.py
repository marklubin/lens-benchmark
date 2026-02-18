"""SQLite-based adapter variants: FTS, Embedding, and Hybrid.

Provides retrieval strategies sharing the same SQLite episode storage:
- sqlite-fts: BM25 keyword search (FTS5)
- sqlite-embedding: Semantic search via Ollama embeddings
- sqlite-hybrid: RRF fusion of FTS + Ollama embeddings
- sqlite-embedding-openai: Semantic search via OpenAI embeddings
- sqlite-hybrid-openai: RRF fusion of FTS + OpenAI embeddings
"""
from __future__ import annotations

import json
import math
import os
import sqlite3
import urllib.request

from lens.adapters.base import (
    CapabilityManifest,
    Document,
    FilterField,
    MemoryAdapter,
    SearchResult,
)
from lens.adapters.registry import register_adapter
from lens.adapters.sqlite import SQLiteAdapter, _fts5_escape


# ---------------------------------------------------------------------------
# FTS-only variant (thin wrapper around SQLiteAdapter)
# ---------------------------------------------------------------------------


@register_adapter("sqlite-fts")
class SQLiteFTSAdapter(SQLiteAdapter):
    """SQLite FTS5 adapter. Identical to base SQLiteAdapter, registered as sqlite-fts."""

    def reset(self, scope_id: str) -> None:
        """Reset scope. Relies on DELETE trigger for FTS cleanup."""
        cur = self._conn.cursor()
        cur.execute("DELETE FROM episodes WHERE scope_id = ?", (scope_id,))
        self._conn.commit()


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------

_DEFAULT_OLLAMA_URL = "http://localhost:11434/api/embed"
_DEFAULT_EMBED_MODEL = "nomic-embed-text"
_MAX_EMBED_CHARS = 6000  # Conservative limit for nomic-embed-text (8192 token context)

_DEFAULT_OPENAI_EMBED_MODEL = "text-embedding-3-small"
_OPENAI_EMBED_URL = "https://api.openai.com/v1/embeddings"


def _embed_texts(
    texts: list[str],
    model: str = _DEFAULT_EMBED_MODEL,
    ollama_url: str = _DEFAULT_OLLAMA_URL,
    _max_retries: int = 3,
) -> list[list[float]]:
    """Call Ollama embedding API. Returns one vector per input text."""
    import logging
    import time

    log = logging.getLogger(__name__)
    # Truncate texts that exceed model context length
    texts = [t[:_MAX_EMBED_CHARS] if len(t) > _MAX_EMBED_CHARS else t for t in texts]
    body = json.dumps({"model": model, "input": texts}).encode()
    for attempt in range(_max_retries):
        req = urllib.request.Request(
            ollama_url, data=body, headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                return json.loads(resp.read())["embeddings"]
        except urllib.error.HTTPError as e:
            resp_body = ""
            try:
                resp_body = e.read().decode(errors="replace")
            except Exception:
                pass
            log.warning(
                "Ollama embed HTTP %s (attempt %d/%d, %d texts, %d bytes): %s",
                e.code, attempt + 1, _max_retries, len(texts), len(body), resp_body[:500],
            )
            if attempt == _max_retries - 1:
                raise
            time.sleep(2 * (attempt + 1))
        except (urllib.error.URLError, OSError) as e:
            log.warning(
                "Ollama embed network error (attempt %d/%d): %s",
                attempt + 1, _max_retries, e,
            )
            if attempt == _max_retries - 1:
                raise
            time.sleep(2 * (attempt + 1))


def _embed_texts_openai(
    texts: list[str],
    model: str = _DEFAULT_OPENAI_EMBED_MODEL,
    api_key: str | None = None,
    _max_retries: int = 3,
) -> list[list[float]]:
    """Call OpenAI embedding API. Returns one vector per input text."""
    import logging
    import time

    log = logging.getLogger(__name__)
    api_key = api_key or os.environ.get("OPENAI_API_KEY") or os.environ.get("LENS_LLM_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API key required: set OPENAI_API_KEY or LENS_LLM_API_KEY")

    body = json.dumps({"model": model, "input": texts}).encode()
    for attempt in range(_max_retries):
        req = urllib.request.Request(
            _OPENAI_EMBED_URL,
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                result = json.loads(resp.read())
                # OpenAI returns {"data": [{"embedding": [...], "index": 0}, ...]}
                sorted_data = sorted(result["data"], key=lambda d: d["index"])
                return [d["embedding"] for d in sorted_data]
        except urllib.error.HTTPError as e:
            resp_body = ""
            try:
                resp_body = e.read().decode(errors="replace")
            except Exception:
                pass
            log.warning(
                "OpenAI embed HTTP %s (attempt %d/%d): %s",
                e.code, attempt + 1, _max_retries, resp_body[:500],
            )
            if attempt == _max_retries - 1:
                raise
            time.sleep(1 * (attempt + 1))
        except (urllib.error.URLError, OSError) as e:
            log.warning(
                "OpenAI embed network error (attempt %d/%d): %s",
                attempt + 1, _max_retries, e,
            )
            if attempt == _max_retries - 1:
                raise
            time.sleep(1 * (attempt + 1))


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors (pure Python)."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# ---------------------------------------------------------------------------
# Embedding adapter
# ---------------------------------------------------------------------------


@register_adapter("sqlite-embedding")
class SQLiteEmbeddingAdapter(MemoryAdapter):
    """SQLite adapter with Ollama embedding-based semantic search.

    Stores episodes in the same schema as SQLiteAdapter, plus an
    episode_embeddings table for vector search.
    """

    def __init__(
        self,
        db_path: str = ":memory:",
        embed_model: str = _DEFAULT_EMBED_MODEL,
        ollama_url: str = _DEFAULT_OLLAMA_URL,
    ) -> None:
        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        self._embed_model = embed_model
        self._ollama_url = ollama_url
        self._ensure_tables()

    def _ensure_tables(self) -> None:
        cur = self._conn.cursor()
        cur.executescript("""
            CREATE TABLE IF NOT EXISTS episodes (
                episode_id TEXT PRIMARY KEY,
                scope_id   TEXT NOT NULL,
                timestamp  TEXT NOT NULL,
                text       TEXT NOT NULL,
                meta       TEXT DEFAULT '{}'
            );
            CREATE TABLE IF NOT EXISTS episode_embeddings (
                episode_id TEXT PRIMARY KEY REFERENCES episodes(episode_id),
                embedding  TEXT NOT NULL
            );
        """)
        self._conn.commit()

    def reset(self, scope_id: str) -> None:
        cur = self._conn.cursor()
        cur.execute(
            "DELETE FROM episode_embeddings WHERE episode_id IN "
            "(SELECT episode_id FROM episodes WHERE scope_id = ?)",
            (scope_id,),
        )
        cur.execute("DELETE FROM episodes WHERE scope_id = ?", (scope_id,))
        self._conn.commit()

    def ingest(
        self,
        episode_id: str,
        scope_id: str,
        timestamp: str,
        text: str,
        meta: dict | None = None,
    ) -> None:
        cur = self._conn.cursor()
        cur.execute(
            "INSERT OR REPLACE INTO episodes (episode_id, scope_id, timestamp, text, meta) "
            "VALUES (?, ?, ?, ?, ?)",
            (episode_id, scope_id, timestamp, text, json.dumps(meta or {})),
        )
        # Compute and store embedding
        vectors = _embed_texts([text], model=self._embed_model, ollama_url=self._ollama_url)
        cur.execute(
            "INSERT OR REPLACE INTO episode_embeddings (episode_id, embedding) VALUES (?, ?)",
            (episode_id, json.dumps(vectors[0])),
        )
        self._conn.commit()

    def search(
        self,
        query: str,
        filters: dict | None = None,
        limit: int | None = None,
    ) -> list[SearchResult]:
        if not query or not query.strip():
            return []

        limit = limit or 10

        # Embed the query
        query_vec = _embed_texts([query], model=self._embed_model, ollama_url=self._ollama_url)[0]

        # Fetch all candidate embeddings
        sql = (
            "SELECT e.episode_id, e.text, e.meta, ee.embedding "
            "FROM episodes e "
            "JOIN episode_embeddings ee ON e.episode_id = ee.episode_id "
            "WHERE 1=1 "
        )
        params: list = []
        if filters:
            if "scope_id" in filters:
                sql += "AND e.scope_id = ? "
                params.append(filters["scope_id"])
            if "start_date" in filters:
                sql += "AND e.timestamp >= ? "
                params.append(filters["start_date"])
            if "end_date" in filters:
                sql += "AND e.timestamp <= ? "
                params.append(filters["end_date"])

        cur = self._conn.cursor()
        cur.execute(sql, params)
        rows = cur.fetchall()

        # Score by cosine similarity
        scored = []
        for row in rows:
            emb = json.loads(row["embedding"])
            sim = cosine_similarity(query_vec, emb)
            meta = json.loads(row["meta"]) if row["meta"] else {}
            scored.append(
                SearchResult(
                    ref_id=row["episode_id"],
                    text=row["text"][:500],
                    score=sim,
                    metadata=meta,
                )
            )

        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[:limit]

    def retrieve(self, ref_id: str) -> Document | None:
        cur = self._conn.cursor()
        cur.execute(
            "SELECT episode_id, text, meta FROM episodes WHERE episode_id = ?",
            (ref_id,),
        )
        row = cur.fetchone()
        if row is None:
            return None
        meta = json.loads(row["meta"]) if row["meta"] else {}
        return Document(ref_id=row["episode_id"], text=row["text"], metadata=meta)

    def get_capabilities(self) -> CapabilityManifest:
        return CapabilityManifest(
            search_modes=["semantic"],
            filter_fields=[
                FilterField(name="scope_id", field_type="string", description="Filter by scope ID"),
                FilterField(name="start_date", field_type="string", description="Filter episodes after this ISO date"),
                FilterField(name="end_date", field_type="string", description="Filter episodes before this ISO date"),
            ],
            max_results_per_search=10,
            supports_date_range=True,
            extra_tools=[],
        )


# ---------------------------------------------------------------------------
# Hybrid adapter (RRF fusion of FTS + Embedding)
# ---------------------------------------------------------------------------


@register_adapter("sqlite-hybrid")
class SQLiteHybridAdapter(MemoryAdapter):
    """SQLite adapter combining FTS5 keyword search and embedding semantic search.

    Results are fused via Reciprocal Rank Fusion (RRF).
    """

    def __init__(
        self,
        db_path: str = ":memory:",
        embed_model: str = _DEFAULT_EMBED_MODEL,
        ollama_url: str = _DEFAULT_OLLAMA_URL,
        rrf_k: int = 60,
    ) -> None:
        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        self._embed_model = embed_model
        self._ollama_url = ollama_url
        self._rrf_k = rrf_k
        self._ensure_tables()

    def _ensure_tables(self) -> None:
        cur = self._conn.cursor()
        cur.executescript("""
            CREATE TABLE IF NOT EXISTS episodes (
                episode_id TEXT PRIMARY KEY,
                scope_id   TEXT NOT NULL,
                timestamp  TEXT NOT NULL,
                text       TEXT NOT NULL,
                meta       TEXT DEFAULT '{}'
            );
            CREATE VIRTUAL TABLE IF NOT EXISTS episodes_fts
                USING fts5(episode_id, text, content=episodes, content_rowid=rowid);

            CREATE TRIGGER IF NOT EXISTS episodes_ai AFTER INSERT ON episodes BEGIN
                INSERT INTO episodes_fts(rowid, episode_id, text)
                    VALUES (new.rowid, new.episode_id, new.text);
            END;
            CREATE TRIGGER IF NOT EXISTS episodes_ad AFTER DELETE ON episodes BEGIN
                INSERT INTO episodes_fts(episodes_fts, rowid, episode_id, text)
                    VALUES ('delete', old.rowid, old.episode_id, old.text);
            END;

            CREATE TABLE IF NOT EXISTS episode_embeddings (
                episode_id TEXT PRIMARY KEY REFERENCES episodes(episode_id),
                embedding  TEXT NOT NULL
            );
        """)
        self._conn.commit()

    def reset(self, scope_id: str) -> None:
        cur = self._conn.cursor()
        # Delete embeddings first
        cur.execute(
            "DELETE FROM episode_embeddings WHERE episode_id IN "
            "(SELECT episode_id FROM episodes WHERE scope_id = ?)",
            (scope_id,),
        )
        # DELETE FROM episodes triggers episodes_ad which cleans up FTS
        cur.execute("DELETE FROM episodes WHERE scope_id = ?", (scope_id,))
        self._conn.commit()

    def ingest(
        self,
        episode_id: str,
        scope_id: str,
        timestamp: str,
        text: str,
        meta: dict | None = None,
    ) -> None:
        cur = self._conn.cursor()
        cur.execute(
            "INSERT OR REPLACE INTO episodes (episode_id, scope_id, timestamp, text, meta) "
            "VALUES (?, ?, ?, ?, ?)",
            (episode_id, scope_id, timestamp, text, json.dumps(meta or {})),
        )
        # Compute and store embedding
        vectors = _embed_texts([text], model=self._embed_model, ollama_url=self._ollama_url)
        cur.execute(
            "INSERT OR REPLACE INTO episode_embeddings (episode_id, embedding) VALUES (?, ?)",
            (episode_id, json.dumps(vectors[0])),
        )
        self._conn.commit()

    def _fts_search(
        self, query: str, filters: dict | None, limit: int,
    ) -> list[SearchResult]:
        """Run FTS5 keyword search."""
        safe_query = _fts5_escape(query)
        if not safe_query:
            return []

        sql = (
            "SELECT e.episode_id, e.text, e.meta, bm25(episodes_fts) AS rank "
            "FROM episodes_fts f "
            "JOIN episodes e ON e.episode_id = f.episode_id "
            "WHERE episodes_fts MATCH ? "
        )
        params: list = [safe_query]
        if filters:
            if "scope_id" in filters:
                sql += "AND e.scope_id = ? "
                params.append(filters["scope_id"])
            if "start_date" in filters:
                sql += "AND e.timestamp >= ? "
                params.append(filters["start_date"])
            if "end_date" in filters:
                sql += "AND e.timestamp <= ? "
                params.append(filters["end_date"])
        sql += "ORDER BY rank LIMIT ?"
        params.append(limit)

        cur = self._conn.cursor()
        try:
            cur.execute(sql, params)
        except sqlite3.OperationalError:
            return []

        results = []
        for row in cur.fetchall():
            meta = json.loads(row["meta"]) if row["meta"] else {}
            results.append(
                SearchResult(
                    ref_id=row["episode_id"],
                    text=row["text"][:500],
                    score=abs(row["rank"]),
                    metadata=meta,
                )
            )
        return results

    def _embedding_search(
        self, query: str, filters: dict | None, limit: int,
    ) -> list[SearchResult]:
        """Run embedding-based semantic search."""
        query_vec = _embed_texts([query], model=self._embed_model, ollama_url=self._ollama_url)[0]

        sql = (
            "SELECT e.episode_id, e.text, e.meta, ee.embedding "
            "FROM episodes e "
            "JOIN episode_embeddings ee ON e.episode_id = ee.episode_id "
            "WHERE 1=1 "
        )
        params: list = []
        if filters:
            if "scope_id" in filters:
                sql += "AND e.scope_id = ? "
                params.append(filters["scope_id"])
            if "start_date" in filters:
                sql += "AND e.timestamp >= ? "
                params.append(filters["start_date"])
            if "end_date" in filters:
                sql += "AND e.timestamp <= ? "
                params.append(filters["end_date"])

        cur = self._conn.cursor()
        cur.execute(sql, params)
        rows = cur.fetchall()

        scored = []
        for row in rows:
            emb = json.loads(row["embedding"])
            sim = cosine_similarity(query_vec, emb)
            meta = json.loads(row["meta"]) if row["meta"] else {}
            scored.append(
                SearchResult(
                    ref_id=row["episode_id"],
                    text=row["text"][:500],
                    score=sim,
                    metadata=meta,
                )
            )
        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[:limit]

    def search(
        self,
        query: str,
        filters: dict | None = None,
        limit: int | None = None,
    ) -> list[SearchResult]:
        if not query or not query.strip():
            return []

        limit = limit or 10
        # Fetch more from each source for better RRF fusion
        fetch_limit = limit * 3

        fts_results = self._fts_search(query, filters, fetch_limit)
        emb_results = self._embedding_search(query, filters, fetch_limit)

        return _rrf_merge(fts_results, emb_results, k=self._rrf_k, limit=limit)

    def retrieve(self, ref_id: str) -> Document | None:
        cur = self._conn.cursor()
        cur.execute(
            "SELECT episode_id, text, meta FROM episodes WHERE episode_id = ?",
            (ref_id,),
        )
        row = cur.fetchone()
        if row is None:
            return None
        meta = json.loads(row["meta"]) if row["meta"] else {}
        return Document(ref_id=row["episode_id"], text=row["text"], metadata=meta)

    def get_capabilities(self) -> CapabilityManifest:
        return CapabilityManifest(
            search_modes=["keyword", "semantic"],
            filter_fields=[
                FilterField(name="scope_id", field_type="string", description="Filter by scope ID"),
                FilterField(name="start_date", field_type="string", description="Filter episodes after this ISO date"),
                FilterField(name="end_date", field_type="string", description="Filter episodes before this ISO date"),
            ],
            max_results_per_search=10,
            supports_date_range=True,
            extra_tools=[],
        )


# ---------------------------------------------------------------------------
# RRF merge utility
# ---------------------------------------------------------------------------


def _rrf_merge(
    results_a: list[SearchResult],
    results_b: list[SearchResult],
    k: int = 60,
    limit: int = 10,
) -> list[SearchResult]:
    """Merge two ranked lists using Reciprocal Rank Fusion.

    RRF score(d) = 1/(k + rank_a) + 1/(k + rank_b)
    Documents not present in a list get rank = len(list) + 1.
    """
    # Build ref_id -> SearchResult lookup (keep best text/meta from either)
    all_results: dict[str, SearchResult] = {}
    for r in results_a:
        all_results[r.ref_id] = r
    for r in results_b:
        if r.ref_id not in all_results:
            all_results[r.ref_id] = r

    # Build rank maps (1-indexed)
    ranks_a = {r.ref_id: i + 1 for i, r in enumerate(results_a)}
    ranks_b = {r.ref_id: i + 1 for i, r in enumerate(results_b)}

    default_rank_a = len(results_a) + 1
    default_rank_b = len(results_b) + 1

    scored: list[tuple[float, SearchResult]] = []
    for ref_id, result in all_results.items():
        ra = ranks_a.get(ref_id, default_rank_a)
        rb = ranks_b.get(ref_id, default_rank_b)
        rrf_score = 1.0 / (k + ra) + 1.0 / (k + rb)
        scored.append((
            rrf_score,
            SearchResult(ref_id=result.ref_id, text=result.text, score=rrf_score, metadata=result.metadata),
        ))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [s[1] for s in scored[:limit]]


# ---------------------------------------------------------------------------
# OpenAI embedding variants
# ---------------------------------------------------------------------------


@register_adapter("sqlite-embedding-openai")
class SQLiteEmbeddingOpenAIAdapter(SQLiteEmbeddingAdapter):
    """SQLite adapter with OpenAI embedding-based semantic search."""

    def __init__(
        self,
        db_path: str = ":memory:",
        embed_model: str = _DEFAULT_OPENAI_EMBED_MODEL,
        api_key: str | None = None,
    ) -> None:
        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        self._embed_model = embed_model
        self._api_key = api_key
        self._ensure_tables()

    def _embed(self, texts: list[str]) -> list[list[float]]:
        return _embed_texts_openai(texts, model=self._embed_model, api_key=self._api_key)

    def ingest(
        self,
        episode_id: str,
        scope_id: str,
        timestamp: str,
        text: str,
        meta: dict | None = None,
    ) -> None:
        cur = self._conn.cursor()
        cur.execute(
            "INSERT OR REPLACE INTO episodes (episode_id, scope_id, timestamp, text, meta) "
            "VALUES (?, ?, ?, ?, ?)",
            (episode_id, scope_id, timestamp, text, json.dumps(meta or {})),
        )
        vectors = self._embed([text])
        cur.execute(
            "INSERT OR REPLACE INTO episode_embeddings (episode_id, embedding) VALUES (?, ?)",
            (episode_id, json.dumps(vectors[0])),
        )
        self._conn.commit()

    def search(
        self,
        query: str,
        filters: dict | None = None,
        limit: int | None = None,
    ) -> list[SearchResult]:
        if not query or not query.strip():
            return []

        limit = limit or 10
        query_vec = self._embed([query])[0]

        sql = (
            "SELECT e.episode_id, e.text, e.meta, ee.embedding "
            "FROM episodes e "
            "JOIN episode_embeddings ee ON e.episode_id = ee.episode_id "
            "WHERE 1=1 "
        )
        params: list = []
        if filters:
            if "scope_id" in filters:
                sql += "AND e.scope_id = ? "
                params.append(filters["scope_id"])
            if "start_date" in filters:
                sql += "AND e.timestamp >= ? "
                params.append(filters["start_date"])
            if "end_date" in filters:
                sql += "AND e.timestamp <= ? "
                params.append(filters["end_date"])

        cur = self._conn.cursor()
        cur.execute(sql, params)
        rows = cur.fetchall()

        scored = []
        for row in rows:
            emb = json.loads(row["embedding"])
            sim = cosine_similarity(query_vec, emb)
            meta = json.loads(row["meta"]) if row["meta"] else {}
            scored.append(
                SearchResult(
                    ref_id=row["episode_id"],
                    text=row["text"][:500],
                    score=sim,
                    metadata=meta,
                )
            )

        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[:limit]


@register_adapter("sqlite-hybrid-openai")
class SQLiteHybridOpenAIAdapter(SQLiteHybridAdapter):
    """SQLite hybrid adapter with OpenAI embeddings + FTS5."""

    def __init__(
        self,
        db_path: str = ":memory:",
        embed_model: str = _DEFAULT_OPENAI_EMBED_MODEL,
        api_key: str | None = None,
        rrf_k: int = 60,
    ) -> None:
        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        self._embed_model = embed_model
        self._api_key = api_key
        self._rrf_k = rrf_k
        self._ensure_tables()

    def _embed(self, texts: list[str]) -> list[list[float]]:
        return _embed_texts_openai(texts, model=self._embed_model, api_key=self._api_key)

    def ingest(
        self,
        episode_id: str,
        scope_id: str,
        timestamp: str,
        text: str,
        meta: dict | None = None,
    ) -> None:
        cur = self._conn.cursor()
        cur.execute(
            "INSERT OR REPLACE INTO episodes (episode_id, scope_id, timestamp, text, meta) "
            "VALUES (?, ?, ?, ?, ?)",
            (episode_id, scope_id, timestamp, text, json.dumps(meta or {})),
        )
        vectors = self._embed([text])
        cur.execute(
            "INSERT OR REPLACE INTO episode_embeddings (episode_id, embedding) VALUES (?, ?)",
            (episode_id, json.dumps(vectors[0])),
        )
        self._conn.commit()

    def _embedding_search(
        self, query: str, filters: dict | None, limit: int,
    ) -> list[SearchResult]:
        """Run embedding-based semantic search via OpenAI."""
        query_vec = self._embed([query])[0]

        sql = (
            "SELECT e.episode_id, e.text, e.meta, ee.embedding "
            "FROM episodes e "
            "JOIN episode_embeddings ee ON e.episode_id = ee.episode_id "
            "WHERE 1=1 "
        )
        params: list = []
        if filters:
            if "scope_id" in filters:
                sql += "AND e.scope_id = ? "
                params.append(filters["scope_id"])
            if "start_date" in filters:
                sql += "AND e.timestamp >= ? "
                params.append(filters["start_date"])
            if "end_date" in filters:
                sql += "AND e.timestamp <= ? "
                params.append(filters["end_date"])

        cur = self._conn.cursor()
        cur.execute(sql, params)
        rows = cur.fetchall()

        scored = []
        for row in rows:
            emb = json.loads(row["embedding"])
            sim = cosine_similarity(query_vec, emb)
            meta = json.loads(row["meta"]) if row["meta"] else {}
            scored.append(
                SearchResult(
                    ref_id=row["episode_id"],
                    text=row["text"][:500],
                    score=sim,
                    metadata=meta,
                )
            )
        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[:limit]

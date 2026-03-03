"""Hopping Context Windows adapter for LENS.

Incremental summarization: instead of re-summarizing ALL episodes at each
checkpoint (compaction), hopping maintains a rolling summary. When the buffer
of new episodes exceeds a configurable token threshold, a compaction step
merges them into the existing summary. This can fire multiple times per
checkpoint if episodes are large.

Three variants:
- hopping:        summary-only (like compaction, but incremental)
- hopping-rag:    summary + semantic search on raw episodes
- hopping-hybrid: summary + RRF-fused (FTS + embedding) search on raw episodes

Environment variables:
    LENS_LLM_API_KEY / OPENAI_API_KEY   — API key for the summarization LLM
    LENS_LLM_API_BASE / OPENAI_BASE_URL — Base URL for the LLM API
    LENS_LLM_MODEL                      — Model name for compaction LLM
    HOPPING_MAX_TOKENS                  — Max tokens for summary output (default 2000)
    HOPPING_COMPACT_THRESHOLD           — Token threshold for triggering compaction (default 50000)
    HOPPING_COMPACT_BATCH               — Max episodes per compaction step (default 4)
    LENS_EMBED_API_KEY / OPENAI_API_KEY — API key for embeddings (RAG/hybrid)
    LENS_EMBED_BASE_URL                 — Base URL for embedding API (RAG/hybrid)
    LENS_EMBED_MODEL                    — Embedding model name (RAG/hybrid)
"""
from __future__ import annotations

import json
import logging
import os
import re
import sqlite3

try:
    from openai import OpenAI as _OpenAI
except ImportError:
    _OpenAI = None  # type: ignore[assignment,misc]

from lens.adapters.base import (
    CapabilityManifest,
    Document,
    ExtraTool,
    MemoryAdapter,
    SearchResult,
)
from lens.adapters.registry import register_adapter
from lens.adapters.sqlite import _fts5_escape
from lens.adapters.sqlite_variants import (
    _embed_texts_openai,
    _rrf_merge,
    cosine_similarity,
)

logger = logging.getLogger(__name__)

_EP_ID_RE = re.compile(r"\[([^\]]+)\]")

_DEFAULT_OPENAI_EMBED_MODEL = "text-embedding-3-small"


def _strip_provider_prefix(model: str) -> str:
    """Convert 'together/Qwen/Qwen3-...' -> 'Qwen/Qwen3-...' for OpenAI API."""
    if "/" in model and model.startswith(("together/", "openai/")):
        return model.split("/", 1)[1]
    return model


_HOPPING_SYSTEM = (
    "You are a memory compaction agent. You maintain a running summary of "
    "sequential episode logs. Each update, you receive your previous summary "
    "and a batch of new episodes. Merge the new information into a single "
    "updated summary."
)

_HOPPING_INITIAL_TMPL = """\
EPISODES ({n} total, {first_ts} to {last_ts}):

{episodes_block}

COMPRESSION OBJECTIVE:
Compress these episodes into a summary. Cite [episode_id] for specific data points. \
Preserve numeric values exactly. Focus on patterns and changes across episodes rather \
than repeating each entry. Prioritise information that reveals trends, anomalies, or \
cause-and-effect relationships.

Max output: approximately {max_tokens} tokens.

SUMMARY:"""

_HOPPING_INCREMENTAL_TMPL = """\
EXISTING SUMMARY:
{previous_summary}

NEW EPISODES ({n} new, {first_ts} to {last_ts}):

{episodes_block}

UPDATE OBJECTIVE:
Merge the new episodes into the existing summary. Cite [episode_id] for specific data \
points. Preserve numeric values exactly. Keep important details from the existing \
summary — do not drop information just because it is older. Add new patterns, changes, \
and data points from the new episodes. If new data contradicts or updates old data, \
note both the old and new values.

Max output: approximately {max_tokens} tokens.

UPDATED SUMMARY:"""


# ---------------------------------------------------------------------------
# Base: HoppingAdapter (summary-only)
# ---------------------------------------------------------------------------


@register_adapter("hopping")
class HoppingAdapter(MemoryAdapter):
    """Hopping context windows — incremental summarization.

    At each checkpoint, new episodes are merged with the previous summary
    rather than re-summarizing all episodes from scratch. This tests whether
    incremental compaction preserves more signal than stop-the-world compaction.
    """

    requires_metering: bool = True  # LLM calls in prepare()

    def __init__(self) -> None:
        self._episodes: list[dict] = []
        self._new_episodes: list[dict] = []
        self._summary: str = ""
        self._cited_episode_ids: list[str] = []
        self._scope_id: str | None = None
        self._max_tokens = int(os.environ.get("HOPPING_MAX_TOKENS", "2000"))
        # Rolling compaction thresholds
        self._compact_threshold = int(os.environ.get("HOPPING_COMPACT_THRESHOLD", "80000"))
        self._compact_batch_size = int(os.environ.get("HOPPING_COMPACT_BATCH", "4"))
        self._scope_episodes: dict[str, list[dict]] = {}
        self._oai_client: object | None = None  # lazy-init OpenAI client
        self._model: str = ""

    def reset(self, scope_id: str) -> None:
        self._scope_episodes.pop(scope_id, None)
        self._episodes = []
        self._new_episodes = []
        self._summary = ""
        self._cited_episode_ids = []
        self._scope_id = scope_id
        for eps in self._scope_episodes.values():
            self._episodes.extend(eps)

    def ingest(
        self,
        episode_id: str,
        scope_id: str,
        timestamp: str,
        text: str,
        meta: dict | None = None,
    ) -> None:
        """Buffer episode — instant, no I/O."""
        ep = {
            "episode_id": episode_id,
            "scope_id": scope_id,
            "timestamp": timestamp,
            "text": text,
            "meta": meta or {},
        }
        self._episodes.append(ep)
        self._new_episodes.append(ep)
        self._scope_episodes.setdefault(scope_id, []).append(ep)

    def _ensure_client(self):
        """Lazy-init the OpenAI client and model name."""
        if self._oai_client is not None:
            return
        if _OpenAI is None:
            logger.error("openai package required for hopping adapter")
            return
        api_key = (
            os.environ.get("LENS_LLM_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
            or "dummy"
        )
        base_url = (
            os.environ.get("LENS_LLM_API_BASE")
            or os.environ.get("OPENAI_BASE_URL")
        )
        model_raw = os.environ.get("LENS_LLM_MODEL", "gpt-4o-mini")
        self._model = _strip_provider_prefix(model_raw)
        client_kwargs: dict = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        self._oai_client = _OpenAI(**client_kwargs)

    def _compact_batch(self, batch: list[dict]) -> None:
        """Run one compaction step: merge a batch of episodes into _summary."""
        self._ensure_client()
        if self._oai_client is None:
            return

        lines = [f"[{ep['episode_id']}] {ep['timestamp']}: {ep['text']}" for ep in batch]
        episodes_block = "\n\n".join(lines)
        est_input_tokens = self._estimate_tokens(episodes_block) + self._estimate_tokens(self._summary)
        logger.info(
            "Hopping _compact_batch: %d episodes, ~%d est input tokens, summary_so_far=%d chars",
            len(batch), est_input_tokens, len(self._summary),
        )
        first_ts = batch[0]["timestamp"]
        last_ts = batch[-1]["timestamp"]

        if self._summary:
            user_msg = _HOPPING_INCREMENTAL_TMPL.format(
                previous_summary=self._summary,
                n=len(batch),
                first_ts=first_ts,
                last_ts=last_ts,
                episodes_block=episodes_block,
                max_tokens=self._max_tokens,
            )
        else:
            user_msg = _HOPPING_INITIAL_TMPL.format(
                n=len(batch),
                first_ts=first_ts,
                last_ts=last_ts,
                episodes_block=episodes_block,
                max_tokens=self._max_tokens,
            )

        try:
            resp = self._oai_client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": _HOPPING_SYSTEM},
                    {"role": "user", "content": user_msg},
                ],
                max_tokens=self._max_tokens,
                temperature=0.0,
            )
            self._summary = resp.choices[0].message.content or ""
            logger.info(
                "Hopping compacted %d episodes → %d char summary",
                len(batch), len(self._summary),
            )
        except Exception as e:
            logger.error("Hopping LLM call failed: %s", e)

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Rough token estimate: ~3.5 chars per token (conservative for Llama tokenizer)."""
        return int(len(text) / 3.5)

    def prepare(self, scope_id: str, checkpoint: int) -> None:
        """Rolling compaction: process new episodes in batches.

        Splits _new_episodes into batches of at most _compact_batch episodes,
        each also respecting _compact_threshold token budget. Each batch is
        merged into the running summary via an LLM call.
        """
        if not self._new_episodes:
            return

        # Split into batches respecting both count and token limits
        batches: list[list[dict]] = []
        current_batch: list[dict] = []
        current_tokens = 0

        for ep in self._new_episodes:
            ep_tokens = self._estimate_tokens(ep["text"])
            # Start new batch if adding this episode would exceed limits
            if current_batch and (
                len(current_batch) >= self._compact_batch_size
                or current_tokens + ep_tokens > self._compact_threshold
            ):
                batches.append(current_batch)
                current_batch = []
                current_tokens = 0
            current_batch.append(ep)
            current_tokens += ep_tokens

        if current_batch:
            batches.append(current_batch)

        logger.info(
            "Hopping prepare: %d new episodes → %d compaction steps "
            "(threshold=%d, batch=%d)",
            len(self._new_episodes), len(batches),
            self._compact_threshold, self._compact_batch_size,
        )

        for batch in batches:
            self._compact_batch(batch)

        # Parse cited episode IDs from final summary
        self._cited_episode_ids = _EP_ID_RE.findall(self._summary)

        # Clear new episodes buffer
        self._new_episodes = []

    def search(
        self,
        query: str,
        filters: dict | None = None,
        limit: int | None = None,
    ) -> list[SearchResult]:
        if self._summary:
            return [SearchResult(
                ref_id="hopping_summary",
                text=self._summary[:500],
                score=1.0,
                metadata={"type": "hopping_summary", "cited_episodes": len(self._cited_episode_ids)},
            )]
        if self._episodes:
            cap = limit or 10
            results: list[SearchResult] = []
            for ep in self._episodes[:cap]:
                results.append(SearchResult(
                    ref_id=ep["episode_id"],
                    text=ep["text"][:500],
                    score=0.5,
                    metadata=ep.get("meta", {}),
                ))
            return results
        return []

    def retrieve(self, ref_id: str) -> Document | None:
        if ref_id == "hopping_summary":
            if self._summary:
                return Document(ref_id="hopping_summary", text=self._summary)
            return None
        for ep in self._episodes:
            if ep["episode_id"] == ref_id:
                return Document(
                    ref_id=ref_id,
                    text=ep["text"],
                    metadata=ep.get("meta", {}),
                )
        return None

    def get_synthetic_refs(self) -> list[tuple[str, str]]:
        if self._summary:
            return [("hopping_summary", self._summary)]
        return []

    def get_capabilities(self) -> CapabilityManifest:
        return CapabilityManifest(
            search_modes=["hopping"],
            max_results_per_search=1,
            extra_tools=[
                ExtraTool(
                    name="batch_retrieve",
                    description=(
                        "Retrieve multiple documents by their reference IDs in a single call. "
                        "PREFER this over calling memory_retrieve multiple times. "
                        "Valid ref_ids: 'hopping_summary' (the full summary), or original "
                        "episode IDs cited in the summary (e.g. 'scope_07_ep_005')."
                    ),
                    parameters={
                        "type": "object",
                        "properties": {
                            "ref_ids": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of reference IDs to retrieve.",
                            },
                        },
                        "required": ["ref_ids"],
                    },
                ),
            ],
        )

    def call_extended_tool(self, tool_name: str, arguments: dict) -> object:
        if tool_name == "batch_retrieve":
            ref_ids = arguments.get("ref_ids", [])
            docs = []
            for ref_id in ref_ids:
                doc = self.retrieve(ref_id)
                if doc is not None:
                    docs.append(doc.to_dict())
            return {"documents": docs, "count": len(docs)}
        return super().call_extended_tool(tool_name, arguments)

    def get_cache_state(self) -> dict | None:
        if not self._summary:
            return None
        return {
            "summary": self._summary,
            "episodes": self._episodes,
            "new_episodes": self._new_episodes,
            "cited_episode_ids": self._cited_episode_ids,
            "scope_id": self._scope_id,
        }

    def restore_cache_state(self, state: dict) -> bool:
        try:
            self._summary = state["summary"]
            self._episodes = state.get("episodes", [])
            self._new_episodes = state.get("new_episodes", [])
            self._cited_episode_ids = state.get("cited_episode_ids", [])
            self._scope_id = state.get("scope_id")
            self._scope_episodes = {}
            for ep in self._episodes:
                sid = ep.get("scope_id", self._scope_id)
                self._scope_episodes.setdefault(sid, []).append(ep)
            logger.info(
                "Restored Hopping cache: %d episodes, summary=%d chars",
                len(self._episodes),
                len(self._summary),
            )
            return True
        except Exception as e:
            logger.warning("Failed to restore Hopping cache: %s", e)
            return False


# ---------------------------------------------------------------------------
# HoppingRAGAdapter: summary + semantic search on raw episodes
# ---------------------------------------------------------------------------


@register_adapter("hopping-rag")
class HoppingRAGAdapter(HoppingAdapter):
    """Hopping summary + semantic search on raw episode embeddings.

    Search returns the summary first, followed by top-k embedding results
    from raw episodes. This tests whether combining incremental summarization
    with direct episode retrieval improves over summary-only.
    """

    def __init__(self) -> None:
        super().__init__()
        self._conn = sqlite3.connect(":memory:", check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._embed_model = os.environ.get(
            "LENS_EMBED_MODEL", _DEFAULT_OPENAI_EMBED_MODEL
        )
        self._embed_api_key: str | None = None
        self._embed_base_url = os.environ.get("LENS_EMBED_BASE_URL")
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

    def _embed(self, texts: list[str]) -> list[list[float]]:
        return _embed_texts_openai(
            texts, model=self._embed_model, api_key=self._embed_api_key,
            base_url=self._embed_base_url,
        )

    def reset(self, scope_id: str) -> None:
        super().reset(scope_id)
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
        super().ingest(episode_id, scope_id, timestamp, text, meta)
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
        """Semantic search on raw episode embeddings."""
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
        results: list[SearchResult] = []
        # Summary first
        if self._summary:
            results.append(SearchResult(
                ref_id="hopping_summary",
                text=self._summary[:500],
                score=1.0,
                metadata={"type": "hopping_summary", "cited_episodes": len(self._cited_episode_ids)},
            ))
        # Then top-k embedding results
        rag_limit = (limit or 5) - len(results)
        if rag_limit > 0:
            emb_results = self._embedding_search(query, filters, rag_limit)
            results.extend(emb_results)
        return results

    def retrieve(self, ref_id: str) -> Document | None:
        # Check summary first
        doc = super().retrieve(ref_id)
        if doc is not None:
            return doc
        # Fall back to SQLite
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
            search_modes=["hopping", "semantic"],
            max_results_per_search=5,
            extra_tools=[
                ExtraTool(
                    name="batch_retrieve",
                    description=(
                        "Retrieve multiple documents by their reference IDs in a single call. "
                        "PREFER this over calling memory_retrieve multiple times. "
                        "Valid ref_ids: 'hopping_summary' (the full summary), or original "
                        "episode IDs (e.g. 'scope_07_ep_005')."
                    ),
                    parameters={
                        "type": "object",
                        "properties": {
                            "ref_ids": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of reference IDs to retrieve.",
                            },
                        },
                        "required": ["ref_ids"],
                    },
                ),
            ],
        )

    def get_cache_state(self) -> dict | None:
        base = super().get_cache_state()
        if base is None:
            return None
        # Add SQLite episode data for restore
        cur = self._conn.cursor()
        cur.execute("SELECT episode_id, embedding FROM episode_embeddings")
        embeddings = {row["episode_id"]: row["embedding"] for row in cur.fetchall()}
        base["embeddings"] = embeddings
        return base

    def restore_cache_state(self, state: dict) -> bool:
        if not super().restore_cache_state(state):
            return False
        try:
            # Restore SQLite episode storage
            cur = self._conn.cursor()
            for ep in self._episodes:
                cur.execute(
                    "INSERT OR REPLACE INTO episodes (episode_id, scope_id, timestamp, text, meta) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (ep["episode_id"], ep["scope_id"], ep["timestamp"], ep["text"],
                     json.dumps(ep.get("meta", {}))),
                )
            embeddings = state.get("embeddings", {})
            for ep_id, emb_json in embeddings.items():
                cur.execute(
                    "INSERT OR REPLACE INTO episode_embeddings (episode_id, embedding) VALUES (?, ?)",
                    (ep_id, emb_json),
                )
            self._conn.commit()
            return True
        except Exception as e:
            logger.warning("Failed to restore HoppingRAG SQLite state: %s", e)
            return False


# ---------------------------------------------------------------------------
# HoppingHybridAdapter: summary + RRF(FTS + embedding) on raw episodes
# ---------------------------------------------------------------------------


@register_adapter("hopping-hybrid")
class HoppingHybridAdapter(HoppingRAGAdapter):
    """Hopping summary + RRF-fused (FTS5 + embedding) search on raw episodes.

    Adds FTS5 keyword search to the RAG variant and fuses with RRF for
    better recall on exact terms.
    """

    def __init__(self) -> None:
        # Don't call HoppingRAGAdapter.__init__ directly — we need different tables
        HoppingAdapter.__init__(self)
        self._conn = sqlite3.connect(":memory:", check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._embed_model = os.environ.get(
            "LENS_EMBED_MODEL", _DEFAULT_OPENAI_EMBED_MODEL
        )
        self._embed_api_key: str | None = None
        self._embed_base_url = os.environ.get("LENS_EMBED_BASE_URL")
        self._rrf_k = 60
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
        HoppingAdapter.reset(self, scope_id)
        cur = self._conn.cursor()
        cur.execute(
            "DELETE FROM episode_embeddings WHERE episode_id IN "
            "(SELECT episode_id FROM episodes WHERE scope_id = ?)",
            (scope_id,),
        )
        cur.execute("DELETE FROM episodes WHERE scope_id = ?", (scope_id,))
        self._conn.commit()

    def _fts_search(
        self, query: str, filters: dict | None, limit: int,
    ) -> list[SearchResult]:
        """BM25 keyword search on full episodes."""
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
            results.append(SearchResult(
                ref_id=row["episode_id"],
                text=row["text"][:500],
                score=abs(row["rank"]),
                metadata=meta,
            ))
        return results

    def search(
        self,
        query: str,
        filters: dict | None = None,
        limit: int | None = None,
    ) -> list[SearchResult]:
        results: list[SearchResult] = []
        # Summary first
        if self._summary:
            results.append(SearchResult(
                ref_id="hopping_summary",
                text=self._summary[:500],
                score=1.0,
                metadata={"type": "hopping_summary", "cited_episodes": len(self._cited_episode_ids)},
            ))

        # RRF-fused FTS + embedding results
        rag_limit = (limit or 5) - len(results)
        if rag_limit > 0:
            fetch_limit = rag_limit * 3
            fts_results = self._fts_search(query, filters, fetch_limit)
            emb_results = self._embedding_search(query, filters, fetch_limit)
            fused = _rrf_merge(fts_results, emb_results, k=self._rrf_k, limit=rag_limit)
            results.extend(fused)
        return results

    def get_capabilities(self) -> CapabilityManifest:
        return CapabilityManifest(
            search_modes=["hopping", "keyword", "semantic"],
            max_results_per_search=5,
            extra_tools=[
                ExtraTool(
                    name="batch_retrieve",
                    description=(
                        "Retrieve multiple documents by their reference IDs in a single call. "
                        "PREFER this over calling memory_retrieve multiple times. "
                        "Valid ref_ids: 'hopping_summary' (the full summary), or original "
                        "episode IDs (e.g. 'scope_07_ep_005')."
                    ),
                    parameters={
                        "type": "object",
                        "properties": {
                            "ref_ids": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of reference IDs to retrieve.",
                            },
                        },
                        "required": ["ref_ids"],
                    },
                ),
            ],
        )

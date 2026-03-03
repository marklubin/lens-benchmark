"""Hierarchical Summarization adapter for LENS.

Multi-level memory index built incrementally during prepare():

Level 0: Raw episodes (stored in-memory or SQLite)
Level 1: Per-episode summaries (~200 words each)
Level 2: Group summaries (GROUP_SIZE episodes per group, ~400 words each)
Level 3: Global summary (~800 words, updated incrementally)

Two variants:
- hierarchical:        summary-only (search returns global summary, retrieve drills down)
- hierarchical-hybrid: summary + RRF-fused (FTS5 + embedding) search across all levels

Environment variables:
    LENS_LLM_API_KEY / OPENAI_API_KEY   — API key for the summarization LLM
    LENS_LLM_API_BASE / OPENAI_BASE_URL — Base URL for the LLM API
    LENS_LLM_MODEL                      — Model name for summarization LLM
    HIERARCHICAL_GROUP_SIZE             — Episodes per group summary (default 4)
    HIERARCHICAL_MAX_TOKENS             — Max tokens for summary output (default 2000)
    LENS_EMBED_API_KEY / OPENAI_API_KEY — API key for embeddings (hybrid)
    LENS_EMBED_BASE_URL                 — Base URL for embedding API (hybrid)
    LENS_EMBED_MODEL                    — Embedding model name (hybrid)
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

logger = logging.getLogger(__name__)

_EP_ID_RE = re.compile(r"\[([^\]]+)\]")

_DEFAULT_OPENAI_EMBED_MODEL = "text-embedding-3-small"

GROUP_SIZE = int(os.environ.get("HIERARCHICAL_GROUP_SIZE", "4"))


def _strip_provider_prefix(model: str) -> str:
    """Convert 'together/Qwen/...' -> 'Qwen/...' for OpenAI API."""
    if "/" in model and model.startswith(("together/", "openai/")):
        return model.split("/", 1)[1]
    return model


# ---------------------------------------------------------------------------
# LLM Prompts
# ---------------------------------------------------------------------------

_EPISODE_SUMMARY_PROMPT = """\
Summarize this episode log in ~200 words. Preserve all entity names, \
numeric values, timestamps, and specific actions. Focus on WHO did WHAT \
and any changes from normal patterns.

EPISODE [{episode_id}]:
{text}

SUMMARY:"""

_GROUP_SUMMARY_PROMPT = """\
You have summaries of {n} consecutive episodes. Synthesize them into a \
~400 word group summary. Highlight patterns, progressions, and changes \
across episodes. Cite [episode_id] for specific claims.

EPISODE SUMMARIES:
{summaries_block}

GROUP SUMMARY:"""

_GLOBAL_SUMMARY_INITIAL_PROMPT = """\
GROUP SUMMARIES:
{groups_block}

Create a global summary (~800 words) synthesizing all information. \
Highlight overall patterns, progressions, and key events. \
Cite [episode_id] for specific claims.

GLOBAL SUMMARY:"""

_GLOBAL_SUMMARY_INCREMENTAL_PROMPT = """\
EXISTING GLOBAL SUMMARY:
{previous_global}

NEW GROUP SUMMARIES:
{new_groups_block}

Update the global summary incorporating the new information. ~800 words. \
Preserve important details from the existing summary. Cite [episode_id].

UPDATED GLOBAL SUMMARY:"""

_SUMMARIZATION_SYSTEM = (
    "You are a memory summarization agent. You create concise, accurate "
    "summaries preserving all entity names, numeric values, and specific "
    "actions. Never editorialize — report facts only."
)


# ---------------------------------------------------------------------------
# HierarchicalAdapter (summary-only)
# ---------------------------------------------------------------------------


@register_adapter("hierarchical")
class HierarchicalAdapter(MemoryAdapter):
    """Multi-level hierarchical summarization adapter.

    Builds 4 levels of summaries incrementally during prepare():
    - L0: Raw episodes
    - L1: Per-episode summaries
    - L2: Group summaries (GROUP_SIZE episodes per group)
    - L3: Global summary

    Search returns the global summary. Retrieve drills down by ref_id.
    """

    requires_metering: bool = True  # LLM calls in prepare()

    def __init__(self) -> None:
        self._episodes: list[dict] = []
        self._new_episodes: list[dict] = []
        self._episode_summaries: dict[str, str] = {}  # episode_id -> L1 summary
        self._group_summaries: dict[str, str] = {}  # group_N -> L2 summary
        self._group_episode_ids: dict[str, list[str]] = {}  # group_N -> [episode_ids]
        self._global_summary: str = ""
        self._scope_id: str | None = None
        self._scope_episodes: dict[str, list[dict]] = {}
        self._oai_client: object | None = None
        self._model: str = ""
        self._max_tokens = int(os.environ.get("HIERARCHICAL_MAX_TOKENS", "2000"))
        self._group_size = GROUP_SIZE
        self._last_processed_count: int = 0  # track which episodes have L1 summaries
        self._dirty_groups: set[str] = set()  # groups that need re-summarization

    def reset(self, scope_id: str) -> None:
        self._scope_episodes.pop(scope_id, None)
        self._episodes = []
        self._new_episodes = []
        self._episode_summaries = {}
        self._group_summaries = {}
        self._group_episode_ids = {}
        self._global_summary = ""
        self._scope_id = scope_id
        self._last_processed_count = 0
        self._dirty_groups = set()
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
            logger.error("openai package required for hierarchical adapter")
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

    def _llm_call(self, user_msg: str) -> str:
        """Single LLM call with retry."""
        self._ensure_client()
        if self._oai_client is None:
            return ""
        try:
            resp = self._oai_client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": _SUMMARIZATION_SYSTEM},
                    {"role": "user", "content": user_msg},
                ],
                max_tokens=self._max_tokens,
                temperature=0.0,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            logger.error("Hierarchical LLM call failed: %s", e)
            return ""

    def _episode_to_group_id(self, index: int) -> str:
        """Map episode index to group ID."""
        return f"group_{index // self._group_size}"

    def prepare(self, scope_id: str, checkpoint: int) -> None:
        """Build/update hierarchical summaries for new episodes."""
        if not self._new_episodes:
            return

        logger.info(
            "Hierarchical prepare: %d new episodes, %d total",
            len(self._new_episodes), len(self._episodes),
        )

        # Step 1: Generate L1 summaries for new episodes
        for ep in self._new_episodes:
            ep_id = ep["episode_id"]
            if ep_id in self._episode_summaries:
                continue
            prompt = _EPISODE_SUMMARY_PROMPT.format(
                episode_id=ep_id,
                text=ep["text"][:12000],  # truncate very long episodes
            )
            summary = self._llm_call(prompt)
            self._episode_summaries[ep_id] = summary
            logger.debug("L1 summary for %s: %d chars", ep_id, len(summary))

            # Mark the affected group as dirty
            ep_index = next(
                (i for i, e in enumerate(self._episodes) if e["episode_id"] == ep_id),
                len(self._episodes) - 1,
            )
            group_id = self._episode_to_group_id(ep_index)
            self._dirty_groups.add(group_id)

        # Step 2: Regenerate L2 group summaries for dirty groups
        for group_id in sorted(self._dirty_groups):
            group_idx = int(group_id.split("_")[1])
            start = group_idx * self._group_size
            end = min(start + self._group_size, len(self._episodes))
            group_eps = self._episodes[start:end]

            ep_ids = [ep["episode_id"] for ep in group_eps]
            self._group_episode_ids[group_id] = ep_ids

            summaries_block = "\n\n".join(
                f"[{eid}]: {self._episode_summaries.get(eid, '(no summary)')}"
                for eid in ep_ids
            )
            prompt = _GROUP_SUMMARY_PROMPT.format(
                n=len(ep_ids),
                summaries_block=summaries_block,
            )
            summary = self._llm_call(prompt)
            self._group_summaries[group_id] = summary
            logger.debug("L2 summary for %s (%d eps): %d chars", group_id, len(ep_ids), len(summary))

        # Step 3: Update L3 global summary
        if self._dirty_groups:
            if self._global_summary:
                new_groups_block = "\n\n".join(
                    f"[{gid}]: {self._group_summaries.get(gid, '')}"
                    for gid in sorted(self._dirty_groups)
                    if gid in self._group_summaries
                )
                prompt = _GLOBAL_SUMMARY_INCREMENTAL_PROMPT.format(
                    previous_global=self._global_summary,
                    new_groups_block=new_groups_block,
                )
            else:
                groups_block = "\n\n".join(
                    f"[{gid}]: {self._group_summaries.get(gid, '')}"
                    for gid in sorted(self._group_summaries)
                )
                prompt = _GLOBAL_SUMMARY_INITIAL_PROMPT.format(
                    groups_block=groups_block,
                )
            self._global_summary = self._llm_call(prompt)
            logger.info(
                "L3 global summary updated: %d chars, %d groups",
                len(self._global_summary), len(self._group_summaries),
            )

        # Clear state
        self._new_episodes = []
        self._dirty_groups = set()

    def search(
        self,
        query: str,
        filters: dict | None = None,
        limit: int | None = None,
    ) -> list[SearchResult]:
        results: list[SearchResult] = []
        if self._global_summary:
            results.append(SearchResult(
                ref_id="global_summary",
                text=self._global_summary[:500],
                score=1.0,
                metadata={"type": "global_summary", "level": 3},
            ))
        # Also return group summaries as search results
        for gid in sorted(self._group_summaries):
            if len(results) >= (limit or 10):
                break
            summary = self._group_summaries[gid]
            results.append(SearchResult(
                ref_id=gid,
                text=summary[:300],
                score=0.8,
                metadata={
                    "type": "group_summary",
                    "level": 2,
                    "episode_ids": self._group_episode_ids.get(gid, []),
                },
            ))
        return results

    def retrieve(self, ref_id: str) -> Document | None:
        # Global summary
        if ref_id == "global_summary":
            if self._global_summary:
                return Document(ref_id="global_summary", text=self._global_summary)
            return None

        # Group summary
        if ref_id in self._group_summaries:
            ep_ids = self._group_episode_ids.get(ref_id, [])
            return Document(
                ref_id=ref_id,
                text=self._group_summaries[ref_id],
                metadata={"type": "group_summary", "episode_ids": ep_ids},
            )

        # Episode summary
        if ref_id.startswith("summary_"):
            ep_id = ref_id[len("summary_"):]
            if ep_id in self._episode_summaries:
                return Document(
                    ref_id=ref_id,
                    text=self._episode_summaries[ep_id],
                    metadata={"type": "episode_summary"},
                )
            return None

        # Raw episode
        for ep in self._episodes:
            if ep["episode_id"] == ref_id:
                return Document(
                    ref_id=ref_id,
                    text=ep["text"],
                    metadata=ep.get("meta", {}),
                )
        return None

    def get_synthetic_refs(self) -> list[tuple[str, str]]:
        refs: list[tuple[str, str]] = []
        if self._global_summary:
            refs.append(("global_summary", self._global_summary))
        for gid, summary in self._group_summaries.items():
            refs.append((gid, summary))
        for ep_id, summary in self._episode_summaries.items():
            refs.append((f"summary_{ep_id}", summary))
        return refs

    def get_capabilities(self) -> CapabilityManifest:
        return CapabilityManifest(
            search_modes=["hierarchical"],
            max_results_per_search=10,
            extra_tools=[
                ExtraTool(
                    name="batch_retrieve",
                    description=(
                        "Retrieve multiple documents by their reference IDs in a single call. "
                        "PREFER this over calling memory_retrieve multiple times. "
                        "Valid ref_ids: 'global_summary' (L3), 'group_N' (L2 group summaries), "
                        "'summary_<episode_id>' (L1 episode summaries), or raw episode IDs."
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
        if not self._global_summary and not self._episode_summaries:
            return None
        return {
            "episodes": self._episodes,
            "new_episodes": self._new_episodes,
            "episode_summaries": self._episode_summaries,
            "group_summaries": self._group_summaries,
            "group_episode_ids": self._group_episode_ids,
            "global_summary": self._global_summary,
            "scope_id": self._scope_id,
            "last_processed_count": self._last_processed_count,
        }

    def restore_cache_state(self, state: dict) -> bool:
        try:
            self._episodes = state["episodes"]
            self._new_episodes = state.get("new_episodes", [])
            self._episode_summaries = state.get("episode_summaries", {})
            self._group_summaries = state.get("group_summaries", {})
            self._group_episode_ids = state.get("group_episode_ids", {})
            self._global_summary = state.get("global_summary", "")
            self._scope_id = state.get("scope_id")
            self._last_processed_count = state.get("last_processed_count", 0)
            self._scope_episodes = {}
            for ep in self._episodes:
                sid = ep.get("scope_id", self._scope_id)
                self._scope_episodes.setdefault(sid, []).append(ep)
            logger.info(
                "Restored Hierarchical cache: %d episodes, %d L1, %d L2, global=%d chars",
                len(self._episodes),
                len(self._episode_summaries),
                len(self._group_summaries),
                len(self._global_summary),
            )
            return True
        except Exception as e:
            logger.warning("Failed to restore Hierarchical cache: %s", e)
            return False


# ---------------------------------------------------------------------------
# HierarchicalHybridAdapter: multi-level + RRF(FTS + embedding) search
# ---------------------------------------------------------------------------


@register_adapter("hierarchical-hybrid")
class HierarchicalHybridAdapter(HierarchicalAdapter):
    """Hierarchical summaries + RRF-fused search across all levels.

    Adds FTS5 + embedding search on raw episodes, episode summaries,
    and group summaries. Results are RRF-fused across all levels.
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
        self._rrf_k = 60
        self._ensure_tables()

    def _ensure_tables(self) -> None:
        cur = self._conn.cursor()
        cur.executescript("""
            -- Level 0: raw episodes
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

            -- Level 1: episode summaries
            CREATE TABLE IF NOT EXISTS episode_summaries (
                episode_id TEXT PRIMARY KEY,
                summary    TEXT NOT NULL
            );
            CREATE VIRTUAL TABLE IF NOT EXISTS episode_summaries_fts
                USING fts5(episode_id, summary);
            CREATE TABLE IF NOT EXISTS episode_summary_embeddings (
                episode_id TEXT PRIMARY KEY,
                embedding  TEXT NOT NULL
            );

            -- Level 2: group summaries
            CREATE TABLE IF NOT EXISTS group_summaries (
                group_id    TEXT PRIMARY KEY,
                summary     TEXT NOT NULL,
                episode_ids TEXT DEFAULT '[]'
            );
            CREATE VIRTUAL TABLE IF NOT EXISTS group_summaries_fts
                USING fts5(group_id, summary);
            CREATE TABLE IF NOT EXISTS group_summary_embeddings (
                group_id  TEXT PRIMARY KEY,
                embedding TEXT NOT NULL
            );
        """)
        self._conn.commit()

    def _embed(self, texts: list[str]) -> list[list[float]]:
        from lens.adapters.sqlite_variants import _embed_texts_openai
        return _embed_texts_openai(
            texts, model=self._embed_model, api_key=self._embed_api_key,
            base_url=self._embed_base_url,
        )

    def reset(self, scope_id: str) -> None:
        super().reset(scope_id)
        cur = self._conn.cursor()
        # Clean all tables for this scope
        cur.execute(
            "DELETE FROM episode_embeddings WHERE episode_id IN "
            "(SELECT episode_id FROM episodes WHERE scope_id = ?)",
            (scope_id,),
        )
        cur.execute("DELETE FROM episodes WHERE scope_id = ?", (scope_id,))
        # Clear all summary tables (they'll be rebuilt in prepare)
        cur.execute("DELETE FROM episode_summaries")
        cur.execute("DELETE FROM episode_summaries_fts")
        cur.execute("DELETE FROM episode_summary_embeddings")
        cur.execute("DELETE FROM group_summaries")
        cur.execute("DELETE FROM group_summaries_fts")
        cur.execute("DELETE FROM group_summary_embeddings")
        self._conn.commit()

    def ingest(
        self,
        episode_id: str,
        scope_id: str,
        timestamp: str,
        text: str,
        meta: dict | None = None,
    ) -> None:
        # Buffer in parent
        super().ingest(episode_id, scope_id, timestamp, text, meta)
        # Store raw episode in SQLite + embed
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

    def prepare(self, scope_id: str, checkpoint: int) -> None:
        """Build hierarchical summaries, then index them in SQLite."""
        # Run parent prepare (generates L1, L2, L3 summaries)
        super().prepare(scope_id, checkpoint)

        # Index L1 summaries in SQLite
        cur = self._conn.cursor()
        for ep_id, summary in self._episode_summaries.items():
            cur.execute(
                "INSERT OR REPLACE INTO episode_summaries (episode_id, summary) VALUES (?, ?)",
                (ep_id, summary),
            )
            cur.execute(
                "INSERT OR REPLACE INTO episode_summaries_fts (episode_id, summary) VALUES (?, ?)",
                (ep_id, summary),
            )
        # Embed L1 summaries
        new_l1 = [
            (ep_id, s) for ep_id, s in self._episode_summaries.items()
            if s  # skip empty summaries
        ]
        if new_l1:
            ep_ids_l1, texts_l1 = zip(*new_l1)
            vectors = self._embed(list(texts_l1))
            for ep_id, vec in zip(ep_ids_l1, vectors):
                cur.execute(
                    "INSERT OR REPLACE INTO episode_summary_embeddings (episode_id, embedding) "
                    "VALUES (?, ?)",
                    (ep_id, json.dumps(vec)),
                )

        # Index L2 group summaries
        for gid, summary in self._group_summaries.items():
            ep_ids = self._group_episode_ids.get(gid, [])
            cur.execute(
                "INSERT OR REPLACE INTO group_summaries (group_id, summary, episode_ids) "
                "VALUES (?, ?, ?)",
                (gid, summary, json.dumps(ep_ids)),
            )
            cur.execute(
                "INSERT OR REPLACE INTO group_summaries_fts (group_id, summary) VALUES (?, ?)",
                (gid, summary),
            )
        # Embed L2 summaries
        new_l2 = [
            (gid, s) for gid, s in self._group_summaries.items()
            if s
        ]
        if new_l2:
            gids, texts_l2 = zip(*new_l2)
            vectors = self._embed(list(texts_l2))
            for gid, vec in zip(gids, vectors):
                cur.execute(
                    "INSERT OR REPLACE INTO group_summary_embeddings (group_id, embedding) "
                    "VALUES (?, ?)",
                    (gid, json.dumps(vec)),
                )

        self._conn.commit()

    def search(
        self,
        query: str,
        filters: dict | None = None,
        limit: int | None = None,
    ) -> list[SearchResult]:
        from lens.adapters.sqlite import _fts5_escape
        from lens.adapters.sqlite_variants import _rrf_merge, cosine_similarity

        cap = limit or 5
        results: list[SearchResult] = []

        # Always include global summary as first result
        if self._global_summary:
            results.append(SearchResult(
                ref_id="global_summary",
                text=self._global_summary[:500],
                score=1.0,
                metadata={"type": "global_summary", "level": 3},
            ))

        # Collect FTS results across all levels
        fts_results: list[SearchResult] = []
        safe_query = _fts5_escape(query)

        if safe_query:
            cur = self._conn.cursor()

            # FTS on raw episodes (L0)
            try:
                cur.execute(
                    "SELECT e.episode_id, e.text, e.meta, bm25(episodes_fts) AS rank "
                    "FROM episodes_fts f "
                    "JOIN episodes e ON e.episode_id = f.episode_id "
                    "WHERE episodes_fts MATCH ? "
                    "ORDER BY rank LIMIT ?",
                    (safe_query, cap * 3),
                )
                for row in cur.fetchall():
                    meta = json.loads(row["meta"]) if row["meta"] else {}
                    meta["level"] = 0
                    fts_results.append(SearchResult(
                        ref_id=row["episode_id"],
                        text=row["text"][:500],
                        score=abs(row["rank"]),
                        metadata=meta,
                    ))
            except sqlite3.OperationalError:
                pass

            # FTS on episode summaries (L1)
            try:
                cur.execute(
                    "SELECT episode_id, summary, bm25(episode_summaries_fts) AS rank "
                    "FROM episode_summaries_fts "
                    "WHERE episode_summaries_fts MATCH ? "
                    "ORDER BY rank LIMIT ?",
                    (safe_query, cap * 3),
                )
                for row in cur.fetchall():
                    fts_results.append(SearchResult(
                        ref_id=f"summary_{row['episode_id']}",
                        text=row["summary"][:300],
                        score=abs(row["rank"]),
                        metadata={"type": "episode_summary", "level": 1},
                    ))
            except sqlite3.OperationalError:
                pass

            # FTS on group summaries (L2)
            try:
                cur.execute(
                    "SELECT group_id, summary, bm25(group_summaries_fts) AS rank "
                    "FROM group_summaries_fts "
                    "WHERE group_summaries_fts MATCH ? "
                    "ORDER BY rank LIMIT ?",
                    (safe_query, cap * 3),
                )
                for row in cur.fetchall():
                    fts_results.append(SearchResult(
                        ref_id=row["group_id"],
                        text=row["summary"][:300],
                        score=abs(row["rank"]),
                        metadata={"type": "group_summary", "level": 2},
                    ))
            except sqlite3.OperationalError:
                pass

        # Collect embedding results across all levels
        emb_results: list[SearchResult] = []
        try:
            query_vec = self._embed([query])[0]
        except Exception as e:
            logger.warning("Embedding query failed, FTS-only: %s", e)
            query_vec = None

        if query_vec is not None:
            cur = self._conn.cursor()

            # Embedding on raw episodes (L0)
            cur.execute(
                "SELECT e.episode_id, e.text, e.meta, ee.embedding "
                "FROM episodes e "
                "JOIN episode_embeddings ee ON e.episode_id = ee.episode_id"
            )
            scored_l0 = []
            for row in cur.fetchall():
                emb = json.loads(row["embedding"])
                sim = cosine_similarity(query_vec, emb)
                meta = json.loads(row["meta"]) if row["meta"] else {}
                meta["level"] = 0
                scored_l0.append(SearchResult(
                    ref_id=row["episode_id"],
                    text=row["text"][:500],
                    score=sim,
                    metadata=meta,
                ))
            scored_l0.sort(key=lambda r: r.score, reverse=True)
            emb_results.extend(scored_l0[:cap * 3])

            # Embedding on episode summaries (L1)
            cur.execute(
                "SELECT episode_id, embedding FROM episode_summary_embeddings"
            )
            scored_l1 = []
            for row in cur.fetchall():
                emb = json.loads(row["embedding"])
                sim = cosine_similarity(query_vec, emb)
                # Get summary text for display
                ep_id = row["episode_id"]
                summary = self._episode_summaries.get(ep_id, "")
                scored_l1.append(SearchResult(
                    ref_id=f"summary_{ep_id}",
                    text=summary[:300],
                    score=sim,
                    metadata={"type": "episode_summary", "level": 1},
                ))
            scored_l1.sort(key=lambda r: r.score, reverse=True)
            emb_results.extend(scored_l1[:cap * 3])

            # Embedding on group summaries (L2)
            cur.execute(
                "SELECT group_id, embedding FROM group_summary_embeddings"
            )
            scored_l2 = []
            for row in cur.fetchall():
                emb = json.loads(row["embedding"])
                sim = cosine_similarity(query_vec, emb)
                gid = row["group_id"]
                summary = self._group_summaries.get(gid, "")
                scored_l2.append(SearchResult(
                    ref_id=gid,
                    text=summary[:300],
                    score=sim,
                    metadata={"type": "group_summary", "level": 2},
                ))
            scored_l2.sort(key=lambda r: r.score, reverse=True)
            emb_results.extend(scored_l2[:cap * 3])

        # RRF-fuse FTS + embedding results
        remaining = cap - len(results)
        if remaining > 0 and (fts_results or emb_results):
            fused = _rrf_merge(fts_results, emb_results, k=self._rrf_k, limit=remaining)
            results.extend(fused)

        return results

    def retrieve(self, ref_id: str) -> Document | None:
        # Check parent (global, group, episode summary, raw episode)
        doc = super().retrieve(ref_id)
        if doc is not None:
            return doc
        # Fall back to SQLite for raw episodes
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
            search_modes=["hierarchical", "keyword", "semantic"],
            max_results_per_search=10,
            extra_tools=[
                ExtraTool(
                    name="batch_retrieve",
                    description=(
                        "Retrieve multiple documents by their reference IDs in a single call. "
                        "PREFER this over calling memory_retrieve multiple times. "
                        "Valid ref_ids: 'global_summary' (L3), 'group_N' (L2 group summaries), "
                        "'summary_<episode_id>' (L1 episode summaries), or raw episode IDs."
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
        # Add SQLite episode embeddings for restore
        cur = self._conn.cursor()
        cur.execute("SELECT episode_id, embedding FROM episode_embeddings")
        base["episode_embeddings"] = {
            row["episode_id"]: row["embedding"] for row in cur.fetchall()
        }
        cur.execute("SELECT episode_id, embedding FROM episode_summary_embeddings")
        base["episode_summary_embeddings"] = {
            row["episode_id"]: row["embedding"] for row in cur.fetchall()
        }
        cur.execute("SELECT group_id, embedding FROM group_summary_embeddings")
        base["group_summary_embeddings"] = {
            row["group_id"]: row["embedding"] for row in cur.fetchall()
        }
        return base

    def restore_cache_state(self, state: dict) -> bool:
        if not super().restore_cache_state(state):
            return False
        try:
            cur = self._conn.cursor()
            # Restore raw episodes
            for ep in self._episodes:
                cur.execute(
                    "INSERT OR REPLACE INTO episodes "
                    "(episode_id, scope_id, timestamp, text, meta) VALUES (?, ?, ?, ?, ?)",
                    (ep["episode_id"], ep["scope_id"], ep["timestamp"], ep["text"],
                     json.dumps(ep.get("meta", {}))),
                )
            # Restore episode embeddings
            for ep_id, emb_json in state.get("episode_embeddings", {}).items():
                cur.execute(
                    "INSERT OR REPLACE INTO episode_embeddings (episode_id, embedding) "
                    "VALUES (?, ?)",
                    (ep_id, emb_json),
                )
            # Restore L1 summaries + embeddings
            for ep_id, summary in self._episode_summaries.items():
                cur.execute(
                    "INSERT OR REPLACE INTO episode_summaries (episode_id, summary) "
                    "VALUES (?, ?)",
                    (ep_id, summary),
                )
                cur.execute(
                    "INSERT OR REPLACE INTO episode_summaries_fts (episode_id, summary) "
                    "VALUES (?, ?)",
                    (ep_id, summary),
                )
            for ep_id, emb_json in state.get("episode_summary_embeddings", {}).items():
                cur.execute(
                    "INSERT OR REPLACE INTO episode_summary_embeddings "
                    "(episode_id, embedding) VALUES (?, ?)",
                    (ep_id, emb_json),
                )
            # Restore L2 summaries + embeddings
            for gid, summary in self._group_summaries.items():
                ep_ids = self._group_episode_ids.get(gid, [])
                cur.execute(
                    "INSERT OR REPLACE INTO group_summaries "
                    "(group_id, summary, episode_ids) VALUES (?, ?, ?)",
                    (gid, summary, json.dumps(ep_ids)),
                )
                cur.execute(
                    "INSERT OR REPLACE INTO group_summaries_fts (group_id, summary) "
                    "VALUES (?, ?)",
                    (gid, summary),
                )
            for gid, emb_json in state.get("group_summary_embeddings", {}).items():
                cur.execute(
                    "INSERT OR REPLACE INTO group_summary_embeddings "
                    "(group_id, embedding) VALUES (?, ?)",
                    (gid, emb_json),
                )
            self._conn.commit()
            return True
        except Exception as e:
            logger.warning("Failed to restore HierarchicalHybrid SQLite state: %s", e)
            return False

"""ResponseCache — SQLite-backed cache for LLM and embedding responses.

Two tables (llm_responses, embed_responses) with content-addressed keys.
WAL mode and check_same_thread=False for concurrent access.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from pathlib import Path
from typing import Any

# ── Key field tuples ─────────────────────────────────────────────────────
LLM_KEY_FIELDS: tuple[str, ...] = (
    "model", "messages", "tools", "tool_choice", "temperature", "seed", "max_tokens",
)
EMBED_KEY_FIELDS: tuple[str, ...] = ("model", "input")


# ── Helpers ──────────────────────────────────────────────────────────────

def _to_json(obj: Any) -> str:
    """Serialize *obj* to a JSON string.

    Handles objects with ``model_dump()`` (pydantic), ``to_dict()`` (legacy),
    plain dicts/lists, and plain strings.
    """
    if hasattr(obj, "model_dump"):
        obj = obj.model_dump()
    elif hasattr(obj, "to_dict"):
        obj = obj.to_dict()
    return json.dumps(obj, sort_keys=True, default=str)


def _try_json(s: str) -> Any:
    """Parse *s* as JSON.  Return the raw string on failure."""
    try:
        return json.loads(s)
    except (json.JSONDecodeError, TypeError):
        return s


# ── SQL DDL ──────────────────────────────────────────────────────────────

_CREATE_LLM = """
CREATE TABLE IF NOT EXISTS llm_responses (
    key               TEXT PRIMARY KEY,
    model             TEXT,
    request           TEXT,
    response          TEXT,
    created_at        REAL,
    latency_ms        REAL,
    prompt_tokens     INTEGER,
    completion_tokens INTEGER,
    hit_count         INTEGER DEFAULT 0
)
"""

_CREATE_EMBED = """
CREATE TABLE IF NOT EXISTS embed_responses (
    key          TEXT PRIMARY KEY,
    model        TEXT,
    request      TEXT,
    response     TEXT,
    created_at   REAL,
    latency_ms   REAL,
    token_count  INTEGER,
    hit_count    INTEGER DEFAULT 0
)
"""


# ── ResponseCache ────────────────────────────────────────────────────────

class ResponseCache:
    """Content-addressed SQLite cache for LLM and embedding API responses."""

    def __init__(self, db_path: str | Path) -> None:
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(_CREATE_LLM)
        self._conn.execute(_CREATE_EMBED)
        self._conn.commit()

    # ── Key computation ──────────────────────────────────────────────────

    @staticmethod
    def cache_key(request: dict[str, Any], key_fields: tuple[str, ...]) -> str:
        """Compute a content-addressed cache key from *request*.

        Only fields listed in *key_fields* participate.  The canonical
        representation is ``json.dumps(sorted_subset, sort_keys=True)``.
        """
        canonical = {k: request[k] for k in sorted(key_fields) if k in request}
        digest = hashlib.sha256(
            json.dumps(canonical, sort_keys=True, default=str).encode()
        ).hexdigest()[:16]
        return digest

    @staticmethod
    def llm_key(request: dict[str, Any]) -> str:
        """Shorthand: compute a cache key using LLM key fields."""
        return ResponseCache.cache_key(request, LLM_KEY_FIELDS)

    @staticmethod
    def embed_key(request: dict[str, Any]) -> str:
        """Shorthand: compute a cache key using embedding key fields."""
        return ResponseCache.cache_key(request, EMBED_KEY_FIELDS)

    # ── LLM operations ───────────────────────────────────────────────────

    def get_llm(self, key: str) -> dict[str, Any] | None:
        """Fetch an LLM response by *key*.  Returns ``None`` on miss.

        On hit, ``hit_count`` is incremented *before* the row is returned.
        """
        self._conn.execute(
            "UPDATE llm_responses SET hit_count = hit_count + 1 WHERE key = ?",
            (key,),
        )
        self._conn.commit()
        row = self._conn.execute(
            "SELECT key, model, request, response, created_at, latency_ms, "
            "prompt_tokens, completion_tokens, hit_count "
            "FROM llm_responses WHERE key = ?",
            (key,),
        ).fetchone()
        if row is None:
            return None
        return {
            "key": row[0],
            "model": row[1],
            "request": _try_json(row[2]),
            "response": _try_json(row[3]),
            "created_at": row[4],
            "latency_ms": row[5],
            "prompt_tokens": row[6],
            "completion_tokens": row[7],
            "hit_count": row[8],
        }

    def put_llm(
        self,
        key: str,
        *,
        model: str,
        request: Any,
        response: Any,
        latency_ms: float,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> None:
        """Insert or replace an LLM response entry."""
        self._conn.execute(
            "INSERT OR REPLACE INTO llm_responses "
            "(key, model, request, response, created_at, latency_ms, "
            "prompt_tokens, completion_tokens, hit_count) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0)",
            (
                key,
                model,
                _to_json(request),
                _to_json(response),
                time.time(),
                latency_ms,
                prompt_tokens,
                completion_tokens,
            ),
        )
        self._conn.commit()

    # ── Embed operations ─────────────────────────────────────────────────

    def get_embed(self, key: str) -> dict[str, Any] | None:
        """Fetch an embedding response by *key*.  Returns ``None`` on miss.

        On hit, ``hit_count`` is incremented *before* the row is returned.
        """
        self._conn.execute(
            "UPDATE embed_responses SET hit_count = hit_count + 1 WHERE key = ?",
            (key,),
        )
        self._conn.commit()
        row = self._conn.execute(
            "SELECT key, model, request, response, created_at, latency_ms, "
            "token_count, hit_count "
            "FROM embed_responses WHERE key = ?",
            (key,),
        ).fetchone()
        if row is None:
            return None
        return {
            "key": row[0],
            "model": row[1],
            "request": _try_json(row[2]),
            "response": _try_json(row[3]),
            "created_at": row[4],
            "latency_ms": row[5],
            "token_count": row[6],
            "hit_count": row[7],
        }

    def put_embed(
        self,
        key: str,
        *,
        model: str,
        request: Any,
        response: Any,
        latency_ms: float,
        token_count: int,
    ) -> None:
        """Insert or replace an embedding response entry."""
        self._conn.execute(
            "INSERT OR REPLACE INTO embed_responses "
            "(key, model, request, response, created_at, latency_ms, "
            "token_count, hit_count) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, 0)",
            (
                key,
                model,
                _to_json(request),
                _to_json(response),
                time.time(),
                latency_ms,
                token_count,
            ),
        )
        self._conn.commit()

    # ── Stats ────────────────────────────────────────────────────────────

    def llm_stats(self) -> dict[str, Any]:
        """Aggregate statistics for the LLM response cache."""
        row = self._conn.execute(
            "SELECT COUNT(*), COALESCE(SUM(prompt_tokens), 0), "
            "COALESCE(SUM(completion_tokens), 0), COALESCE(SUM(latency_ms), 0.0) "
            "FROM llm_responses"
        ).fetchone()
        return {
            "total_entries": row[0],
            "total_prompt_tokens": row[1],
            "total_completion_tokens": row[2],
            "total_latency_ms": row[3],
        }

    def embed_stats(self) -> dict[str, Any]:
        """Aggregate statistics for the embedding response cache."""
        row = self._conn.execute(
            "SELECT COUNT(*), COALESCE(SUM(token_count), 0), "
            "COALESCE(SUM(latency_ms), 0.0) "
            "FROM embed_responses"
        ).fetchone()
        return {
            "total_entries": row[0],
            "total_tokens": row[1],
            "total_latency_ms": row[2],
        }

    # ── Lifecycle ────────────────────────────────────────────────────────

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        self._conn.close()

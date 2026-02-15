from __future__ import annotations

import sqlite3

from lens.adapters.base import (
    CapabilityManifest,
    Document,
    FilterField,
    MemoryAdapter,
    SearchResult,
)
from lens.adapters.registry import register_adapter


@register_adapter("sqlite")
class SQLiteAdapter(MemoryAdapter):
    """SQLite + FTS5 memory adapter. Stdlib only, no external dependencies.

    Uses an in-memory SQLite database by default. FTS5 provides BM25-ranked
    full-text search over ingested episode text.
    """

    def __init__(self, db_path: str = ":memory:") -> None:
        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
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
        """)
        self._conn.commit()

    def reset(self, scope_id: str) -> None:
        cur = self._conn.cursor()
        # Get rowids to delete from FTS
        cur.execute("SELECT rowid, episode_id, text FROM episodes WHERE scope_id = ?", (scope_id,))
        rows = cur.fetchall()
        for row in rows:
            cur.execute(
                "INSERT INTO episodes_fts(episodes_fts, rowid, episode_id, text) "
                "VALUES ('delete', ?, ?, ?)",
                (row[0], row[1], row[2]),
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
        import json

        cur = self._conn.cursor()
        cur.execute(
            "INSERT OR REPLACE INTO episodes (episode_id, scope_id, timestamp, text, meta) "
            "VALUES (?, ?, ?, ?, ?)",
            (episode_id, scope_id, timestamp, text, json.dumps(meta or {})),
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
        # Escape FTS5 special characters and build a simple query
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
            # Malformed FTS query — return empty
            return []

        results = []
        for row in cur.fetchall():
            import json

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

    def retrieve(self, ref_id: str) -> Document | None:
        import json

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
            search_modes=["keyword"],
            filter_fields=[
                FilterField(name="scope_id", field_type="string", description="Filter by scope ID"),
                FilterField(name="start_date", field_type="string", description="Filter episodes after this ISO date"),
                FilterField(name="end_date", field_type="string", description="Filter episodes before this ISO date"),
            ],
            max_results_per_search=10,
            supports_date_range=True,
            extra_tools=[],
        )


def _fts5_escape(query: str) -> str:
    """Escape a user query for FTS5 MATCH.

    Wraps each word in double-quotes to prevent FTS5 syntax errors
    from special characters like *, -, etc.
    """
    words = query.split()
    if not words:
        return ""
    # Quote each token and join with OR for broader matching
    escaped = " OR ".join(f'"{w}"' for w in words if w.strip())
    return escaped

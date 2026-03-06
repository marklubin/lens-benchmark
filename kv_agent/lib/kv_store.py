import sqlite3
import os
from typing import Any, Dict, List, Optional
import json
import re
import logging
from nltk.stem import PorterStemmer


class KVStore:
    """SQLite-based key-value store for agent memory."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.stemmer = PorterStemmer()
        self._init_db()

    def _init_db(self):
        """Initialize the database schema."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS kv_store (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_updated_at ON kv_store (updated_at)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_key ON kv_store (key)
            """)

    def _canonicalize_key(self, entity_type: str, entity_name: str) -> str:
        """Canonicalize key format: {ENTITY_TYPE}:{ENTITY_NAME} in uppercase with stemming."""
        # Convert to uppercase
        entity_type = entity_type.upper()
        entity_name = entity_name.upper()

        # Remove special characters and extra whitespace
        entity_name = re.sub(r"[^A-Z0-9\s]", "", entity_name)
        entity_name = re.sub(r"\s+", " ", entity_name).strip()

        # Apply stemming if enabled
        words = entity_name.split()
        stemmed_words = [self.stemmer.stem(word) for word in words]
        entity_name = " ".join(stemmed_words)

        # Use shortest well-understood naming convention
        # Common abbreviations
        abbreviations = {
            "SAN FRANCISCO": "SAN_FRANCISCO",
            "NEW YORK": "NYC",
            "LOS ANGELES": "LA",
            "WASHINGTON DC": "DC",
            "UNITED STATES": "USA",
            "UNITED KINGDOM": "UK",
        }

        for full, abbr in abbreviations.items():
            if entity_name == full:
                entity_name = abbr
                break

        # Replace spaces with underscores
        entity_name = entity_name.replace(" ", "_")

        # Limit length
        entity_name = entity_name[:32]

        return f"{entity_type}:{entity_name}"

    def read(self, key: str) -> Optional[Any]:
        """Read a value from the store."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT value FROM kv_store WHERE key = ?", (key,))
            row = cursor.fetchone()
            if row:
                return json.loads(row[0])
            return None

    def write(self, key: str, value: Any) -> bool:
        """Write a value to the store."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO kv_store (key, value, updated_at)
                    VALUES (?, ?, CURRENT_TIMESTAMP)
                """,
                    (key, json.dumps(value)),
                )
            return True
        except Exception as e:
            logging.error(f"Failed to write to KV store: {e}")
            return False

    def search(self, pattern: str) -> List[Dict[str, Any]]:
        """Search for keys matching a pattern."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT key, value, updated_at FROM kv_store WHERE key LIKE ? ORDER BY updated_at DESC",
                (f"%{pattern}%",),
            )
            results = []
            for row in cursor.fetchall():
                results.append({"key": row[0], "value": json.loads(row[1]), "updated_at": row[2]})
            return results

    def get_all_keys(self) -> List[str]:
        """Get all keys in the store."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT key FROM kv_store ORDER BY updated_at DESC")
            return [row[0] for row in cursor.fetchall()]

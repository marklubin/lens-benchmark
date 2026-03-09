"""Tests for bench.cache — ResponseCache with LLM and embedding tables."""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

import pytest

from bench.cache import ResponseCache, _to_json, _try_json


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

class TestCacheSchema:
    def test_creates_tables_on_init(self, tmp_db: Path) -> None:
        cache = ResponseCache(tmp_db)
        conn = sqlite3.connect(str(tmp_db))
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        conn.close()
        cache.close()
        assert "llm_responses" in tables
        assert "embed_responses" in tables

    def test_wal_mode_enabled(self, tmp_db: Path) -> None:
        cache = ResponseCache(tmp_db)
        conn = sqlite3.connect(str(tmp_db))
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        conn.close()
        cache.close()
        assert mode == "wal"


# ---------------------------------------------------------------------------
# Cache key
# ---------------------------------------------------------------------------

LLM_KEY_FIELDS = ("model", "messages", "tools", "tool_choice", "temperature", "seed", "max_tokens")
EMBED_KEY_FIELDS = ("model", "input")


class TestCacheKey:
    def test_same_request_same_key(self) -> None:
        req = {"model": "gpt-4", "messages": [{"role": "user", "content": "hi"}]}
        k1 = ResponseCache.cache_key(req, LLM_KEY_FIELDS)
        k2 = ResponseCache.cache_key(req, LLM_KEY_FIELDS)
        assert k1 == k2

    def test_different_request_different_key(self) -> None:
        req_a = {"model": "gpt-4", "messages": [{"role": "user", "content": "hi"}]}
        req_b = {"model": "gpt-4", "messages": [{"role": "user", "content": "bye"}]}
        assert ResponseCache.cache_key(req_a, LLM_KEY_FIELDS) != ResponseCache.cache_key(req_b, LLM_KEY_FIELDS)

    def test_key_excludes_ephemeral_fields(self) -> None:
        base = {"model": "gpt-4", "messages": [{"role": "user", "content": "hi"}]}
        with_ephemeral = {**base, "stream": True, "timeout": 30, "top_p": 0.9}
        assert ResponseCache.cache_key(base, LLM_KEY_FIELDS) == ResponseCache.cache_key(with_ephemeral, LLM_KEY_FIELDS)

    def test_key_is_deterministic_across_key_order(self) -> None:
        """Dict insertion order must not affect the cache key."""
        req_a = {"model": "gpt-4", "messages": [], "temperature": 0.7}
        req_b = {"temperature": 0.7, "model": "gpt-4", "messages": []}
        assert ResponseCache.cache_key(req_a, LLM_KEY_FIELDS) == ResponseCache.cache_key(req_b, LLM_KEY_FIELDS)


# ---------------------------------------------------------------------------
# Static shorthand helpers
# ---------------------------------------------------------------------------

class TestKeyShorthands:
    def test_llm_key_matches_cache_key(self) -> None:
        req = {"model": "gpt-4", "messages": [{"role": "user", "content": "hi"}]}
        assert ResponseCache.llm_key(req) == ResponseCache.cache_key(req, LLM_KEY_FIELDS)

    def test_embed_key_matches_cache_key(self) -> None:
        req = {"model": "text-embedding-3-small", "input": "hello"}
        assert ResponseCache.embed_key(req) == ResponseCache.cache_key(req, EMBED_KEY_FIELDS)


# ---------------------------------------------------------------------------
# LLM cache operations
# ---------------------------------------------------------------------------

class TestLLMCacheOps:
    def test_miss_returns_none(self, tmp_db: Path) -> None:
        cache = ResponseCache(tmp_db)
        assert cache.get_llm("nonexistent_key") is None
        cache.close()

    def test_put_then_get_returns_entry(self, tmp_db: Path) -> None:
        cache = ResponseCache(tmp_db)
        req = {"model": "gpt-4", "messages": [{"role": "user", "content": "hi"}]}
        resp = {"id": "chatcmpl-1", "choices": [{"message": {"content": "hello"}}]}
        key = ResponseCache.llm_key(req)

        cache.put_llm(
            key,
            model="gpt-4",
            request=req,
            response=resp,
            latency_ms=123.4,
            prompt_tokens=10,
            completion_tokens=5,
        )

        entry = cache.get_llm(key)
        assert entry is not None
        assert entry["key"] == key
        assert entry["model"] == "gpt-4"
        assert entry["request"] == req
        assert entry["response"] == resp
        assert entry["latency_ms"] == pytest.approx(123.4)
        assert entry["prompt_tokens"] == 10
        assert entry["completion_tokens"] == 5
        assert isinstance(entry["created_at"], float)
        cache.close()

    def test_hit_count_increments(self, tmp_db: Path) -> None:
        cache = ResponseCache(tmp_db)
        req = {"model": "gpt-4", "messages": [{"role": "user", "content": "count"}]}
        key = ResponseCache.llm_key(req)

        cache.put_llm(
            key,
            model="gpt-4",
            request=req,
            response={"choices": []},
            latency_ms=50.0,
            prompt_tokens=5,
            completion_tokens=2,
        )

        # After put, hit_count should be 0.
        entry0 = cache.get_llm(key)
        assert entry0 is not None
        assert entry0["hit_count"] == 1  # first get increments from 0 -> 1

        entry1 = cache.get_llm(key)
        assert entry1 is not None
        assert entry1["hit_count"] == 2  # second get increments from 1 -> 2

        cache.close()


# ---------------------------------------------------------------------------
# Embed cache operations
# ---------------------------------------------------------------------------

class TestEmbedCacheOps:
    def test_miss_returns_none(self, tmp_db: Path) -> None:
        cache = ResponseCache(tmp_db)
        assert cache.get_embed("nonexistent_key") is None
        cache.close()

    def test_put_then_get_returns_entry(self, tmp_db: Path) -> None:
        cache = ResponseCache(tmp_db)
        req = {"model": "text-embedding-3-small", "input": "hello world"}
        resp = {"data": [{"embedding": [0.1, 0.2, 0.3]}]}
        key = ResponseCache.embed_key(req)

        cache.put_embed(
            key,
            model="text-embedding-3-small",
            request=req,
            response=resp,
            latency_ms=45.6,
            token_count=3,
        )

        entry = cache.get_embed(key)
        assert entry is not None
        assert entry["key"] == key
        assert entry["model"] == "text-embedding-3-small"
        assert entry["request"] == req
        assert entry["response"] == resp
        assert entry["latency_ms"] == pytest.approx(45.6)
        assert entry["token_count"] == 3
        assert isinstance(entry["created_at"], float)
        cache.close()


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

class TestCacheStats:
    def test_llm_stats(self, tmp_db: Path) -> None:
        cache = ResponseCache(tmp_db)
        for i in range(2):
            cache.put_llm(
                f"key_{i}",
                model="gpt-4",
                request={"i": i},
                response={"r": i},
                latency_ms=100.0 * (i + 1),
                prompt_tokens=10 * (i + 1),
                completion_tokens=5 * (i + 1),
            )

        stats = cache.llm_stats()
        assert stats["total_entries"] == 2
        assert stats["total_prompt_tokens"] == 30  # 10 + 20
        assert stats["total_completion_tokens"] == 15  # 5 + 10
        assert stats["total_latency_ms"] == pytest.approx(300.0)  # 100 + 200
        cache.close()

    def test_embed_stats(self, tmp_db: Path) -> None:
        cache = ResponseCache(tmp_db)
        for i in range(2):
            cache.put_embed(
                f"key_{i}",
                model="embed-v1",
                request={"i": i},
                response={"r": i},
                latency_ms=50.0 * (i + 1),
                token_count=8 * (i + 1),
            )

        stats = cache.embed_stats()
        assert stats["total_entries"] == 2
        assert stats["total_tokens"] == 24  # 8 + 16
        assert stats["total_latency_ms"] == pytest.approx(150.0)  # 50 + 100
        cache.close()


# ---------------------------------------------------------------------------
# Corruption resilience
# ---------------------------------------------------------------------------

class TestCacheCorruption:
    def test_get_on_corrupt_json_returns_entry(self, tmp_db: Path) -> None:
        """Directly insert non-JSON request field; get_llm must still return the row."""
        cache = ResponseCache(tmp_db)

        # Bypass the cache API and insert a row with invalid JSON in request.
        conn = sqlite3.connect(str(tmp_db))
        conn.execute(
            """INSERT INTO llm_responses
               (key, model, request, response, created_at, latency_ms,
                prompt_tokens, completion_tokens, hit_count)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            ("corrupt_key", "gpt-4", "NOT-VALID-JSON{{", '{"ok": true}', time.time(), 99.0, 7, 3, 0),
        )
        conn.commit()
        conn.close()

        entry = cache.get_llm("corrupt_key")
        assert entry is not None
        assert entry["key"] == "corrupt_key"
        # The corrupt request comes back as the raw string.
        assert entry["request"] == "NOT-VALID-JSON{{"
        # The valid response is parsed normally.
        assert entry["response"] == {"ok": True}
        cache.close()


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_to_json_dict(self) -> None:
        assert json.loads(_to_json({"a": 1})) == {"a": 1}

    def test_to_json_str(self) -> None:
        assert _to_json("raw string") == '"raw string"'

    def test_to_json_model_dump(self) -> None:
        class FakeModel:
            def model_dump(self):
                return {"x": 42}

        assert json.loads(_to_json(FakeModel())) == {"x": 42}

    def test_to_json_to_dict(self) -> None:
        class Legacy:
            def to_dict(self):
                return {"y": 99}

        assert json.loads(_to_json(Legacy())) == {"y": 99}

    def test_try_json_valid(self) -> None:
        assert _try_json('{"a": 1}') == {"a": 1}

    def test_try_json_invalid(self) -> None:
        assert _try_json("NOT JSON") == "NOT JSON"


# ---------------------------------------------------------------------------
# Concurrent write safety
# ---------------------------------------------------------------------------

import threading


class TestCacheConcurrency:
    def test_concurrent_writes_no_corruption(self, tmp_db: Path) -> None:
        cache = ResponseCache(tmp_db)
        errors: list[Exception] = []

        def writer(n: int) -> None:
            try:
                for i in range(20):
                    key = f"thread{n}_entry{i}"
                    cache.put_llm(
                        key,
                        model="m",
                        request={"i": i},
                        response={"r": n},
                        latency_ms=1.0,
                        prompt_tokens=1,
                        completion_tokens=1,
                    )
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(n,)) for n in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Concurrent write errors: {errors}"
        stats = cache.llm_stats()
        assert stats["total_entries"] == 80  # 4 threads * 20 entries
        cache.close()

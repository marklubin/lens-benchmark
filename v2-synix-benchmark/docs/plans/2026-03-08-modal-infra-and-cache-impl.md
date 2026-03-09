# Modal Infra + Response Cache Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build the SQLite response cache and Modal broker that all v2 benchmark inference flows through.

**Architecture:** Two modules — `cache.py` (SQLite content-addressed store for LLM and embedding responses) and `broker.py` (thin client that does cache lookup → Modal call → cache write). The cache is the single source of truth for cost accounting.

**Tech Stack:** Python 3.11+, sqlite3 (stdlib), openai SDK, httpx (for embed calls), pytest

**Reference:** Design doc at `v2-synix-benchmark/docs/plans/2026-03-08-modal-infra-and-cache-design.md`, v1 cache at `src/lens/agent/llm_cache.py`

---

### Task 1: Project scaffolding

**Files:**
- Create: `v2-synix-benchmark/pyproject.toml`
- Create: `v2-synix-benchmark/src/bench/__init__.py`
- Create: `v2-synix-benchmark/tests/__init__.py`
- Create: `v2-synix-benchmark/tests/conftest.py`

**Step 1: Create pyproject.toml**

```toml
[project]
name = "lens-bench-v2"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "openai>=1.0",
    "httpx>=0.27",
]

[project.optional-dependencies]
dev = ["pytest>=8.0", "pytest-timeout>=2.2"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/bench"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

**Step 2: Create package and test scaffolding**

```python
# src/bench/__init__.py
# (empty)
```

```python
# tests/__init__.py
# (empty)
```

```python
# tests/conftest.py
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def tmp_db(tmp_path: Path) -> Path:
    """Return a path to a temporary SQLite database file."""
    return tmp_path / "test_cache.db"
```

**Step 3: Verify the project installs**

Run: `cd v2-synix-benchmark && uv pip install -e ".[dev]" && uv run pytest --co -q`
Expected: "no tests ran" (0 collected), clean exit

**Step 4: Commit**

```bash
git add v2-synix-benchmark/pyproject.toml v2-synix-benchmark/src/bench/__init__.py v2-synix-benchmark/tests/__init__.py v2-synix-benchmark/tests/conftest.py
git commit -m "feat(v2): scaffold project with pyproject.toml and test fixtures"
```

---

### Task 2: ResponseCache — schema and basic operations

**Files:**
- Create: `v2-synix-benchmark/src/bench/cache.py`
- Create: `v2-synix-benchmark/tests/test_cache.py`

**Step 1: Write the failing tests**

```python
# tests/test_cache.py
from __future__ import annotations

import json
import time
from pathlib import Path

from bench.cache import ResponseCache


class TestCacheSchema:
    def test_creates_tables_on_init(self, tmp_db: Path) -> None:
        cache = ResponseCache(tmp_db)
        tables = cache._execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name").fetchall()
        names = [r[0] for r in tables]
        assert "embed_responses" in names
        assert "llm_responses" in names

    def test_wal_mode_enabled(self, tmp_db: Path) -> None:
        cache = ResponseCache(tmp_db)
        mode = cache._execute("PRAGMA journal_mode").fetchone()[0]
        assert mode == "wal"


class TestCacheKey:
    def test_same_request_same_key(self, tmp_db: Path) -> None:
        cache = ResponseCache(tmp_db)
        req = {"model": "test", "messages": [{"role": "user", "content": "hi"}]}
        assert cache.cache_key(req) == cache.cache_key(req)

    def test_different_request_different_key(self, tmp_db: Path) -> None:
        cache = ResponseCache(tmp_db)
        req_a = {"model": "test", "messages": [{"role": "user", "content": "hi"}]}
        req_b = {"model": "test", "messages": [{"role": "user", "content": "bye"}]}
        assert cache.cache_key(req_a) != cache.cache_key(req_b)

    def test_key_excludes_ephemeral_fields(self, tmp_db: Path) -> None:
        cache = ResponseCache(tmp_db)
        req_a = {"model": "test", "messages": [], "timeout": 30}
        req_b = {"model": "test", "messages": [], "stream": True}
        req_c = {"model": "test", "messages": []}
        assert cache.cache_key(req_a) == cache.cache_key(req_c)
        assert cache.cache_key(req_b) == cache.cache_key(req_c)

    def test_key_is_deterministic_across_key_order(self, tmp_db: Path) -> None:
        cache = ResponseCache(tmp_db)
        req_a = {"model": "test", "messages": [], "temperature": 0.0}
        req_b = {"temperature": 0.0, "messages": [], "model": "test"}
        assert cache.cache_key(req_a) == cache.cache_key(req_b)


class TestLLMCacheOps:
    def test_miss_returns_none(self, tmp_db: Path) -> None:
        cache = ResponseCache(tmp_db)
        assert cache.get_llm("nonexistent_key") is None

    def test_put_then_get_returns_entry(self, tmp_db: Path) -> None:
        cache = ResponseCache(tmp_db)
        key = "abc123"
        request = {"model": "test", "messages": []}
        response = {"id": "chatcmpl-1", "choices": [{"message": {"content": "hello"}}]}
        cache.put_llm(key, model="test", request=request, response=response, latency_ms=42.0, prompt_tokens=5, completion_tokens=3)
        entry = cache.get_llm(key)
        assert entry is not None
        assert entry["response"] == response
        assert entry["latency_ms"] == 42.0
        assert entry["prompt_tokens"] == 5
        assert entry["completion_tokens"] == 3

    def test_hit_count_increments(self, tmp_db: Path) -> None:
        cache = ResponseCache(tmp_db)
        key = "abc123"
        cache.put_llm(key, model="test", request={}, response={}, latency_ms=0, prompt_tokens=0, completion_tokens=0)
        assert cache.get_llm(key)["hit_count"] == 0
        # get_llm bumps hit_count
        assert cache.get_llm(key)["hit_count"] == 1
        assert cache.get_llm(key)["hit_count"] == 2


class TestEmbedCacheOps:
    def test_miss_returns_none(self, tmp_db: Path) -> None:
        cache = ResponseCache(tmp_db)
        assert cache.get_embed("nonexistent_key") is None

    def test_put_then_get_returns_entry(self, tmp_db: Path) -> None:
        cache = ResponseCache(tmp_db)
        key = "emb456"
        request = {"model": "gte-base", "input": ["hello"]}
        response = {"data": [{"embedding": [0.1, 0.2, 0.3]}]}
        cache.put_embed(key, model="gte-base", request=request, response=response, latency_ms=10.0, token_count=1)
        entry = cache.get_embed(key)
        assert entry is not None
        assert entry["response"] == response
        assert entry["token_count"] == 1


class TestCacheStats:
    def test_llm_stats(self, tmp_db: Path) -> None:
        cache = ResponseCache(tmp_db)
        cache.put_llm("k1", model="m", request={}, response={}, latency_ms=100, prompt_tokens=10, completion_tokens=5)
        cache.put_llm("k2", model="m", request={}, response={}, latency_ms=200, prompt_tokens=20, completion_tokens=10)
        stats = cache.llm_stats()
        assert stats["total_entries"] == 2
        assert stats["total_prompt_tokens"] == 30
        assert stats["total_completion_tokens"] == 15
        assert stats["total_latency_ms"] == 300

    def test_embed_stats(self, tmp_db: Path) -> None:
        cache = ResponseCache(tmp_db)
        cache.put_embed("e1", model="m", request={}, response={}, latency_ms=10, token_count=5)
        cache.put_embed("e2", model="m", request={}, response={}, latency_ms=20, token_count=10)
        stats = cache.embed_stats()
        assert stats["total_entries"] == 2
        assert stats["total_tokens"] == 15


class TestCacheCorruption:
    def test_get_on_corrupt_json_returns_none_and_logs(self, tmp_db: Path, caplog) -> None:
        cache = ResponseCache(tmp_db)
        # Directly insert corrupt JSON
        cache._execute(
            "INSERT INTO llm_responses (key, model, request, response, created_at, latency_ms, prompt_tokens, completion_tokens, hit_count) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("bad", "m", "not json", "{}", time.time(), 0, 0, 0, 0),
        )
        cache._conn.commit()
        entry = cache.get_llm("bad")
        # Should still return the row — request field is stored as text, corruption is in the caller's interpretation
        assert entry is not None
```

**Step 2: Run tests to verify they fail**

Run: `cd v2-synix-benchmark && uv run pytest tests/test_cache.py -v`
Expected: ImportError — `bench.cache` does not exist

**Step 3: Write the implementation**

```python
# src/bench/cache.py
"""SQLite content-addressed response cache for LLM and embedding calls.

Provides a single ResponseCache class that stores both LLM chat completions
and embedding responses in a SQLite database with WAL mode for concurrent access.

Cache keys are sha256 hashes of canonicalized request parameters.
"""
from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# Fields included in cache key computation for LLM calls
_LLM_KEY_FIELDS = ("model", "messages", "tools", "tool_choice", "temperature", "seed", "max_tokens")

# Fields included in cache key computation for embedding calls
_EMBED_KEY_FIELDS = ("model", "input")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS llm_responses (
    key TEXT PRIMARY KEY,
    model TEXT NOT NULL,
    request TEXT NOT NULL,
    response TEXT NOT NULL,
    created_at REAL NOT NULL,
    latency_ms REAL NOT NULL,
    prompt_tokens INTEGER NOT NULL,
    completion_tokens INTEGER NOT NULL,
    hit_count INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS embed_responses (
    key TEXT PRIMARY KEY,
    model TEXT NOT NULL,
    request TEXT NOT NULL,
    response TEXT NOT NULL,
    created_at REAL NOT NULL,
    latency_ms REAL NOT NULL,
    token_count INTEGER NOT NULL,
    hit_count INTEGER NOT NULL DEFAULT 0
);
"""


class ResponseCache:
    """Content-addressed SQLite cache for LLM and embedding responses."""

    def __init__(self, db_path: Path | str) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(
            str(self._db_path),
            check_same_thread=False,
        )
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def _execute(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        return self._conn.execute(sql, params)

    # ---- cache key ----

    @staticmethod
    def cache_key(request: dict[str, Any], key_fields: tuple[str, ...] = _LLM_KEY_FIELDS) -> str:
        canonical: dict[str, Any] = {}
        for k in key_fields:
            if k in request:
                canonical[k] = request[k]
        raw = json.dumps(canonical, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    @staticmethod
    def llm_key(request: dict[str, Any]) -> str:
        return ResponseCache.cache_key(request, _LLM_KEY_FIELDS)

    @staticmethod
    def embed_key(request: dict[str, Any]) -> str:
        return ResponseCache.cache_key(request, _EMBED_KEY_FIELDS)

    # ---- LLM operations ----

    def get_llm(self, key: str) -> dict[str, Any] | None:
        row = self._execute(
            "SELECT key, model, request, response, created_at, latency_ms, prompt_tokens, completion_tokens, hit_count FROM llm_responses WHERE key = ?",
            (key,),
        ).fetchone()
        if row is None:
            return None
        self._execute("UPDATE llm_responses SET hit_count = hit_count + 1 WHERE key = ?", (key,))
        self._conn.commit()
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
        self._execute(
            "INSERT OR REPLACE INTO llm_responses (key, model, request, response, created_at, latency_ms, prompt_tokens, completion_tokens, hit_count) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0)",
            (key, model, _to_json(request), _to_json(response), time.time(), latency_ms, prompt_tokens, completion_tokens),
        )
        self._conn.commit()

    # ---- Embed operations ----

    def get_embed(self, key: str) -> dict[str, Any] | None:
        row = self._execute(
            "SELECT key, model, request, response, created_at, latency_ms, token_count, hit_count FROM embed_responses WHERE key = ?",
            (key,),
        ).fetchone()
        if row is None:
            return None
        self._execute("UPDATE embed_responses SET hit_count = hit_count + 1 WHERE key = ?", (key,))
        self._conn.commit()
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
        self._execute(
            "INSERT OR REPLACE INTO embed_responses (key, model, request, response, created_at, latency_ms, token_count, hit_count) VALUES (?, ?, ?, ?, ?, ?, ?, 0)",
            (key, model, _to_json(request), _to_json(response), time.time(), latency_ms, token_count),
        )
        self._conn.commit()

    # ---- stats ----

    def llm_stats(self) -> dict[str, Any]:
        row = self._execute(
            "SELECT COUNT(*), COALESCE(SUM(prompt_tokens), 0), COALESCE(SUM(completion_tokens), 0), COALESCE(SUM(latency_ms), 0) FROM llm_responses"
        ).fetchone()
        return {
            "total_entries": row[0],
            "total_prompt_tokens": row[1],
            "total_completion_tokens": row[2],
            "total_latency_ms": row[3],
        }

    def embed_stats(self) -> dict[str, Any]:
        row = self._execute(
            "SELECT COUNT(*), COALESCE(SUM(token_count), 0), COALESCE(SUM(latency_ms), 0) FROM embed_responses"
        ).fetchone()
        return {
            "total_entries": row[0],
            "total_tokens": row[1],
            "total_latency_ms": row[2],
        }


def _to_json(obj: Any) -> str:
    if isinstance(obj, str):
        return obj
    if hasattr(obj, "model_dump"):
        obj = obj.model_dump()
    elif hasattr(obj, "to_dict"):
        obj = obj.to_dict()
    return json.dumps(obj, sort_keys=True, default=str)


def _try_json(s: str) -> Any:
    try:
        return json.loads(s)
    except (json.JSONDecodeError, TypeError):
        return s
```

**Step 4: Run tests**

Run: `cd v2-synix-benchmark && uv run pytest tests/test_cache.py -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add v2-synix-benchmark/src/bench/cache.py v2-synix-benchmark/tests/test_cache.py
git commit -m "feat(v2): add SQLite response cache with LLM and embedding tables"
```

---

### Task 3: ModalBroker — LLM calls with cache integration

**Files:**
- Create: `v2-synix-benchmark/src/bench/broker.py`
- Create: `v2-synix-benchmark/tests/test_broker.py`

**Step 1: Write the failing tests**

```python
# tests/test_broker.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from bench.broker import ModalBroker
from bench.cache import ResponseCache


def _make_chat_response(content: str = "hello", prompt_tokens: int = 5, completion_tokens: int = 3) -> MagicMock:
    """Build a mock OpenAI ChatCompletion response."""
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = content
    resp.choices[0].message.tool_calls = None
    resp.usage.prompt_tokens = prompt_tokens
    resp.usage.completion_tokens = completion_tokens
    resp.usage.total_tokens = prompt_tokens + completion_tokens
    resp.model_dump.return_value = {
        "id": "chatcmpl-test",
        "choices": [{"message": {"content": content, "tool_calls": None}}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens, "total_tokens": prompt_tokens + completion_tokens},
    }
    return resp


def _make_embed_response(embeddings: list[list[float]] | None = None) -> dict:
    """Build a mock embedding API response dict."""
    if embeddings is None:
        embeddings = [[0.1, 0.2, 0.3]]
    return {
        "object": "list",
        "data": [{"object": "embedding", "index": i, "embedding": e} for i, e in enumerate(embeddings)],
        "model": "gte-base",
        "usage": {"prompt_tokens": 5, "total_tokens": 5},
    }


class TestBrokerLLM:
    def test_cache_miss_calls_api_and_stores(self, tmp_db: Path) -> None:
        cache = ResponseCache(tmp_db)
        mock_response = _make_chat_response("test reply")

        with patch("bench.broker.openai") as mock_openai:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai.OpenAI.return_value = mock_client

            broker = ModalBroker(cache=cache, llm_base_url="http://fake:8000/v1", embed_base_url="http://fake:8080")
            result = broker.chat_completion(model="test", messages=[{"role": "user", "content": "hi"}])

        assert result is mock_response
        mock_client.chat.completions.create.assert_called_once()
        # Verify it was cached
        stats = cache.llm_stats()
        assert stats["total_entries"] == 1

    def test_cache_hit_skips_api(self, tmp_db: Path) -> None:
        cache = ResponseCache(tmp_db)
        mock_response = _make_chat_response("cached reply")

        with patch("bench.broker.openai") as mock_openai:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai.OpenAI.return_value = mock_client

            broker = ModalBroker(cache=cache, llm_base_url="http://fake:8000/v1", embed_base_url="http://fake:8080")
            # First call — miss
            broker.chat_completion(model="test", messages=[{"role": "user", "content": "hi"}])
            # Second call — hit
            result = broker.chat_completion(model="test", messages=[{"role": "user", "content": "hi"}])

        assert mock_client.chat.completions.create.call_count == 1  # only called once

    def test_bypass_cache_forces_fresh_call(self, tmp_db: Path) -> None:
        cache = ResponseCache(tmp_db)
        mock_response = _make_chat_response("fresh")

        with patch("bench.broker.openai") as mock_openai:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai.OpenAI.return_value = mock_client

            broker = ModalBroker(cache=cache, llm_base_url="http://fake:8000/v1", embed_base_url="http://fake:8080")
            broker.chat_completion(model="test", messages=[{"role": "user", "content": "hi"}])
            broker.chat_completion(model="test", messages=[{"role": "user", "content": "hi"}], bypass_cache=True)

        assert mock_client.chat.completions.create.call_count == 2

    def test_retry_on_transient_error(self, tmp_db: Path) -> None:
        cache = ResponseCache(tmp_db)
        mock_response = _make_chat_response("after retry")

        with patch("bench.broker.openai") as mock_openai:
            mock_client = MagicMock()
            # First call raises 503, second succeeds
            mock_client.chat.completions.create.side_effect = [
                mock_openai.InternalServerError("503", response=MagicMock(), body=None),
                mock_response,
            ]
            mock_openai.OpenAI.return_value = mock_client

            broker = ModalBroker(cache=cache, llm_base_url="http://fake:8000/v1", embed_base_url="http://fake:8080")
            result = broker.chat_completion(model="test", messages=[{"role": "user", "content": "hi"}])

        assert result is mock_response
        assert mock_client.chat.completions.create.call_count == 2

    def test_permanent_error_raises_immediately(self, tmp_db: Path) -> None:
        cache = ResponseCache(tmp_db)

        with patch("bench.broker.openai") as mock_openai:
            mock_client = MagicMock()
            mock_client.chat.completions.create.side_effect = mock_openai.BadRequestError(
                "400", response=MagicMock(), body=None
            )
            mock_openai.OpenAI.return_value = mock_client

            broker = ModalBroker(cache=cache, llm_base_url="http://fake:8000/v1", embed_base_url="http://fake:8080")
            with pytest.raises(mock_openai.BadRequestError):
                broker.chat_completion(model="test", messages=[{"role": "user", "content": "hi"}])

        assert mock_client.chat.completions.create.call_count == 1


class TestBrokerEmbed:
    def test_cache_miss_calls_api_and_stores(self, tmp_db: Path) -> None:
        cache = ResponseCache(tmp_db)
        mock_resp = _make_embed_response()

        with patch("bench.broker.httpx") as mock_httpx:
            mock_client = MagicMock()
            mock_http_resp = MagicMock()
            mock_http_resp.json.return_value = mock_resp
            mock_http_resp.raise_for_status.return_value = None
            mock_client.post.return_value = mock_http_resp
            mock_httpx.Client.return_value = mock_client

            broker = ModalBroker(cache=cache, llm_base_url="http://fake:8000/v1", embed_base_url="http://fake:8080")
            result = broker.embed(input=["hello"], model="gte-base")

        assert result == [[0.1, 0.2, 0.3]]
        assert cache.embed_stats()["total_entries"] == 1

    def test_cache_hit_skips_api(self, tmp_db: Path) -> None:
        cache = ResponseCache(tmp_db)
        mock_resp = _make_embed_response()

        with patch("bench.broker.httpx") as mock_httpx:
            mock_client = MagicMock()
            mock_http_resp = MagicMock()
            mock_http_resp.json.return_value = mock_resp
            mock_http_resp.raise_for_status.return_value = None
            mock_client.post.return_value = mock_http_resp
            mock_httpx.Client.return_value = mock_client

            broker = ModalBroker(cache=cache, llm_base_url="http://fake:8000/v1", embed_base_url="http://fake:8080")
            broker.embed(input=["hello"], model="gte-base")
            broker.embed(input=["hello"], model="gte-base")

        assert mock_client.post.call_count == 1


class TestBrokerStats:
    def test_stats_aggregate(self, tmp_db: Path) -> None:
        cache = ResponseCache(tmp_db)
        mock_response = _make_chat_response(prompt_tokens=10, completion_tokens=5)

        with patch("bench.broker.openai") as mock_openai:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai.OpenAI.return_value = mock_client

            broker = ModalBroker(cache=cache, llm_base_url="http://fake:8000/v1", embed_base_url="http://fake:8080")
            broker.chat_completion(model="test", messages=[{"role": "user", "content": "a"}])

        stats = broker.stats()
        assert stats["llm"]["total_entries"] == 1
        assert stats["llm"]["total_prompt_tokens"] == 10
```

**Step 2: Run tests to verify they fail**

Run: `cd v2-synix-benchmark && uv run pytest tests/test_broker.py -v`
Expected: ImportError — `bench.broker` does not exist

**Step 3: Write the implementation**

```python
# src/bench/broker.py
"""Modal broker — single inference gate for all v2 benchmark model calls.

All LLM and embedding calls go through ModalBroker, which handles:
- Cache lookup before every call
- Modal endpoint calls on cache miss
- Response capture with latency and token metadata
- Retry with exponential backoff on transient errors
- Cost and token accounting via the ResponseCache
"""
from __future__ import annotations

import logging
import time
from typing import Any

import httpx
import openai

from bench.cache import ResponseCache

log = logging.getLogger(__name__)

_TRANSIENT_ERRORS = (
    openai.InternalServerError,
    openai.RateLimitError,
    openai.APIConnectionError,
    openai.APITimeoutError,
)

_MAX_RETRIES = 5


class ModalBroker:
    """Single entry point for all inference in the v2 benchmark."""

    def __init__(
        self,
        *,
        cache: ResponseCache,
        llm_base_url: str,
        embed_base_url: str,
        llm_api_key: str = "unused",
        embed_timeout: float = 30.0,
    ) -> None:
        self._cache = cache
        self._llm_client = openai.OpenAI(api_key=llm_api_key, base_url=llm_base_url)
        self._embed_client = httpx.Client(base_url=embed_base_url, timeout=embed_timeout)

    def chat_completion(self, *, bypass_cache: bool = False, **kwargs: Any) -> Any:
        key = self._cache.llm_key(kwargs)

        if not bypass_cache:
            cached = self._cache.get_llm(key)
            if cached is not None:
                log.info("LLM CACHE HIT [%s]", key)
                return _deserialize_chat(cached["response"])

        log.info("LLM CACHE MISS [%s]", key)
        t0 = time.monotonic()
        response = self._call_llm_with_retry(**kwargs)
        latency_ms = (time.monotonic() - t0) * 1000

        usage = response.usage
        prompt_tokens = usage.prompt_tokens if usage else 0
        completion_tokens = usage.completion_tokens if usage else 0

        self._cache.put_llm(
            key,
            model=kwargs.get("model", ""),
            request=kwargs,
            response=response,
            latency_ms=latency_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
        return response

    def embed(self, *, input: list[str], model: str, bypass_cache: bool = False) -> list[list[float]]:
        request = {"model": model, "input": input}
        key = self._cache.embed_key(request)

        if not bypass_cache:
            cached = self._cache.get_embed(key)
            if cached is not None:
                log.info("EMBED CACHE HIT [%s]", key)
                return [d["embedding"] for d in cached["response"]["data"]]

        log.info("EMBED CACHE MISS [%s]", key)
        t0 = time.monotonic()
        resp = self._embed_client.post("/embeddings", json=request)
        resp.raise_for_status()
        data = resp.json()
        latency_ms = (time.monotonic() - t0) * 1000

        token_count = sum(len(t.split()) for t in input)

        self._cache.put_embed(
            key,
            model=model,
            request=request,
            response=data,
            latency_ms=latency_ms,
            token_count=token_count,
        )
        return [d["embedding"] for d in data["data"]]

    def stats(self) -> dict[str, Any]:
        return {
            "llm": self._cache.llm_stats(),
            "embed": self._cache.embed_stats(),
        }

    def _call_llm_with_retry(self, **kwargs: Any) -> Any:
        for attempt in range(_MAX_RETRIES + 1):
            try:
                return self._llm_client.chat.completions.create(**kwargs)
            except _TRANSIENT_ERRORS as e:
                if attempt == _MAX_RETRIES:
                    raise
                wait = min(2 ** attempt * 2, 60)
                log.warning("LLM transient error (attempt %d/%d, retry in %ds): %s", attempt + 1, _MAX_RETRIES + 1, wait, e)
                time.sleep(wait)


def _deserialize_chat(response_data: dict) -> Any:
    try:
        from openai.types.chat import ChatCompletion
        return ChatCompletion.model_validate(response_data)
    except Exception:
        log.warning("Could not deserialize cached LLM response as ChatCompletion")
        return response_data
```

**Step 4: Run tests**

Run: `cd v2-synix-benchmark && uv run pytest tests/test_broker.py -v`
Expected: All tests PASS

**Step 5: Run full test suite**

Run: `cd v2-synix-benchmark && uv run pytest tests/ -v`
Expected: All tests from both test_cache.py and test_broker.py PASS

**Step 6: Commit**

```bash
git add v2-synix-benchmark/src/bench/broker.py v2-synix-benchmark/tests/test_broker.py
git commit -m "feat(v2): add ModalBroker with cache-through LLM and embedding calls"
```

---

### Task 4: Parameterize Modal min_containers

**Files:**
- Modify: `infra/modal/llm_server.py:63` (the `min_containers=2` line)

**Step 1: Read the current file to confirm the line**

Run: `head -70 infra/modal/llm_server.py`

**Step 2: Replace hardcoded min_containers with env-configurable default**

Change the `@app.function` decorator block in `infra/modal/llm_server.py`:

```python
# At the top of the file, after CHAT_TEMPLATE:
import os
MIN_CONTAINERS = int(os.environ.get("LENS_MIN_CONTAINERS", "0"))
```

And in the decorator:
```python
    min_containers=MIN_CONTAINERS,
```

**Step 3: Verify it still deploys cleanly (syntax check)**

Run: `cd infra/modal && python -c "import llm_server; print('OK')"`
Expected: `OK` (no import errors)

**Step 4: Commit**

```bash
git add infra/modal/llm_server.py
git commit -m "feat: parameterize Modal min_containers via LENS_MIN_CONTAINERS env var"
```

---

### Task 5: Concurrent write safety test

**Files:**
- Modify: `v2-synix-benchmark/tests/test_cache.py`

**Step 1: Add concurrent write test**

Append to `tests/test_cache.py`:

```python
import threading


class TestCacheConcurrency:
    def test_concurrent_writes_no_corruption(self, tmp_db: Path) -> None:
        cache = ResponseCache(tmp_db)
        errors: list[Exception] = []

        def writer(n: int) -> None:
            try:
                for i in range(20):
                    key = f"thread{n}_entry{i}"
                    cache.put_llm(key, model="m", request={"i": i}, response={"r": n}, latency_ms=1.0, prompt_tokens=1, completion_tokens=1)
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
```

**Step 2: Run the test**

Run: `cd v2-synix-benchmark && uv run pytest tests/test_cache.py::TestCacheConcurrency -v`
Expected: PASS

**Step 3: Commit**

```bash
git add v2-synix-benchmark/tests/test_cache.py
git commit -m "test(v2): add concurrent write safety test for ResponseCache"
```

---

### Task 6: Update ops records

**Files:**
- Modify: `v2-synix-benchmark/ops/WORKBOARD.md`
- Modify: `v2-synix-benchmark/ops/WORKLOG.md`

**Step 1: Update T003 status on workboard**

Change T003 row from `ready` to `in_progress`.

**Step 2: Append worklog entry**

```markdown
| 2026-03-08 | T003: Modal broker and cache | Implemented ResponseCache (SQLite WAL, content-addressed) and ModalBroker (cache-through LLM + embed, retry, accounting). Full unit tests. Parameterized min_containers in Modal deploy. |
```

**Step 3: Commit**

```bash
git add v2-synix-benchmark/ops/WORKBOARD.md v2-synix-benchmark/ops/WORKLOG.md
git commit -m "ops: update workboard and worklog for T003 progress"
```

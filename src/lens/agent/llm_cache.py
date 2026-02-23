"""Transparent LLM call memoization layer.

Drop-in replacement for an OpenAI client that caches every
``client.chat.completions.create()`` call to disk as JSON files.

Cache key: sha256(json.dumps(canonicalized_kwargs))[:16]
Cache location: {cache_dir}/{key}.json
Each entry stores: {request, response, timestamp, hit_count}

Usage:
    from openai import OpenAI
    from lens.agent.llm_cache import CachingOpenAIClient

    real = OpenAI(api_key=..., base_url=...)
    client = CachingOpenAIClient(real, cache_dir="results/llm_cache")

    # This is now cached transparently:
    response = client.chat.completions.create(model=..., messages=...)

Ported from synix-bench with minimal changes.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


@dataclass
class CacheStats:
    """Tracks cache hit/miss counts for the session."""

    hits: int = 0
    misses: int = 0
    errors: int = 0

    @property
    def total(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        return self.hits / self.total if self.total > 0 else 0.0


class CachingOpenAIClient:
    """Drop-in replacement for OpenAI client with transparent disk caching.

    Proxies attribute access so ``client.chat.completions.create()`` works
    identically, but caches every call to disk.  All other attributes
    (``client.models``, ``client.embeddings``, etc.) pass through to the
    real client.

    The cache is content-addressed: same (model + messages + tools +
    temperature + extra_body) = same key = cache hit.
    """

    def __init__(
        self,
        client: Any,
        cache_dir: Path | str,
        *,
        enabled: bool = True,
    ) -> None:
        self._real_client = client
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._enabled = enabled
        self.stats = CacheStats()
        self.chat = _ChatNamespace(self)

    # ---- proxy everything else to the real client ----
    def __getattr__(self, name: str) -> Any:
        return getattr(self._real_client, name)

    # ---- cache key ----
    @staticmethod
    def _cache_key(kwargs: dict) -> str:
        """Compute a deterministic cache key from the API call kwargs.

        Canonical form: sorted JSON of (model, messages, tools, temperature,
        extra_body, seed).  We deliberately exclude ephemeral fields like
        ``stream`` or ``timeout``.
        """
        canonical: dict[str, Any] = {}
        for k in ("model", "messages", "tools", "temperature", "seed",
                   "extra_body", "max_tokens", "tool_choice"):
            if k in kwargs:
                canonical[k] = kwargs[k]
        raw = json.dumps(canonical, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    # ---- disk I/O ----
    def _cache_path(self, key: str) -> Path:
        return self._cache_dir / f"{key}.json"

    def _load(self, key: str) -> dict | None:
        """Load a cached entry by key. Returns None on miss."""
        path = self._cache_path(key)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text())
            # Bump hit count
            data["hit_count"] = data.get("hit_count", 0) + 1
            data["last_hit"] = time.time()
            path.write_text(json.dumps(data, indent=2, default=str))
            return data
        except (json.JSONDecodeError, OSError) as e:
            log.warning("Cache read error for %s: %s", key, e)
            self.stats.errors += 1
            return None

    def _save(self, key: str, kwargs: dict, response: Any) -> None:
        """Persist a cache entry to disk."""
        # Serialize the response -- OpenAI SDK objects have .model_dump()
        if hasattr(response, "model_dump"):
            resp_data = response.model_dump()
        elif hasattr(response, "to_dict"):
            resp_data = response.to_dict()
        else:
            resp_data = str(response)

        entry = {
            "cache_key": key,
            "timestamp": time.time(),
            "hit_count": 0,
            "request": _sanitize_request(kwargs),
            "response": resp_data,
        }
        path = self._cache_path(key)
        try:
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(entry, indent=2, default=str))
            os.replace(tmp, path)
        except OSError as e:
            log.warning("Cache write error for %s: %s", key, e)
            self.stats.errors += 1

    def _deserialize(self, cached: dict) -> Any:
        """Reconstruct an OpenAI ChatCompletion from cached JSON."""
        resp_data = cached["response"]
        try:
            from openai.types.chat import ChatCompletion
            return ChatCompletion.model_validate(resp_data)
        except Exception:
            log.warning("Could not deserialize cached response as ChatCompletion, using dict")
            return _DictResponse(resp_data)


class _ChatNamespace:
    """Proxy for ``client.chat``."""

    def __init__(self, parent: CachingOpenAIClient) -> None:
        self.completions = _CompletionsNamespace(parent)


class _CompletionsNamespace:
    """Proxy for ``client.chat.completions``."""

    def __init__(self, parent: CachingOpenAIClient) -> None:
        self._parent = parent

    def create(self, **kwargs: Any) -> Any:
        """Cached wrapper around ``chat.completions.create()``."""
        if not self._parent._enabled:
            return self._parent._real_client.chat.completions.create(**kwargs)

        key = self._parent._cache_key(kwargs)

        cached = self._parent._load(key)
        if cached is not None:
            self._parent.stats.hits += 1
            log.info("CACHE HIT [%s] (hits=%d)", key, cached.get("hit_count", 1))
            return self._parent._deserialize(cached)

        # Cache miss — call the real API
        self._parent.stats.misses += 1
        t0 = time.monotonic()
        response = self._parent._real_client.chat.completions.create(**kwargs)
        latency_ms = (time.monotonic() - t0) * 1000
        log.info("CACHE MISS [%s] (%.0fms)", key, latency_ms)

        self._parent._save(key, kwargs, response)
        return response


class _DictResponse:
    """Minimal wrapper around a cached response dict for fallback deserialization."""

    def __init__(self, data: dict) -> None:
        self._data = data

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        val = self._data.get(name)
        if isinstance(val, dict):
            return _DictResponse(val)
        if isinstance(val, list):
            return [_DictResponse(v) if isinstance(v, dict) else v for v in val]
        return val


def _sanitize_request(kwargs: dict) -> dict:
    """Strip large/binary fields from request for storage."""
    out: dict = {}
    for k, v in kwargs.items():
        if k in ("stream", "timeout"):
            continue
        out[k] = v
    return out

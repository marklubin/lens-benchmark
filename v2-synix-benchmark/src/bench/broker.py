"""ModalBroker — single inference gate for all v2 benchmark model calls.

Wraps OpenAI SDK for LLM calls and httpx for embedding calls,
with configurable cache-through semantics via ResponseCache.

Cache behavior is controlled at construction time:
    cache_enabled: master switch — False means no caching at all (no DB created)
    cache_dir:     path to the SQLite DB — change this to namespace caches
    cache_llm:     cache LLM chat completions (default True)
    cache_embed:   cache embedding calls (default True)
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import httpx
import openai

from bench.cache import ResponseCache

logger = logging.getLogger(__name__)


def _deserialize_chat(response_data: dict) -> Any:
    """Attempt to deserialize cached response data into a ChatCompletion object."""
    try:
        from openai.types.chat import ChatCompletion
        return ChatCompletion.model_validate(response_data)
    except Exception:
        return response_data


class ModalBroker:
    """Cache-through broker for LLM and embedding inference calls.

    All model calls in the v2 benchmark flow through this class.
    It handles caching, retries with exponential backoff, and
    token/latency accounting.
    """

    _TRANSIENT_ERRORS: tuple[str, ...] = (
        "InternalServerError",
        "RateLimitError",
        "APIConnectionError",
        "APITimeoutError",
    )
    _MAX_RETRIES: int = 5

    def __init__(
        self,
        *,
        llm_base_url: str,
        embed_base_url: str,
        llm_api_key: str = "unused",
        embed_timeout: float = 30.0,
        cache_enabled: bool = True,
        cache_dir: str | Path = "./cache",
        cache_llm: bool = True,
        cache_embed: bool = True,
        # Allow injecting a pre-built cache (for tests / advanced use)
        cache: ResponseCache | None = None,
        # Extra body params merged into every LLM request (e.g. chat_template_kwargs)
        default_extra_body: dict[str, Any] | None = None,
    ) -> None:
        self._cache_enabled = cache_enabled
        self._cache_llm = cache_llm and cache_enabled
        self._cache_embed = cache_embed and cache_enabled

        if cache is not None:
            self._cache = cache
        elif cache_enabled:
            db_path = Path(cache_dir) / "responses.db"
            db_path.parent.mkdir(parents=True, exist_ok=True)
            self._cache = ResponseCache(db_path)
        else:
            self._cache = None  # type: ignore[assignment]

        self._llm_base_url = llm_base_url
        self._llm_api_key = llm_api_key
        self._llm_client = openai.OpenAI(base_url=llm_base_url, api_key=llm_api_key)
        self._embed_client = httpx.Client(base_url=embed_base_url, timeout=embed_timeout)
        self._embed_base_url = embed_base_url
        self._default_extra_body = default_extra_body or {}

    # ── LLM ───────────────────────────────────────────────────────────────

    def chat_completion(self, *, bypass_cache: bool = False, **kwargs: Any) -> Any:
        """Run a chat completion, checking the cache first unless bypassed.

        On cache miss, calls the LLM API with retry logic, captures latency
        and token usage, and stores the result in the cache.
        """
        # Merge default extra_body (e.g. chat_template_kwargs for disabling thinking)
        if self._default_extra_body:
            existing = kwargs.get("extra_body", {}) or {}
            kwargs["extra_body"] = {**self._default_extra_body, **existing}

        use_cache = self._cache_llm and not bypass_cache

        if use_cache:
            cache_key = self._cache.llm_key(kwargs)
            hit = self._cache.get_llm(cache_key)
            if hit is not None:
                logger.debug("LLM cache hit: %s", cache_key)
                return _deserialize_chat(hit["response"])

        t0 = time.time()
        response = self._call_llm_with_retry(**kwargs)
        t1 = time.time()
        latency_ms = (t1 - t0) * 1000.0

        if self._cache_llm:
            prompt_tokens = getattr(response.usage, "prompt_tokens", 0) or 0
            completion_tokens = getattr(response.usage, "completion_tokens", 0) or 0
            model = kwargs.get("model", getattr(response, "model", "unknown"))
            if not use_cache:
                cache_key = self._cache.llm_key(kwargs)

            self._cache.put_llm(
                cache_key,
                model=model,
                request=kwargs,
                response=response,
                latency_ms=latency_ms,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )

        return response

    def _call_llm_with_retry(self, **kwargs: Any) -> Any:
        """Call the LLM API with up to 5 retries on transient errors.

        Uses exponential backoff: min(2^attempt * 2, 60) seconds.
        Permanent errors (e.g. BadRequestError) raise immediately.
        """
        transient = tuple(
            getattr(openai, name)
            for name in self._TRANSIENT_ERRORS
            if hasattr(openai, name)
        )

        last_exc: Exception | None = None
        for attempt in range(self._MAX_RETRIES):
            try:
                return self._llm_client.chat.completions.create(**kwargs)
            except Exception as exc:
                if transient and isinstance(exc, transient):
                    last_exc = exc
                    backoff = min(2 ** attempt * 2, 60)
                    logger.warning(
                        "Transient LLM error (attempt %d/%d), retrying in %ds: %s",
                        attempt + 1, self._MAX_RETRIES, backoff, exc,
                    )
                    time.sleep(backoff)
                else:
                    raise

        # If we exhausted retries, raise the last transient error
        raise last_exc  # type: ignore[misc]

    # ── Embeddings ────────────────────────────────────────────────────────

    def embed(
        self,
        *,
        input: list[str],
        model: str,
        bypass_cache: bool = False,
    ) -> list[list[float]]:
        """Compute embeddings, checking the cache first unless bypassed.

        On cache miss, POSTs to the embed endpoint, captures latency,
        estimates token count, and stores the result in the cache.
        """
        use_cache = self._cache_embed and not bypass_cache
        request = {"model": model, "input": input}

        if use_cache:
            cache_key = self._cache.embed_key(request)
            hit = self._cache.get_embed(cache_key)
            if hit is not None:
                logger.debug("Embed cache hit: %s", cache_key)
                data = hit["response"]
                return [d["embedding"] for d in data["data"]]

        t0 = time.time()
        http_response = self._embed_client.post("/embeddings", json=request)
        http_response.raise_for_status()
        data = http_response.json()
        t1 = time.time()
        latency_ms = (t1 - t0) * 1000.0

        if self._cache_embed:
            token_count = sum(len(t.split()) for t in input)
            if not use_cache:
                cache_key = self._cache.embed_key(request)

            self._cache.put_embed(
                cache_key,
                model=model,
                request=request,
                response=data,
                latency_ms=latency_ms,
                token_count=token_count,
            )

        return [d["embedding"] for d in data["data"]]

    # ── Stats ─────────────────────────────────────────────────────────────

    def stats(self) -> dict[str, Any]:
        """Return aggregate cache statistics for LLM and embedding calls."""
        if not self._cache_enabled:
            return {"llm": {}, "embed": {}, "cache_enabled": False}
        return {
            "llm": self._cache.llm_stats(),
            "embed": self._cache.embed_stats(),
            "cache_enabled": True,
            "cache_llm": self._cache_llm,
            "cache_embed": self._cache_embed,
        }

"""Tests for bench.broker — ModalBroker with cache-through LLM and embedding calls."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from bench.cache import ResponseCache


# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------

def _make_chat_response(
    content: str,
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
    model: str = "test-model",
) -> MagicMock:
    """Return a MagicMock that mimics an openai ChatCompletion."""
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = content
    resp.usage.prompt_tokens = prompt_tokens
    resp.usage.completion_tokens = completion_tokens
    resp.model = model
    resp.model_dump.return_value = {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }
    return resp


def _make_embed_response(embeddings: list[list[float]]) -> dict[str, Any]:
    """Return a dict that mimics the JSON body of an embeddings API response."""
    return {
        "object": "list",
        "data": [{"object": "embedding", "embedding": e, "index": i} for i, e in enumerate(embeddings)],
        "model": "test-embed-model",
        "usage": {"prompt_tokens": 5, "total_tokens": 5},
    }


# ---------------------------------------------------------------------------
# TestBrokerLLM
# ---------------------------------------------------------------------------

class TestBrokerLLM:
    """LLM chat_completion through the broker."""

    def test_cache_miss_calls_api_and_stores(self, tmp_db: Path) -> None:
        cache = ResponseCache(tmp_db)
        mock_response = _make_chat_response("hello world")

        with (
            patch("bench.broker.openai") as mock_openai,
            patch("bench.broker.time") as mock_time,
        ):
            mock_client = MagicMock()
            mock_openai.OpenAI.return_value = mock_client
            mock_client.chat.completions.create.return_value = mock_response
            mock_time.time.side_effect = [0.0, 0.1]  # start, end for latency
            mock_time.sleep = MagicMock()  # no-op sleep

            from bench.broker import ModalBroker

            broker = ModalBroker(
                cache=cache,
                llm_base_url="http://fake-llm",
                embed_base_url="http://fake-embed",
            )
            result = broker.chat_completion(
                model="test-model",
                messages=[{"role": "user", "content": "hi"}],
            )

        assert result.choices[0].message.content == "hello world"
        mock_client.chat.completions.create.assert_called_once()

        # Verify stored in cache
        stats = cache.llm_stats()
        assert stats["total_entries"] == 1
        cache.close()

    def test_cache_hit_skips_api(self, tmp_db: Path) -> None:
        cache = ResponseCache(tmp_db)
        mock_response = _make_chat_response("cached answer")

        with (
            patch("bench.broker.openai") as mock_openai,
            patch("bench.broker.time") as mock_time,
        ):
            mock_client = MagicMock()
            mock_openai.OpenAI.return_value = mock_client
            mock_client.chat.completions.create.return_value = mock_response
            mock_time.time.side_effect = [0.0, 0.1, 0.0, 0.1]
            mock_time.sleep = MagicMock()

            from bench.broker import ModalBroker

            broker = ModalBroker(
                cache=cache,
                llm_base_url="http://fake-llm",
                embed_base_url="http://fake-embed",
            )
            kwargs = {"model": "test-model", "messages": [{"role": "user", "content": "same"}]}
            _ = broker.chat_completion(**kwargs)
            result2 = broker.chat_completion(**kwargs)

        # API called only once — second call served from cache
        assert mock_client.chat.completions.create.call_count == 1
        # Result still contains the right data
        assert result2["choices"][0]["message"]["content"] == "cached answer" or (
            hasattr(result2, "choices") and result2.choices[0].message.content == "cached answer"
        )
        cache.close()

    def test_bypass_cache_forces_fresh_call(self, tmp_db: Path) -> None:
        cache = ResponseCache(tmp_db)
        mock_response = _make_chat_response("fresh")

        with (
            patch("bench.broker.openai") as mock_openai,
            patch("bench.broker.time") as mock_time,
        ):
            mock_client = MagicMock()
            mock_openai.OpenAI.return_value = mock_client
            mock_client.chat.completions.create.return_value = mock_response
            mock_time.time.side_effect = [0.0, 0.1, 0.0, 0.1]
            mock_time.sleep = MagicMock()

            from bench.broker import ModalBroker

            broker = ModalBroker(
                cache=cache,
                llm_base_url="http://fake-llm",
                embed_base_url="http://fake-embed",
            )
            kwargs = {"model": "test-model", "messages": [{"role": "user", "content": "same"}]}
            _ = broker.chat_completion(**kwargs)
            _ = broker.chat_completion(bypass_cache=True, **kwargs)

        # API called twice because bypass_cache=True skips cache lookup
        assert mock_client.chat.completions.create.call_count == 2
        cache.close()

    def test_retry_on_transient_error(self, tmp_db: Path) -> None:
        cache = ResponseCache(tmp_db)
        mock_response = _make_chat_response("after retry")

        with (
            patch("bench.broker.openai") as mock_openai,
            patch("bench.broker.time") as mock_time,
        ):
            mock_client = MagicMock()
            mock_openai.OpenAI.return_value = mock_client

            # Configure the transient exception class so isinstance() checks work
            transient_exc = Exception("server error")
            mock_openai.InternalServerError = type("InternalServerError", (Exception,), {})
            mock_openai.RateLimitError = type("RateLimitError", (Exception,), {})
            mock_openai.APIConnectionError = type("APIConnectionError", (Exception,), {})
            mock_openai.APITimeoutError = type("APITimeoutError", (Exception,), {})
            mock_openai.BadRequestError = type("BadRequestError", (Exception,), {})

            err = mock_openai.InternalServerError("server error")
            mock_client.chat.completions.create.side_effect = [err, mock_response]
            mock_time.time.side_effect = [0.0, 0.1]
            mock_time.sleep = MagicMock()

            from bench.broker import ModalBroker

            broker = ModalBroker(
                cache=cache,
                llm_base_url="http://fake-llm",
                embed_base_url="http://fake-embed",
            )
            result = broker.chat_completion(
                model="test-model",
                messages=[{"role": "user", "content": "retry me"}],
            )

        assert result.choices[0].message.content == "after retry"
        assert mock_client.chat.completions.create.call_count == 2
        # sleep was called for backoff
        mock_time.sleep.assert_called_once()
        cache.close()

    def test_permanent_error_raises_immediately(self, tmp_db: Path) -> None:
        cache = ResponseCache(tmp_db)

        with (
            patch("bench.broker.openai") as mock_openai,
            patch("bench.broker.time") as mock_time,
        ):
            mock_client = MagicMock()
            mock_openai.OpenAI.return_value = mock_client

            mock_openai.InternalServerError = type("InternalServerError", (Exception,), {})
            mock_openai.RateLimitError = type("RateLimitError", (Exception,), {})
            mock_openai.APIConnectionError = type("APIConnectionError", (Exception,), {})
            mock_openai.APITimeoutError = type("APITimeoutError", (Exception,), {})
            mock_openai.BadRequestError = type("BadRequestError", (Exception,), {})

            err = mock_openai.BadRequestError("bad input")
            mock_client.chat.completions.create.side_effect = err
            mock_time.time.side_effect = [0.0, 0.1]
            mock_time.sleep = MagicMock()

            from bench.broker import ModalBroker

            broker = ModalBroker(
                cache=cache,
                llm_base_url="http://fake-llm",
                embed_base_url="http://fake-embed",
            )

            with pytest.raises(Exception, match="bad input"):
                broker.chat_completion(
                    model="test-model",
                    messages=[{"role": "user", "content": "bad"}],
                )

        assert mock_client.chat.completions.create.call_count == 1
        cache.close()


# ---------------------------------------------------------------------------
# TestBrokerEmbed
# ---------------------------------------------------------------------------

class TestBrokerEmbed:
    """Embedding calls through the broker."""

    def test_cache_miss_calls_api_and_stores(self, tmp_db: Path) -> None:
        cache = ResponseCache(tmp_db)
        embed_data = _make_embed_response([[0.1, 0.2], [0.3, 0.4]])

        with (
            patch("bench.broker.openai") as mock_openai,
            patch("bench.broker.httpx") as mock_httpx,
            patch("bench.broker.time") as mock_time,
        ):
            mock_openai.OpenAI.return_value = MagicMock()
            mock_http_client = MagicMock()
            mock_httpx.Client.return_value = mock_http_client

            mock_http_response = MagicMock()
            mock_http_response.json.return_value = embed_data
            mock_http_response.raise_for_status = MagicMock()
            mock_http_client.post.return_value = mock_http_response

            mock_time.time.side_effect = [0.0, 0.1]
            mock_time.sleep = MagicMock()

            from bench.broker import ModalBroker

            broker = ModalBroker(
                cache=cache,
                llm_base_url="http://fake-llm",
                embed_base_url="http://fake-embed",
            )
            result = broker.embed(
                input=["hello world", "goodbye"],
                model="test-embed-model",
            )

        assert result == [[0.1, 0.2], [0.3, 0.4]]
        mock_http_client.post.assert_called_once()

        stats = cache.embed_stats()
        assert stats["total_entries"] == 1
        cache.close()

    def test_cache_hit_skips_api(self, tmp_db: Path) -> None:
        cache = ResponseCache(tmp_db)
        embed_data = _make_embed_response([[0.5, 0.6]])

        with (
            patch("bench.broker.openai") as mock_openai,
            patch("bench.broker.httpx") as mock_httpx,
            patch("bench.broker.time") as mock_time,
        ):
            mock_openai.OpenAI.return_value = MagicMock()
            mock_http_client = MagicMock()
            mock_httpx.Client.return_value = mock_http_client

            mock_http_response = MagicMock()
            mock_http_response.json.return_value = embed_data
            mock_http_response.raise_for_status = MagicMock()
            mock_http_client.post.return_value = mock_http_response

            mock_time.time.side_effect = [0.0, 0.1, 0.0, 0.1]
            mock_time.sleep = MagicMock()

            from bench.broker import ModalBroker

            broker = ModalBroker(
                cache=cache,
                llm_base_url="http://fake-llm",
                embed_base_url="http://fake-embed",
            )
            _ = broker.embed(input=["hello"], model="test-embed-model")
            result2 = broker.embed(input=["hello"], model="test-embed-model")

        # API called only once — second call served from cache
        assert mock_http_client.post.call_count == 1
        assert result2 == [[0.5, 0.6]]
        cache.close()


# ---------------------------------------------------------------------------
# TestBrokerStats
# ---------------------------------------------------------------------------

class TestBrokerStats:
    """Broker stats aggregation."""

    def test_stats_aggregate(self, tmp_db: Path) -> None:
        cache = ResponseCache(tmp_db)
        mock_response = _make_chat_response("stats test", prompt_tokens=15, completion_tokens=8)

        with (
            patch("bench.broker.openai") as mock_openai,
            patch("bench.broker.time") as mock_time,
        ):
            mock_client = MagicMock()
            mock_openai.OpenAI.return_value = mock_client
            mock_client.chat.completions.create.return_value = mock_response
            mock_time.time.side_effect = [0.0, 0.1]
            mock_time.sleep = MagicMock()

            from bench.broker import ModalBroker

            broker = ModalBroker(
                cache=cache,
                llm_base_url="http://fake-llm",
                embed_base_url="http://fake-embed",
            )
            _ = broker.chat_completion(
                model="test-model",
                messages=[{"role": "user", "content": "stats"}],
            )

            stats = broker.stats()

        assert "llm" in stats
        assert "embed" in stats
        assert stats["llm"]["total_entries"] == 1
        assert stats["llm"]["total_prompt_tokens"] == 15
        assert stats["llm"]["total_completion_tokens"] == 8
        assert stats["embed"]["total_entries"] == 0
        cache.close()


# ---------------------------------------------------------------------------
# TestBrokerCacheConfig
# ---------------------------------------------------------------------------

class TestBrokerCacheConfig:
    """Cache configuration: enable/disable, per-call-type, namespacing."""

    def test_cache_disabled_no_db_created(self, tmp_path: Path) -> None:
        cache_dir = tmp_path / "should_not_exist"
        mock_response = _make_chat_response("no cache")

        with (
            patch("bench.broker.openai") as mock_openai,
            patch("bench.broker.time") as mock_time,
        ):
            mock_client = MagicMock()
            mock_openai.OpenAI.return_value = mock_client
            mock_client.chat.completions.create.return_value = mock_response
            mock_time.time.side_effect = [0.0, 0.1, 0.0, 0.1]
            mock_time.sleep = MagicMock()

            from bench.broker import ModalBroker

            broker = ModalBroker(
                llm_base_url="http://fake-llm",
                embed_base_url="http://fake-embed",
                cache_enabled=False,
                cache_dir=str(cache_dir),
            )
            # Two identical calls — both should hit API since cache is off
            broker.chat_completion(model="m", messages=[{"role": "user", "content": "hi"}])
            broker.chat_completion(model="m", messages=[{"role": "user", "content": "hi"}])

        assert mock_client.chat.completions.create.call_count == 2
        assert not (cache_dir / "responses.db").exists()

    def test_cache_disabled_stats_returns_empty(self, tmp_path: Path) -> None:
        with patch("bench.broker.openai") as mock_openai:
            mock_openai.OpenAI.return_value = MagicMock()

            from bench.broker import ModalBroker

            broker = ModalBroker(
                llm_base_url="http://fake-llm",
                embed_base_url="http://fake-embed",
                cache_enabled=False,
                cache_dir=str(tmp_path / "none"),
            )
            stats = broker.stats()

        assert stats["cache_enabled"] is False
        assert stats["llm"] == {}
        assert stats["embed"] == {}

    def test_cache_llm_false_skips_llm_caching(self, tmp_db: Path) -> None:
        cache = ResponseCache(tmp_db)
        mock_response = _make_chat_response("not cached")

        with (
            patch("bench.broker.openai") as mock_openai,
            patch("bench.broker.time") as mock_time,
        ):
            mock_client = MagicMock()
            mock_openai.OpenAI.return_value = mock_client
            mock_client.chat.completions.create.return_value = mock_response
            mock_time.time.side_effect = [0.0, 0.1, 0.0, 0.1]
            mock_time.sleep = MagicMock()

            from bench.broker import ModalBroker

            broker = ModalBroker(
                cache=cache,
                llm_base_url="http://fake-llm",
                embed_base_url="http://fake-embed",
                cache_llm=False,
            )
            broker.chat_completion(model="m", messages=[{"role": "user", "content": "a"}])
            broker.chat_completion(model="m", messages=[{"role": "user", "content": "a"}])

        # Both calls hit API — LLM cache disabled
        assert mock_client.chat.completions.create.call_count == 2
        assert cache.llm_stats()["total_entries"] == 0
        cache.close()

    def test_cache_embed_false_skips_embed_caching(self, tmp_db: Path) -> None:
        cache = ResponseCache(tmp_db)
        embed_data = _make_embed_response([[0.1, 0.2]])

        with (
            patch("bench.broker.openai") as mock_openai,
            patch("bench.broker.httpx") as mock_httpx,
            patch("bench.broker.time") as mock_time,
        ):
            mock_openai.OpenAI.return_value = MagicMock()
            mock_http_client = MagicMock()
            mock_httpx.Client.return_value = mock_http_client

            mock_http_response = MagicMock()
            mock_http_response.json.return_value = embed_data
            mock_http_response.raise_for_status = MagicMock()
            mock_http_client.post.return_value = mock_http_response

            mock_time.time.side_effect = [0.0, 0.1, 0.0, 0.1]
            mock_time.sleep = MagicMock()

            from bench.broker import ModalBroker

            broker = ModalBroker(
                cache=cache,
                llm_base_url="http://fake-llm",
                embed_base_url="http://fake-embed",
                cache_embed=False,
            )
            broker.embed(input=["hello"], model="m")
            broker.embed(input=["hello"], model="m")

        # Both calls hit API — embed cache disabled
        assert mock_http_client.post.call_count == 2
        assert cache.embed_stats()["total_entries"] == 0
        cache.close()

    def test_cache_llm_false_but_embed_still_cached(self, tmp_db: Path) -> None:
        cache = ResponseCache(tmp_db)
        mock_response = _make_chat_response("llm no cache")
        embed_data = _make_embed_response([[0.5, 0.6]])

        with (
            patch("bench.broker.openai") as mock_openai,
            patch("bench.broker.httpx") as mock_httpx,
            patch("bench.broker.time") as mock_time,
        ):
            mock_client = MagicMock()
            mock_openai.OpenAI.return_value = mock_client
            mock_client.chat.completions.create.return_value = mock_response

            mock_http_client = MagicMock()
            mock_httpx.Client.return_value = mock_http_client
            mock_http_response = MagicMock()
            mock_http_response.json.return_value = embed_data
            mock_http_response.raise_for_status = MagicMock()
            mock_http_client.post.return_value = mock_http_response

            mock_time.time.side_effect = [0.0, 0.1, 0.0, 0.1, 0.0, 0.1, 0.0, 0.1]
            mock_time.sleep = MagicMock()

            from bench.broker import ModalBroker

            broker = ModalBroker(
                cache=cache,
                llm_base_url="http://fake-llm",
                embed_base_url="http://fake-embed",
                cache_llm=False,
                cache_embed=True,
            )
            # LLM: two identical calls, both should hit API
            broker.chat_completion(model="m", messages=[{"role": "user", "content": "x"}])
            broker.chat_completion(model="m", messages=[{"role": "user", "content": "x"}])
            # Embed: two identical calls, second should be cached
            broker.embed(input=["test"], model="m")
            broker.embed(input=["test"], model="m")

        assert mock_client.chat.completions.create.call_count == 2  # no LLM cache
        assert mock_http_client.post.call_count == 1  # embed cached
        cache.close()

    def test_cache_dir_namespaces_independently(self, tmp_path: Path) -> None:
        mock_response = _make_chat_response("namespaced")

        with (
            patch("bench.broker.openai") as mock_openai,
            patch("bench.broker.time") as mock_time,
        ):
            mock_client = MagicMock()
            mock_openai.OpenAI.return_value = mock_client
            mock_client.chat.completions.create.return_value = mock_response
            mock_time.time.side_effect = [0.0, 0.1, 0.0, 0.1]
            mock_time.sleep = MagicMock()

            from bench.broker import ModalBroker

            # Broker A with namespace "study_01"
            broker_a = ModalBroker(
                llm_base_url="http://fake-llm",
                embed_base_url="http://fake-embed",
                cache_dir=tmp_path / "study_01",
            )
            broker_a.chat_completion(model="m", messages=[{"role": "user", "content": "q"}])

            # Broker B with namespace "study_02" — should NOT see A's cache
            mock_client.chat.completions.create.reset_mock()
            mock_time.time.side_effect = [0.0, 0.1]

            broker_b = ModalBroker(
                llm_base_url="http://fake-llm",
                embed_base_url="http://fake-embed",
                cache_dir=tmp_path / "study_02",
            )
            broker_b.chat_completion(model="m", messages=[{"role": "user", "content": "q"}])

        # Both called API — different namespaces don't share
        assert mock_client.chat.completions.create.call_count == 1  # only B's call (A's was before reset)
        assert (tmp_path / "study_01" / "responses.db").exists()
        assert (tmp_path / "study_02" / "responses.db").exists()

    def test_stats_includes_config_flags(self, tmp_db: Path) -> None:
        cache = ResponseCache(tmp_db)

        with patch("bench.broker.openai") as mock_openai:
            mock_openai.OpenAI.return_value = MagicMock()

            from bench.broker import ModalBroker

            broker = ModalBroker(
                cache=cache,
                llm_base_url="http://fake-llm",
                embed_base_url="http://fake-embed",
                cache_llm=True,
                cache_embed=False,
            )
            stats = broker.stats()

        assert stats["cache_enabled"] is True
        assert stats["cache_llm"] is True
        assert stats["cache_embed"] is False
        cache.close()

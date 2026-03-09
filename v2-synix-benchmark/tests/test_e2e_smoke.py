"""E2E smoke test — hits live Modal endpoints through the broker + cache.

Run manually:
    cd v2-synix-benchmark && uv run pytest tests/test_e2e_smoke.py -v -s

Requires live Modal servers. Skip with: pytest -k "not e2e"
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

from bench.broker import ModalBroker
from bench.cache import ResponseCache

LLM_URL = "https://synix--lens-llm-llm-serve.modal.run/v1"
EMBED_URL = "https://synix--lens-embed-serve.modal.run"
LLM_MODEL = "Qwen/Qwen3.5-35B-A3B"
EMBED_MODEL = "Alibaba-NLP/gte-modernbert-base"


@pytest.fixture
def broker(tmp_path: Path) -> ModalBroker:
    cache = ResponseCache(tmp_path / "smoke.db")
    return ModalBroker(
        cache=cache,
        llm_base_url=LLM_URL,
        embed_base_url=EMBED_URL,
        embed_timeout=60.0,
    )


@pytest.mark.skipif(
    os.environ.get("LENS_E2E") != "1",
    reason="Set LENS_E2E=1 to run live endpoint tests",
)
class TestE2ESmoke:
    @pytest.mark.timeout(120)
    def test_llm_basic_completion(self, broker: ModalBroker) -> None:
        """LLM returns a non-empty response."""
        resp = broker.chat_completion(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": "Say hello in exactly 3 words."}],
            max_tokens=32,
            temperature=0.0,
        )
        content = resp.choices[0].message.content
        print(f"\nLLM response: {content}")
        assert content and len(content.strip()) > 0

    @pytest.mark.timeout(120)
    def test_llm_tool_call(self, broker: ModalBroker) -> None:
        """LLM produces a parsed tool_call (not raw XML in content)."""
        resp = broker.chat_completion(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": "Search for clinical trial data"}],
            tools=[{
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "Search the memory store",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                    },
                },
            }],
            max_tokens=256,
            temperature=0.0,
        )
        msg = resp.choices[0].message
        content = msg.content or ""
        tool_calls = msg.tool_calls or []
        print(f"\nContent: {content[:200]}")
        print(f"Tool calls: {tool_calls}")

        # The critical check: tool calls should be parsed, not raw XML in content
        assert "<tool_call>" not in content, "Tool call XML leaked into content — parser not matching"
        assert len(tool_calls) > 0, "No tool_calls returned — model didn't call the tool"
        assert tool_calls[0].function.name == "search"

    @pytest.mark.timeout(120)
    def test_llm_cache_hit(self, broker: ModalBroker) -> None:
        """Second identical call is a cache hit (no API call)."""
        kwargs = dict(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": "What is 2+2?"}],
            max_tokens=16,
            temperature=0.0,
        )
        broker.chat_completion(**kwargs)
        stats_after_miss = broker.stats()

        broker.chat_completion(**kwargs)
        stats_after_hit = broker.stats()

        # Entry count shouldn't change on cache hit
        assert stats_after_hit["llm"]["total_entries"] == stats_after_miss["llm"]["total_entries"]
        print(f"\nCache stats: {stats_after_hit}")

    @pytest.mark.timeout(60)
    def test_embed_basic(self, broker: ModalBroker) -> None:
        """Embedding returns vectors of the right shape."""
        vectors = broker.embed(input=["hello world", "test query"], model=EMBED_MODEL)
        print(f"\nEmbedding dims: {len(vectors[0])}")
        assert len(vectors) == 2
        assert len(vectors[0]) == 768  # gte-modernbert-base is 768-dim
        assert all(isinstance(v, float) for v in vectors[0])

    @pytest.mark.timeout(60)
    def test_embed_cache_hit(self, broker: ModalBroker) -> None:
        """Second identical embed call is a cache hit."""
        broker.embed(input=["cache test"], model=EMBED_MODEL)
        broker.embed(input=["cache test"], model=EMBED_MODEL)
        stats = broker.stats()
        assert stats["embed"]["total_entries"] == 1
        print(f"\nEmbed cache stats: {stats['embed']}")

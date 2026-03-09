"""E2E smoke test — hits live Modal endpoints through the broker + cache.

Run manually:
    cd v2-synix-benchmark && LENS_E2E=1 uv run pytest tests/test_e2e_smoke.py -v -s

Requires live Modal servers. Skip with: pytest -k "not e2e"
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from bench.broker import ModalBroker
from bench.warmup import wait_for_modal

LLM_URL = "https://synix--lens-llm-llm-serve.modal.run/v1"
EMBED_URL = "https://synix--lens-embed-serve.modal.run"
LLM_MODEL = "Qwen/Qwen3.5-35B-A3B"
EMBED_MODEL = "Alibaba-NLP/gte-modernbert-base"


@pytest.fixture(scope="module")
def warm_endpoints() -> dict[str, float]:
    """Wait for Modal endpoints to be ready before any tests run."""
    return wait_for_modal(
        llm_base_url=LLM_URL,
        embed_base_url=EMBED_URL,
        timeout=300.0,
    )


@pytest.fixture
def broker(tmp_path: Path) -> ModalBroker:
    return ModalBroker(
        llm_base_url=LLM_URL,
        embed_base_url=EMBED_URL,
        cache_dir=tmp_path / "smoke_cache",
        embed_timeout=60.0,
    )


@pytest.mark.skipif(
    os.environ.get("LENS_E2E") != "1",
    reason="Set LENS_E2E=1 to run live endpoint tests",
)
class TestE2ESmoke:
    @pytest.mark.timeout(120)
    def test_llm_basic_completion(self, warm_endpoints: dict, broker: ModalBroker) -> None:
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
    def test_llm_tool_call(self, warm_endpoints: dict, broker: ModalBroker) -> None:
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
    def test_llm_cache_hit(self, warm_endpoints: dict, broker: ModalBroker) -> None:
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
    def test_embed_basic(self, warm_endpoints: dict, broker: ModalBroker) -> None:
        """Embedding returns vectors of the right shape."""
        vectors = broker.embed(input=["hello world", "test query"], model=EMBED_MODEL)
        print(f"\nEmbedding dims: {len(vectors[0])}")
        assert len(vectors) == 2
        assert len(vectors[0]) == 768  # gte-modernbert-base is 768-dim
        assert all(isinstance(v, float) for v in vectors[0])

    @pytest.mark.timeout(60)
    def test_embed_cache_hit(self, warm_endpoints: dict, broker: ModalBroker) -> None:
        """Second identical embed call is a cache hit."""
        broker.embed(input=["cache test"], model=EMBED_MODEL)
        broker.embed(input=["cache test"], model=EMBED_MODEL)
        stats = broker.stats()
        assert stats["embed"]["total_entries"] == 1
        print(f"\nEmbed cache stats: {stats['embed']}")


@pytest.mark.skipif(
    os.environ.get("LENS_E2E") != "1",
    reason="Set LENS_E2E=1 to run live endpoint tests",
)
class TestE2ECachePersistence:
    """Verify cache survives broker restart and namespace isolation works."""

    @pytest.mark.timeout(120)
    def test_cache_survives_broker_restart(
        self, warm_endpoints: dict, tmp_path: Path,
    ) -> None:
        """Data cached by broker A is available to broker B on same cache_dir."""
        cache_dir = tmp_path / "persist_test"

        # Broker A: make a call, data goes to cache
        broker_a = ModalBroker(
            llm_base_url=LLM_URL,
            embed_base_url=EMBED_URL,
            cache_dir=cache_dir,
            embed_timeout=60.0,
        )
        broker_a.embed(input=["persistence check"], model=EMBED_MODEL)
        stats_a = broker_a.stats()
        assert stats_a["embed"]["total_entries"] == 1

        # Broker B: new instance, same cache_dir — should see the cached entry
        broker_b = ModalBroker(
            llm_base_url=LLM_URL,
            embed_base_url=EMBED_URL,
            cache_dir=cache_dir,
            embed_timeout=60.0,
        )
        stats_b = broker_b.stats()
        assert stats_b["embed"]["total_entries"] == 1

        # Calling with same args should be a cache hit (no extra entry)
        broker_b.embed(input=["persistence check"], model=EMBED_MODEL)
        assert broker_b.stats()["embed"]["total_entries"] == 1
        print("\nCache persistence: OK — broker B sees broker A's data")

    @pytest.mark.timeout(120)
    def test_namespace_isolation(
        self, warm_endpoints: dict, tmp_path: Path,
    ) -> None:
        """Different cache_dir = isolated cache, no cross-contamination."""
        broker_x = ModalBroker(
            llm_base_url=LLM_URL,
            embed_base_url=EMBED_URL,
            cache_dir=tmp_path / "ns_x",
            embed_timeout=60.0,
        )
        broker_y = ModalBroker(
            llm_base_url=LLM_URL,
            embed_base_url=EMBED_URL,
            cache_dir=tmp_path / "ns_y",
            embed_timeout=60.0,
        )

        broker_x.embed(input=["namespace test"], model=EMBED_MODEL)
        assert broker_x.stats()["embed"]["total_entries"] == 1
        assert broker_y.stats()["embed"]["total_entries"] == 0
        print("\nNamespace isolation: OK — broker Y doesn't see broker X's data")

    @pytest.mark.timeout(120)
    def test_cache_disabled_no_db_created(self, tmp_path: Path) -> None:
        """When cache_enabled=False, no SQLite file is created."""
        cache_dir = tmp_path / "should_not_exist"
        broker = ModalBroker(
            llm_base_url=LLM_URL,
            embed_base_url=EMBED_URL,
            cache_dir=cache_dir,
            cache_enabled=False,
        )
        assert not cache_dir.exists()
        stats = broker.stats()
        assert stats["cache_enabled"] is False
        print("\nCache disabled: OK — no DB created")

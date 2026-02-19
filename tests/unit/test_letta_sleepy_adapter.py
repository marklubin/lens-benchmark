"""Tests for the LettaSleepy memory adapter.

Uses MockLettaClient (shared pattern with test_letta_adapter.py) plus a
MockOpenAI client to avoid requiring live servers.
"""
from __future__ import annotations

import uuid
from unittest.mock import MagicMock, patch

import pytest

from lens.adapters.base import CapabilityManifest, Document, SearchResult
from lens.adapters.registry import get_adapter


# ---------------------------------------------------------------------------
# Mock Letta client (same as letta adapter tests)
# ---------------------------------------------------------------------------


class _MockPassage:
    def __init__(self, text: str):
        self.text = text
        self.content = text
        self.timestamp = "2024-01-01T00:00:00"
        self.id = str(uuid.uuid4())
        self.tags = []


class _MockSearchResult:
    def __init__(self, text: str):
        self.content = text
        self.text = text
        self.timestamp = "2024-01-01T00:00:00"
        self.id = None


class _MockPassageSearchResponse:
    def __init__(self, results: list):
        self.results = results


class _MockPassagesNamespace:
    def __init__(self):
        self._store: dict[str, list[_MockPassage]] = {}

    def create(self, agent_id: str, text: str):
        p = _MockPassage(text)
        self._store.setdefault(agent_id, []).append(p)
        return [p]

    def search(self, agent_id: str, query: str) -> _MockPassageSearchResponse:
        passages = self._store.get(agent_id, [])
        q = query.lower()
        matches = [
            _MockSearchResult(p.text)
            for p in passages
            if q in p.text.lower() or any(w in p.text.lower() for w in q.split())
        ]
        return _MockPassageSearchResponse(matches)

    def list(self, agent_id: str, limit: int = 500):
        return list(self._store.get(agent_id, []))


class _MockAgent:
    def __init__(self, name: str):
        self.name = name
        self.id = str(uuid.uuid4())


class _MockAgentsNamespace:
    def __init__(self):
        self._agents: dict[str, _MockAgent] = {}
        self.passages = _MockPassagesNamespace()

    def create(self, name: str, model: str, embedding: str, memory_blocks: list):
        a = _MockAgent(name)
        self._agents[a.id] = a
        return a

    def delete(self, agent_id: str):
        self._agents.pop(agent_id, None)
        self.passages._store.pop(agent_id, None)

    def list(self):
        return list(self._agents.values())


class MockLettaClient:
    def __init__(self):
        self.agents = _MockAgentsNamespace()


# ---------------------------------------------------------------------------
# Mock OpenAI client for sleep cycle
# ---------------------------------------------------------------------------


def _make_mock_openai(synthesis_text: str = "SYNTHESIS: ep_001 shows baseline, ep_002 shows spike."):
    """Return a mock OpenAI client whose completions return synthesis_text."""
    choice = MagicMock()
    choice.message.content = synthesis_text
    completion = MagicMock()
    completion.choices = [choice]

    oai_instance = MagicMock()
    oai_instance.chat.completions.create.return_value = completion

    oai_cls = MagicMock(return_value=oai_instance)
    return oai_cls, oai_instance


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _make_adapter(variant: int = 2, env_overrides: dict | None = None):
    """Create a LettaSleepyAdapter with mocked Letta client and env vars."""
    from lens.adapters.letta_sleepy import LettaSleepyAdapter

    mock_client = MockLettaClient()
    env = {"LETTA_SLEEP_VARIANT": str(variant), **(env_overrides or {})}

    with patch.dict("os.environ", env, clear=False):
        adapter = LettaSleepyAdapter()
    adapter._client = mock_client
    return adapter


def _ingested_adapter(variant: int = 2, n: int = 3) -> "LettaSleepyAdapter":
    """Helper: adapter with reset + n ingested episodes."""
    adapter = _make_adapter(variant=variant)
    adapter.reset("scope_test")
    for i in range(1, n + 1):
        adapter.ingest(
            episode_id=f"ep_{i:03d}",
            scope_id="scope_test",
            timestamp=f"2024-01-0{i}T00:00:00",
            text=f"System report {i}: metric_a={i * 10}, metric_b={i * 5}.",
        )
    return adapter


# ---------------------------------------------------------------------------
# Tests: Registration
# ---------------------------------------------------------------------------


class TestLettaSleepyRegistration:
    def test_registered_by_name(self):
        cls = get_adapter("letta-sleepy")
        assert cls.__name__ == "LettaSleepyAdapter"

    def test_distinct_from_letta(self):
        assert get_adapter("letta-sleepy") is not get_adapter("letta")


# ---------------------------------------------------------------------------
# Tests: reset()
# ---------------------------------------------------------------------------


class TestLettaSleepyReset:
    def test_reset_creates_agent(self):
        adapter = _make_adapter()
        adapter.reset("scope_x")
        assert adapter._agent_id is not None

    def test_reset_clears_synthesis(self):
        adapter = _make_adapter()
        adapter._synthesis = "old synthesis"
        adapter.reset("scope_x")
        assert adapter._synthesis == ""

    def test_reset_clears_text_cache(self):
        adapter = _ingested_adapter()
        assert len(adapter._text_cache) == 3
        adapter.reset("scope_new")
        assert adapter._text_cache == {}

    def test_reset_deletes_previous_agent(self):
        adapter = _make_adapter()
        adapter.reset("scope_a")
        first_id = adapter._agent_id
        adapter.reset("scope_a")
        # Old agent should be gone from mock store
        assert first_id not in adapter._client.agents._agents

    def test_reset_agent_name_contains_sleepy(self):
        adapter = _make_adapter()
        adapter.reset("scope_z")
        agent = adapter._client.agents._agents[adapter._agent_id]
        assert "sleepy" in agent.name
        assert "scope_z" in agent.name


# ---------------------------------------------------------------------------
# Tests: ingest()
# ---------------------------------------------------------------------------


class TestLettaSleepyIngest:
    def test_ingest_stores_passage(self):
        adapter = _ingested_adapter(n=2)
        passages = adapter._client.agents.passages._store.get(adapter._agent_id, [])
        assert len(passages) == 2

    def test_ingest_passage_contains_episode_id(self):
        adapter = _ingested_adapter(n=1)
        passages = adapter._client.agents.passages._store[adapter._agent_id]
        assert "[ep_001]" in passages[0].text

    def test_ingest_caches_text(self):
        adapter = _ingested_adapter(n=2)
        assert "ep_001" in adapter._text_cache
        assert "ep_002" in adapter._text_cache

    def test_ingest_requires_reset(self):
        adapter = _make_adapter()
        with pytest.raises(Exception):
            adapter.ingest("ep_001", "scope_x", "2024-01-01", "text")


# ---------------------------------------------------------------------------
# Tests: prepare() — variant 0 (control, no sleep)
# ---------------------------------------------------------------------------


class TestLettaSleepyPrepareV0:
    def test_v0_skips_sleep(self):
        adapter = _ingested_adapter(variant=0)
        oai_cls, oai_instance = _make_mock_openai()
        with patch("lens.adapters.letta_sleepy._OpenAI", oai_cls):
            adapter.prepare("scope_test", checkpoint=5)
        oai_instance.chat.completions.create.assert_not_called()
        assert adapter._synthesis == ""

    def test_v0_search_has_no_synthesis_result(self):
        adapter = _ingested_adapter(variant=0)
        results = adapter.search("metric_a")
        assert not any(r.ref_id == "synthesis" for r in results)


# ---------------------------------------------------------------------------
# Tests: prepare() — variants 1-3 (sleep active)
# ---------------------------------------------------------------------------


class TestLettaSleepyPrepareActive:
    @pytest.mark.parametrize("variant", [1, 2, 3])
    def test_prepare_calls_llm(self, variant):
        adapter = _ingested_adapter(variant=variant)
        oai_cls, oai_instance = _make_mock_openai("synth result")
        with patch("lens.adapters.letta_sleepy._OpenAI", oai_cls):
            adapter.prepare("scope_test", checkpoint=3)
        oai_instance.chat.completions.create.assert_called_once()

    @pytest.mark.parametrize("variant", [1, 2, 3])
    def test_prepare_stores_synthesis(self, variant):
        adapter = _ingested_adapter(variant=variant)
        oai_cls, _ = _make_mock_openai("SYNTHESIS TEXT")
        with patch("lens.adapters.letta_sleepy._OpenAI", oai_cls):
            adapter.prepare("scope_test", checkpoint=3)
        assert adapter._synthesis == "SYNTHESIS TEXT"

    def test_prepare_passes_passages_to_llm(self):
        adapter = _ingested_adapter(variant=2, n=2)
        oai_cls, oai_instance = _make_mock_openai("synth")
        with patch("lens.adapters.letta_sleepy._OpenAI", oai_cls):
            adapter.prepare("scope_test", checkpoint=2)
        call_kwargs = oai_instance.chat.completions.create.call_args
        messages = call_kwargs.kwargs.get("messages", []) or []
        user_msg = next((m["content"] for m in messages if m["role"] == "user"), "")
        assert "[ep_001]" in user_msg or "ep_001" in user_msg

    def test_v1_objective_in_prompt(self):
        adapter = _ingested_adapter(variant=1)
        oai_cls, oai_instance = _make_mock_openai("s")
        with patch("lens.adapters.letta_sleepy._OpenAI", oai_cls):
            adapter.prepare("scope_test", checkpoint=3)
        user_msg = oai_instance.chat.completions.create.call_args.kwargs["messages"][1]["content"]
        assert "comprehensive summary" in user_msg.lower() or "summarise" in user_msg.lower() or "summarize" in user_msg.lower()

    def test_v2_objective_in_prompt(self):
        adapter = _ingested_adapter(variant=2)
        oai_cls, oai_instance = _make_mock_openai("s")
        with patch("lens.adapters.letta_sleepy._OpenAI", oai_cls):
            adapter.prepare("scope_test", checkpoint=3)
        user_msg = oai_instance.chat.completions.create.call_args.kwargs["messages"][1]["content"]
        assert "patterns" in user_msg.lower() or "anomalies" in user_msg.lower()

    def test_v3_objective_in_prompt(self):
        adapter = _ingested_adapter(variant=3)
        oai_cls, oai_instance = _make_mock_openai("s")
        with patch("lens.adapters.letta_sleepy._OpenAI", oai_cls):
            adapter.prepare("scope_test", checkpoint=3)
        user_msg = oai_instance.chat.completions.create.call_args.kwargs["messages"][1]["content"]
        assert "changed" in user_msg.lower() or "causal" in user_msg.lower() or "transitions" in user_msg.lower()

    def test_v1_v2_v3_prompts_differ(self):
        """Each variant must produce a distinct user prompt."""
        prompts = {}
        for variant in (1, 2, 3):
            adapter = _ingested_adapter(variant=variant)
            oai_cls, oai_instance = _make_mock_openai("s")
            with patch("lens.adapters.letta_sleepy._OpenAI", oai_cls):
                adapter.prepare("scope_test", checkpoint=3)
            user_msg = oai_instance.chat.completions.create.call_args.kwargs["messages"][1]["content"]
            prompts[variant] = user_msg
        assert prompts[1] != prompts[2]
        assert prompts[2] != prompts[3]
        assert prompts[1] != prompts[3]

    def test_prepare_llm_failure_is_nonfatal(self):
        adapter = _ingested_adapter(variant=2)
        oai_instance = MagicMock()
        oai_instance.chat.completions.create.side_effect = RuntimeError("timeout")
        oai_cls = MagicMock(return_value=oai_instance)
        with patch("lens.adapters.letta_sleepy._OpenAI", oai_cls):
            adapter.prepare("scope_test", checkpoint=3)  # must not raise
        assert adapter._synthesis == ""

    def test_prepare_no_passages_skips_sleep(self):
        """prepare() should not call LLM if no passages have been ingested."""
        adapter = _make_adapter(variant=2)
        adapter.reset("scope_empty")
        oai_cls, oai_instance = _make_mock_openai("s")
        with patch("lens.adapters.letta_sleepy._OpenAI", oai_cls):
            adapter.prepare("scope_empty", checkpoint=0)
        oai_instance.chat.completions.create.assert_not_called()

    def test_synthesis_truncated_to_max(self):
        long_synthesis = "X" * 5000
        adapter = _ingested_adapter(variant=2)
        oai_cls, _ = _make_mock_openai(long_synthesis)
        with patch("lens.adapters.letta_sleepy._OpenAI", oai_cls):
            adapter.prepare("scope_test", checkpoint=3)
        assert len(adapter._synthesis) <= 3000

    def test_prepare_uses_lens_api_key(self):
        adapter = _ingested_adapter(variant=2)
        oai_cls, _ = _make_mock_openai("s")
        env = {"LENS_LLM_API_KEY": "mykey123", "LENS_LLM_API_BASE": "https://example.com/v1"}
        with patch("lens.adapters.letta_sleepy._OpenAI", oai_cls), patch.dict("os.environ", env):
            adapter.prepare("scope_test", checkpoint=3)
        oai_cls.assert_called_once()
        call_kwargs = oai_cls.call_args.kwargs
        assert call_kwargs.get("api_key") == "mykey123"
        assert call_kwargs.get("base_url") == "https://example.com/v1"

    def test_prepare_strips_provider_prefix_from_model(self):
        adapter = _make_adapter(variant=2, env_overrides={"LETTA_LLM_MODEL": "together/Qwen/Qwen3-235B"})
        adapter.reset("scope_test")
        adapter.ingest("ep_001", "scope_test", "2024-01-01", "data")
        oai_cls, oai_instance = _make_mock_openai("s")
        with patch("lens.adapters.letta_sleepy._OpenAI", oai_cls):
            adapter.prepare("scope_test", checkpoint=1)
        model_used = oai_instance.chat.completions.create.call_args.kwargs["model"]
        assert model_used == "Qwen/Qwen3-235B"
        assert not model_used.startswith("together/")


# ---------------------------------------------------------------------------
# Tests: search()
# ---------------------------------------------------------------------------


class TestLettaSleepySearch:
    def test_search_without_synthesis_returns_passage_results(self):
        adapter = _ingested_adapter(variant=0)
        results = adapter.search("metric_a")
        assert len(results) > 0
        assert all(r.ref_id != "synthesis" for r in results)

    def test_search_with_synthesis_prepends_it(self):
        adapter = _ingested_adapter(variant=2)
        adapter._synthesis = "Consolidated: ep_001 shows baseline."
        results = adapter.search("metric_a")
        assert results[0].ref_id == "synthesis"

    def test_search_synthesis_text_in_result(self):
        adapter = _ingested_adapter(variant=2)
        adapter._synthesis = "Consolidated: ep_001 shows baseline."
        results = adapter.search("metric_a")
        syn = next(r for r in results if r.ref_id == "synthesis")
        assert "Consolidated" in syn.text

    def test_search_total_within_limit(self):
        adapter = _ingested_adapter(variant=2, n=5)
        adapter._synthesis = "some synthesis"
        results = adapter.search("metric_a", limit=5)
        assert len(results) <= 5

    def test_search_empty_query_returns_empty(self):
        adapter = _ingested_adapter()
        adapter._synthesis = "some synthesis"
        assert adapter.search("") == []
        assert adapter.search("   ") == []

    def test_search_no_agent_returns_empty(self):
        adapter = _make_adapter()
        assert adapter.search("query") == []

    def test_search_synthesis_score_is_moderate(self):
        adapter = _ingested_adapter()
        adapter._synthesis = "synth"
        results = adapter.search("metric_a")
        syn = next(r for r in results if r.ref_id == "synthesis")
        # Score should be between 0 and 1 (not dominating)
        assert 0.0 < syn.score <= 1.0

    def test_search_returns_searchresult_instances(self):
        adapter = _ingested_adapter()
        adapter._synthesis = "synth"
        results = adapter.search("metric_a")
        assert all(isinstance(r, SearchResult) for r in results)


# ---------------------------------------------------------------------------
# Tests: retrieve()
# ---------------------------------------------------------------------------


class TestLettaSleepyRetrieve:
    def test_retrieve_episode_from_cache(self):
        adapter = _ingested_adapter(n=2)
        doc = adapter.retrieve("ep_001")
        assert isinstance(doc, Document)
        assert "ep_001" not in doc.text  # text cache stores raw text, not prefixed

    def test_retrieve_synthesis(self):
        adapter = _ingested_adapter()
        adapter._synthesis = "My synthesis text"
        doc = adapter.retrieve("synthesis")
        assert isinstance(doc, Document)
        assert doc.ref_id == "synthesis"
        assert "My synthesis text" in doc.text

    def test_retrieve_synthesis_empty_returns_none(self):
        adapter = _ingested_adapter()
        # No synthesis set
        assert adapter.retrieve("synthesis") is None

    def test_retrieve_missing_episode_returns_none(self):
        adapter = _ingested_adapter(n=1)
        assert adapter.retrieve("ep_999") is None


# ---------------------------------------------------------------------------
# Tests: get_capabilities()
# ---------------------------------------------------------------------------


class TestLettaSleepyCapabilities:
    @pytest.mark.parametrize("variant", [0, 1, 2, 3])
    def test_capabilities_include_variant(self, variant):
        adapter = _make_adapter(variant=variant)
        caps = adapter.get_capabilities()
        assert isinstance(caps, CapabilityManifest)
        assert any(f"v{variant}" in m for m in caps.search_modes)

    def test_capabilities_has_batch_retrieve(self):
        adapter = _make_adapter()
        caps = adapter.get_capabilities()
        tool_names = [t.name for t in caps.extra_tools]
        assert "batch_retrieve" in tool_names

    def test_capabilities_mentions_synthesis_in_batch_retrieve(self):
        adapter = _make_adapter()
        caps = adapter.get_capabilities()
        bt = next(t for t in caps.extra_tools if t.name == "batch_retrieve")
        assert "synthesis" in bt.description.lower()


# ---------------------------------------------------------------------------
# Tests: batch_retrieve extended tool
# ---------------------------------------------------------------------------


class TestLettaSleepyBatchRetrieve:
    def test_batch_retrieve_returns_episodes(self):
        adapter = _ingested_adapter(n=3)
        result = adapter.call_extended_tool("batch_retrieve", {"ref_ids": ["ep_001", "ep_002"]})
        assert result["count"] == 2
        assert len(result["documents"]) == 2

    def test_batch_retrieve_includes_synthesis(self):
        adapter = _ingested_adapter(n=2)
        adapter._synthesis = "Some synthesis"
        result = adapter.call_extended_tool(
            "batch_retrieve", {"ref_ids": ["synthesis", "ep_001"]}
        )
        assert result["count"] == 2
        ref_ids = [d["ref_id"] for d in result["documents"]]
        assert "synthesis" in ref_ids
        assert "ep_001" in ref_ids

    def test_batch_retrieve_skips_missing(self):
        adapter = _ingested_adapter(n=1)
        result = adapter.call_extended_tool("batch_retrieve", {"ref_ids": ["ep_999"]})
        assert result["count"] == 0

    def test_batch_retrieve_empty_list(self):
        adapter = _ingested_adapter()
        result = adapter.call_extended_tool("batch_retrieve", {"ref_ids": []})
        assert result["count"] == 0
        assert result["documents"] == []


# ---------------------------------------------------------------------------
# Tests: _strip_provider_prefix helper
# ---------------------------------------------------------------------------


class TestStripProviderPrefix:
    def test_strips_together_prefix(self):
        from lens.adapters.letta_sleepy import _strip_provider_prefix
        assert _strip_provider_prefix("together/Qwen/Qwen3-235B") == "Qwen/Qwen3-235B"

    def test_no_prefix_unchanged(self):
        from lens.adapters.letta_sleepy import _strip_provider_prefix
        assert _strip_provider_prefix("gpt-4o") == "gpt-4o"

    def test_openai_prefix(self):
        from lens.adapters.letta_sleepy import _strip_provider_prefix
        assert _strip_provider_prefix("openai/gpt-4o-mini") == "gpt-4o-mini"

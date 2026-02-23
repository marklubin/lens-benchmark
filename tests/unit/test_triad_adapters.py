"""Unit tests for Triad Memory Protocol adapters (panel + pairs).

Tests both triad adapters covering:
- Registration in the adapter registry
- Lifecycle: reset → ingest → prepare → search → retrieve
- Buffering without LLM calls
- Notebook state management
- Fallback behavior when no LLM is configured
- Mocked LLM paths for prepare() and search()
- v1 object store meta-schema prompts and operations
- Pairs cross-reference consult structure
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch
import pytest

from lens.adapters.base import (
    CapabilityManifest,
    Document,
    SearchResult,
)
from lens.adapters.registry import get_adapter, list_adapters
from lens.adapters.triad import (
    FACETS_4,
    _TriadBase,
    _complete,
    _strip_provider_prefix,
)
from lens.adapters.triad_v1 import (
    TriadV1PanelAdapter,
    TriadV1PairsAdapter,
    _V1_RECORD_SYSTEMS,
    _V1_CONSULT_SYSTEMS,
    _V1_SYNTHESIS_SYSTEM,
    _V1_PAIR_DESCRIPTIONS,
    _V1_PAIR_CONSULT_SYSTEM,
    _V1_PAIR_SYNTHESIS_SYSTEM,
    _build_v1_synthesis_user,
    _build_v1_pair_synthesis_user,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SAMPLE_EPISODES = [
    ("ep-1", "scope-a", "2024-01-01T09:00:00", "Alice is a software engineer at Acme Corp."),
    ("ep-2", "scope-a", "2024-01-02T10:00:00", "Bob is Alice's manager. He mentors her."),
    ("ep-3", "scope-a", "2024-01-03T11:00:00", "Alice's debugging caused the server outage to be fixed in 2 hours."),
]


def _ingest_samples(adapter: _TriadBase) -> None:
    adapter.reset("scope-a")
    for eid, sid, ts, text in SAMPLE_EPISODES:
        adapter.ingest(eid, sid, ts, text)


def _make_mock_completion(responses: list[str] | None = None) -> MagicMock:
    """Create a mock OpenAI client whose chat.completions.create returns canned responses."""
    idx = {"i": 0}
    defaults = responses or ["mock response"]

    def side_effect(**kwargs):
        text = defaults[idx["i"] % len(defaults)]
        idx["i"] += 1
        msg = MagicMock()
        msg.content = text
        choice = MagicMock()
        choice.message = msg
        resp = MagicMock()
        resp.choices = [choice]
        return resp

    client = MagicMock()
    client.chat.completions.create = MagicMock(side_effect=side_effect)
    return client


def _setup_mock(adapter: _TriadBase, responses: list[str]) -> MagicMock:
    """Inject a mock client and patch _init_client so prepare() doesn't overwrite it."""
    mock_client = _make_mock_completion(responses)
    adapter._oai = mock_client
    adapter._model = "test-model"
    adapter._init_client = lambda: None  # type: ignore[assignment]
    return mock_client


ALL_ADAPTER_CLASSES = [
    TriadV1PanelAdapter,
    TriadV1PairsAdapter,
]

ALL_ADAPTER_NAMES = [
    "triadv1-panel",
    "triadv1-pairs",
]


# ===========================================================================
# Registry tests
# ===========================================================================


class TestTriadRegistry:
    def test_both_registered(self):
        adapters = list_adapters()
        for name in ALL_ADAPTER_NAMES:
            assert name in adapters, f"{name} not registered"

    @pytest.mark.parametrize("name,cls", [
        ("triadv1-panel", TriadV1PanelAdapter),
        ("triadv1-pairs", TriadV1PairsAdapter),
    ])
    def test_get_adapter_returns_correct_class(self, name: str, cls: type):
        assert get_adapter(name) is cls


# ===========================================================================
# _strip_provider_prefix
# ===========================================================================


class TestStripProviderPrefix:
    def test_strips_openai_prefix(self):
        assert _strip_provider_prefix("openai/gpt-4o") == "gpt-4o"

    def test_strips_together_prefix(self):
        assert _strip_provider_prefix("together/meta-llama/Llama-3-70b") == "meta-llama/Llama-3-70b"

    def test_leaves_bare_model_alone(self):
        assert _strip_provider_prefix("gpt-4o-mini") == "gpt-4o-mini"

    def test_leaves_unknown_prefix_alone(self):
        assert _strip_provider_prefix("anthropic/claude-3") == "anthropic/claude-3"


# ===========================================================================
# Synthesis prompt builders
# ===========================================================================


class TestSynthesisPrompts:
    def test_v1_synthesis_user_builder(self):
        responses = {
            "entity": "ent resp", "relation": "rel resp",
            "event": "evt resp", "cause": "cau resp",
        }
        u = _build_v1_synthesis_user("Why?", FACETS_4, responses)
        assert "ENTITY STORE SPECIALIST:" in u
        assert "RELATION STORE SPECIALIST:" in u
        assert "EVENT STORE SPECIALIST:" in u
        assert "CAUSE STORE SPECIALIST:" in u
        assert "ent resp" in u
        assert "ACCOMMODATE" in u

    def test_missing_response_defaults(self):
        u = _build_v1_synthesis_user("Q?", FACETS_4, {})
        assert "(no response)" in u


# ===========================================================================
# Shared base behavior (tested via each concrete subclass)
# ===========================================================================


@pytest.fixture(params=ALL_ADAPTER_CLASSES, ids=["panel", "pairs"])
def adapter(request) -> _TriadBase:
    return request.param()


class TestTriadBaseLifecycle:
    """Tests that apply to both triad adapters."""

    def test_reset_clears_state(self, adapter: _TriadBase):
        _ingest_samples(adapter)
        assert len(adapter._episodes) == 3
        adapter.reset("scope-b")
        assert adapter._episodes == []
        for key in adapter._notebook_keys:
            assert adapter._notebooks[key] == "(empty)"

    def test_ingest_buffers_episodes(self, adapter: _TriadBase):
        _ingest_samples(adapter)
        assert len(adapter._episodes) == 3
        assert adapter._episodes[0]["episode_id"] == "ep-1"
        assert adapter._episodes[2]["text"] == SAMPLE_EPISODES[2][3]

    def test_ingest_stores_meta(self, adapter: _TriadBase):
        adapter.reset("s")
        adapter.ingest("ep-x", "s", "2024-01-01", "text", meta={"tag": "test"})
        assert adapter._episodes[0]["meta"] == {"tag": "test"}

    def test_ingest_defaults_meta_to_empty(self, adapter: _TriadBase):
        adapter.reset("s")
        adapter.ingest("ep-x", "s", "2024-01-01", "text")
        assert adapter._episodes[0]["meta"] == {}

    def test_retrieve_episode_by_id(self, adapter: _TriadBase):
        _ingest_samples(adapter)
        doc = adapter.retrieve("ep-2")
        assert doc is not None
        assert isinstance(doc, Document)
        assert "Bob is Alice's manager" in doc.text

    def test_retrieve_nonexistent_returns_none(self, adapter: _TriadBase):
        _ingest_samples(adapter)
        assert adapter.retrieve("ep-999") is None

    def test_retrieve_notebook_when_empty(self, adapter: _TriadBase):
        adapter.reset("s")
        key = adapter._notebook_keys[0]
        doc = adapter.retrieve(f"notebook-{key}")
        assert doc is not None
        assert doc.text == "(empty)"

    def test_retrieve_notebook_invalid_facet(self, adapter: _TriadBase):
        adapter.reset("s")
        assert adapter.retrieve("notebook-nonexistent") is None

    def test_get_capabilities(self, adapter: _TriadBase):
        caps = adapter.get_capabilities()
        assert isinstance(caps, CapabilityManifest)
        assert "synthesis" in caps.search_modes
        assert caps.max_results_per_search == 1

    def test_synthetic_refs_empty_before_prepare(self, adapter: _TriadBase):
        _ingest_samples(adapter)
        assert adapter.get_synthetic_refs() == []

    def test_synthetic_refs_after_notebook_update(self, adapter: _TriadBase):
        adapter.reset("s")
        key = adapter._notebook_keys[0]
        adapter._notebooks[key] = "Updated notebook content"
        refs = adapter.get_synthetic_refs()
        assert len(refs) >= 1
        ref_ids = [r[0] for r in refs]
        assert f"notebook-{key}" in ref_ids

    def test_adapter_label_set(self, adapter: _TriadBase):
        assert adapter._adapter_label != ""
        assert adapter._adapter_label.startswith("triad")


# ===========================================================================
# Notebook key configuration
# ===========================================================================


class TestNotebookKeys:
    def test_panel_has_four_stores(self):
        a = TriadV1PanelAdapter()
        assert a._notebook_keys == FACETS_4
        assert len(a._notebook_keys) == 4

    def test_pairs_has_four_stores(self):
        assert TriadV1PairsAdapter()._notebook_keys == FACETS_4

    def test_4facet_ordering(self):
        """Developmental order: entity → relation → event → cause."""
        assert FACETS_4 == ("entity", "relation", "event", "cause")


# ===========================================================================
# Fallback behavior (no LLM configured → returns raw episodes)
# ===========================================================================


class TestFallbackSearch:
    """When no LLM client is available, search() should fall back to raw episodes."""

    @pytest.mark.parametrize("cls", ALL_ADAPTER_CLASSES, ids=["panel", "pairs"])
    def test_fallback_returns_episodes(self, cls: type):
        a = cls()
        _ingest_samples(a)
        results = a.search("anything")
        assert len(results) == 3
        assert all(isinstance(r, SearchResult) for r in results)
        assert results[0].ref_id == "ep-1"
        assert results[0].score == 0.5

    def test_fallback_respects_limit(self):
        a = TriadV1PanelAdapter()
        _ingest_samples(a)
        results = a.search("anything", limit=2)
        assert len(results) == 2


# ===========================================================================
# _complete() helper
# ===========================================================================


class TestCompleteHelper:
    def test_calls_openai_correctly(self):
        mock_client = _make_mock_completion(["hello world"])
        result = _complete(mock_client, "gpt-4o", "system msg", "user msg", max_tokens=100)
        assert result == "hello world"
        mock_client.chat.completions.create.assert_called_once_with(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "system msg"},
                {"role": "user", "content": "user msg"},
            ],
            max_tokens=100,
            temperature=0.0,
        )

    def test_returns_empty_on_none_content(self):
        mock_client = MagicMock()
        msg = MagicMock()
        msg.content = None
        choice = MagicMock()
        choice.message = msg
        resp = MagicMock()
        resp.choices = [choice]
        mock_client.chat.completions.create.return_value = resp

        result = _complete(mock_client, "m", "s", "u")
        assert result == ""


# ===========================================================================
# _init_client env var handling
# ===========================================================================


class TestInitClient:
    @patch.dict("os.environ", {
        "LENS_LLM_API_KEY": "test-key",
        "LENS_LLM_API_BASE": "http://localhost:8080/v1",
        "LENS_LLM_MODEL": "together/meta-llama/Llama-3-70b",
    })
    @patch("lens.adapters.triad._OpenAI")
    def test_uses_env_vars(self, mock_openai_cls):
        a = TriadV1PanelAdapter()
        a.reset("s")
        a._init_client()
        mock_openai_cls.assert_called_once_with(
            api_key="test-key",
            base_url="http://localhost:8080/v1",
        )
        assert a._model == "meta-llama/Llama-3-70b"

    @patch.dict("os.environ", {}, clear=True)
    @patch("lens.adapters.triad._OpenAI")
    def test_defaults(self, mock_openai_cls):
        a = TriadV1PanelAdapter()
        a.reset("s")
        a._init_client()
        mock_openai_cls.assert_called_once_with(api_key="dummy")
        assert a._model == "gpt-4o-mini"

    @patch("lens.adapters.triad._OpenAI", None)
    def test_raises_without_openai_package(self):
        a = TriadV1PanelAdapter()
        a.reset("s")
        with pytest.raises(RuntimeError, match="openai package required"):
            a._init_client()


# ===========================================================================
# v1 prompt validation
# ===========================================================================


class TestTriadV1Prompts:
    """Verify v1 prompts contain required meta-schema fields and operations."""

    def test_record_systems_contain_operations(self):
        for key, prompt in _V1_RECORD_SYSTEMS.items():
            assert "INSTANTIATE" in prompt, f"{key} missing INSTANTIATE"
            assert "UPDATE" in prompt, f"{key} missing UPDATE"
            assert "ACCOMMODATE" in prompt, f"{key} missing ACCOMMODATE"

    def test_record_systems_contain_meta_schema_fields(self):
        for key, prompt in _V1_RECORD_SYSTEMS.items():
            assert "identity" in prompt, f"{key} missing identity"
            assert "schema" in prompt, f"{key} missing schema"
            assert "interface" in prompt, f"{key} missing interface"
            assert "state" in prompt, f"{key} missing state"
            assert "lifecycle" in prompt, f"{key} missing lifecycle"

    def test_consult_systems_reference_object_store(self):
        for key, prompt in _V1_CONSULT_SYSTEMS.items():
            assert "object store" in prompt.lower(), f"{key} missing object store ref"

    def test_consult_systems_mention_accommodate(self):
        for key, prompt in _V1_CONSULT_SYSTEMS.items():
            assert "ACCOMMODATE" in prompt, f"{key} missing ACCOMMODATE"

    def test_synthesis_mentions_accommodate(self):
        assert "ACCOMMODATE" in _V1_SYNTHESIS_SYSTEM

    def test_synthesis_mentions_cross_references(self):
        assert "cross-reference" in _V1_SYNTHESIS_SYSTEM.lower()


# ===========================================================================
# Panel with mocked LLM
# ===========================================================================


class TestTriadV1PanelWithMockedLLM:
    def test_prepare_updates_all_stores(self):
        a = TriadV1PanelAdapter()
        a.reset("s")
        a.ingest("ep-1", "s", "2024-01-01", "Alice met Bob.")

        mock_client = _setup_mock(a, ["Updated store"])
        a.prepare("s", 1)

        assert mock_client.chat.completions.create.call_count == len(FACETS_4)
        for f in FACETS_4:
            assert a._notebooks[f] == "Updated store"

    def test_search_consults_then_synthesizes(self):
        a = TriadV1PanelAdapter()
        a.reset("s")

        call_count = {"n": 0}

        def side_effect(**kwargs):
            call_count["n"] += 1
            msg = MagicMock()
            msg.content = f"Response {call_count['n']}"
            choice = MagicMock()
            choice.message = msg
            resp = MagicMock()
            resp.choices = [choice]
            return resp

        mock_client = MagicMock()
        mock_client.chat.completions.create = MagicMock(side_effect=side_effect)
        a._oai = mock_client
        a._model = "test-model"
        for f in FACETS_4:
            a._notebooks[f] = f"Content for {f}"

        results = a.search("Tell me about Alice")
        assert len(results) == 1
        assert results[0].ref_id == "triadv1-panel-answer"
        # 4 consults + 1 synthesis = 5
        assert call_count["n"] == 5

    def test_synthesis_prompt_mentions_accommodate(self):
        a = TriadV1PanelAdapter()
        a.reset("s")

        captured_systems: list[str] = []

        def side_effect(**kwargs):
            captured_systems.append(kwargs["messages"][0]["content"])
            msg = MagicMock()
            msg.content = "Answer"
            choice = MagicMock()
            choice.message = msg
            resp = MagicMock()
            resp.choices = [choice]
            return resp

        mock_client = MagicMock()
        mock_client.chat.completions.create = MagicMock(side_effect=side_effect)
        a._oai = mock_client
        a._model = "test-model"
        for f in FACETS_4:
            a._notebooks[f] = f"Content for {f}"

        a.search("question")

        synthesis_system = captured_systems[-1]
        assert "ACCOMMODATE" in synthesis_system

    def test_consult_uses_higher_max_tokens(self):
        a = TriadV1PanelAdapter()
        a.reset("s")
        mock_client = _make_mock_completion(["resp"])
        a._oai = mock_client
        a._model = "test-model"
        for f in FACETS_4:
            a._notebooks[f] = f"Content for {f}"

        a.search("q")

        calls = mock_client.chat.completions.create.call_args_list
        for call in calls[:4]:
            assert call.kwargs["max_tokens"] == 2000
        assert calls[-1].kwargs["max_tokens"] == 2500

    def test_prepare_uses_object_store_prompt(self):
        a = TriadV1PanelAdapter()
        _ingest_samples(a)
        mock_client = _setup_mock(a, ["updated store"])

        a.prepare("scope-a", 1)

        first_call = mock_client.chat.completions.create.call_args_list[0]
        system_msg = first_call.kwargs["messages"][0]["content"]
        user_msg = first_call.kwargs["messages"][1]["content"]
        assert "object-store" in system_msg.lower()
        assert "CURRENT OBJECT STORE:" in user_msg

    def test_search_uses_object_store_user_template(self):
        a = TriadV1PanelAdapter()
        a.reset("s")
        a._oai = _make_mock_completion(["The answer"])
        a._model = "test-model"
        for f in FACETS_4:
            a._notebooks[f] = "some objects"

        a.search("question")

        first_call = a._oai.chat.completions.create.call_args_list[0]
        user_msg = first_call.kwargs["messages"][1]["content"]
        assert "OBJECT STORE:" in user_msg

    def test_prepare_handles_llm_error(self):
        a = TriadV1PanelAdapter()
        _ingest_samples(a)
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = RuntimeError("API down")
        a._oai = mock_client
        a._model = "test-model"
        a._init_client = lambda: None  # type: ignore[assignment]

        a.prepare("scope-a", 1)
        for f in FACETS_4:
            assert a._notebooks[f] == "(empty)"

    def test_search_handles_llm_error(self):
        a = TriadV1PanelAdapter()
        _ingest_samples(a)
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = RuntimeError("API down")
        a._oai = mock_client
        a._model = "test-model"
        for f in FACETS_4:
            a._notebooks[f] = "some content"

        results = a.search("q")
        assert len(results) == 3
        assert results[0].score == 0.5


# ===========================================================================
# Pairs prompts and helpers
# ===========================================================================


class TestTriadV1PairsPrompts:
    def test_pair_descriptions_cover_all_combinations(self):
        from itertools import combinations
        expected = list(combinations(FACETS_4, 2))
        assert len(_V1_PAIR_DESCRIPTIONS) == 6
        for pair in expected:
            assert pair in _V1_PAIR_DESCRIPTIONS, f"Missing pair: {pair}"

    def test_pair_consult_system_has_placeholders(self):
        assert "{facet_a}" in _V1_PAIR_CONSULT_SYSTEM
        assert "{facet_b}" in _V1_PAIR_CONSULT_SYSTEM
        assert "{pair_desc}" in _V1_PAIR_CONSULT_SYSTEM

    def test_pair_consult_system_mentions_accommodate(self):
        assert "ACCOMMODATE" in _V1_PAIR_CONSULT_SYSTEM

    def test_pair_synthesis_system_mentions_pairings(self):
        assert "6" in _V1_PAIR_SYNTHESIS_SYSTEM
        assert "pairing" in _V1_PAIR_SYNTHESIS_SYSTEM.lower()

    def test_pair_synthesis_user_builder(self):
        pairs = [("entity", "relation"), ("event", "cause")]
        responses = {
            ("entity", "relation"): "er resp",
            ("event", "cause"): "vx resp",
        }
        u = _build_v1_pair_synthesis_user("Why?", pairs, responses)
        assert "ENTITY\u00d7RELATION" in u
        assert "EVENT\u00d7CAUSE" in u
        assert "er resp" in u
        assert "vx resp" in u
        assert "Why?" in u

    def test_pair_synthesis_missing_response(self):
        pairs = [("entity", "relation")]
        u = _build_v1_pair_synthesis_user("Q?", pairs, {})
        assert "(no response)" in u


# ===========================================================================
# Pairs with mocked LLM
# ===========================================================================


class TestTriadV1PairsWithMockedLLM:
    def test_search_6_consults_plus_synthesis(self):
        """6 pair consults + 1 synthesis = 7 calls."""
        a = TriadV1PairsAdapter()
        a.reset("s")

        call_count = {"n": 0}

        def side_effect(**kwargs):
            call_count["n"] += 1
            msg = MagicMock()
            msg.content = f"Response {call_count['n']}"
            choice = MagicMock()
            choice.message = msg
            resp = MagicMock()
            resp.choices = [choice]
            return resp

        mock_client = MagicMock()
        mock_client.chat.completions.create = MagicMock(side_effect=side_effect)
        a._oai = mock_client
        a._model = "test-model"
        for f in FACETS_4:
            a._notebooks[f] = f"Content for {f}"

        results = a.search("Tell me about the relationships")
        assert len(results) == 1
        assert results[0].ref_id == "triadv1-pairs-answer"
        # C(4,2) = 6 pair consults + 1 synthesis = 7
        assert call_count["n"] == 7

    def test_consult_receives_two_stores(self):
        """Each pair consult should receive both facet stores."""
        a = TriadV1PairsAdapter()
        a.reset("s")

        captured_users: list[str] = []

        def side_effect(**kwargs):
            captured_users.append(kwargs["messages"][1]["content"])
            msg = MagicMock()
            msg.content = "Answer"
            choice = MagicMock()
            choice.message = msg
            resp = MagicMock()
            resp.choices = [choice]
            return resp

        mock_client = MagicMock()
        mock_client.chat.completions.create = MagicMock(side_effect=side_effect)
        a._oai = mock_client
        a._model = "test-model"
        for f in FACETS_4:
            a._notebooks[f] = f"Store-{f}"

        a.search("question")

        for user_msg in captured_users[:6]:
            assert "OBJECT STORE:" in user_msg

    def test_synthesis_uses_pair_prompt(self):
        """Synthesis should use the pair-specific synthesis system prompt."""
        a = TriadV1PairsAdapter()
        a.reset("s")

        captured_systems: list[str] = []

        def side_effect(**kwargs):
            captured_systems.append(kwargs["messages"][0]["content"])
            msg = MagicMock()
            msg.content = "Answer"
            choice = MagicMock()
            choice.message = msg
            resp = MagicMock()
            resp.choices = [choice]
            return resp

        mock_client = MagicMock()
        mock_client.chat.completions.create = MagicMock(side_effect=side_effect)
        a._oai = mock_client
        a._model = "test-model"
        for f in FACETS_4:
            a._notebooks[f] = f"Content for {f}"

        a.search("question")

        synthesis_system = captured_systems[-1]
        assert "6" in synthesis_system
        assert "cross-reference" in synthesis_system.lower()

    def test_max_tokens(self):
        a = TriadV1PairsAdapter()
        a.reset("s")
        mock_client = _make_mock_completion(["resp"])
        a._oai = mock_client
        a._model = "test-model"
        for f in FACETS_4:
            a._notebooks[f] = f"Content for {f}"

        a.search("q")

        calls = mock_client.chat.completions.create.call_args_list
        for call in calls[:6]:
            assert call.kwargs["max_tokens"] == 2000
        assert calls[-1].kwargs["max_tokens"] == 2500

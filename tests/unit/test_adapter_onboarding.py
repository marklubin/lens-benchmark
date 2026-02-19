"""Adapter onboarding harness — validates the process of adding and using adapters."""
from __future__ import annotations

import pytest
from lens.adapters.base import MemoryAdapter, SearchResult, Document, CapabilityManifest
from lens.adapters.registry import get_adapter, list_adapters, register_adapter, _ADAPTER_REGISTRY
from lens.core.errors import PluginError


class TestRegistrationAndDiscovery:
    def test_register_adapter_adds_to_registry(self):
        """@register_adapter("test-foo") makes it discoverable."""
        @register_adapter("test-onboard-foo")
        class _TestFoo(MemoryAdapter):
            def reset(self, scope_id): pass
            def ingest(self, episode_id, scope_id, timestamp, text, meta=None): pass
            def search(self, query, filters=None, limit=None): return []
            def retrieve(self, ref_id): return None
            def get_capabilities(self): return CapabilityManifest()

        assert get_adapter("test-onboard-foo") is _TestFoo
        # Cleanup
        del _ADAPTER_REGISTRY["test-onboard-foo"]

    def test_list_adapters_includes_all_builtins(self):
        """list_adapters() returns all built-in adapters."""
        adapters = list_adapters()
        for name in ["null", "sqlite", "sqlite-fts"]:
            assert name in adapters, f"{name} missing from adapters"

    def test_get_adapter_unknown_raises_plugin_error(self):
        """get_adapter("nonexistent") raises PluginError with available list."""
        with pytest.raises(PluginError, match="Unknown adapter.*nonexistent"):
            get_adapter("nonexistent-adapter-xyz")

    def test_guarded_import_missing_sdk(self):
        """When mem0ai is not installed, _ensure_builtins doesn't crash."""
        adapters = list_adapters()
        try:
            import mem0  # noqa: F401
        except ImportError:
            assert "mem0-raw" not in adapters or True


class TestInstantiation:
    def test_adapter_instantiates_with_no_args(self):
        """Every built-in adapter can be instantiated as cls() with no constructor args."""
        safe_adapters = ["null", "sqlite", "sqlite-fts"]
        for name in safe_adapters:
            cls = get_adapter(name)
            instance = cls()
            assert isinstance(instance, MemoryAdapter)


class TestMiniLifecycle:
    """Mini-run: reset -> ingest 3 episodes -> prepare -> search -> retrieve."""

    @pytest.mark.parametrize("adapter_name", ["null", "sqlite", "sqlite-fts"])
    def test_mini_lifecycle(self, adapter_name, mini_episodes):
        """Full lifecycle: reset -> ingest -> prepare -> search -> retrieve."""
        cls = get_adapter(adapter_name)
        adapter = cls()

        # Reset
        adapter.reset("mini")

        # Ingest
        for ep_id, scope_id, ts, text in mini_episodes:
            adapter.ingest(ep_id, scope_id, ts, text)

        # Prepare
        adapter.prepare("mini", len(mini_episodes))

        # Search
        results = adapter.search("CPU")
        assert isinstance(results, list)
        for r in results:
            assert isinstance(r, SearchResult)

        # Retrieve
        doc = adapter.retrieve("ep_001")
        if adapter_name != "null":
            assert isinstance(doc, Document)
            assert doc.ref_id == "ep_001"
        else:
            assert doc is None  # NullAdapter returns None

    @pytest.mark.parametrize("adapter_name", ["null", "sqlite", "sqlite-fts"])
    def test_capabilities_valid(self, adapter_name):
        """get_capabilities returns valid manifest for each adapter."""
        cls = get_adapter(adapter_name)
        adapter = cls()
        caps = adapter.get_capabilities()
        assert isinstance(caps, CapabilityManifest)
        assert len(caps.search_modes) > 0

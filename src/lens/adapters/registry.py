from __future__ import annotations

import importlib.metadata
from typing import TYPE_CHECKING

from lens.core.errors import PluginError

if TYPE_CHECKING:
    from lens.adapters.base import MemoryAdapter

_ADAPTER_REGISTRY: dict[str, type[MemoryAdapter]] = {}


def register_adapter(name: str):
    """Decorator to register a built-in adapter class.

    Usage:
        @register_adapter("null")
        class NullAdapter(MemoryAdapter):
            ...
    """

    def decorator(cls: type[MemoryAdapter]) -> type[MemoryAdapter]:
        _ADAPTER_REGISTRY[name] = cls
        return cls

    return decorator


def _discover_entrypoints() -> None:
    """Load external adapters from 'lens.adapters' entry points."""
    try:
        eps = importlib.metadata.entry_points(group="lens.adapters")
    except TypeError:
        # Python 3.11 compat
        eps = importlib.metadata.entry_points().get("lens.adapters", [])

    for ep in eps:
        if ep.name not in _ADAPTER_REGISTRY:
            try:
                cls = ep.load()
                _ADAPTER_REGISTRY[ep.name] = cls
            except Exception as e:
                raise PluginError(f"Failed to load adapter {ep.name!r}: {e}") from e


def get_adapter(name: str) -> type[MemoryAdapter]:
    """Get an adapter class by name. Discovers entrypoints on first miss."""
    if name in _ADAPTER_REGISTRY:
        return _ADAPTER_REGISTRY[name]

    # Lazy-load built-ins
    _ensure_builtins()

    if name in _ADAPTER_REGISTRY:
        return _ADAPTER_REGISTRY[name]

    # Try entrypoints
    _discover_entrypoints()

    if name in _ADAPTER_REGISTRY:
        return _ADAPTER_REGISTRY[name]

    available = sorted(_ADAPTER_REGISTRY.keys())
    msg = f"Unknown adapter: {name!r}. Available: {available}"
    raise PluginError(msg)


def list_adapters() -> dict[str, type[MemoryAdapter]]:
    """Return all registered adapters (built-in + entrypoints)."""
    _ensure_builtins()
    _discover_entrypoints()
    return dict(_ADAPTER_REGISTRY)


def _ensure_builtins() -> None:
    """Import built-in adapter modules to trigger @register_adapter decorators."""
    # These imports trigger the decorators
    import lens.adapters.null  # noqa: F401
    import lens.adapters.sqlite  # noqa: F401
    import lens.adapters.sqlite_variants  # noqa: F401
    try:
        import lens.adapters.mem0  # noqa: F401
    except ImportError:
        pass  # mem0ai not installed
    try:
        import lens.adapters.letta  # noqa: F401
    except ImportError:
        pass  # letta-client not installed
    try:
        import lens.adapters.hindsight  # noqa: F401
    except ImportError:
        pass  # hindsight-client not installed
    try:
        import lens.adapters.letta_sleepy  # noqa: F401
    except ImportError:
        pass  # letta-client not installed
    try:
        import lens.adapters.graphiti_adapter  # noqa: F401
    except ImportError:
        pass  # graphiti-core not installed
    try:
        import lens.adapters.cognee_adapter  # noqa: F401
    except ImportError:
        pass  # cognee not installed
    try:
        import lens.adapters.compaction  # noqa: F401
    except ImportError:
        pass  # openai not installed
    try:
        import lens.adapters.triad_v1  # noqa: F401
    except ImportError:
        pass  # openai not installed

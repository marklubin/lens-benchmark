from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lens.scorer.base import BaseMetric

_METRIC_REGISTRY: dict[str, type[BaseMetric]] = {}


def register_metric(name: str):
    """Decorator to register a scoring metric.

    Usage:
        @register_metric("evidence_validity")
        class EvidenceValidity(BaseMetric):
            ...
    """

    def decorator(cls: type[BaseMetric]) -> type[BaseMetric]:
        _METRIC_REGISTRY[name] = cls
        return cls

    return decorator


def get_metric(name: str) -> type[BaseMetric]:
    """Get a metric class by name."""
    _ensure_builtins()
    if name not in _METRIC_REGISTRY:
        available = sorted(_METRIC_REGISTRY.keys())
        msg = f"Unknown metric: {name!r}. Available: {available}"
        raise KeyError(msg)
    return _METRIC_REGISTRY[name]


def list_metrics() -> dict[str, type[BaseMetric]]:
    """Return all registered metrics."""
    _ensure_builtins()
    return dict(_METRIC_REGISTRY)


def _ensure_builtins() -> None:
    """Import built-in metric modules to trigger @register_metric decorators."""
    import lens.scorer.tier1  # noqa: F401
    import lens.scorer.tier2  # noqa: F401

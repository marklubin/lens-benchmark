from __future__ import annotations

from abc import ABC, abstractmethod

from lens.core.models import MetricResult, RunResult


class BaseMetric(ABC):
    """Abstract base class for scoring metrics."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Metric name (e.g. 'evidence_validity')."""

    @property
    @abstractmethod
    def tier(self) -> int:
        """Metric tier (1=mechanical, 2=stability, 3=future)."""

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description."""

    @abstractmethod
    def compute(self, result: RunResult) -> MetricResult:
        """Compute the metric over a full run result."""

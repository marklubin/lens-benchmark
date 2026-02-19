"""LLM metering proxy for tracking adapter-internal token usage."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class MeteringUsage:
    """Accumulated LLM usage statistics from the metering proxy."""
    total_calls: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_tokens: int = 0
    avg_latency_ms: float = 0.0
    by_model: dict[str, int] = field(default_factory=dict)  # model -> token count

    def to_dict(self) -> dict:
        return {
            "total_calls": self.total_calls,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_tokens,
            "avg_latency_ms": self.avg_latency_ms,
            "by_model": dict(self.by_model),
        }

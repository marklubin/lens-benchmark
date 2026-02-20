from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class LLMConfig:
    """Configuration for the LLM used by the runner."""

    provider: str = "mock"  # "mock", "anthropic", "openai"
    model: str = "gpt-4o-mini"
    api_base: str | None = None
    api_key: str | None = None
    seed: int = 42
    temperature: float = 0.0

    @classmethod
    def from_dict(cls, d: dict) -> LLMConfig:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def resolve_env(self) -> LLMConfig:
        """Override fields from environment variables."""
        return LLMConfig(
            provider=os.environ.get("LENS_LLM_PROVIDER", self.provider),
            model=os.environ.get("LENS_LLM_MODEL", self.model),
            api_base=os.environ.get("LENS_LLM_API_BASE", self.api_base),
            api_key=os.environ.get("LENS_LLM_API_KEY", self.api_key),
            seed=int(os.environ.get("LENS_LLM_SEED", str(self.seed))),
            temperature=float(os.environ.get("LENS_LLM_TEMPERATURE", str(self.temperature))),
        )


@dataclass
class AgentBudgetConfig:
    """Budget configuration for the agent's per-question execution."""

    preset: str = "standard"
    max_turns: int = 10
    max_tool_calls: int = 20
    max_payload_bytes: int = 65536
    max_latency_per_call_ms: float = 5000
    max_agent_tokens: int = 32768
    ingest_max_latency_ms: float = 200
    max_cumulative_result_tokens: int = 0  # 0 = unlimited

    @classmethod
    def fast(cls) -> AgentBudgetConfig:
        """Minimal budget for quick tests."""
        return cls(
            preset="fast",
            max_turns=5,
            max_tool_calls=10,
            max_payload_bytes=32768,
            max_latency_per_call_ms=5000,
            max_agent_tokens=4096,
            ingest_max_latency_ms=200,
        )

    @classmethod
    def standard(cls) -> AgentBudgetConfig:
        """Standard budget for normal runs."""
        return cls(
            preset="standard",
            max_turns=10,
            max_tool_calls=20,
            max_payload_bytes=65536,
            max_latency_per_call_ms=5000,
            max_agent_tokens=32768,
            ingest_max_latency_ms=200,
        )

    @classmethod
    def extended(cls) -> AgentBudgetConfig:
        """Extended budget for thorough evaluation."""
        return cls(
            preset="extended",
            max_turns=20,
            max_tool_calls=50,
            max_payload_bytes=131072,
            max_latency_per_call_ms=10000,
            max_agent_tokens=65536,
            ingest_max_latency_ms=200,
        )

    @classmethod
    def constrained_4k(cls) -> AgentBudgetConfig:
        """Constrained budget — 4K token cap on cumulative tool results."""
        return cls(
            preset="constrained-4k",
            max_turns=6,
            max_tool_calls=12,
            max_payload_bytes=65536,
            max_latency_per_call_ms=5000,
            max_agent_tokens=16384,
            ingest_max_latency_ms=200,
            max_cumulative_result_tokens=4096,
        )

    @classmethod
    def constrained_2k(cls) -> AgentBudgetConfig:
        """Constrained budget — 2K token cap on cumulative tool results."""
        return cls(
            preset="constrained-2k",
            max_turns=6,
            max_tool_calls=12,
            max_payload_bytes=65536,
            max_latency_per_call_ms=5000,
            max_agent_tokens=16384,
            ingest_max_latency_ms=200,
            max_cumulative_result_tokens=2048,
        )

    @classmethod
    def from_preset(cls, name: str) -> AgentBudgetConfig:
        presets = {
            "fast": cls.fast,
            "standard": cls.standard,
            "extended": cls.extended,
            "constrained-4k": cls.constrained_4k,
            "constrained-2k": cls.constrained_2k,
        }
        if name not in presets:
            msg = f"Unknown agent budget preset: {name!r}. Choose from: {list(presets)}"
            raise ValueError(msg)
        return presets[name]()

    @classmethod
    def from_dict(cls, d: dict) -> AgentBudgetConfig:
        preset = d.get("preset", "standard")
        config = cls.from_preset(preset)
        for key in (
            "max_turns", "max_tool_calls", "max_payload_bytes",
            "max_latency_per_call_ms", "max_agent_tokens", "ingest_max_latency_ms",
            "max_cumulative_result_tokens",
        ):
            if key in d:
                setattr(config, key, d[key])
        return config


@dataclass
class RunConfig:
    """Top-level run configuration."""

    adapter: str = "null"
    dataset: str = ""
    output_dir: str = "output"
    agent_budget: AgentBudgetConfig = field(default_factory=AgentBudgetConfig.standard)
    llm: LLMConfig = field(default_factory=LLMConfig)
    checkpoints: list[int] = field(default_factory=lambda: [10, 20, 40, 80])
    seed: int = 42
    parallel_questions: int = 1  # Number of questions to answer concurrently
    cache_dir: str | None = None  # Directory for adapter state caching

    @classmethod
    def from_dict(cls, d: dict) -> RunConfig:
        agent_budget = (
            AgentBudgetConfig.from_dict(d["agent_budget"])
            if "agent_budget" in d
            else AgentBudgetConfig.standard()
        )
        llm = LLMConfig.from_dict(d["llm"]) if "llm" in d else LLMConfig()
        return cls(
            adapter=d.get("adapter", "null"),
            dataset=d.get("dataset", ""),
            output_dir=d.get("output_dir", "output"),
            agent_budget=agent_budget,
            llm=llm.resolve_env(),
            checkpoints=d.get("checkpoints", [10, 20, 40, 80]),
            seed=d.get("seed", 42),
            parallel_questions=d.get("parallel_questions", 1),
            cache_dir=d.get("cache_dir"),
        )

    def to_dict(self) -> dict:
        budget_dict: dict = {
            "preset": self.agent_budget.preset,
            "max_turns": self.agent_budget.max_turns,
            "max_tool_calls": self.agent_budget.max_tool_calls,
            "max_agent_tokens": self.agent_budget.max_agent_tokens,
        }
        if self.agent_budget.max_cumulative_result_tokens > 0:
            budget_dict["max_cumulative_result_tokens"] = (
                self.agent_budget.max_cumulative_result_tokens
            )
        d = {
            "adapter": self.adapter,
            "dataset": self.dataset,
            "output_dir": self.output_dir,
            "agent_budget": budget_dict,
            "llm": {
                "provider": self.llm.provider,
                "model": self.llm.model,
                "seed": self.llm.seed,
                "temperature": self.llm.temperature,
            },
            "checkpoints": self.checkpoints,
            "seed": self.seed,
        }
        if self.parallel_questions > 1:
            d["parallel_questions"] = self.parallel_questions
        if self.cache_dir is not None:
            d["cache_dir"] = self.cache_dir
        return d

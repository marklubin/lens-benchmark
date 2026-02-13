from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class LLMConfig:
    """Configuration for the LLM used by the runner."""

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
            model=os.environ.get("LENS_LLM_MODEL", self.model),
            api_base=os.environ.get("LENS_LLM_API_BASE", self.api_base),
            api_key=os.environ.get("LENS_LLM_API_KEY", self.api_key),
            seed=int(os.environ.get("LENS_LLM_SEED", str(self.seed))),
            temperature=float(os.environ.get("LENS_LLM_TEMPERATURE", str(self.temperature))),
        )


@dataclass
class BudgetPreset:
    """Budget limits for a single method call."""

    max_llm_calls: int = 0
    max_tokens: int = 0
    max_latency_ms: float = float("inf")

    @classmethod
    def from_dict(cls, d: dict) -> BudgetPreset:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class BudgetConfig:
    """Budget configuration controlling adapter resource usage.

    Two named presets are supported: 'fast' and 'standard'.
    """

    preset: str = "standard"

    # Per-method budgets
    ingest: BudgetPreset = field(
        default_factory=lambda: BudgetPreset(max_llm_calls=0, max_tokens=0, max_latency_ms=200)
    )
    refresh: BudgetPreset = field(
        default_factory=lambda: BudgetPreset(
            max_llm_calls=999999, max_tokens=999999, max_latency_ms=float("inf")
        )
    )
    core: BudgetPreset = field(default_factory=BudgetPreset)
    search: BudgetPreset = field(default_factory=BudgetPreset)

    # Evidence constraints
    max_evidence_episodes: int = 8

    @classmethod
    def fast(cls) -> BudgetConfig:
        """Pure retrieval — no LLM at serve time."""
        return cls(
            preset="fast",
            ingest=BudgetPreset(max_llm_calls=0, max_tokens=0, max_latency_ms=200),
            refresh=BudgetPreset(
                max_llm_calls=999999, max_tokens=999999, max_latency_ms=float("inf")
            ),
            core=BudgetPreset(max_llm_calls=0, max_tokens=0, max_latency_ms=800),
            search=BudgetPreset(max_llm_calls=0, max_tokens=0, max_latency_ms=500),
            max_evidence_episodes=8,
        )

    @classmethod
    def standard(cls) -> BudgetConfig:
        """Allows one synthesis call in core()."""
        return cls(
            preset="standard",
            ingest=BudgetPreset(max_llm_calls=0, max_tokens=0, max_latency_ms=200),
            refresh=BudgetPreset(
                max_llm_calls=999999, max_tokens=999999, max_latency_ms=float("inf")
            ),
            core=BudgetPreset(max_llm_calls=1, max_tokens=800, max_latency_ms=800),
            search=BudgetPreset(max_llm_calls=0, max_tokens=0, max_latency_ms=500),
            max_evidence_episodes=8,
        )

    @classmethod
    def from_preset(cls, name: str) -> BudgetConfig:
        presets = {"fast": cls.fast, "standard": cls.standard}
        if name not in presets:
            msg = f"Unknown budget preset: {name!r}. Choose from: {list(presets)}"
            raise ValueError(msg)
        return presets[name]()

    @classmethod
    def from_dict(cls, d: dict) -> BudgetConfig:
        preset = d.get("preset", "standard")
        config = cls.from_preset(preset)
        if "ingest" in d:
            config.ingest = BudgetPreset.from_dict(d["ingest"])
        if "refresh" in d:
            config.refresh = BudgetPreset.from_dict(d["refresh"])
        if "core" in d:
            config.core = BudgetPreset.from_dict(d["core"])
        if "search" in d:
            config.search = BudgetPreset.from_dict(d["search"])
        if "max_evidence_episodes" in d:
            config.max_evidence_episodes = d["max_evidence_episodes"]
        return config


@dataclass
class RunConfig:
    """Top-level run configuration."""

    adapter: str = "null"
    dataset: str = ""
    output_dir: str = "output"
    budget: BudgetConfig = field(default_factory=BudgetConfig.standard)
    llm: LLMConfig = field(default_factory=LLMConfig)
    checkpoints: list[int] = field(default_factory=lambda: [10, 20, 40, 80])
    search_queries: list[str] = field(default_factory=list)
    core_k: int = 10
    search_k: int = 10
    seed: int = 42

    @classmethod
    def from_dict(cls, d: dict) -> RunConfig:
        budget = BudgetConfig.from_dict(d["budget"]) if "budget" in d else BudgetConfig.standard()
        llm = LLMConfig.from_dict(d["llm"]) if "llm" in d else LLMConfig()
        return cls(
            adapter=d.get("adapter", "null"),
            dataset=d.get("dataset", ""),
            output_dir=d.get("output_dir", "output"),
            budget=budget,
            llm=llm.resolve_env(),
            checkpoints=d.get("checkpoints", [10, 20, 40, 80]),
            search_queries=d.get("search_queries", []),
            core_k=d.get("core_k", 10),
            search_k=d.get("search_k", 10),
            seed=d.get("seed", 42),
        )

    def to_dict(self) -> dict:
        return {
            "adapter": self.adapter,
            "dataset": self.dataset,
            "output_dir": self.output_dir,
            "budget": {
                "preset": self.budget.preset,
                "max_evidence_episodes": self.budget.max_evidence_episodes,
            },
            "llm": {
                "model": self.llm.model,
                "seed": self.llm.seed,
                "temperature": self.llm.temperature,
            },
            "checkpoints": self.checkpoints,
            "search_queries": self.search_queries,
            "core_k": self.core_k,
            "search_k": self.search_k,
            "seed": self.seed,
        }

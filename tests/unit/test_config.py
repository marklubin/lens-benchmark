from __future__ import annotations

import pytest

from lens.core.config import BudgetConfig, LLMConfig, RunConfig


class TestBudgetConfig:
    def test_fast_preset(self):
        config = BudgetConfig.fast()
        assert config.preset == "fast"
        assert config.core.max_llm_calls == 0
        assert config.search.max_llm_calls == 0

    def test_standard_preset(self):
        config = BudgetConfig.standard()
        assert config.preset == "standard"
        assert config.core.max_llm_calls == 1
        assert config.core.max_tokens == 800

    def test_from_preset(self):
        config = BudgetConfig.from_preset("fast")
        assert config.preset == "fast"

    def test_invalid_preset(self):
        with pytest.raises(ValueError, match="Unknown budget preset"):
            BudgetConfig.from_preset("turbo")

    def test_from_dict(self):
        d = {"preset": "fast"}
        config = BudgetConfig.from_dict(d)
        assert config.preset == "fast"


class TestLLMConfig:
    def test_defaults(self):
        config = LLMConfig()
        assert config.model == "gpt-4o-mini"
        assert config.seed == 42

    def test_from_dict(self):
        config = LLMConfig.from_dict({"model": "gpt-4o", "seed": 99})
        assert config.model == "gpt-4o"
        assert config.seed == 99

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("LENS_LLM_MODEL", "claude-3-haiku")
        config = LLMConfig().resolve_env()
        assert config.model == "claude-3-haiku"


class TestRunConfig:
    def test_defaults(self):
        config = RunConfig()
        assert config.adapter == "null"
        assert config.seed == 42
        assert config.checkpoints == [10, 20, 40, 80]

    def test_from_dict(self):
        d = {
            "adapter": "rolling",
            "dataset": "test.json",
            "budget": {"preset": "fast"},
            "checkpoints": [5, 10],
        }
        config = RunConfig.from_dict(d)
        assert config.adapter == "rolling"
        assert config.budget.preset == "fast"
        assert config.checkpoints == [5, 10]

    def test_to_dict(self):
        config = RunConfig()
        d = config.to_dict()
        assert d["adapter"] == "null"
        assert "budget" in d
        assert "llm" in d

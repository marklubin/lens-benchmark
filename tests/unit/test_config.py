from __future__ import annotations

import pytest

from lens.core.config import AgentBudgetConfig, LLMConfig, RunConfig


class TestAgentBudgetConfig:
    def test_fast_preset(self):
        config = AgentBudgetConfig.fast()
        assert config.preset == "fast"
        assert config.max_turns == 5
        assert config.max_tool_calls == 10
        assert config.max_agent_tokens == 4096

    def test_standard_preset(self):
        config = AgentBudgetConfig.standard()
        assert config.preset == "standard"
        assert config.max_turns == 10
        assert config.max_tool_calls == 20
        assert config.max_agent_tokens == 32768

    def test_extended_preset(self):
        config = AgentBudgetConfig.extended()
        assert config.preset == "extended"
        assert config.max_turns == 20
        assert config.max_tool_calls == 50
        assert config.max_agent_tokens == 65536

    def test_from_preset(self):
        config = AgentBudgetConfig.from_preset("fast")
        assert config.preset == "fast"

    def test_invalid_preset(self):
        with pytest.raises(ValueError, match="Unknown agent budget preset"):
            AgentBudgetConfig.from_preset("turbo")

    def test_from_dict(self):
        d = {"preset": "fast", "max_turns": 3}
        config = AgentBudgetConfig.from_dict(d)
        assert config.preset == "fast"
        assert config.max_turns == 3  # overridden


class TestLLMConfig:
    def test_defaults(self):
        config = LLMConfig()
        assert config.model == "gpt-4o-mini"
        assert config.provider == "mock"
        assert config.seed == 42

    def test_from_dict(self):
        config = LLMConfig.from_dict({"model": "gpt-4o", "seed": 99, "provider": "openai"})
        assert config.model == "gpt-4o"
        assert config.seed == 99
        assert config.provider == "openai"

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("LENS_LLM_MODEL", "claude-3-haiku")
        monkeypatch.setenv("LENS_LLM_PROVIDER", "anthropic")
        config = LLMConfig().resolve_env()
        assert config.model == "claude-3-haiku"
        assert config.provider == "anthropic"


class TestRunConfig:
    def test_defaults(self):
        config = RunConfig()
        assert config.adapter == "null"
        assert config.seed == 42
        assert config.checkpoints == [10, 20, 40, 80]
        assert config.agent_budget.preset == "standard"

    def test_from_dict(self):
        d = {
            "adapter": "rolling",
            "dataset": "test.json",
            "agent_budget": {"preset": "fast"},
            "checkpoints": [5, 10],
        }
        config = RunConfig.from_dict(d)
        assert config.adapter == "rolling"
        assert config.agent_budget.preset == "fast"
        assert config.checkpoints == [5, 10]

    def test_to_dict(self):
        config = RunConfig()
        d = config.to_dict()
        assert d["adapter"] == "null"
        assert "agent_budget" in d
        assert "llm" in d
        assert d["llm"]["provider"] == "mock"

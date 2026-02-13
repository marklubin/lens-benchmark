from __future__ import annotations

import pytest

from lens.core.config import BudgetConfig
from lens.core.errors import BudgetExceededError
from lens.runner.budget import BudgetTracker, BudgetedLLM


class TestBudgetedLLM:
    def test_fast_blocks_core_calls(self):
        config = BudgetConfig.fast()
        llm = BudgetedLLM(config)
        llm.set_context("core", "core")
        with pytest.raises(BudgetExceededError, match="llm_calls"):
            llm.complete("test prompt")

    def test_standard_allows_one_core_call(self):
        config = BudgetConfig.standard()
        llm = BudgetedLLM(config)
        llm.set_context("core", "core")
        result = llm.complete("test prompt", max_tokens=100)
        assert isinstance(result, str)

    def test_standard_blocks_second_core_call(self):
        config = BudgetConfig.standard()
        llm = BudgetedLLM(config)
        llm.set_context("core", "core")
        llm.complete("first call", max_tokens=100)
        with pytest.raises(BudgetExceededError, match="llm_calls"):
            llm.complete("second call", max_tokens=100)

    def test_refresh_allows_many_calls(self):
        config = BudgetConfig.standard()
        llm = BudgetedLLM(config)
        llm.set_context("refresh", "refresh")
        for _ in range(100):
            llm.complete("call", max_tokens=10)

    def test_ingest_blocks_all_calls(self):
        config = BudgetConfig.standard()
        llm = BudgetedLLM(config)
        llm.set_context("ingest", "ingest")
        with pytest.raises(BudgetExceededError):
            llm.complete("test")

    def test_calls_remaining(self):
        config = BudgetConfig.standard()
        llm = BudgetedLLM(config)
        llm.set_context("core", "core")
        assert llm.calls_remaining == 1
        llm.complete("call", max_tokens=100)
        assert llm.calls_remaining == 0


class TestBudgetTracker:
    def test_tracks_usage(self):
        config = BudgetConfig.standard()
        tracker = BudgetTracker()
        llm = BudgetedLLM(config, tracker)

        llm.set_context("refresh", "refresh")
        llm.complete("prompt", max_tokens=50)
        llm.complete("prompt", max_tokens=50)

        assert tracker.total_calls == 2
        assert len(tracker.by_phase("refresh")) == 2

    def test_summary(self):
        config = BudgetConfig.standard()
        tracker = BudgetTracker()
        llm = BudgetedLLM(config, tracker)

        llm.set_context("refresh", "refresh")
        llm.complete("prompt", max_tokens=50)

        summary = tracker.summary()
        assert summary["total_calls"] == 1
        assert "refresh" in summary["by_phase"]

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from lens.core.errors import (
    AdapterError,
    AntiCheatError,
    BudgetExceededError,
    ConfigError,
    DatasetError,
    EvidenceError,
    LatencyExceededError,
    LensError,
    PluginError,
    ScoringError,
    ValidationError,
    atomic_write,
)


class TestErrorHierarchy:
    def test_all_inherit_from_lens_error(self):
        errors = [
            ConfigError("config"),
            AdapterError("adapter"),
            BudgetExceededError("core", "core", "calls", 1, 2),
            LatencyExceededError("core", 800, 1200),
            ValidationError("validation"),
            EvidenceError("ep_001", "quote"),
            AntiCheatError("cheat"),
            DatasetError("dataset"),
            ScoringError("scoring"),
            PluginError("plugin"),
        ]
        for err in errors:
            assert isinstance(err, LensError)

    def test_budget_exceeded_message(self):
        err = BudgetExceededError("core", "core", "llm_calls", 1, 2)
        assert "core" in str(err)
        assert "llm_calls" in str(err)

    def test_evidence_error_message(self):
        err = EvidenceError("ep_001", "a long quote that should be truncated")
        assert "ep_001" in str(err)


class TestAtomicWrite:
    def test_success(self, tmp_path):
        target = tmp_path / "test.json"
        with atomic_write(target) as tmp:
            tmp.write_text('{"key": "value"}')
        assert target.exists()
        assert target.read_text() == '{"key": "value"}'

    def test_failure_cleans_up(self, tmp_path):
        target = tmp_path / "test.json"
        with pytest.raises(RuntimeError):
            with atomic_write(target) as tmp:
                tmp.write_text("partial")
                raise RuntimeError("intentional failure")
        assert not target.exists()

    def test_creates_parent_dirs(self, tmp_path):
        target = tmp_path / "a" / "b" / "test.json"
        with atomic_write(target) as tmp:
            tmp.write_text("data")
        assert target.exists()

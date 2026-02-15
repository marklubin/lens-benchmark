from __future__ import annotations

from pathlib import Path

import pytest

from lens.adapters.sqlite import SQLiteAdapter
from lens.agent.llm_client import MockLLMClient
from lens.core.config import AgentBudgetConfig, RunConfig
from lens.core.logging import LensLogger, Verbosity
from lens.datasets.loader import load_episodes, load_questions, load_smoke_dataset
from lens.runner.runner import RunEngine
from lens.scorer.engine import ScorerEngine


class TestSQLiteE2E:
    """Integration test: full run with SQLite adapter + mock LLM against smoke dataset."""

    def test_sqlite_adapter_smoke(self, tmp_path: Path):
        data = load_smoke_dataset()
        episodes = load_episodes(data)
        questions = load_questions(data)

        config = RunConfig(
            adapter="sqlite",
            agent_budget=AgentBudgetConfig.fast(),
            checkpoints=[5, 10],
        )

        logger = LensLogger(Verbosity.QUIET)
        engine = RunEngine(config, logger, llm_client=MockLLMClient())
        result = engine.run(episodes, questions=questions)
        result.dataset_version = data["version"]

        # Verify structure
        assert result.run_id
        assert result.adapter == "sqlite"
        assert len(result.scopes) == 2

        for scope in result.scopes:
            assert len(scope.checkpoints) >= 1
            for cp in scope.checkpoints:
                assert cp.validation_errors == []
                for qr in cp.question_results:
                    assert qr.answer.answer_text

        # Score
        scorer = ScorerEngine(logger=logger)
        scorecard = scorer.score(result)
        assert scorecard.composite_score >= 0.0

        # Save artifacts — verify per-run directory
        out = engine.save_artifacts(result, tmp_path / "output")
        assert result.run_id in str(out)
        assert (out / "run_manifest.json").exists()
        assert (out / "config.json").exists()
        assert (out / "log.jsonl").exists()

        # Verify log.jsonl has content
        log_lines = (out / "log.jsonl").read_text().strip().split("\n")
        assert len(log_lines) > 0
        import json
        first_log = json.loads(log_lines[0])
        assert "step" in first_log

    def test_sqlite_search_returns_results(self):
        """Verify the SQLite adapter returns actual search results during a run."""
        adapter = SQLiteAdapter()
        adapter.reset("s1")

        # Ingest episodes
        adapter.ingest("ep1", "s1", "2024-01-01T00:00:00", "The patient reported headaches and fatigue.")
        adapter.ingest("ep2", "s1", "2024-01-02T00:00:00", "Blood pressure was elevated at 150/95.")
        adapter.ingest("ep3", "s1", "2024-01-03T00:00:00", "Started medication for hypertension.")

        # Search should find relevant results
        results = adapter.search("headaches fatigue")
        assert len(results) >= 1
        assert results[0].ref_id == "ep1"

        results = adapter.search("blood pressure")
        assert len(results) >= 1
        assert results[0].ref_id == "ep2"

    def test_per_run_output_structure(self, tmp_path: Path):
        """Verify the per-run output directory structure."""
        data = load_smoke_dataset()
        episodes = load_episodes(data)
        questions = load_questions(data)

        config = RunConfig(
            adapter="sqlite",
            agent_budget=AgentBudgetConfig.fast(),
            checkpoints=[5, 10],
        )

        logger = LensLogger(Verbosity.QUIET)
        engine = RunEngine(config, logger, llm_client=MockLLMClient())
        result = engine.run(episodes, questions=questions)
        result.dataset_version = data["version"]

        out = engine.save_artifacts(result, tmp_path / "output")

        # Should be output/<run_id>/
        assert out.name == result.run_id
        assert out.parent.name == "output"

        # Verify all expected files
        assert (out / "run_manifest.json").exists()
        assert (out / "config.json").exists()
        assert (out / "log.jsonl").exists()
        assert (out / "scopes").is_dir()

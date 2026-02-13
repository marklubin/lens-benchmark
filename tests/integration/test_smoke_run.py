from __future__ import annotations

from pathlib import Path

import pytest

from lens.core.config import BudgetConfig, RunConfig
from lens.core.logging import LensLogger, Verbosity
from lens.datasets.loader import load_episodes, load_smoke_dataset
from lens.runner.runner import RunEngine
from lens.scorer.engine import ScorerEngine


class TestSmokeRun:
    """Integration test: full run with null adapter against smoke dataset."""

    def test_null_adapter_smoke(self, tmp_path: Path):
        data = load_smoke_dataset()
        episodes = load_episodes(data)

        config = RunConfig(
            adapter="null",
            budget=BudgetConfig.standard(),
            checkpoints=[5, 10],
            search_queries=["anxiety patterns"],
        )

        logger = LensLogger(Verbosity.QUIET)
        engine = RunEngine(config, logger)
        result = engine.run(episodes)
        result.dataset_version = data["version"]

        # Verify structure
        assert result.run_id
        assert result.adapter == "null"
        assert len(result.personas) == 2

        for persona in result.personas:
            assert len(persona.checkpoints) >= 1
            for cp in persona.checkpoints:
                assert cp.insights == []  # Null adapter returns nothing
                assert cp.validation_errors == []

        # Score
        scorer = ScorerEngine(logger=logger)
        scorecard = scorer.score(result)
        # Null adapter gets BC=1.0 (no violations), all other metrics=0
        # Composite = BC_weight(0.10) * 1.0 / total_present_weight
        assert scorecard.composite_score == pytest.approx(0.10, abs=0.01)

        # Save artifacts
        out = engine.save_artifacts(result, tmp_path / "output")
        assert (out / "run_manifest.json").exists()
        assert (out / "config.json").exists()
        assert (out / "budget_report.json").exists()

    def test_full_pipeline(self, tmp_path: Path):
        """Test run -> score -> report pipeline."""
        import json

        from lens.artifacts.bundle import load_run_result, load_scorecard
        from lens.core.errors import atomic_write
        from lens.report.markdown import generate_markdown_report

        data = load_smoke_dataset()
        episodes = load_episodes(data)

        config = RunConfig(
            adapter="null",
            budget=BudgetConfig.standard(),
            checkpoints=[5, 10],
            search_queries=["test query"],
        )

        logger = LensLogger(Verbosity.QUIET)
        engine = RunEngine(config, logger)
        result = engine.run(episodes)
        result.dataset_version = data["version"]

        # Save artifacts
        out = engine.save_artifacts(result, tmp_path / "run_output")

        # Load artifacts back
        loaded = load_run_result(out)
        assert loaded.run_id == result.run_id
        assert loaded.adapter == "null"

        # Score
        scorer = ScorerEngine(logger=logger)
        scorecard = scorer.score(loaded)

        scores_dir = out / "scores"
        scores_dir.mkdir(parents=True, exist_ok=True)
        with atomic_write(scores_dir / "scorecard.json") as tmp:
            tmp.write_text(json.dumps(scorecard.to_dict(), indent=2))

        # Load scorecard
        loaded_sc = load_scorecard(out)
        assert loaded_sc is not None
        assert loaded_sc.composite_score == pytest.approx(0.10, abs=0.01)

        # Generate report
        report = generate_markdown_report(loaded_sc)
        assert "LENS Benchmark Report" in report
        assert "null" in report

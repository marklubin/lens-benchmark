from __future__ import annotations

import json
from pathlib import Path

import click
from rich.console import Console

console = Console()


@click.command()
@click.option("--run", "run_dir", required=True, type=click.Path(exists=True), help="Run output directory")
@click.option("--out", "output_path", default=None, help="Output scorecard path")
@click.option("--tier", type=int, default=None, help="Only compute metrics of this tier")
@click.option("-v", "--verbose", count=True)
def score(run_dir: str, output_path: str | None, tier: int | None, verbose: int) -> None:
    """Score a benchmark run."""
    from lens.artifacts.bundle import load_run_result
    from lens.core.errors import atomic_write
    from lens.core.logging import LensLogger, Verbosity
    from lens.scorer.engine import ScorerEngine

    verbosity = Verbosity(min(verbose + 1, 3))
    logger = LensLogger(verbosity)

    logger.info(f"Loading run from {run_dir}")
    result = load_run_result(run_dir)

    scorer = ScorerEngine(tier_filter=tier, logger=logger)
    scorecard = scorer.score(result)

    # Determine output path
    if output_path is None:
        scores_dir = Path(run_dir) / "scores"
        scores_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(scores_dir / "scorecard.json")

    with atomic_write(output_path) as tmp:
        tmp.write_text(json.dumps(scorecard.to_dict(), indent=2))

    console.print(f"\n[bold green]Scoring complete![/bold green] Scorecard: {output_path}")
    console.print(f"Composite score: [bold]{scorecard.composite_score:.4f}[/bold]")

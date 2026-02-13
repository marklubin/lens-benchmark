from __future__ import annotations

import click
from rich.console import Console

from lens import __version__

console = Console()


@click.group()
@click.version_option(version=__version__, prog_name="lens")
def cli() -> None:
    """LENS: Longitudinal Evidence-backed Narrative Signals benchmark."""


# Import and register subcommands
from lens.cli.run_commands import run  # noqa: E402
from lens.cli.score_commands import score  # noqa: E402
from lens.cli.report_commands import report  # noqa: E402
from lens.cli.compare_commands import compare  # noqa: E402
from lens.cli.list_commands import adapters, metrics  # noqa: E402

cli.add_command(run)
cli.add_command(score)
cli.add_command(report)
cli.add_command(compare)
cli.add_command(adapters)
cli.add_command(metrics)


@cli.command()
def smoke() -> None:
    """Quick sanity check with bundled dataset + null adapter."""
    from lens.core.config import RunConfig
    from lens.core.logging import LensLogger, Verbosity
    from lens.datasets.loader import load_episodes, load_smoke_dataset
    from lens.runner.runner import RunEngine
    from lens.scorer.engine import ScorerEngine

    logger = LensLogger(Verbosity.NORMAL)
    logger.info("Running smoke test with null adapter...")

    data = load_smoke_dataset()
    episodes = load_episodes(data)

    config = RunConfig(
        adapter="null",
        budget=RunConfig.__dataclass_fields__["budget"].default_factory(),
        checkpoints=[5, 10],
        search_queries=data.get("search_queries", ["test query"]),
    )

    engine = RunEngine(config, logger)
    result = engine.run(episodes)
    result.dataset_version = data.get("version", "smoke")

    scorer = ScorerEngine(logger=logger)
    scorecard = scorer.score(result)

    console.print()
    console.print(f"[bold green]Smoke test passed![/bold green]")
    console.print(f"Composite score: {scorecard.composite_score:.4f} (expected ~0.0 for null adapter)")


if __name__ == "__main__":
    cli()

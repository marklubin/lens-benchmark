from __future__ import annotations

import json
from pathlib import Path

import click
from rich.console import Console

console = Console()


@click.command()
@click.option("--dataset", required=True, type=click.Path(exists=True), help="Path to dataset JSON")
@click.option("--adapter", default="null", help="Adapter name")
@click.option("--config", "config_path", type=click.Path(exists=True), help="Config JSON file")
@click.option("--out", "output_dir", default="output", help="Output directory")
@click.option("--budget", default="standard", type=click.Choice(["fast", "standard"]))
@click.option("--seed", default=42, type=int)
@click.option("-v", "--verbose", count=True, help="Increase verbosity")
def run(
    dataset: str,
    adapter: str,
    config_path: str | None,
    output_dir: str,
    budget: str,
    seed: int,
    verbose: int,
) -> None:
    """Run the LENS benchmark against a memory adapter."""
    from lens.core.config import BudgetConfig, RunConfig
    from lens.core.logging import LensLogger, Verbosity
    from lens.datasets.loader import get_dataset_version, get_search_queries, load_dataset, load_episodes
    from lens.runner.runner import RunEngine

    verbosity = Verbosity(min(verbose + 1, 3))
    logger = LensLogger(verbosity)

    # Load config
    if config_path:
        config_data = json.loads(Path(config_path).read_text())
        config = RunConfig.from_dict(config_data)
    else:
        config = RunConfig(
            adapter=adapter,
            dataset=dataset,
            output_dir=output_dir,
            budget=BudgetConfig.from_preset(budget),
            seed=seed,
        )

    # Load dataset
    logger.info(f"Loading dataset from {dataset}")
    data = load_dataset(dataset)
    episodes = load_episodes(data)
    config.search_queries = get_search_queries(data)

    # Run
    engine = RunEngine(config, logger)
    result = engine.run(episodes)
    result.dataset_version = get_dataset_version(data)

    # Save artifacts
    out_path = engine.save_artifacts(result, output_dir)
    console.print(f"\n[bold green]Run complete![/bold green] Artifacts: {out_path}")

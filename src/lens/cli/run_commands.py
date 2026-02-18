from __future__ import annotations

import json
from pathlib import Path

import click
from rich.console import Console

console = Console()


def _default_dataset() -> str:
    """Path to the bundled smoke dataset."""
    return str(Path(__file__).resolve().parent.parent / "datasets" / "smoke" / "smoke_dataset.json")


@click.command()
@click.option("--dataset", default=None, type=click.Path(exists=True), help="Path to dataset JSON (default: bundled smoke dataset)")
@click.option("--adapter", default="sqlite", help="Adapter name")
@click.option("--config", "config_path", type=click.Path(exists=True), help="Config JSON file")
@click.option("--out", "output_dir", default="output", help="Output directory")
@click.option("--budget", default="standard", type=click.Choice(["fast", "standard", "extended"]))
@click.option("--seed", default=42, type=int)
@click.option("--provider", default=None, help="LLM provider (mock, openai)")
@click.option("--model", default=None, help="LLM model name")
@click.option("-v", "--verbose", count=True, help="Increase verbosity")
def run(
    dataset: str | None,
    adapter: str,
    config_path: str | None,
    output_dir: str,
    budget: str,
    seed: int,
    provider: str | None,
    model: str | None,
    verbose: int,
) -> None:
    """Run the LENS benchmark against a memory adapter."""
    from lens.agent.client_factory import create_llm_client
    from lens.core.config import AgentBudgetConfig, RunConfig
    from lens.core.logging import LensLogger, Verbosity
    from lens.datasets.loader import (
        get_dataset_version,
        load_dataset,
        load_episodes,
        load_questions,
    )
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
            dataset=dataset or "",
            output_dir=output_dir,
            agent_budget=AgentBudgetConfig.from_preset(budget),
            seed=seed,
        )

    # CLI --dataset overrides config; fall back to config.dataset; last resort: smoke
    dataset = dataset or config.dataset or _default_dataset()

    # CLI overrides for LLM config
    if provider is not None:
        config.llm.provider = provider
    if model is not None:
        config.llm.model = model

    # Resolve env vars into LLM config
    config.llm = config.llm.resolve_env()

    # Create LLM client from config
    llm_client = create_llm_client(config.llm)

    # Load dataset
    logger.info(f"Loading dataset from {dataset}")
    data = load_dataset(dataset)
    episodes = load_episodes(data)
    questions = load_questions(data)

    # Run
    engine = RunEngine(config, logger, llm_client=llm_client)
    result = engine.run(episodes, questions=questions)
    result.dataset_version = get_dataset_version(data)

    # Save artifacts
    out_path = engine.save_artifacts(result, output_dir)
    console.print(f"\n[bold green]Run complete![/bold green] Artifacts: {out_path}")

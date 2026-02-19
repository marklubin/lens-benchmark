from __future__ import annotations

import json
import os
from pathlib import Path

import click
from rich.console import Console

console = Console()


def _make_openai_judge(model: str, api_key: str | None = None):
    """Create a judge_fn callable that uses OpenAI chat completions."""
    import openai

    key = api_key or os.environ.get("OPENAI_API_KEY") or os.environ.get("LENS_LLM_API_KEY")
    if not key:
        raise click.ClickException(
            "Judge requires an API key. Set OPENAI_API_KEY or LENS_LLM_API_KEY."
        )
    base_url = os.environ.get("LENS_LLM_API_BASE") or os.environ.get("OPENAI_BASE_URL")
    client_kwargs: dict = {"api_key": key}
    if base_url:
        client_kwargs["base_url"] = base_url
    client = openai.OpenAI(**client_kwargs)

    def judge_fn(prompt: str) -> str:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=10,
        )
        return resp.choices[0].message.content or "TIE"

    return judge_fn


@click.command()
@click.option("--run", "run_dir", required=True, type=click.Path(exists=True), help="Run output directory")
@click.option("--out", "output_path", default=None, help="Output scorecard path")
@click.option("--tier", type=int, default=None, help="Only compute metrics of this tier")
@click.option("--judge-model", default=None, help="OpenAI model for LLM judge (e.g. gpt-4o-mini)")
@click.option("--no-gate", is_flag=True, help="Disable tier-1 hard gates (budget/grounding)")
@click.option("-v", "--verbose", count=True)
def score(run_dir: str, output_path: str | None, tier: int | None, judge_model: str | None, no_gate: bool, verbose: int) -> None:
    """Score a benchmark run."""
    from lens.artifacts.bundle import load_run_result
    from lens.core.errors import atomic_write
    from lens.core.logging import LensLogger, Verbosity
    from lens.scorer.engine import ScorerEngine

    verbosity = Verbosity(min(verbose + 1, 3))
    logger = LensLogger(verbosity)

    logger.info(f"Loading run from {run_dir}")
    result = load_run_result(run_dir)

    judge_fn = None
    if judge_model:
        logger.info(f"Using LLM judge: {judge_model}")
        judge_fn = _make_openai_judge(judge_model)

    gate_thresholds = {} if no_gate else None
    scorer = ScorerEngine(
        tier_filter=tier, logger=logger, judge_fn=judge_fn,
        gate_thresholds=gate_thresholds,
    )
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

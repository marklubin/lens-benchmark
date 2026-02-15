from __future__ import annotations

from pathlib import Path

import click
from rich.console import Console

console = Console()


@click.command()
@click.argument("runs", nargs=-1, required=True, type=click.Path(exists=True))
@click.option("--format", "fmt", default="markdown", type=click.Choice(["markdown", "json"]))
@click.option("--out", "output_path", default=None, help="Output file path")
def compare(runs: tuple[str, ...], fmt: str, output_path: str | None) -> None:
    """Compare multiple benchmark runs."""
    from lens.artifacts.bundle import load_scorecard
    from lens.core.errors import LensError, atomic_write
    from lens.report.json_report import save_comparison
    from lens.report.markdown import generate_comparison_report

    scorecards = []
    for run_dir in runs:
        sc = load_scorecard(run_dir)
        if sc is None:
            raise LensError(f"No scorecard found in {run_dir}. Run 'lens score' first.")
        scorecards.append(sc)

    if fmt == "markdown":
        content = generate_comparison_report(scorecards)
        out_path = Path(output_path or "comparison.md")
        with atomic_write(out_path) as tmp:
            tmp.write_text(content)
        console.print(content)
    else:
        out_path = Path(output_path or "comparison.json")
        save_comparison(scorecards, out_path)

    console.print(f"\n[bold green]Comparison saved:[/bold green] {out_path}")

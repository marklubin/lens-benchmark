from __future__ import annotations

from pathlib import Path

import click
from rich.console import Console

console = Console()


@click.command()
@click.option("--run", "run_dir", required=True, type=click.Path(exists=True), help="Run output directory")
@click.option("--format", "fmt", default="markdown", type=click.Choice(["markdown", "json", "html"]))
def report(run_dir: str, fmt: str) -> None:
    """Generate a report from a scored run."""
    from lens.artifacts.bundle import load_scorecard
    from lens.core.errors import LensError, atomic_write
    from lens.report.html_report import generate_html_report
    from lens.report.markdown import generate_markdown_report

    scorecard = load_scorecard(run_dir)
    if scorecard is None:
        raise LensError(
            f"No scorecard found in {run_dir}. Run 'lens score' first."
        )

    report_dir = Path(run_dir) / "report"
    report_dir.mkdir(parents=True, exist_ok=True)

    if fmt == "markdown":
        content = generate_markdown_report(scorecard)
        out_path = report_dir / "report.md"
        with atomic_write(out_path) as tmp:
            tmp.write_text(content)
    elif fmt == "html":
        content = generate_html_report(scorecard)
        out_path = report_dir / "report.html"
        with atomic_write(out_path) as tmp:
            tmp.write_text(content)
    else:
        import json

        out_path = report_dir / "report.json"
        with atomic_write(out_path) as tmp:
            tmp.write_text(json.dumps(scorecard.to_dict(), indent=2))

    console.print(f"[bold green]Report generated:[/bold green] {out_path}")

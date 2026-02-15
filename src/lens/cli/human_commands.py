from __future__ import annotations

import uuid
from pathlib import Path

import click
from rich.console import Console

console = Console()


@click.command()
@click.option("--dataset", required=True, type=click.Path(exists=True), help="Path to dataset JSON")
@click.option("--port", default=8000, type=int, help="Server port")
@click.option("--out", default="output", type=click.Path(), help="Output directory")
@click.option("--run-id", default=None, type=str, help="Resume a previous run (omit to start new)")
def human(dataset: str, port: int, out: str, run_id: str | None) -> None:
    """Run the human benchmark web harness."""
    from lens.datasets.loader import load_dataset, load_episodes, load_questions, get_dataset_version
    from lens.human.server import serve
    from lens.human.state import HumanBenchmarkState, discover_checkpoints

    data = load_dataset(dataset)
    episodes = load_episodes(data)
    questions = load_questions(data)
    version = get_dataset_version(data)

    total_episodes = sum(len(eps) for eps in episodes.values())
    total_questions = len(questions)
    total_scopes = len(episodes)

    output_dir = Path(out)

    if run_id is None:
        run_id = uuid.uuid4().hex[:12]
        state = HumanBenchmarkState.initialize(run_id, dataset, data)
        state_path = output_dir / run_id / "human_state.json"
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state.save(state_path)
        console.print(f"[bold]New run:[/bold] {run_id}")
    else:
        state_path = output_dir / run_id / "human_state.json"
        if not state_path.exists():
            console.print(f"[red]No state found at {state_path}[/red]")
            raise SystemExit(1)
        state = HumanBenchmarkState.load(state_path)
        console.print(f"[bold]Resuming run:[/bold] {run_id}")

    console.print(
        f"Loaded dataset: {total_scopes} scopes, "
        f"{total_episodes} episodes, {total_questions} questions"
    )
    console.print(f"Open [bold blue]http://localhost:{port}[/bold blue] in your browser to begin.")
    console.print("Press Ctrl+C to pause (progress is auto-saved).")
    console.print()

    serve(data, output_dir, run_id, port)

from __future__ import annotations

from pathlib import Path

import click
from rich.console import Console

console = Console()


@click.command()
@click.argument("spec_path", type=click.Path(exists=True))
@click.option("--provider", default="openai", help="LLM provider (openai)")
@click.option("--model", default="gpt-4o", help="LLM model name")
@click.option("--api-key", default=None, help="API key (or set LENS_LLM_API_KEY)")
@click.option("--api-base", default=None, help="API base URL")
@click.option("-v", "--verbose", is_flag=True, help="Verbose output")
def generate(
    spec_path: str,
    provider: str,
    model: str,
    api_key: str | None,
    api_base: str | None,
    verbose: bool,
) -> None:
    """Generate episodes from a scope spec YAML file.

    SPEC_PATH is the path to the spec.yaml file.
    """
    from lens.datagen.generator import generate_scope

    def _log(msg: str) -> None:
        if verbose:
            console.print(f"[dim]{msg}[/dim]")

    console.print(f"[bold]Generating scope from:[/bold] {spec_path}")

    gen_dir = generate_scope(
        spec_path=spec_path,
        provider=provider,
        model=model,
        api_key=api_key,
        api_base=api_base,
        log_fn=_log,
    )

    console.print(f"\n[bold green]Generation complete![/bold green] Output: {gen_dir}")


@click.command(name="compile")
@click.option(
    "--scope", "scope_dirs", multiple=True, required=True,
    type=click.Path(exists=True),
    help="Scope directory (repeatable)",
)
@click.option("--version", "dataset_version", required=True, help="Dataset version string")
@click.option("--output", "output_path", required=True, type=click.Path(), help="Output JSON path")
def compile_cmd(
    scope_dirs: tuple[str, ...],
    dataset_version: str,
    output_path: str,
) -> None:
    """Compile generated scopes into a single dataset JSON.

    Assembles episodes and questions from multiple scope directories into
    the dataset format expected by 'lens run'.
    """
    from lens.datagen.compiler import compile_dataset

    console.print(f"[bold]Compiling {len(scope_dirs)} scope(s) into dataset...[/bold]")

    result_path = compile_dataset(
        scope_dirs=list(scope_dirs),
        version=dataset_version,
        output_path=output_path,
    )

    console.print(f"\n[bold green]Dataset compiled![/bold green] Output: {result_path}")


@click.command()
@click.argument("scope_dir", type=click.Path(exists=True))
@click.option("--contamination/--no-contamination", default=True, help="Run contamination check")
@click.option("--naive-baseline/--no-naive-baseline", default=True, help="Run naive baseline")
@click.option("--provider", default="openai", help="LLM provider (openai)")
@click.option("--model", default="gpt-4o", help="LLM model name")
@click.option("--api-key", default=None, help="API key (or set LENS_LLM_API_KEY)")
@click.option("--api-base", default=None, help="API base URL")
@click.option("-v", "--verbose", is_flag=True, help="Verbose output")
def verify(
    scope_dir: str,
    contamination: bool,
    naive_baseline: bool,
    provider: str,
    model: str,
    api_key: str | None,
    api_base: str | None,
    verbose: bool,
) -> None:
    """Run verification checks on a generated scope.

    SCOPE_DIR is the path to the scope directory (containing spec.yaml and generated/).
    """
    from lens.datagen.verifier import verify_scope

    def _log(msg: str) -> None:
        if verbose:
            console.print(f"[dim]{msg}[/dim]")

    console.print(f"[bold]Verifying scope:[/bold] {scope_dir}")

    results = verify_scope(
        scope_dir=scope_dir,
        contamination=contamination,
        naive_baseline=naive_baseline,
        provider=provider,
        model=model,
        api_key=api_key,
        api_base=api_base,
        log_fn=_log,
    )

    # Display results
    if "contamination" in results:
        c = results["contamination"]
        status = "[green]PASS[/green]" if c["summary"] == "pass" else "[red]FAIL[/red]"
        console.print(f"\nContamination check: {status}")
        if "questions" in c:
            for q in c["questions"]:
                flag = "[red]CONTAMINATED[/red]" if q["contaminated"] else "[green]clean[/green]"
                console.print(
                    f"  {q['question_id']}: max single-ep coverage "
                    f"{q['max_single_episode_coverage']:.1%} {flag}"
                )

    if "naive_baseline" in results:
        nb = results["naive_baseline"]
        console.print(f"\nNaive baseline scores:")
        if isinstance(nb.get("summary"), dict):
            for qtype, avg in nb["summary"].items():
                console.print(f"  {qtype}: {avg:.1%} avg fact coverage")
        if "questions" in nb:
            for q in nb["questions"]:
                console.print(
                    f"  {q['question_id']} ({q['question_type']}): "
                    f"{q['fact_coverage']:.1%}"
                )

    console.print(f"\n[bold green]Verification complete![/bold green]")

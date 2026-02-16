from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import click
from rich.console import Console

console = Console()


@click.command()
@click.argument("spec_path", type=click.Path(exists=True))
@click.option("--provider", default="openai", help="LLM provider (openai)")
@click.option("--model", default="gpt-4o-mini", help="LLM model name")
@click.option("--api-key", default=None, help="API key (or set LENS_LLM_API_KEY)")
@click.option("--api-base", default=None, help="API base URL")
@click.option("--concurrency", "-j", default=8, type=int, help="Concurrent LLM workers")
@click.option("--validate/--no-validate", default=True, help="Run validators after build")
@click.option("-v", "--verbose", is_flag=True, help="Verbose output")
@click.option("--legacy", is_flag=True, help="Use legacy generator (no synix)")
def generate(
    spec_path: str,
    provider: str,
    model: str,
    api_key: str | None,
    api_base: str | None,
    concurrency: int,
    validate: bool,
    verbose: bool,
    legacy: bool,
) -> None:
    """Generate episodes from a scope spec YAML file.

    SPEC_PATH is the path to the spec.yaml file.

    By default uses the synix pipeline. Pass --legacy to use the original
    generator.
    """
    if legacy:
        _generate_legacy(spec_path, provider, model, api_key, api_base, verbose)
        return

    scope_dir = Path(spec_path).parent
    pipeline_path = Path(__file__).parent.parent / "datagen" / "synix" / "pipeline.py"

    console.print(f"[bold]Generating scope from:[/bold] {spec_path}")
    console.print(f"[dim]Pipeline: {pipeline_path}[/dim]")

    env = {
        **os.environ,
        "LENS_SCOPE_DIR": str(scope_dir),
        "LENS_BUILD_DIR": str(scope_dir / "generated"),
        "LENS_LLM_PROVIDER": provider,
        "LENS_LLM_MODEL": model,
    }

    cmd = [
        "uvx", "--with", "pyyaml", "synix", "build",
        str(pipeline_path),
        "--source-dir", str(scope_dir),
        "--build-dir", str(scope_dir / "generated"),
        "-j", str(concurrency),
    ]
    if validate:
        cmd.append("--validate")
    if verbose:
        cmd.extend(["-vv"])

    try:
        subprocess.run(cmd, env=env, check=True)
        console.print(f"\n[bold green]Generation complete![/bold green] Output: {scope_dir / 'generated'}")
    except subprocess.CalledProcessError as e:
        console.print(f"\n[bold red]Generation failed![/bold red] Exit code: {e.returncode}")
        raise SystemExit(e.returncode)
    except FileNotFoundError:
        console.print("[bold red]Error:[/bold red] 'uvx' not found. Install uv: https://docs.astral.sh/uv/")
        raise SystemExit(1)


def _generate_legacy(
    spec_path: str,
    provider: str,
    model: str,
    api_key: str | None,
    api_base: str | None,
    verbose: bool,
) -> None:
    """Fallback to the original generator (no synix)."""
    from lens.datagen.generator import generate_scope

    def _log(msg: str) -> None:
        if verbose:
            console.print(f"[dim]{msg}[/dim]")

    console.print(f"[bold]Generating scope from:[/bold] {spec_path} [dim](legacy mode)[/dim]")

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
@click.option("--model", default="gpt-4o-mini", help="LLM model name")
@click.option("--api-key", default=None, help="API key (or set LENS_LLM_API_KEY)")
@click.option("--api-base", default=None, help="API base URL")
@click.option("-v", "--verbose", is_flag=True, help="Verbose output")
@click.option("--legacy", is_flag=True, help="Use legacy verifier (no synix)")
def verify(
    scope_dir: str,
    contamination: bool,
    naive_baseline: bool,
    provider: str,
    model: str,
    api_key: str | None,
    api_base: str | None,
    verbose: bool,
    legacy: bool,
) -> None:
    """Run verification checks on a generated scope.

    SCOPE_DIR is the path to the scope directory (containing spec.yaml and generated/).

    By default uses the synix validator pipeline. Pass --legacy to use the
    original verifier.
    """
    if legacy:
        _verify_legacy(scope_dir, contamination, naive_baseline, provider, model, api_key, api_base, verbose)
        return

    pipeline_path = Path(__file__).parent.parent / "datagen" / "synix" / "pipeline.py"

    console.print(f"[bold]Verifying scope:[/bold] {scope_dir}")

    env = {
        **os.environ,
        "LENS_SCOPE_DIR": scope_dir,
        "LENS_BUILD_DIR": str(Path(scope_dir) / "generated"),
        "LENS_LLM_PROVIDER": provider,
        "LENS_LLM_MODEL": model,
    }

    cmd = [
        "uvx", "--with", "pyyaml", "synix", "validate",
        str(pipeline_path),
    ]
    if verbose:
        cmd.append("-v")

    try:
        subprocess.run(cmd, env=env, check=True)
        console.print(f"\n[bold green]Verification complete![/bold green]")
    except subprocess.CalledProcessError as e:
        console.print(f"\n[bold red]Verification failed![/bold red] Exit code: {e.returncode}")
        raise SystemExit(e.returncode)
    except FileNotFoundError:
        console.print("[bold red]Error:[/bold red] 'uvx' not found. Install uv: https://docs.astral.sh/uv/")
        raise SystemExit(1)


def _verify_legacy(
    scope_dir: str,
    contamination: bool,
    naive_baseline: bool,
    provider: str,
    model: str,
    api_key: str | None,
    api_base: str | None,
    verbose: bool,
) -> None:
    """Fallback to the original verifier."""
    import json

    from lens.datagen.spec import load_spec
    from lens.datagen.verifier import verify_scope
    from lens.datagen.verify_report import generate_verification_report

    def _log(msg: str) -> None:
        if verbose:
            console.print(f"[dim]{msg}[/dim]")

    console.print(f"[bold]Verifying scope:[/bold] {scope_dir} [dim](legacy mode)[/dim]")

    scope_path = Path(scope_dir)
    gen_dir = scope_path / "generated"

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

    # Generate HTML verification report
    spec = load_spec(scope_path / "spec.yaml")
    manifest_path = gen_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    html = generate_verification_report(results, spec, manifest)
    report_path = gen_dir / "verification_report.html"
    report_path.write_text(html)

    console.print(f"\nResults: {gen_dir / 'verification.json'}")
    console.print(f"Report:  {report_path}")
    console.print(f"\n[bold green]Verification complete![/bold green]")


@click.command()
@click.argument("scope_dir", type=click.Path(exists=True))
@click.option("-v", "--verbose", is_flag=True, help="Verbose output")
def release(scope_dir: str, verbose: bool) -> None:
    """Produce final outputs from synix build artifacts + validator results.

    Reads synix build artifacts and validator results, produces legacy JSON
    files (episodes.json, distractors.json, questions.json), manifest,
    verification report, and HTML report.
    """
    from lens.datagen.synix.release import run_release

    console.print(f"[bold]Releasing scope:[/bold] {scope_dir}")

    gen_dir = run_release(scope_dir, verbose=verbose)

    console.print(f"\n[bold green]Release complete![/bold green] Output: {gen_dir}")

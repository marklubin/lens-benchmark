from __future__ import annotations

import json
import os
import re
from pathlib import Path

import click
from rich.console import Console

console = Console()

# Strip Qwen3-style <think>...</think> tags from model output
_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def _strip_think_tags(text: str) -> str:
    """Remove <think>...</think> blocks from Qwen3 reasoning output."""
    return _THINK_RE.sub("", text).strip()


def _make_openai_judge(model: str, api_key: str | None = None, cache_dir: str | None = None):
    """Create a judge_fn callable that uses OpenAI chat completions.

    If cache_dir is provided, all judge LLM calls are cached to disk
    so re-scoring the same run with the same judge model costs zero API calls.
    """
    import openai

    # LENS_JUDGE_* vars take priority — allows judge to use a different provider
    # than the agent LLM (e.g., judge on Together AI while agent on Cerebras).
    key = (
        api_key
        or os.environ.get("LENS_JUDGE_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or os.environ.get("LENS_LLM_API_KEY")
    )
    if not key:
        raise click.ClickException(
            "Judge requires an API key. Set LENS_JUDGE_API_KEY, OPENAI_API_KEY, or LENS_LLM_API_KEY."
        )
    base_url = (
        os.environ.get("LENS_JUDGE_API_BASE")
        or os.environ.get("LENS_LLM_API_BASE")
        or os.environ.get("OPENAI_BASE_URL")
    )
    client_kwargs: dict = {"api_key": key}
    if base_url:
        client_kwargs["base_url"] = base_url
    client = openai.OpenAI(**client_kwargs)

    # Wrap with caching layer — judge calls are deterministic (temperature=0)
    # so identical inputs always produce the same output.
    if cache_dir:
        from lens.agent.llm_cache import CachingOpenAIClient
        client = CachingOpenAIClient(client, cache_dir=cache_dir)

    # Detect Qwen3 models that default to thinking mode
    _is_thinking_model = "qwen3" in model.lower()

    def judge_fn(prompt: str) -> str:
        # Append /no_think for Qwen3 to avoid wasting tokens on reasoning
        content = prompt + "\n/no_think" if _is_thinking_model else prompt
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": content}],
            temperature=0.0,
            max_tokens=20 if _is_thinking_model else 10,
        )
        raw = resp.choices[0].message.content or "TIE"
        return _strip_think_tags(raw) if _is_thinking_model else raw

    return judge_fn, client


def _make_baseline_llm_fn(model: str, api_key: str | None = None, cache_dir: str | None = None):
    """Create a baseline_fn callable for naive baseline generation.

    If cache_dir is provided, all baseline LLM calls are cached to disk.
    """
    import openai

    key = api_key or os.environ.get("OPENAI_API_KEY") or os.environ.get("LENS_LLM_API_KEY")
    if not key:
        raise click.ClickException(
            "Baseline requires an API key. Set OPENAI_API_KEY or LENS_LLM_API_KEY."
        )
    base_url = os.environ.get("LENS_LLM_API_BASE") or os.environ.get("OPENAI_BASE_URL")
    client_kwargs: dict = {"api_key": key}
    if base_url:
        client_kwargs["base_url"] = base_url
    client = openai.OpenAI(**client_kwargs)

    if cache_dir:
        from lens.agent.llm_cache import CachingOpenAIClient
        client = CachingOpenAIClient(client, cache_dir=cache_dir)

    _is_thinking_model = "qwen3" in model.lower()

    def llm_fn(system_prompt: str, user_prompt: str) -> str:
        content = user_prompt + "\n/no_think" if _is_thinking_model else user_prompt
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ],
            temperature=0.0,
            max_tokens=1024,
        )
        raw = resp.choices[0].message.content or ""
        return _strip_think_tags(raw) if _is_thinking_model else raw

    return llm_fn


def _load_run_config(run_dir: str) -> dict | None:
    """Load config.json from a run directory."""
    config_path = Path(run_dir) / "config.json"
    if config_path.exists():
        try:
            return json.loads(config_path.read_text())
        except (json.JSONDecodeError, OSError):
            return None
    return None


def _load_episodes_for_baseline(dataset_path: str):
    """Load episodes from the dataset file for naive baseline generation."""
    from lens.datasets.loader import load_dataset, load_episodes
    data = load_dataset(dataset_path)
    scopes = load_episodes(data)
    # Flatten: NaiveBaselineGenerator expects a flat list of Episode objects
    all_episodes = []
    for eps in scopes.values():
        all_episodes.extend(eps)
    return all_episodes


@click.command()
@click.option("--run", "run_dir", required=True, type=click.Path(exists=True), help="Run output directory")
@click.option("--out", "output_path", default=None, help="Output scorecard path")
@click.option("--tier", type=int, default=None, help="Only compute metrics of this tier")
@click.option("--judge-model", default=None, help="OpenAI model for LLM judge (e.g. gpt-4o-mini)")
@click.option("--no-gate", is_flag=True, help="Disable tier-1 hard gates (budget/grounding)")
@click.option("--no-baseline", is_flag=True, help="Skip naive baseline generation (faster re-score)")
@click.option("--parallel-judge", type=int, default=1, help="Concurrent judge calls (use >1 with self-hosted vLLM)")
@click.option("--position-swap-audit", "swap_audit_n", type=int, default=0,
              help="Run position-swap reliability audit on N random judgments")
@click.option("-v", "--verbose", count=True)
def score(run_dir: str, output_path: str | None, tier: int | None, judge_model: str | None, no_gate: bool, no_baseline: bool, parallel_judge: int, swap_audit_n: int, verbose: int) -> None:
    """Score a benchmark run."""
    from lens.artifacts.bundle import load_run_result
    from lens.core.errors import atomic_write
    from lens.core.logging import LensLogger, Verbosity
    from lens.scorer.engine import ScorerEngine

    verbosity = Verbosity(min(verbose + 1, 3))
    logger = LensLogger(verbosity)

    logger.info(f"Loading run from {run_dir}")
    result = load_run_result(run_dir)

    # Cache dir for judge LLM calls — re-scoring the same run costs zero API calls
    judge_cache_dir = str(Path(run_dir) / "scores" / "judge_cache")

    judge_fn = None
    if judge_model:
        logger.info(f"Using LLM judge: {judge_model}")
        judge_fn, _judge_client = _make_openai_judge(
            judge_model, cache_dir=judge_cache_dir
        )

    # Build naive baseline generator if judge is enabled and not skipped
    baseline_generator = None
    if judge_model and not no_baseline:
        run_config = _load_run_config(run_dir)
        dataset_path = run_config.get("dataset") if run_config else None
        agent_model = run_config.get("llm", {}).get("model") if run_config else None

        if dataset_path and agent_model:
            # Resolve dataset path relative to project root
            ds_path = Path(dataset_path)
            if not ds_path.is_absolute():
                # Try relative to run dir, then cwd
                candidates = [Path(run_dir) / ds_path, Path.cwd() / ds_path]
                ds_path = next((p for p in candidates if p.exists()), ds_path)

            if ds_path.exists():
                logger.info(f"Generating naive baseline with model: {agent_model}")
                from lens.scorer.naive_baseline import NaiveBaselineGenerator

                # Read max_cumulative_result_tokens from run config for fair comparison
                max_result_tokens = (
                    run_config.get("agent_budget", {}).get("max_cumulative_result_tokens", 0)
                    if run_config else 0
                )

                # Cache baseline LLM calls too — same content-addressed cache
                baseline_cache_dir = str(Path(run_dir) / "scores" / "baseline_llm_cache")
                baseline_llm_fn = _make_baseline_llm_fn(
                    agent_model, cache_dir=baseline_cache_dir
                )
                episodes = _load_episodes_for_baseline(str(ds_path))
                cache_dir = Path(run_dir) / "scores"
                baseline_generator = NaiveBaselineGenerator(
                    llm_fn=baseline_llm_fn,
                    episodes=episodes,
                    cache_dir=cache_dir,
                    model_id=agent_model,
                    max_result_tokens=max_result_tokens,
                    max_workers=parallel_judge,
                )
            else:
                logger.info(f"Dataset not found at {ds_path}, skipping naive baseline")
        else:
            logger.info("Run config missing dataset/model, skipping naive baseline")

    gate_thresholds = {} if no_gate else None
    if parallel_judge > 1:
        logger.info(f"Parallel judge workers: {parallel_judge}")
    scorer = ScorerEngine(
        tier_filter=tier, logger=logger, judge_fn=judge_fn,
        gate_thresholds=gate_thresholds,
        baseline_generator=baseline_generator,
        max_judge_workers=parallel_judge,
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

    # Print timing summary
    budget_metric = next(
        (m for m in scorecard.metrics if m.name == "budget_compliance"), None
    )
    if budget_metric and budget_metric.details.get("avg_wall_time_ms"):
        avg_s = budget_metric.details["avg_wall_time_ms"] / 1000
        total_min = budget_metric.details.get("total_wall_time_minutes", 0)
        console.print(f"Timing: {avg_s:.1f}s avg/question, {total_min:.1f} min total")

    # Print naive baseline advantage if computed
    nba_metric = next(
        (m for m in scorecard.metrics if m.name == "naive_baseline_advantage"), None
    )
    if nba_metric and not nba_metric.details.get("not_configured"):
        console.print(
            f"Naive baseline advantage: [bold]{nba_metric.value:.4f}[/bold] "
            f"(win:{nba_metric.details.get('win_rate', 0):.0%} "
            f"loss:{nba_metric.details.get('loss_rate', 0):.0%} "
            f"tie:{nba_metric.details.get('tie_rate', 0):.0%})"
        )

    # Position-swap reliability audit
    if swap_audit_n > 0 and judge_fn:
        from lens.scorer.judge import position_swap_audit

        console.print(f"\n[bold]Running position-swap audit ({swap_audit_n} samples)...[/bold]")
        audit = position_swap_audit(
            scored_run_dir=run_dir,
            judge_fn=judge_fn,
            n_samples=swap_audit_n,
            max_workers=parallel_judge,
        )
        if "error" in audit:
            console.print(f"[red]Audit error: {audit['error']}[/red]")
        else:
            console.print(
                f"Position-swap agreement: {audit['agreement_pct']:.1%} "
                f"({audit['agree']}/{audit['total']})"
            )
            console.print(f"Cohen's kappa: [bold]{audit['cohens_kappa']:.4f}[/bold]")
            console.print(f"Position bias (A-win rate): {audit['position_bias']:.1%}")
            if audit["cohens_kappa"] >= 0.8:
                console.print("[green]Reliability: GOOD (kappa >= 0.8)[/green]")
            elif audit["cohens_kappa"] >= 0.6:
                console.print("[yellow]Reliability: MODERATE (0.6 <= kappa < 0.8)[/yellow]")
            else:
                console.print("[red]Reliability: POOR (kappa < 0.6)[/red]")

            # Save audit results
            audit_path = Path(run_dir) / "scores" / "position_swap_audit.json"
            audit_path.parent.mkdir(parents=True, exist_ok=True)
            # Don't save full details to keep file manageable
            audit_summary = {k: v for k, v in audit.items() if k != "details"}
            audit_path.write_text(json.dumps(audit_summary, indent=2))
            console.print(f"Audit saved to {audit_path}")

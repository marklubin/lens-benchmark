#!/usr/bin/env python3
"""
Benchmark orchestrator for LENS experiment matrix.

Reads an experiment manifest (JSON), runs each experiment via `uv run lens run`,
auto-scores each completed run, and tracks state for resume capability.

Serial adapters (letta, letta-sleepy) are run sequentially to avoid race conditions.
Parallel adapters (mem0-raw, chunked-hybrid, hindsight) can run concurrently.

Usage:
    python scripts/benchmark_orchestrator.py [OPTIONS]

Options:
    --manifest PATH       Experiment manifest (default: experiments/matrix.json)
    --state PATH          State file for resume (default: experiments/matrix_state.json)
    --max-parallel N      Max concurrent parallel experiments (default: 3)
    --filter PATTERN      Only run experiments matching substring
    --dry-run             Print what would run, no execution
    --reset-failed        Reset failed -> pending before running
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SERIAL_ADAPTERS = {"letta", "letta-sleepy", "cognee"}
DEFAULT_MANIFEST = "experiments/matrix.json"
DEFAULT_STATE = "experiments/matrix_state.json"

# ---------------------------------------------------------------------------
# State helpers
# ---------------------------------------------------------------------------

_state_lock = Lock()


def _load_state(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    text = p.read_text()
    if not text.strip():
        return {}
    return json.loads(text)


def _save_state(state: dict, path: str) -> None:
    """Atomic write: write to .tmp then os.replace."""
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, path)


def _update_state(state: dict, path: str, key: str, updates: dict) -> None:
    """Thread-safe state update."""
    with _state_lock:
        entry = state.setdefault(key, {})
        entry.update(updates)
        _save_state(state, path)


# ---------------------------------------------------------------------------
# Environment loading
# ---------------------------------------------------------------------------

def _load_env_key(name: str) -> str:
    """Load a key from environment or .env file."""
    key = os.environ.get(name, "")
    if not key:
        env_path = Path(".env")
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                line = line.strip()
                if line.startswith(f"{name}="):
                    key = line.split("=", 1)[1].strip("\"'")
                    break
    return key


def _load_modal_config() -> dict[str, str]:
    """Load Modal LLM and embedding URLs from environment / .env."""
    llm_url = _load_env_key("MODAL_LLM_URL")
    embed_url = _load_env_key("MODAL_EMBED_URL")
    api_key = _load_env_key("MODAL_API_KEY")
    llm_model = _load_env_key("MODAL_LLM_MODEL")

    if not llm_url:
        print("ERROR: MODAL_LLM_URL not set. Add it to .env or environment.", file=sys.stderr)
        sys.exit(1)
    if not embed_url:
        print("ERROR: MODAL_EMBED_URL not set. Add it to .env or environment.", file=sys.stderr)
        sys.exit(1)

    return {"llm_url": llm_url, "embed_url": embed_url, "api_key": api_key, "llm_model": llm_model}


def _build_run_env(modal_config: dict[str, str], extra: dict[str, str] | None = None) -> dict:
    """Build environment for a subprocess from Modal config."""
    env = {**os.environ}
    api_key = modal_config["api_key"] or "unused"

    env["LENS_LLM_API_KEY"] = api_key
    env["LENS_LLM_API_BASE"] = modal_config["llm_url"]
    env["LENS_LLM_MODEL"] = modal_config.get("llm_model") or ""
    env["OPENAI_API_KEY"] = api_key
    env["OPENAI_BASE_URL"] = modal_config["llm_url"]
    env["LENS_EMBED_BASE_URL"] = modal_config["embed_url"]
    env["LENS_EMBED_API_KEY"] = api_key

    if extra:
        env.update(extra)
    return env


# ---------------------------------------------------------------------------
# Run / Score subprocesses
# ---------------------------------------------------------------------------

RUN_ID_PATTERN = re.compile(r"Artifacts:\s*\S*/([a-f0-9]{12})")
COMPOSITE_PATTERN = re.compile(r"Composite\s+score:\s+([0-9.]+)")


def _run_experiment(experiment: dict, env: dict) -> tuple[str | None, str]:
    """Run `uv run lens run --config <config>` and return (run_id, stdout+stderr).

    Returns (run_id_or_None, combined_output).
    """
    config_path = experiment["config"]
    cmd = ["uv", "run", "lens", "run", "--config", config_path, "-v"]

    merged_env = {**os.environ, **env}

    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env=merged_env,
        timeout=7200,  # 2 hour timeout per experiment
    )

    combined = proc.stdout + "\n" + proc.stderr

    if proc.returncode != 0:
        raise RuntimeError(
            f"lens run failed (exit {proc.returncode}) for {config_path}:\n{combined}"
        )

    # Extract run_id from output
    match = RUN_ID_PATTERN.search(combined)
    if match:
        return match.group(1), combined

    # Fallback: look for newest directory in output/
    output_dir = Path(experiment.get("output_dir", "output"))
    if output_dir.exists():
        candidates = sorted(output_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
        for c in candidates:
            if c.is_dir() and re.fullmatch(r"[a-f0-9]{12}", c.name):
                return c.name, combined

    return None, combined


def _score_run(run_id: str, experiment: dict, modal_config: dict[str, str]) -> tuple[float | None, str]:
    """Run `uv run lens score --run output/<run_id> --judge-model <model>`.

    Returns (composite_score_or_None, combined_output).
    """
    judge_model = experiment.get("judge_model", "Qwen/Qwen3.5-35B-A3B")

    run_dir = str(Path(experiment.get("output_dir", "output")) / run_id)
    cmd = ["uv", "run", "lens", "score", "--run", run_dir, "--judge-model", judge_model, "-v"]

    api_key = modal_config["api_key"] or "unused"
    score_env = {
        **os.environ,
        "OPENAI_API_KEY": api_key,
        "OPENAI_BASE_URL": modal_config["llm_url"],
        "LENS_JUDGE_API_KEY": api_key,
        "LENS_JUDGE_API_BASE": modal_config["llm_url"],
    }

    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env=score_env,
        timeout=3600,  # 1 hour timeout for scoring
    )

    combined = proc.stdout + "\n" + proc.stderr

    if proc.returncode != 0:
        raise RuntimeError(
            f"lens score failed (exit {proc.returncode}) for run {run_id}:\n{combined}"
        )

    match = COMPOSITE_PATTERN.search(combined)
    if match:
        return float(match.group(1)), combined

    return None, combined


# ---------------------------------------------------------------------------
# Experiment executor
# ---------------------------------------------------------------------------

def _run_and_score(
    name: str,
    experiment: dict,
    state: dict,
    state_path: str,
    modal_config: dict[str, str],
) -> None:
    """Execute a single experiment: run -> score -> update state."""
    now = datetime.now(timezone.utc).isoformat()

    # Mark running
    _update_state(state, state_path, name, {
        "status": "running",
        "started_at": now,
        "error": None,
    })
    _print_status(state)

    try:
        # Build env
        extra = experiment.get("env", {})
        env = _build_run_env(modal_config, extra if extra else None)

        # Run
        run_id, run_output = _run_experiment(experiment, env)

        if run_id is None:
            raise RuntimeError(f"Could not extract run_id from lens run output:\n{run_output}")

        _update_state(state, state_path, name, {
            "status": "scoring",
            "run_id": run_id,
        })
        _print_status(state)

        # Score
        score_val, score_output = _score_run(run_id, experiment, modal_config)

        if score_val is None:
            print(
                f"WARNING: Could not parse composite score for {name} (run {run_id}). "
                f"Setting score=null.",
                file=sys.stderr,
            )

        _update_state(state, state_path, name, {
            "status": "done",
            "score": score_val,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        })
        _print_status(state)

    except Exception as exc:
        print(f"ERROR [{name}]: {exc}", file=sys.stderr)
        _update_state(state, state_path, name, {
            "status": "failed",
            "error": str(exc),
            "completed_at": datetime.now(timezone.utc).isoformat(),
        })
        _print_status(state)


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

_print_lock = Lock()


def _print_status(state: dict) -> None:
    """Print a progress table to stdout."""
    with _print_lock:
        lines = []
        lines.append("")
        header = f"{'Experiment':<35} {'Status':<10} {'Run ID':<14} {'Score':<8}"
        lines.append(header)
        lines.append("\u2500" * len(header))
        for name in sorted(state.keys()):
            entry = state[name]
            status = entry.get("status", "pending")
            run_id = entry.get("run_id", "\u2014")
            score = entry.get("score")
            score_str = f"{score:.4f}" if score is not None else "\u2014"
            lines.append(f"{name:<35} {status:<10} {run_id or '\u2014':<14} {score_str:<8}")
        lines.append("")
        print("\n".join(lines), flush=True)


def _print_summary(state: dict) -> None:
    """Print final summary."""
    done = sum(1 for e in state.values() if e.get("status") == "done")
    failed = sum(1 for e in state.values() if e.get("status") == "failed")
    pending = sum(1 for e in state.values() if e.get("status") in ("pending", "running", "scoring"))
    total = len(state)
    print(f"\nSummary: {done}/{total} done, {failed} failed, {pending} remaining")

    if done > 0:
        scores = [e["score"] for e in state.values() if e.get("status") == "done" and e.get("score") is not None]
        if scores:
            print(f"Score range: {min(scores):.4f} - {max(scores):.4f}, mean: {sum(scores)/len(scores):.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="LENS benchmark orchestrator")
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST, help="Experiment manifest JSON")
    parser.add_argument("--state", default=DEFAULT_STATE, help="State file for resume")
    parser.add_argument("--max-parallel", type=int, default=3, help="Max concurrent parallel experiments")
    parser.add_argument("--filter", dest="filter_pattern", default=None, help="Only run experiments matching substring")
    parser.add_argument("--dry-run", action="store_true", help="Print plan, don't execute")
    parser.add_argument("--reset-failed", action="store_true", help="Reset failed -> pending before running")
    args = parser.parse_args()

    # Load manifest
    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print(f"ERROR: Manifest not found: {manifest_path}", file=sys.stderr)
        sys.exit(1)

    manifest = json.loads(manifest_path.read_text())
    experiments: dict[str, dict] = manifest["experiments"]

    # Propagate top-level judge_model to experiments that don't specify their own
    default_judge = manifest.get("judge_model")
    if default_judge:
        for exp in experiments.values():
            exp.setdefault("judge_model", default_judge)

    # Filter
    if args.filter_pattern:
        experiments = {
            k: v for k, v in experiments.items()
            if args.filter_pattern in k
        }
        print(f"Filter matched {len(experiments)} experiments")

    if not experiments:
        print("No experiments to run.")
        return

    # Load state
    state = _load_state(args.state)

    # Reset failed if requested
    if args.reset_failed:
        for name in list(state.keys()):
            if state[name].get("status") == "failed":
                print(f"Resetting {name}: failed -> pending")
                state[name]["status"] = "pending"
                state[name]["error"] = None
        _save_state(state, args.state)

    # Initialize state for new experiments
    for name in experiments:
        if name not in state:
            state[name] = {"status": "pending"}
    _save_state(state, args.state)

    # Determine what needs to run
    to_run = {
        name: exp for name, exp in experiments.items()
        if state.get(name, {}).get("status") in ("pending", None)
    }

    if not to_run:
        print("All experiments already completed or in progress.")
        _print_status(state)
        _print_summary(state)
        return

    # Split into serial / parallel
    serial_queue = {k: v for k, v in to_run.items() if v.get("adapter") in SERIAL_ADAPTERS}
    parallel_queue = {k: v for k, v in to_run.items() if v.get("adapter") not in SERIAL_ADAPTERS}

    # Dry run
    if args.dry_run:
        print("\n=== DRY RUN ===\n")
        print(f"Serial experiments ({len(serial_queue)}):")
        for name in sorted(serial_queue):
            print(f"  {name}: config={serial_queue[name]['config']}")
        print(f"\nParallel experiments ({len(parallel_queue)}, max_workers={args.max_parallel}):")
        for name in sorted(parallel_queue):
            print(f"  {name}: config={parallel_queue[name]['config']}")
        print(f"\nTotal: {len(to_run)} experiments to run")
        return

    # Load Modal config
    modal_config = _load_modal_config()
    print(f"Using Modal LLM: {modal_config['llm_url']}")
    print(f"Using Modal Embeddings: {modal_config['embed_url']}")

    print(f"\nStarting {len(to_run)} experiments: {len(serial_queue)} serial, {len(parallel_queue)} parallel")
    _print_status(state)

    start_time = time.monotonic()

    # Launch parallel experiments in thread pool
    parallel_futures = {}
    executor = None

    if parallel_queue:
        executor = ThreadPoolExecutor(max_workers=args.max_parallel)
        for name, exp in sorted(parallel_queue.items()):
            future = executor.submit(
                _run_and_score, name, exp, state, args.state, modal_config
            )
            parallel_futures[future] = name

    # Run serial experiments sequentially (on main thread)
    for name in sorted(serial_queue):
        exp = serial_queue[name]
        print(f"\n>>> Serial: {name}")
        _run_and_score(name, exp, state, args.state, modal_config)

    # Wait for parallel experiments
    if parallel_futures:
        for future in as_completed(parallel_futures):
            name = parallel_futures[future]
            try:
                future.result()
            except Exception as exc:
                # _run_and_score already handles errors and updates state,
                # but if something truly unexpected happens, log it.
                print(f"ERROR: Unexpected failure in {name}: {exc}", file=sys.stderr)

    if executor is not None:
        executor.shutdown(wait=True)

    elapsed = time.monotonic() - start_time
    print(f"\nAll experiments finished in {elapsed/60:.1f} minutes")
    _print_status(state)
    _print_summary(state)


if __name__ == "__main__":
    main()

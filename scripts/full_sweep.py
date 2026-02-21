#!/usr/bin/env python3
"""Full sweep orchestrator: 8 adapters × 3 budgets × 6 scopes = 144 runs.

Usage:
    # Smoke test (1 config per adapter, scope 01 standard only)
    python3 scripts/full_sweep.py --smoke-test

    # Full sweep
    python3 scripts/full_sweep.py 2>&1 | tee sweep.log

    # Filter to specific adapter(s)
    python3 scripts/full_sweep.py --filter chunked-hybrid --filter compaction

    # Resume after failure (skips completed runs)
    python3 scripts/full_sweep.py

    # Score only (skip run phase)
    python3 scripts/full_sweep.py --score-only

Environment:
    VLLM_URL        - vLLM endpoint (required)
    PARALLEL_JUDGE  - judge parallelism (default: 32)
    PARALLEL_Q      - per-run question parallelism (default: 4)
    MAX_ADAPTERS    - max concurrent adapter tracks (default: 8)
"""
from __future__ import annotations

import concurrent.futures
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("sweep")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ADAPTERS = [
    "chunked-hybrid",
    "compaction",
    "cognee",
    "graphiti",
    "letta",
    "letta-sleepy",
    "mem0-raw",
    "hindsight",
]

SCOPES = ["01", "02", "03", "04", "05", "06"]

# Order matters: standard first (builds cache), then constrained (uses cache)
BUDGETS = ["standard", "4k", "2k"]

CONFIGS_DIR = Path("configs")
OUTPUT_DIR = Path("output")
CACHE_DIR = Path(".cache/adapter")
STATE_FILE = Path("sweep_state.json")

# Adapters that must run sequentially (shared Letta server)
SERIAL_ADAPTERS = {"letta", "letta-sleepy"}

# Per-adapter env var overrides (set at container level; these are for the
# agent harness, not the adapter containers themselves).
# VLLM_URL is injected at runtime in build_env().
ADAPTER_ENV: dict[str, dict[str, str]] = {
    "cognee": {
        "ENABLE_BACKEND_ACCESS_CONTROL": "false",
        "COGNEE_LLM_API_KEY": "dummy",
        "COGNEE_LLM_MODEL": "Qwen/Qwen3-32B",
        # COGNEE_LLM_ENDPOINT set dynamically in build_env()
        "COGNEE_EMBED_API_KEY": "dummy",
        "COGNEE_EMBED_MODEL": "nomic-embed-text",
        "COGNEE_EMBED_ENDPOINT": "http://localhost:11434/v1",
        "COGNEE_EMBED_DIMS": "768",
    },
    "graphiti": {
        "GRAPHITI_LLM_API_KEY": "dummy",
        "GRAPHITI_LLM_MODEL": "Qwen/Qwen3-32B",
        # GRAPHITI_LLM_BASE_URL set dynamically in build_env()
        "GRAPHITI_EMBED_API_KEY": "dummy",
        "GRAPHITI_EMBED_MODEL": "nomic-embed-text",
        "GRAPHITI_EMBED_BASE_URL": "http://localhost:11434/v1",
        "GRAPHITI_EMBED_DIM": "768",
    },
    "letta": {
        "LETTA_BASE_URL": "http://localhost:8283",
    },
    "letta-sleepy": {
        "LETTA_BASE_URL": "http://localhost:8283",
        "LETTA_SLEEP_VARIANT": "3",
    },
    "mem0-raw": {
        "MEM0_LLM_API_KEY": "dummy",
        "MEM0_LLM_MODEL": "Qwen/Qwen3-32B",
        # MEM0_LLM_BASE_URL set dynamically in build_env()
        "MEM0_EMBED_API_KEY": "dummy",
        "MEM0_EMBED_MODEL": "nomic-embed-text",
        "MEM0_EMBED_BASE_URL": "http://localhost:11434/v1",
        "MEM0_EMBED_DIMS": "768",
        "MEM0_EMBED_NO_DIMS": "1",
        "OPENAI_API_KEY": "dummy",
    },
}

# AWQ + V1 engine throughput settings — vLLM max-num-seqs=16 supports this
DEFAULT_PARALLEL_Q = 16      # questions per run (AWQ frees KV cache for concurrency)
DEFAULT_PARALLEL_JUDGE = 16  # judge calls (matches max-num-seqs)
DEFAULT_MAX_ADAPTERS = 8     # concurrent adapter tracks

# Hard timeouts — if any adapter exceeds these, it's broken
RUN_TIMEOUT = 300            # 5 minutes per run
SCORE_TIMEOUT = 120          # 2 minutes per score


def config_filename(adapter: str, scope: str, budget: str) -> str:
    base = adapter.replace("-", "_")
    if budget == "standard":
        return f"{base}_scope{scope}.json"
    return f"{base}_scope{scope}_{budget}.json"


# ---------------------------------------------------------------------------
# State tracking
# ---------------------------------------------------------------------------


def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {"completed": {}, "failed": {}, "run_ids": {}}


def save_state(state: dict) -> None:
    STATE_FILE.write_text(json.dumps(state, indent=2) + "\n")


def run_key(adapter: str, scope: str, budget: str) -> str:
    return f"{adapter}/{scope}/{budget}"


def is_completed(state: dict, adapter: str, scope: str, budget: str) -> bool:
    return run_key(adapter, scope, budget) in state["completed"]


def mark_completed(state: dict, adapter: str, scope: str, budget: str, run_id: str) -> None:
    state["completed"][run_key(adapter, scope, budget)] = run_id
    state["run_ids"][run_key(adapter, scope, budget)] = run_id
    save_state(state)


def mark_failed(state: dict, adapter: str, scope: str, budget: str, error: str) -> None:
    state["failed"][run_key(adapter, scope, budget)] = error
    save_state(state)


# ---------------------------------------------------------------------------
# Build environment for a run
# ---------------------------------------------------------------------------


def build_env(adapter: str, embed_url: str | None = None) -> dict[str, str]:
    """Build env vars for a run subprocess."""
    env = dict(os.environ)

    vllm_url = os.environ.get("VLLM_URL", "")
    if not vllm_url:
        log.error("VLLM_URL not set!")
        sys.exit(1)

    embed_base = embed_url or os.environ.get("EMBED_URL", "http://localhost:11434/v1")

    # Common env vars for the agent harness
    env["LENS_LLM_API_KEY"] = "dummy"
    env["LENS_LLM_API_BASE"] = vllm_url
    env["LENS_LLM_MODEL"] = "Qwen/Qwen3-32B"

    # Embedding
    env["LENS_EMBED_API_KEY"] = "dummy"
    env["LENS_EMBED_BASE_URL"] = embed_base
    env["LENS_EMBED_MODEL"] = "nomic-embed-text"

    # Per-adapter overrides
    extra = ADAPTER_ENV.get(adapter, {})
    env.update(extra)

    # Override embed URLs for adapters that have their own env vars
    for key in list(env.keys()):
        if key.endswith("_EMBED_BASE_URL") or key.endswith("_EMBED_ENDPOINT"):
            if key.startswith("LENS_"):
                continue
            env[key] = embed_base

    # Inject dynamic vLLM URL for adapters that need it
    if adapter == "mem0-raw":
        env["MEM0_LLM_BASE_URL"] = vllm_url
    elif adapter == "graphiti":
        env["GRAPHITI_LLM_BASE_URL"] = vllm_url
    elif adapter == "cognee":
        env["COGNEE_LLM_ENDPOINT"] = vllm_url

    return env


# ---------------------------------------------------------------------------
# Single run execution
# ---------------------------------------------------------------------------


def find_run_id(output: str) -> str | None:
    """Extract run_id from CLI output (looks for 'Artifacts: output/<run_id>')."""
    for line in output.split("\n"):
        if "Artifacts:" in line:
            # Format: "Run complete! Artifacts: output/run_20260220_..."
            parts = line.split("Artifacts:")
            if len(parts) == 2:
                artifact_path = parts[1].strip()
                return Path(artifact_path).name
    return None


def execute_run(
    adapter: str,
    scope: str,
    budget: str,
    parallel_q: int = 4,
    use_cache: bool = False,
    embed_url: str | None = None,
) -> str | None:
    """Execute a single benchmark run. Returns run_id or None on failure."""
    fname = config_filename(adapter, scope, budget)
    config_path = CONFIGS_DIR / fname

    if not config_path.exists():
        log.error("Config not found: %s", config_path)
        return None

    cmd = [
        "uv", "run", "lens", "run",
        "--config", str(config_path),
        "--parallel-questions", str(parallel_q),
        "-v",
    ]
    if use_cache:
        cmd.extend(["--cache-dir", str(CACHE_DIR)])

    env = build_env(adapter, embed_url=embed_url)

    log.info("START  %s/%s/%s — %s", adapter, scope, budget, fname)
    t0 = time.time()

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=env,
            timeout=RUN_TIMEOUT,
        )
        elapsed = time.time() - t0

        if result.returncode != 0:
            log.error(
                "FAILED %s/%s/%s (%.0fs)\nstdout: %s\nstderr: %s",
                adapter, scope, budget, elapsed,
                result.stdout[-2000:] if result.stdout else "",
                result.stderr[-2000:] if result.stderr else "",
            )
            return None

        run_id = find_run_id(result.stdout)
        log.info("DONE   %s/%s/%s → %s (%.0fs)", adapter, scope, budget, run_id, elapsed)
        return run_id

    except subprocess.TimeoutExpired:
        log.error(
            "TIMEOUT EXCEEDED: %s/%s/%s at %ds — adapter is broken, fix it",
            adapter, scope, budget, RUN_TIMEOUT,
        )
        return None
    except Exception as e:
        log.error("ERROR  %s/%s/%s: %s", adapter, scope, budget, e)
        return None


# ---------------------------------------------------------------------------
# Adapter track: run all 18 configs for one adapter serially
# ---------------------------------------------------------------------------


def run_adapter_track(
    adapter: str,
    state: dict,
    scopes: list[str],
    budgets: list[str],
    parallel_q: int,
    embed_url: str | None = None,
) -> list[str]:
    """Run all scope × budget configs for one adapter. Returns list of run_ids."""
    run_ids = []

    for scope in scopes:
        for i, budget in enumerate(budgets):
            key = run_key(adapter, scope, budget)

            if is_completed(state, adapter, scope, budget):
                log.info("SKIP   %s (already completed)", key)
                existing_id = state["run_ids"].get(key, "")
                if existing_id:
                    run_ids.append(existing_id)
                continue

            # First budget (standard) = fresh run with cache save
            # Subsequent budgets (4k, 2k) = try cached state
            use_cache = True  # Always enable cache dir
            rid = execute_run(
                adapter, scope, budget, parallel_q,
                use_cache=use_cache, embed_url=embed_url,
            )

            if rid:
                mark_completed(state, adapter, scope, budget, rid)
                run_ids.append(rid)
            else:
                mark_failed(state, adapter, scope, budget, "run failed")

    return run_ids


# ---------------------------------------------------------------------------
# Scoring phase
# ---------------------------------------------------------------------------


def find_unscored_runs() -> list[str]:
    """Find run directories that have results but no scorecard."""
    unscored = []
    if not OUTPUT_DIR.exists():
        return unscored

    for run_dir in sorted(OUTPUT_DIR.iterdir()):
        if not run_dir.is_dir():
            continue
        # Check for run artifacts (either old run_result.json or new run_manifest.json)
        has_result = (
            (run_dir / "run_result.json").exists()
            or (run_dir / "run_manifest.json").exists()
        )
        has_score = (
            (run_dir / "scorecard.json").exists()
            or (run_dir / "scores" / "scorecard.json").exists()
        )
        if has_result and not has_score:
            unscored.append(str(run_dir))

    return unscored


def score_run(
    run_dir: str,
    parallel_judge: int = 16,
    judge_model: str = "Qwen/Qwen3-32B",
    judge_base_url: str | None = None,
) -> bool:
    """Score a single run with NBA baseline. Returns True on success."""
    vllm_url = judge_base_url or os.environ.get("VLLM_URL", "")
    env = dict(os.environ)
    env["OPENAI_API_KEY"] = "dummy"
    env["OPENAI_BASE_URL"] = vllm_url
    # Baseline generation needs the LLM endpoint
    env["LENS_LLM_API_KEY"] = "dummy"
    env["LENS_LLM_API_BASE"] = vllm_url
    env["LENS_LLM_MODEL"] = judge_model

    cmd = [
        "uv", "run", "lens", "score",
        "--run", run_dir,
        "--judge-model", judge_model,
        "--parallel-judge", str(parallel_judge),
        "-v",
    ]

    log.info("SCORE  %s", run_dir)
    t0 = time.time()

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=env,
            timeout=SCORE_TIMEOUT,
        )
        elapsed = time.time() - t0

        if result.returncode != 0:
            log.error(
                "SCORE FAILED %s (%.0fs)\nstderr: %s",
                run_dir, elapsed,
                result.stderr[-1000:] if result.stderr else "",
            )
            return False

        log.info("SCORED %s (%.0fs)", run_dir, elapsed)
        return True

    except subprocess.TimeoutExpired:
        log.error("SCORE TIMEOUT %s after %ds", run_dir, SCORE_TIMEOUT)
        return False
    except Exception as e:
        log.error("SCORE ERROR %s: %s", run_dir, e)
        return False


def batch_score(
    parallel_judge: int = 16,
    judge_model: str = "Qwen/Qwen3-32B",
    judge_base_url: str | None = None,
) -> None:
    """Score all unscored runs."""
    unscored = find_unscored_runs()
    if not unscored:
        log.info("No unscored runs found.")
        return

    log.info("Scoring %d runs with parallel_judge=%d", len(unscored), parallel_judge)
    succeeded = 0
    failed = 0
    for run_dir in unscored:
        if score_run(run_dir, parallel_judge, judge_model, judge_base_url):
            succeeded += 1
        else:
            failed += 1

    log.info("Scoring complete: %d succeeded, %d failed", succeeded, failed)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Full sweep orchestrator")
    parser.add_argument("--smoke-test", action="store_true", help="Run 1 config per adapter (scope 01, standard)")
    parser.add_argument("--filter", action="append", default=[], help="Only run these adapters")
    parser.add_argument("--score-only", action="store_true", help="Skip run phase, just score")
    parser.add_argument("--max-adapters", type=int, default=None, help="Max concurrent adapter tracks")
    parser.add_argument("--parallel-judge", type=int, default=None, help="Judge parallelism for scoring")
    parser.add_argument("--parallel-q", type=int, default=None, help="Per-run question parallelism")
    parser.add_argument("--scopes", nargs="+", default=None, help="Only these scopes (e.g., 01 02)")
    parser.add_argument("--budgets", nargs="+", default=None, help="Only these budgets (e.g., standard 4k)")
    parser.add_argument("--embed-url", default=None, help="Embedding server URL (default: EMBED_URL env or localhost:11434)")
    parser.add_argument("--judge-model", default="Qwen/Qwen3-32B", help="Model for scoring judge")
    parser.add_argument("--judge-base-url", default=None, help="Base URL for scoring judge (default: VLLM_URL)")
    args = parser.parse_args()

    max_adapters = args.max_adapters or int(os.environ.get("MAX_ADAPTERS", str(DEFAULT_MAX_ADAPTERS)))
    parallel_judge = args.parallel_judge or int(os.environ.get("PARALLEL_JUDGE", str(DEFAULT_PARALLEL_JUDGE)))
    parallel_q = args.parallel_q or int(os.environ.get("PARALLEL_Q", str(DEFAULT_PARALLEL_Q)))

    # Determine what to run
    adapters = args.filter if args.filter else list(ADAPTERS)
    scopes = args.scopes if args.scopes else list(SCOPES)
    budgets = args.budgets if args.budgets else list(BUDGETS)

    if args.smoke_test:
        scopes = ["01"]
        budgets = ["standard"]
        log.info("SMOKE TEST mode: scope 01, standard budget only")

    total = len(adapters) * len(scopes) * len(budgets)
    log.info(
        "Sweep: %d adapters × %d scopes × %d budgets = %d runs",
        len(adapters), len(scopes), len(budgets), total,
    )

    state = load_state()
    already_done = sum(1 for a in adapters for s in scopes for b in budgets
                       if is_completed(state, a, s, b))
    log.info("Already completed: %d/%d", already_done, total)

    if not args.score_only:
        # Phase 1: Run all configs
        # Serial adapters (letta, letta-sleepy) must not run concurrently
        serial = [a for a in adapters if a in SERIAL_ADAPTERS]
        parallel = [a for a in adapters if a not in SERIAL_ADAPTERS]

        # Run parallel adapters concurrently
        embed_url = args.embed_url

        if parallel:
            log.info("Running %d parallel adapter tracks (max_workers=%d)", len(parallel), max_adapters)
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_adapters) as pool:
                futures = {
                    pool.submit(run_adapter_track, a, state, scopes, budgets, parallel_q, embed_url): a
                    for a in parallel
                }

                # Also submit serial adapters one at a time within the same pool
                # but they share a single slot via a separate sequential submission
                serial_future = None
                if serial:
                    def run_serial_adapters():
                        for a in serial:
                            run_adapter_track(a, state, scopes, budgets, parallel_q, embed_url)
                    serial_future = pool.submit(run_serial_adapters)
                    futures[serial_future] = "serial-group"

                for future in concurrent.futures.as_completed(futures):
                    adapter_name = futures[future]
                    try:
                        future.result()
                        log.info("Track completed: %s", adapter_name)
                    except Exception as e:
                        log.error("Track failed: %s — %s", adapter_name, e)
        elif serial:
            # Only serial adapters
            for a in serial:
                run_adapter_track(a, state, scopes, budgets, parallel_q, embed_url)

    # Phase 2: Score all unscored runs
    log.info("=== SCORING PHASE ===")
    batch_score(
        parallel_judge=parallel_judge,
        judge_model=args.judge_model,
        judge_base_url=args.judge_base_url,
    )

    # Summary
    state = load_state()
    log.info(
        "=== SWEEP COMPLETE === %d completed, %d failed",
        len(state["completed"]),
        len(state["failed"]),
    )
    if state["failed"]:
        log.warning("Failed runs:")
        for key, err in state["failed"].items():
            log.warning("  %s: %s", key, err)


if __name__ == "__main__":
    main()

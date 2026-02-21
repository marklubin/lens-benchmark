#!/usr/bin/env python3
"""Pod worker: executes LENS benchmark jobs on a RunPod worker pod.

Reads jobs from $LENS_JOBS (JSON list of {adapter, scope, budget} dicts)
and runs each sequentially: run → score → commit+push.

Usage (on a worker pod, called by pod_setup.sh):
    python3 scripts/pod_worker.py

Environment:
    LENS_JOBS      - JSON list of jobs, e.g. [{"adapter":"graphiti","scope":"01","budget":"standard"}]
    LENS_BRANCH    - Git branch to commit results to (required)
    LENS_GROUP     - Worker group name for commit messages (required)
    VLLM_URL       - vLLM endpoint (default: http://localhost:8000/v1)
    EMBED_URL      - Embedding endpoint (default: http://localhost:11434/v1)
    PARALLEL_Q     - Per-run question parallelism (default: 16)
    PARALLEL_JUDGE - Judge parallelism for scoring (default: 16)
"""
from __future__ import annotations

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
log = logging.getLogger("pod_worker")

# Hard timeouts — if any adapter exceeds these, it's broken
RUN_TIMEOUT = 300    # 5 minutes per run
SCORE_TIMEOUT = 120  # 2 minutes per score


def parse_jobs() -> list[dict]:
    """Parse jobs from LENS_JOBS env var."""
    raw = os.environ.get("LENS_JOBS", "")
    if not raw:
        log.error("LENS_JOBS not set")
        sys.exit(1)

    try:
        jobs = json.loads(raw)
    except json.JSONDecodeError as e:
        log.error("Invalid LENS_JOBS JSON: %s", e)
        sys.exit(1)

    if not isinstance(jobs, list) or not jobs:
        log.error("LENS_JOBS must be a non-empty JSON list")
        sys.exit(1)

    return jobs


def config_filename(adapter: str, scope: str, budget: str) -> str:
    base = adapter.replace("-", "_")
    if budget == "standard":
        return f"{base}_scope{scope}.json"
    return f"{base}_scope{scope}_{budget}.json"


def find_run_id(output: str) -> str | None:
    """Extract run_id from CLI output."""
    for line in output.split("\n"):
        if "Artifacts:" in line:
            parts = line.split("Artifacts:")
            if len(parts) == 2:
                return Path(parts[1].strip()).name
    return None


def run_job(adapter: str, scope: str, budget: str) -> str | None:
    """Execute a single benchmark run. Returns run_id or None."""
    fname = config_filename(adapter, scope, budget)
    config_path = Path("configs") / fname

    if not config_path.exists():
        log.error("Config not found: %s", config_path)
        return None

    parallel_q = int(os.environ.get("PARALLEL_Q", "16"))
    vllm_url = os.environ.get("VLLM_URL", "http://localhost:8000/v1")
    embed_url = os.environ.get("EMBED_URL", "http://localhost:11434/v1")

    cmd = [
        "uv", "run", "lens", "run",
        "--config", str(config_path),
        "--parallel-questions", str(parallel_q),
        "--cache-dir", ".cache/adapter",
        "-v",
    ]

    # Build environment
    env = dict(os.environ)
    env["LENS_LLM_API_KEY"] = "dummy"
    env["LENS_LLM_API_BASE"] = vllm_url
    env["LENS_LLM_MODEL"] = "Qwen/Qwen3-32B"
    env["LENS_EMBED_API_KEY"] = "dummy"
    env["LENS_EMBED_BASE_URL"] = embed_url
    env["LENS_EMBED_MODEL"] = "nomic-embed-text"

    # Adapter-specific env
    adapter_envs = {
        "cognee": {
            "ENABLE_BACKEND_ACCESS_CONTROL": "false",
            "COGNEE_LLM_API_KEY": "dummy",
            "COGNEE_LLM_MODEL": "Qwen/Qwen3-32B",
            "COGNEE_LLM_ENDPOINT": vllm_url,
            "COGNEE_EMBED_API_KEY": "dummy",
            "COGNEE_EMBED_MODEL": "nomic-embed-text",
            "COGNEE_EMBED_ENDPOINT": embed_url,
            "COGNEE_EMBED_DIMS": "768",
        },
        "graphiti": {
            "GRAPHITI_LLM_API_KEY": "dummy",
            "GRAPHITI_LLM_MODEL": "Qwen/Qwen3-32B",
            "GRAPHITI_LLM_BASE_URL": vllm_url,
            "GRAPHITI_EMBED_API_KEY": "dummy",
            "GRAPHITI_EMBED_MODEL": "nomic-embed-text",
            "GRAPHITI_EMBED_BASE_URL": embed_url,
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
            "MEM0_LLM_BASE_URL": vllm_url,
            "MEM0_EMBED_API_KEY": "dummy",
            "MEM0_EMBED_MODEL": "nomic-embed-text",
            "MEM0_EMBED_BASE_URL": embed_url,
            "MEM0_EMBED_DIMS": "768",
            "MEM0_EMBED_NO_DIMS": "1",
            "OPENAI_API_KEY": "dummy",
        },
    }
    env.update(adapter_envs.get(adapter, {}))

    log.info("RUN    %s/%s/%s", adapter, scope, budget)
    t0 = time.time()

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, env=env, timeout=RUN_TIMEOUT,
        )
        elapsed = time.time() - t0

        if result.returncode != 0:
            log.error(
                "FAILED %s/%s/%s (%.0fs)\nstderr: %s",
                adapter, scope, budget, elapsed,
                result.stderr[-2000:] if result.stderr else "",
            )
            return None

        run_id = find_run_id(result.stdout)
        log.info("DONE   %s/%s/%s → %s (%.0fs)", adapter, scope, budget, run_id, elapsed)
        return run_id

    except subprocess.TimeoutExpired:
        log.error(
            "TIMEOUT EXCEEDED: %s/%s/%s at %ds — adapter is broken",
            adapter, scope, budget, RUN_TIMEOUT,
        )
        return None
    except Exception as e:
        log.error("ERROR  %s/%s/%s: %s", adapter, scope, budget, e)
        return None


def score_job(run_dir: str) -> bool:
    """Score a single run with NBA baseline. Returns True on success."""
    vllm_url = os.environ.get("VLLM_URL", "http://localhost:8000/v1")
    judge_model = "Qwen/Qwen3-32B"
    parallel_judge = int(os.environ.get("PARALLEL_JUDGE", "16"))

    env = dict(os.environ)
    env["OPENAI_API_KEY"] = "dummy"
    env["OPENAI_BASE_URL"] = vllm_url
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
            cmd, capture_output=True, text=True, env=env, timeout=SCORE_TIMEOUT,
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


def git_commit_and_push(run_id: str, adapter: str, scope: str, budget: str, scored: bool) -> bool:
    """Commit run artifacts and push to the sweep branch."""
    branch = os.environ.get("LENS_BRANCH", "")
    group = os.environ.get("LENS_GROUP", "worker")

    if not branch:
        log.warning("LENS_BRANCH not set, skipping git push")
        return False

    run_dir = f"output/{run_id}"
    score_tag = " [scored]" if scored else ""
    msg = f"sweep({group}): {adapter}/{scope}/{budget}{score_tag}"

    try:
        subprocess.run(
            ["git", "add", run_dir],
            check=True, capture_output=True, text=True,
        )
        subprocess.run(
            ["git", "commit", "-m", msg],
            check=True, capture_output=True, text=True,
        )
        result = subprocess.run(
            ["git", "push", "origin", branch],
            capture_output=True, text=True, timeout=60,
        )
        if result.returncode != 0:
            log.error("git push failed: %s", result.stderr)
            return False

        log.info("PUSHED %s → %s", run_id, branch)
        return True

    except Exception as e:
        log.error("GIT ERROR for %s: %s", run_id, e)
        return False


def main():
    jobs = parse_jobs()
    branch = os.environ.get("LENS_BRANCH", "")
    group = os.environ.get("LENS_GROUP", "worker")

    log.info("=== LENS Pod Worker ===")
    log.info("Group:  %s", group)
    log.info("Branch: %s", branch)
    log.info("Jobs:   %d", len(jobs))

    # Order: standard first (builds cache), then 4k/2k (uses cache)
    budget_order = {"standard": 0, "4k": 1, "2k": 2}
    jobs.sort(key=lambda j: (j.get("scope", "01"), budget_order.get(j.get("budget", "standard"), 9)))

    completed = 0
    failed = 0
    scored = 0
    results: list[dict] = []

    for i, job in enumerate(jobs, 1):
        adapter = job["adapter"]
        scope = job["scope"]
        budget = job["budget"]

        log.info("=== JOB %d/%d: %s/%s/%s ===", i, len(jobs), adapter, scope, budget)

        # Run
        run_id = run_job(adapter, scope, budget)
        if not run_id:
            failed += 1
            results.append({"adapter": adapter, "scope": scope, "budget": budget, "status": "FAILED"})
            continue

        # Score
        run_dir = f"output/{run_id}"
        score_ok = score_job(run_dir)
        if score_ok:
            scored += 1

        # Commit + push
        if branch:
            git_commit_and_push(run_id, adapter, scope, budget, scored=score_ok)

        completed += 1
        results.append({
            "adapter": adapter, "scope": scope, "budget": budget,
            "status": "SCORED" if score_ok else "RUN_ONLY",
            "run_id": run_id,
        })

    # Final summary
    log.info("=" * 60)
    log.info("=== POD WORKER COMPLETE ===")
    log.info("Completed: %d/%d", completed, len(jobs))
    log.info("Failed:    %d/%d", failed, len(jobs))
    log.info("Scored:    %d/%d", scored, len(jobs))
    log.info("=" * 60)

    for r in results:
        status = r["status"]
        key = f"{r['adapter']}/{r['scope']}/{r['budget']}"
        rid = r.get("run_id", "—")
        log.info("  %-8s %s → %s", status, key, rid)

    # Exit with error if any jobs failed
    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()

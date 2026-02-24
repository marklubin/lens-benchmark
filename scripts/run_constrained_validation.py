#!/usr/bin/env python3
"""Constrained budget validation: full sweep across all adapters × scopes × budgets.

Phased execution with decision gates for course correction.

Phase 1: Scope 01, lightweight adapters (null, chunked-hybrid, compaction) — 6 runs
Phase 2: Scope 01, heavy infra adapters (cognee, graphiti, mem0-raw, hindsight, letta, letta-sleepy) — 12 runs
Phase 3: All scopes (02-06) for adapters that showed signal — up to 90 runs
Phase 4: Full analysis across all data

Usage:
    # Phase 1 — lightweight, no infra needed
    python3 scripts/run_constrained_validation.py --phase 1

    # Phase 2 — needs external services (Letta, FalkorDB, Qdrant, Hindsight)
    python3 scripts/run_constrained_validation.py --phase 2

    # Phase 3 — expand to all scopes for promising adapters
    python3 scripts/run_constrained_validation.py --phase 3

    # Phase 3 but only specific adapters
    python3 scripts/run_constrained_validation.py --phase 3 --adapters chunked-hybrid compaction null

    # Phase 3 specific scopes
    python3 scripts/run_constrained_validation.py --phase 3 --scopes 02 03

    # Re-score all completed runs
    python3 scripts/run_constrained_validation.py --score-only

    # Run analysis
    python3 scripts/run_constrained_validation.py --analyze

    # Show current status
    python3 scripts/run_constrained_validation.py --status

Environment:
    TOGETHER_API_KEY  - Required. Read from .env if not set.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("constrained")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent
CONFIGS_DIR = PROJECT_DIR / "configs"
OUTPUT_DIR = PROJECT_DIR / "output"
RESULTS_DIR = PROJECT_DIR / "results"
CACHE_DIR = PROJECT_DIR / ".cache" / "adapter"
STATE_FILE = PROJECT_DIR / "constrained_validation_state.json"

JUDGE_MODEL = "Qwen/Qwen3-235B-A22B-Instruct-2507-tput"

# Timeouts
RUN_TIMEOUT = 900    # 15 min per run
SCORE_TIMEOUT = 900  # 15 min per score (235B judge on serverless is slow)

ALL_SCOPES = ["01", "02", "03", "04", "05", "06"]
BUDGETS = ["4k", "2k"]
PHASE3D_BUDGETS = ["8k", "16k"]  # Phase 3 with distractors uses larger budgets

# Cerebras API for fast agent inference (3000 tok/s)
CEREBRAS_API_BASE = "https://api.cerebras.ai/v1"

# ---------------------------------------------------------------------------
# Adapter definitions — grouped by infrastructure requirements
# ---------------------------------------------------------------------------

# Group A: no external services, fully concurrent
# Use registry names — these must match the "adapter" field in config JSONs.
LIGHTWEIGHT_ADAPTERS = ["null", "sqlite-chunked-hybrid", "compaction"]

# Group B: external services needed, concurrency constraints
# Env var values containing {TOGETHER_API_KEY} are resolved at runtime in build_env().
HEAVY_ADAPTERS = {
    "cognee": {
        "extra_env": {
            "ENABLE_BACKEND_ACCESS_CONTROL": "false",
            "COGNEE_LLM_API_KEY": "{TOGETHER_API_KEY}",
            "COGNEE_LLM_ENDPOINT": "https://api.together.xyz/v1",
            "COGNEE_LLM_MODEL": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
            "COGNEE_EMBED_API_KEY": "{TOGETHER_API_KEY}",
            "COGNEE_EMBED_ENDPOINT": "https://api.together.xyz/v1",
            "COGNEE_EMBED_MODEL": "Alibaba-NLP/gte-modernbert-base",
            "COGNEE_EMBED_DIMS": "768",
        },
        "serial_group": "cognee",  # cognee uses shared SQLite — must serialize
        "health_check": None,
        "timeout": 3600,  # 60 min — cognify entity extraction via remote API is very slow
    },
    "graphiti": {
        "extra_env": {
            "GRAPHITI_LLM_API_KEY": "{TOGETHER_API_KEY}",
            "GRAPHITI_LLM_BASE_URL": "https://api.together.xyz/v1",
            "GRAPHITI_LLM_MODEL": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
            "GRAPHITI_EMBED_API_KEY": "{TOGETHER_API_KEY}",
            "GRAPHITI_EMBED_BASE_URL": "https://api.together.xyz/v1",
            "GRAPHITI_EMBED_MODEL": "Alibaba-NLP/gte-modernbert-base",
            "GRAPHITI_EMBED_DIM": "768",
        },
        "serial_group": "graphiti",  # shared FalkorDB instance — must serialize
        "health_check": ("http://localhost:6379", "FalkorDB"),
    },
    "mem0-raw": {
        "extra_env": {
            "MEM0_LLM_API_KEY": "{TOGETHER_API_KEY}",
            "MEM0_LLM_BASE_URL": "https://api.together.xyz/v1",
            "MEM0_LLM_MODEL": "Qwen/Qwen3-235B-A22B-Instruct-2507-tput",
            "MEM0_EMBED_API_KEY": "{TOGETHER_API_KEY}",
            "MEM0_EMBED_BASE_URL": "https://api.together.xyz/v1",
            "MEM0_EMBED_MODEL": "Alibaba-NLP/gte-modernbert-base",
            "MEM0_EMBED_DIMS": "768",
            "MEM0_EMBED_NO_DIMS": "1",
        },
        "serial_group": "mem0",  # shared Qdrant collection — must serialize
        "health_check": ("http://localhost:6333", "Qdrant"),
    },
    # hindsight removed from evaluation — 18GB image, 413 batch embed errors,
    # scored barely above null (0.213 AnsQ). See STATUS_REPORT.md session 19.
    "letta": {
        "extra_env": {
            "LETTA_BASE_URL": "http://localhost:8283",
            "LETTA_LLM_MODEL": "together/Qwen/Qwen3-235B-A22B-Instruct-2507-tput",
            "LETTA_EMBED_MODEL": "together-oai/text-embedding-3-small",
        },
        "serial_group": "letta",  # letta & letta-sleepy share server
        "health_check": ("http://localhost:8283/v1/health", "Letta"),
    },
    "letta-sleepy": {
        "extra_env": {
            "LETTA_BASE_URL": "http://localhost:8283",
            "LETTA_SLEEP_VARIANT": "3",
            "LETTA_LLM_MODEL": "together/Qwen/Qwen3-235B-A22B-Instruct-2507-tput",
            "LETTA_EMBED_MODEL": "together-oai/text-embedding-3-small",
        },
        "serial_group": "letta",
        "health_check": ("http://localhost:8283/v1/health", "Letta"),
    },
}

ALL_ADAPTERS = LIGHTWEIGHT_ADAPTERS + list(HEAVY_ADAPTERS.keys())


def config_filename(adapter: str, scope: str, budget: str, distractors: bool = False) -> str:
    """Build config filename from adapter/scope/budget.

    For Phase 3 with distractors, uses 'scope{NN}d' naming convention.
    Falls back to alternate filename conventions if primary doesn't exist.
    """
    base = adapter.replace("-", "_")
    scope_tag = f"scope{scope}d" if distractors else f"scope{scope}"
    primary = f"{base}_{scope_tag}_{budget}.json"
    if (CONFIGS_DIR / primary).exists():
        return primary
    # Fallback: scope01d configs used hyphenated filenames for some adapters
    alt = f"{adapter}_{scope_tag}_{budget}.json"
    if (CONFIGS_DIR / alt).exists():
        return alt
    return primary  # return primary so caller sees the "expected" name in warning


def run_label(adapter: str, scope: str, budget: str) -> str:
    """Canonical label for a run."""
    return f"{adapter}/s{scope}/{budget}"


# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------


def load_env() -> str:
    """Load TOGETHER_API_KEY from environment or .env file."""
    key = os.environ.get("TOGETHER_API_KEY", "")
    if not key:
        env_file = PROJECT_DIR / ".env"
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                line = line.strip()
                if line.startswith("TOGETHER_API_KEY="):
                    key = line.split("=", 1)[1].strip("\"'")
                    break
    if not key:
        log.error("TOGETHER_API_KEY not found in environment or .env")
        sys.exit(1)
    return key


def build_env(api_key: str, extra: dict[str, str] | None = None) -> dict[str, str]:
    """Build environment for a subprocess."""
    env = dict(os.environ)
    env["LENS_LLM_API_KEY"] = api_key
    env["LENS_LLM_API_BASE"] = "https://api.together.xyz/v1"
    env["LENS_LLM_MODEL"] = "Qwen/Qwen3-235B-A22B-Instruct-2507-tput"
    env["LENS_EMBED_API_KEY"] = api_key
    env["LENS_EMBED_BASE_URL"] = "https://api.together.xyz/v1"
    env["LENS_EMBED_MODEL"] = "Alibaba-NLP/gte-modernbert-base"
    env["OPENAI_API_KEY"] = api_key
    env["OPENAI_BASE_URL"] = "https://api.together.xyz/v1"
    # Judge always uses Together AI — even when agent LLM is on a different provider
    env["LENS_JUDGE_API_KEY"] = api_key
    env["LENS_JUDGE_API_BASE"] = "https://api.together.xyz/v1"
    # Enable LLM response caching — replays cached responses on retries
    env["LENS_LLM_CACHE_DIR"] = str(RESULTS_DIR / "llm_cache")
    if extra:
        for k, v in extra.items():
            env[k] = v.replace("{TOGETHER_API_KEY}", api_key)
    return env


# ---------------------------------------------------------------------------
# State tracking (thread-safe)
# ---------------------------------------------------------------------------


def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except (json.JSONDecodeError, OSError) as e:
            log.warning("Failed to load state file, starting fresh: %s", e)
    return {}


# Cross-process file lock for safe concurrent orchestrator instances
_file_lock = None


def _get_file_lock():
    """Lazy-create a file lock for cross-process state file safety."""
    global _file_lock
    if _file_lock is None:
        import fcntl
        lock_path = STATE_FILE.with_suffix(".lock")
        _file_lock = open(lock_path, "w")
    return _file_lock


def save_state(state: dict) -> None:
    """Atomically merge in-memory state into on-disk state file.

    Uses file locking to prevent concurrent orchestrator instances from
    clobbering each other's results.
    """
    import fcntl
    lock_fd = _get_file_lock()
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        # Re-read disk state and merge our updates on top
        disk_state = {}
        if STATE_FILE.exists():
            try:
                disk_state = json.loads(STATE_FILE.read_text())
            except (json.JSONDecodeError, OSError):
                pass
        disk_state.update(state)
        tmp = STATE_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(disk_state, indent=2) + "\n")
        tmp.rename(STATE_FILE)
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)


# ---------------------------------------------------------------------------
# Run execution
# ---------------------------------------------------------------------------


def find_run_id(output: str) -> str | None:
    """Extract run_id from CLI output."""
    for line in output.split("\n"):
        if "Artifacts:" in line or "Artifacts saved to" in line:
            for token in line.split():
                if token.startswith("output/"):
                    return Path(token).name
    return None


def execute_run(config: str, env: dict[str, str], timeout: int = RUN_TIMEOUT) -> str:
    """Execute a benchmark run. Returns run_id. Raises on failure."""
    cmd = [
        "uv", "run", "lens", "run",
        "--config", config,
        "--parallel-questions", "4",
        "--cache-dir", str(CACHE_DIR),
        "-v",
    ]
    result = subprocess.run(
        cmd, capture_output=True, text=True,
        env=env, timeout=timeout, cwd=str(PROJECT_DIR),
    )
    if result.returncode != 0:
        stderr_tail = result.stderr[-5000:] if result.stderr else ""
        stdout_tail = result.stdout[-2000:] if result.stdout else ""
        raise RuntimeError(
            f"Run failed (rc={result.returncode}):\nstdout: {stdout_tail}\nstderr: {stderr_tail}"
        )
    run_id = find_run_id(result.stdout)
    if not run_id:
        raise RuntimeError(f"Could not extract run_id from output:\n{result.stdout[-1000:]}")
    return run_id


def score_run(run_id: str, env: dict[str, str]) -> None:
    """Score a completed run. Raises on failure."""
    run_dir = str(OUTPUT_DIR / run_id)
    cmd = [
        "uv", "run", "lens", "score",
        "--run", run_dir,
        "--judge-model", JUDGE_MODEL,
        "-v",
    ]
    result = subprocess.run(
        cmd, capture_output=True, text=True,
        env=env, timeout=SCORE_TIMEOUT, cwd=str(PROJECT_DIR),
    )
    if result.returncode != 0:
        stderr_tail = result.stderr[-1000:] if result.stderr else ""
        raise RuntimeError(f"Scoring failed (rc={result.returncode}):\n{stderr_tail}")


def extract_scores(run_id: str) -> dict:
    """Extract composite and NBA from scorecard.json."""
    for subpath in ["scores/scorecard.json", "scorecard.json"]:
        path = OUTPUT_DIR / run_id / subpath
        if path.exists():
            scorecard = json.loads(path.read_text())
            composite = scorecard.get("composite_score")
            nba = None
            nba_details = {}
            for m in scorecard.get("metrics", []):
                if isinstance(m, dict) and m.get("name") == "naive_baseline_advantage":
                    nba = m.get("value")
                    nba_details = m.get("details", {})
                    break
            return {
                "composite": composite,
                "nba": nba,
                "nba_per_question": nba_details.get("per_question", []),
            }
    raise FileNotFoundError(f"No scorecard found for {run_id}")


# ---------------------------------------------------------------------------
# Per-run worker
# ---------------------------------------------------------------------------


def run_and_score(
    label: str,
    config: str,
    env: dict[str, str],
    state: dict,
    lock: threading.Lock,
    timeout: int = RUN_TIMEOUT,
) -> None:
    """Execute one benchmark run + score. Thread-safe via lock."""
    with lock:
        existing = state.get(label, {})
        if existing.get("status") == "scored":
            log.info("SKIP   %-35s (already scored, NBA=%.4f)", label, existing.get("nba", 0))
            return
        if existing.get("status") == "run_complete" and existing.get("run_id"):
            log.info("RESUME %-35s (scoring run %s)", label, existing["run_id"])
            run_id = existing["run_id"]
        else:
            run_id = None

    t0 = time.time()
    try:
        if run_id is None:
            log.info("START  %-35s", label)
            run_id = execute_run(config, env, timeout=timeout)
            elapsed = time.time() - t0
            log.info("RUN OK %-35s -> %s (%.0fs)", label, run_id, elapsed)
            with lock:
                state[label] = {"status": "run_complete", "run_id": run_id}
                save_state(state)

        log.info("SCORE  %-35s (%s)", label, run_id)
        score_run(run_id, env)
        metrics = extract_scores(run_id)

        with lock:
            state[label] = {"status": "scored", "run_id": run_id, **metrics}
            save_state(state)

        elapsed = time.time() - t0
        log.info(
            "DONE   %-35s  composite=%.4f  NBA=%.4f  (%.0fs)",
            label, metrics.get("composite") or 0, metrics.get("nba") or 0, elapsed,
        )
    except Exception as e:
        elapsed = time.time() - t0
        log.error("FAILED %-35s after %.0fs: %s", label, elapsed, e)
        with lock:
            state[label] = {"status": "failed", "run_id": run_id, "error": str(e)[:2000]}
            save_state(state)


# ---------------------------------------------------------------------------
# Phase builders
# ---------------------------------------------------------------------------


def build_phase_runs(
    phase: int,
    adapters_filter: list[str] | None,
    scopes_filter: list[str] | None,
    cerebras_key: str | None = None,
) -> list[tuple[str, str, str, dict[str, str]]]:
    """Build list of (label, config, adapter, extra_env) for a phase.

    Returns list of (label, config_path, adapter_name, extra_env).
    """
    runs = []
    distractors = False
    budgets = BUDGETS

    if phase == 1:
        adapters = [a for a in LIGHTWEIGHT_ADAPTERS if not adapters_filter or a in adapters_filter]
        scopes = ["01"]
    elif phase == 2:
        adapters = [a for a in HEAVY_ADAPTERS if not adapters_filter or a in adapters_filter]
        scopes = ["01"]
    elif phase == 3:
        adapters = [a for a in ALL_ADAPTERS if not adapters_filter or a in adapters_filter]
        scopes = [s for s in ALL_SCOPES if s != "01"]  # 02-06, scope 01 done in phase 1+2
        if scopes_filter:
            scopes = [s for s in scopes if s in scopes_filter]
    elif phase == 5:
        # Phase 3 with distractors (120 episodes) — uses Cerebras for agent LLM
        adapters = [a for a in ALL_ADAPTERS if not adapters_filter or a in adapters_filter]
        scopes = scopes_filter if scopes_filter else ALL_SCOPES
        distractors = True
        budgets = PHASE3D_BUDGETS
    else:
        log.error("Unknown phase: %d", phase)
        return []

    for adapter in adapters:
        for scope in scopes:
            for budget in budgets:
                fname = config_filename(adapter, scope, budget, distractors=distractors)
                config_path = CONFIGS_DIR / fname
                if not config_path.exists():
                    log.warning("Config not found, skipping: %s", fname)
                    continue
                label = run_label(adapter, scope, budget)
                extra_env = {}
                if adapter in HEAVY_ADAPTERS:
                    extra_env = dict(HEAVY_ADAPTERS[adapter]["extra_env"])
                # Phase 5 uses Cerebras for agent LLM, Together for judge/embed
                if distractors and cerebras_key:
                    extra_env["LENS_LLM_API_KEY"] = cerebras_key
                    extra_env["LENS_LLM_API_BASE"] = CEREBRAS_API_BASE
                    extra_env["LENS_LLM_MODEL"] = "gpt-oss-120b"
                    extra_env["OPENAI_API_KEY"] = cerebras_key
                    extra_env["OPENAI_BASE_URL"] = CEREBRAS_API_BASE
                    # Route Letta's internal LLM to Cerebras too — Together AI
                    # takes 120-180s on large contexts, exceeding httpx timeouts.
                    # NOTE: Letta server only supports together/letta providers,
                    # NOT cerebras. Keep Letta on Together AI's Qwen3-235B.
                runs.append((label, f"configs/{fname}", adapter, extra_env))

    return runs


def get_serial_group(adapter: str) -> str | None:
    """Get serial group for an adapter (adapters in same group run sequentially)."""
    if adapter in HEAVY_ADAPTERS:
        return HEAVY_ADAPTERS[adapter]["serial_group"]
    return None


# ---------------------------------------------------------------------------
# Health checks
# ---------------------------------------------------------------------------


def check_service_health(url: str, name: str) -> bool:
    """Check if a service is reachable."""
    import urllib.parse
    parsed = urllib.parse.urlparse(url)
    # FalkorDB uses Redis protocol — can't use HTTP
    if parsed.port == 6379 or name.lower() == "falkordb":
        try:
            import socket
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(5)
            s.connect((parsed.hostname or "localhost", parsed.port or 6379))
            s.send(b"PING\r\n")
            resp = s.recv(64)
            s.close()
            return b"+PONG" in resp
        except Exception:
            log.warning("%s not reachable at %s", name, url)
            return False
    try:
        import urllib.request
        with urllib.request.urlopen(url, timeout=5) as resp:
            return resp.status == 200
    except Exception:
        log.warning("%s not reachable at %s", name, url)
        return False


def preflight_checks(api_key: str, runs: list[tuple]) -> bool:
    """Verify configs exist and required services are reachable."""
    ok = True

    # Check all configs exist
    for label, config, adapter, extra_env in runs:
        config_path = PROJECT_DIR / config
        if not config_path.exists():
            log.error("Config missing: %s (%s)", config, label)
            ok = False

    # Check required services for heavy adapters in this batch
    checked_services = set()
    for label, config, adapter, extra_env in runs:
        if adapter in HEAVY_ADAPTERS:
            hc = HEAVY_ADAPTERS[adapter].get("health_check")
            if hc and hc[1] not in checked_services:
                url, name = hc
                if check_service_health(url, name):
                    log.info("%s: OK", name)
                else:
                    log.error("%s not reachable at %s — runs for %s will fail", name, url, adapter)
                    # Don't hard-fail — let individual runs fail and get retried
                checked_services.add(name)

    # Test Together API
    try:
        import urllib.request
        req = urllib.request.Request(
            "https://api.together.xyz/v1/models",
            headers={"Authorization": f"Bearer {api_key}"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            if resp.status == 200:
                log.info("Together API: OK")
            else:
                log.warning("Together API returned status %d", resp.status)
    except Exception as e:
        log.warning("Together API check failed: %s", e)

    return ok


# ---------------------------------------------------------------------------
# Execution engine
# ---------------------------------------------------------------------------


def execute_phase(
    runs: list[tuple[str, str, str, dict[str, str]]],
    api_key: str,
    state: dict,
    lock: threading.Lock,
    max_workers: int = 6,
) -> None:
    """Execute a batch of runs with concurrency constraints.

    Adapters in the same serial_group run sequentially.
    Everything else runs concurrently up to max_workers.
    """
    # Split into serial groups and concurrent runs
    serial_groups: dict[str, list[tuple]] = {}
    concurrent_runs: list[tuple] = []

    for item in runs:
        label, config, adapter, extra_env = item
        sg = get_serial_group(adapter)
        if sg:
            serial_groups.setdefault(sg, []).append(item)
        else:
            concurrent_runs.append(item)

    # Filter out already-scored
    concurrent_runs = [
        r for r in concurrent_runs
        if state.get(r[0], {}).get("status") != "scored"
    ]
    for sg in serial_groups:
        serial_groups[sg] = [
            r for r in serial_groups[sg]
            if state.get(r[0], {}).get("status") != "scored"
        ]
    serial_groups = {k: v for k, v in serial_groups.items() if v}

    total_needed = len(concurrent_runs) + sum(len(v) for v in serial_groups.values())
    if total_needed == 0:
        log.info("All runs already scored — nothing to do")
        return

    log.info(
        "Executing %d runs: %d concurrent + %d in %d serial group(s)",
        total_needed, len(concurrent_runs),
        sum(len(v) for v in serial_groups.values()),
        len(serial_groups),
    )

    def _adapter_timeout(adapter: str) -> int:
        if adapter in HEAVY_ADAPTERS:
            return HEAVY_ADAPTERS[adapter].get("timeout", RUN_TIMEOUT)
        return RUN_TIMEOUT

    def run_serial_group(group_runs: list[tuple]) -> None:
        for label, config, adapter, extra_env in group_runs:
            env = build_env(api_key, extra_env)
            run_and_score(label, config, env, state, lock, timeout=_adapter_timeout(adapter))

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {}

        # Submit concurrent runs
        for label, config, adapter, extra_env in concurrent_runs:
            env = build_env(api_key, extra_env)
            t = _adapter_timeout(adapter)
            f = pool.submit(run_and_score, label, config, env, state, lock, timeout=t)
            futures[f] = label

        # Submit serial groups (each group runs as a single sequential task)
        for sg_name, group_runs in serial_groups.items():
            f = pool.submit(run_serial_group, group_runs)
            futures[f] = f"serial:{sg_name}"

        for future in concurrent.futures.as_completed(futures):
            name = futures[future]
            try:
                future.result()
            except Exception as e:
                log.error("Unexpected error in %s: %s", name, e)


# ---------------------------------------------------------------------------
# Status & summary
# ---------------------------------------------------------------------------


def print_status(state: dict) -> None:
    """Print current state of all runs."""
    if not state:
        print("No runs recorded yet.")
        return

    print(f"\n{'Label':<35} {'Status':<14} {'Composite':>10} {'NBA':>10}")
    print("-" * 75)

    by_adapter: dict[str, list] = {}
    for label, info in sorted(state.items()):
        if not isinstance(info, dict):
            continue
        adapter = label.split("/")[0] if "/" in label else label.rsplit("_", 1)[0]
        by_adapter.setdefault(adapter, []).append((label, info))

    for adapter in sorted(by_adapter):
        for label, info in by_adapter[adapter]:
            status = info.get("status", "unknown")
            composite = info.get("composite")
            nba = info.get("nba")
            comp_str = f"{composite:.4f}" if composite is not None else "---"
            nba_str = f"{nba:.4f}" if nba is not None else "---"
            print(f"{label:<35} {status:<14} {comp_str:>10} {nba_str:>10}")

    # Summary counts
    scored = sum(1 for v in state.values() if isinstance(v, dict) and v.get("status") == "scored")
    failed = sum(1 for v in state.values() if isinstance(v, dict) and v.get("status") == "failed")
    running = sum(1 for v in state.values() if isinstance(v, dict) and v.get("status") == "run_complete")
    print(f"\nTotal: {scored} scored, {failed} failed, {running} run but unscored")


def print_phase_summary(state: dict, phase: int) -> None:
    """Print summary for a completed phase with decision gate."""
    print(f"\n{'=' * 75}")
    print(f"PHASE {phase} SUMMARY")
    print(f"{'=' * 75}")

    scored_runs = {
        k: v for k, v in state.items()
        if isinstance(v, dict) and v.get("status") == "scored"
    }

    if not scored_runs:
        print("No scored runs.")
        return

    # Group by adapter
    by_adapter: dict[str, dict[str, dict]] = {}
    for label, info in scored_runs.items():
        parts = label.split("/")
        if len(parts) == 3:
            adapter, scope, budget = parts[0], parts[1], parts[2]
        else:
            continue
        by_adapter.setdefault(adapter, {})[f"{scope}/{budget}"] = info

    print(f"\n{'Adapter':<20} {'Scope/Budget':<12} {'Composite':>10} {'NBA':>10}")
    print("-" * 55)
    for adapter in sorted(by_adapter):
        for key in sorted(by_adapter[adapter]):
            info = by_adapter[adapter][key]
            comp = info.get("composite")
            nba = info.get("nba")
            comp_str = f"{comp:.4f}" if comp is not None else "---"
            nba_str = f"{nba:.4f}" if nba is not None else "---"
            print(f"{adapter:<20} {key:<12} {comp_str:>10} {nba_str:>10}")

    # Decision gate
    print(f"\n{'=' * 75}")
    print("DECISION GATE")
    print(f"{'=' * 75}")

    # Use smallest budget for decision gating (2k for phases 1-3, 8k for phase 5)
    budget_suffix = "/8k" if any(k.endswith("/8k") for k in scored_runs) else "/2k"
    nba_2k = {
        k: v.get("nba") for k, v in scored_runs.items()
        if k.endswith(budget_suffix) and v.get("nba") is not None and "null" not in k
    }
    null_2k = {
        k: v.get("nba") for k, v in scored_runs.items()
        if k.endswith(budget_suffix) and v.get("nba") is not None and "null" in k
    }

    if nba_2k:
        best_label = max(nba_2k, key=nba_2k.get)
        best_nba = nba_2k[best_label]
        print(f"  Best adapter NBA at 2K: {best_label} = {best_nba:.4f}")
        if null_2k:
            avg_null = sum(null_2k.values()) / len(null_2k)
            print(f"  Null baseline NBA at 2K (avg): {avg_null:.4f}")

        if best_nba > 0.45:
            print("  -> STRONG SIGNAL: retrieval advantage validated at constrained budget")
            print("  -> Recommendation: proceed to expand scopes")
        elif best_nba > 0.30:
            print("  -> MODERATE SIGNAL: some advantage, expand to confirm across scopes")
            print("  -> Recommendation: proceed with promising adapters")
        elif best_nba > 0.20:
            print("  -> WEAK SIGNAL: marginal advantage, review per-question breakdown")
            print("  -> Recommendation: proceed cautiously, focus on best adapters")
        else:
            print("  -> NO SIGNAL: constrained budget alone doesn't help")
            print("  -> Recommendation: investigate alternative hypotheses")
    else:
        print("  No adapter NBA data at 2K — cannot assess")

    # List adapters worth expanding
    if phase in (1, 2):
        worth_expanding = [
            k.split("/")[0] for k, v in nba_2k.items() if v and v > 0.15
        ]
        if worth_expanding:
            print(f"\n  Adapters to expand to all scopes: {', '.join(sorted(set(worth_expanding)))}")
            print(f"  (Plus null as baseline)")
        print()


def export_results(state: dict) -> None:
    """Write results/constrained_validation.json."""
    RESULTS_DIR.mkdir(exist_ok=True)
    out = {
        "experiment": "constrained_budget_validation",
        "judge_model": JUDGE_MODEL,
        "runs": {},
    }
    for label, info in state.items():
        if isinstance(info, dict):
            export = {k: v for k, v in info.items() if k != "nba_per_question"}
            out["runs"][label] = export

    out_path = RESULTS_DIR / "constrained_validation.json"
    out_path.write_text(json.dumps(out, indent=2) + "\n")
    log.info("Results exported to %s", out_path)


# ---------------------------------------------------------------------------
# Score-only mode
# ---------------------------------------------------------------------------


def score_only_mode(api_key: str, state: dict, lock: threading.Lock) -> None:
    """Re-score all runs that have run_ids but aren't scored."""
    env = build_env(api_key)
    to_score = [
        (label, info) for label, info in state.items()
        if isinstance(info, dict) and info.get("run_id") and info.get("status") != "scored"
    ]
    if not to_score:
        log.info("All runs already scored.")
        return

    log.info("Scoring %d runs", len(to_score))
    for label, info in to_score:
        run_id = info["run_id"]
        log.info("SCORE  %-35s (%s)", label, run_id)
        try:
            score_run(run_id, env)
            metrics = extract_scores(run_id)
            with lock:
                state[label] = {"status": "scored", "run_id": run_id, **metrics}
                save_state(state)
            log.info("DONE   %-35s  NBA=%.4f", label, metrics.get("nba") or 0)
        except Exception as e:
            log.error("SCORE FAILED %-35s: %s", label, e)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Constrained budget validation — full sweep",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Phases:
  1  Scope 01, lightweight (null, chunked-hybrid, compaction) — 6 runs
  2  Scope 01, heavy infra (cognee, graphiti, mem0-raw, hindsight, letta, letta-sleepy) — 12 runs
  3  Scopes 02-06, all adapters (or filtered) — up to 90 runs
  4  Analysis only (runs analyze_constrained.py)
  5  All scopes WITH distractors (120 eps), Cerebras agent LLM — up to 108 runs
""",
    )
    parser.add_argument("--phase", type=int, choices=[1, 2, 3, 4, 5], help="Phase to execute (5 = distractors, all scopes)")
    parser.add_argument("--adapters", nargs="+", default=None, help="Filter to specific adapters")
    parser.add_argument("--scopes", nargs="+", default=None, help="Filter to specific scopes")
    parser.add_argument("--cerebras-key", default=None, help="Cerebras API key for Phase 5 agent LLM (or set CEREBRAS_API_KEY env var)")
    parser.add_argument("--max-workers", type=int, default=6, help="Max concurrent runs (default: 6)")
    parser.add_argument("--score-only", action="store_true", help="Re-score completed but unscored runs")
    parser.add_argument("--analyze", action="store_true", help="Run analysis script")
    parser.add_argument("--status", action="store_true", help="Show current status and exit")
    parser.add_argument("--retry-failed", action="store_true", help="Clear failed status so runs are retried")
    args = parser.parse_args()

    # Load state
    state = load_state()
    lock = threading.Lock()

    # Status mode
    if args.status:
        print_status(state)
        return

    # Retry failed — clear failed entries so they get re-run
    if args.retry_failed:
        cleared = 0
        for label in list(state.keys()):
            if isinstance(state[label], dict) and state[label].get("status") == "failed":
                del state[label]
                cleared += 1
        if cleared:
            save_state(state)
            log.info("Cleared %d failed runs — they will be retried", cleared)
        else:
            log.info("No failed runs to clear")
        if not args.phase:
            return

    # Score-only mode
    if args.score_only:
        api_key = load_env()
        score_only_mode(api_key, state, lock)
        print_status(state)
        export_results(state)
        return

    # Analysis mode
    if args.analyze or args.phase == 4:
        log.info("Running analysis...")
        analysis_script = PROJECT_DIR / "scripts" / "analyze_constrained.py"
        if analysis_script.exists():
            subprocess.run([sys.executable, str(analysis_script)], cwd=str(PROJECT_DIR))
        else:
            log.error("Analysis script not found: %s", analysis_script)
        return

    if not args.phase:
        parser.error("--phase is required (1, 2, 3, or 4)")

    # Setup
    api_key = load_env()

    log.info("=" * 75)
    log.info("CONSTRAINED BUDGET VALIDATION — Phase %d", args.phase)
    log.info("=" * 75)
    log.info("State file: %s", STATE_FILE)

    # Cerebras key for Phase 5
    cerebras_key = None
    if args.phase == 5:
        cerebras_key = args.cerebras_key or os.environ.get("CEREBRAS_API_KEY", "")
        if not cerebras_key:
            log.error("Phase 5 requires --cerebras-key or CEREBRAS_API_KEY env var")
            sys.exit(1)

    # Build run list
    runs = build_phase_runs(args.phase, args.adapters, args.scopes, cerebras_key=cerebras_key)
    if not runs:
        log.error("No runs to execute for phase %d (check --adapters / --scopes filters)", args.phase)
        sys.exit(1)

    # Count already done
    already_scored = sum(1 for r in runs if state.get(r[0], {}).get("status") == "scored")
    log.info("Phase %d: %d total runs, %d already scored, %d to go",
             args.phase, len(runs), already_scored, len(runs) - already_scored)

    # Preflight
    preflight_checks(api_key, runs)

    # Execute
    execute_phase(runs, api_key, state, lock, max_workers=args.max_workers)

    # Summary
    print_phase_summary(state, args.phase)
    export_results(state)


if __name__ == "__main__":
    main()

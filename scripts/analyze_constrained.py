#!/usr/bin/env python3
"""Analyze constrained validation results: bootstrap CIs, paired tests, degradation curves.

Usage:
    python3 scripts/analyze_constrained.py
    python3 scripts/analyze_constrained.py --state-file constrained_validation_state.json
    python3 scripts/analyze_constrained.py --scope 01          # single scope
    python3 scripts/analyze_constrained.py --aggregate          # pool all scopes
    python3 scripts/analyze_constrained.py --no-plot

Reads scorecards from completed runs and produces:
  1. Bootstrap CIs on per-question NBA
  2. Paired Wilcoxon: adapter vs null
  3. Budget degradation: 4K vs 2K per adapter
  4. Interaction: differential degradation (adapter holds up better than null?)
  5. Summary table + go/no-go verdict
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("analyze")

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "output"
STATE_FILE = PROJECT_DIR / "constrained_validation_state.json"
RESULTS_DIR = PROJECT_DIR / "results"


# ---------------------------------------------------------------------------
# Label parsing — handles both old (null_4k) and new (null/s01/4k) formats
# ---------------------------------------------------------------------------


def parse_label(label: str) -> tuple[str, str, str] | None:
    """Parse label into (adapter, scope, budget).

    Supports:
      - "adapter/sXX/budget" (new format)
      - "adapter_budget" (old format, scope defaults to "01")
    """
    if "/" in label:
        parts = label.split("/")
        if len(parts) == 3:
            adapter = parts[0]
            scope = parts[1].lstrip("s")
            budget = parts[2]
            return adapter, scope, budget
    # Old format: "null_4k", "chunked_hybrid_2k"
    for suffix in ("_4k", "_2k"):
        if label.endswith(suffix):
            adapter = label[: -len(suffix)]
            budget = suffix.lstrip("_")
            return adapter, "01", budget
    return None


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_per_question_nba(run_id: str) -> list[dict]:
    """Load per-question NBA data from a scorecard."""
    for subpath in ["scores/scorecard.json", "scorecard.json"]:
        path = OUTPUT_DIR / run_id / subpath
        if path.exists():
            scorecard = json.loads(path.read_text())
            for m in scorecard.get("metrics", []):
                if isinstance(m, dict) and m.get("name") == "naive_baseline_advantage":
                    return m.get("details", {}).get("per_question", [])
    return []


def load_state(state_file: Path) -> dict:
    """Load the validation state file."""
    if not state_file.exists():
        log.error("State file not found: %s", state_file)
        sys.exit(1)
    return json.loads(state_file.read_text())


def collect_nba_vectors(
    state: dict,
    scope_filter: str | None = None,
    aggregate: bool = False,
) -> dict[str, np.ndarray]:
    """Collect per-question NBA vectors.

    If scope_filter is set, only include that scope.
    If aggregate is True, pool questions across all scopes for each adapter×budget.
    Otherwise, return per-scope vectors.

    Returns {key: np.array of win_rates}.
    Key format: "adapter/budget" if aggregating or single-scope, else full label.
    """
    # Collect raw data grouped by (adapter, budget, scope)
    raw: dict[tuple[str, str, str], list[float]] = {}

    for label, info in state.items():
        if not isinstance(info, dict) or info.get("status") != "scored":
            continue
        parsed = parse_label(label)
        if parsed is None:
            continue
        adapter, scope, budget = parsed

        if scope_filter and scope != scope_filter:
            continue

        run_id = info.get("run_id")
        if not run_id:
            continue

        per_q = load_per_question_nba(run_id)
        if not per_q:
            log.warning("No per-question NBA for %s (run %s)", label, run_id)
            continue

        rates = [q["win_rate"] for q in sorted(per_q, key=lambda x: x.get("question_id", ""))]
        raw[(adapter, scope, budget)] = rates

    # Build output vectors
    vectors: dict[str, np.ndarray] = {}

    if aggregate or scope_filter:
        # Group by adapter×budget, pool questions
        pooled: dict[tuple[str, str], list[float]] = defaultdict(list)
        for (adapter, scope, budget), rates in raw.items():
            pooled[(adapter, budget)].extend(rates)

        for (adapter, budget), rates in pooled.items():
            key = f"{adapter}/{budget}"
            vectors[key] = np.array(rates)
    else:
        # Per-scope vectors
        for (adapter, scope, budget), rates in raw.items():
            key = f"{adapter}/s{scope}/{budget}"
            vectors[key] = np.array(rates)

    return vectors


def get_adapter_budget(key: str) -> tuple[str, str]:
    """Extract (adapter, budget) from a vector key."""
    parts = key.rsplit("/", 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    return key, ""


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------


def bootstrap_ci(
    data: np.ndarray,
    n_boot: int = 10000,
    ci: float = 0.95,
    rng: np.random.Generator | None = None,
) -> tuple[float, float, float]:
    """Bootstrap mean and CI. Returns (mean, ci_low, ci_high)."""
    if rng is None:
        rng = np.random.default_rng(42)
    n = len(data)
    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        sample = rng.choice(data, size=n, replace=True)
        boot_means[i] = sample.mean()
    alpha = (1 - ci) / 2
    low = np.quantile(boot_means, alpha)
    high = np.quantile(boot_means, 1 - alpha)
    return float(data.mean()), float(low), float(high)


def wilcoxon_test(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Wilcoxon signed-rank test. Falls back to sign test if scipy unavailable."""
    diff = x - y
    try:
        from scipy.stats import wilcoxon
        nonzero = diff[diff != 0]
        if len(nonzero) < 3:
            return 0.0, 1.0
        stat, p = wilcoxon(nonzero)
        return float(stat), float(p)
    except ImportError:
        log.warning("scipy not available — using sign test")
        n_pos = int(np.sum(diff > 0))
        n_neg = int(np.sum(diff < 0))
        n_total = n_pos + n_neg
        if n_total == 0:
            return 0.0, 1.0
        from math import comb
        k = min(n_pos, n_neg)
        p = 2 * sum(comb(n_total, i) * 0.5 ** n_total for i in range(k + 1))
        return float(n_pos - n_neg), min(float(p), 1.0)


def differential_degradation_ci(
    adapter_4k: np.ndarray,
    adapter_2k: np.ndarray,
    null_4k: np.ndarray,
    null_2k: np.ndarray,
    n_boot: int = 10000,
) -> tuple[float, float, float]:
    """Bootstrap CI on (adapter_delta) - (null_delta). Positive = adapter holds better."""
    adapter_delta = adapter_4k - adapter_2k
    null_delta = null_4k - null_2k
    interaction = adapter_delta - null_delta
    return bootstrap_ci(interaction, n_boot=n_boot)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_degradation(
    vectors: dict[str, np.ndarray],
    cis: dict[str, tuple[float, float, float]],
    output_path: Path,
) -> None:
    """Plot NBA vs budget per adapter."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib not available — skipping plot")
        return

    # Discover adapters
    adapters = set()
    for key in vectors:
        adapter, budget = get_adapter_budget(key)
        adapters.add(adapter)

    fig, ax = plt.subplots(figsize=(10, 6))
    budgets = ["4k", "2k"]
    budget_x = {b: i for i, b in enumerate(budgets)}

    colors = [
        "#888888", "#2196F3", "#4CAF50", "#FF9800", "#9C27B0",
        "#F44336", "#00BCD4", "#795548", "#607D8B",
    ]
    markers = ["s", "o", "^", "D", "v", "<", ">", "p", "h"]

    for i, adapter in enumerate(sorted(adapters)):
        xs, ys, errs_low, errs_high = [], [], [], []
        for budget in budgets:
            key = f"{adapter}/{budget}"
            if key in cis:
                mean, lo, hi = cis[key]
                xs.append(budget_x[budget])
                ys.append(mean)
                errs_low.append(mean - lo)
                errs_high.append(hi - mean)
        if xs:
            ax.errorbar(
                xs, ys, yerr=[errs_low, errs_high],
                label=adapter, marker=markers[i % len(markers)],
                color=colors[i % len(colors)],
                capsize=4, linewidth=2, markersize=8,
            )

    ax.axhline(y=0.5, color="red", linestyle="--", alpha=0.5, label="parity (NBA=0.5)")
    ax.set_xticks(list(budget_x.values()))
    ax.set_xticklabels(budgets)
    ax.set_xlabel("Token Budget")
    ax.set_ylabel("NBA (mean +/- 95% CI)")
    ax.set_title("Naive Baseline Advantage vs Token Budget")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    ax.set_ylim(-0.1, 1.1)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
    log.info("Plot saved to %s", output_path)
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Analyze constrained validation results")
    parser.add_argument("--state-file", type=Path, default=STATE_FILE)
    parser.add_argument("--scope", default=None, help="Analyze single scope (e.g., 01)")
    parser.add_argument("--aggregate", action="store_true", help="Pool questions across all scopes")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    state = load_state(args.state_file)

    # Default: aggregate if multi-scope data exists, else single-scope
    use_aggregate = args.aggregate
    if not args.scope and not args.aggregate:
        # Auto-detect: if multiple scopes present, aggregate
        scopes_seen = set()
        for label in state:
            parsed = parse_label(label)
            if parsed:
                scopes_seen.add(parsed[1])
        use_aggregate = len(scopes_seen) > 1
        if use_aggregate:
            log.info("Multiple scopes detected (%s) — aggregating", ", ".join(sorted(scopes_seen)))

    vectors = collect_nba_vectors(state, scope_filter=args.scope, aggregate=use_aggregate or bool(args.scope))

    if not vectors:
        log.error("No scored runs with per-question NBA data found")
        sys.exit(1)

    log.info("Loaded NBA vectors for: %s", ", ".join(sorted(vectors.keys())))

    # -----------------------------------------------------------------------
    # Analysis 1: Bootstrap CIs
    # -----------------------------------------------------------------------
    print(f"\n{'=' * 75}")
    print("ANALYSIS 1: Bootstrap CIs on NBA (10,000 resamples)")
    print(f"{'=' * 75}")
    print(f"{'Label':<30} {'Mean':>8} {'95% CI Low':>12} {'95% CI High':>12} {'N':>5}")
    print("-" * 70)

    cis: dict[str, tuple[float, float, float]] = {}
    for key in sorted(vectors.keys()):
        v = vectors[key]
        mean, lo, hi = bootstrap_ci(v)
        cis[key] = (mean, lo, hi)
        print(f"{key:<30} {mean:>8.4f} {lo:>12.4f} {hi:>12.4f} {len(v):>5}")

    # -----------------------------------------------------------------------
    # Analysis 2: Paired adapter vs null
    # -----------------------------------------------------------------------
    print(f"\n{'=' * 75}")
    print("ANALYSIS 2: Paired Wilcoxon — Adapter vs Null")
    print(f"{'=' * 75}")
    print(f"{'Comparison':<45} {'Stat':>8} {'p-value':>10} {'Sig?':>6}")
    print("-" * 72)

    for budget in ["4k", "2k"]:
        null_key = f"null/{budget}"
        if null_key not in vectors:
            continue
        null_v = vectors[null_key]
        for key, v in sorted(vectors.items()):
            adapter, b = get_adapter_budget(key)
            if adapter == "null" or b != budget:
                continue
            if len(v) != len(null_v):
                # If aggregated, vectors may differ in length — skip paired test
                log.warning("Length mismatch: %s (%d) vs %s (%d)", key, len(v), null_key, len(null_v))
                continue
            stat, p = wilcoxon_test(v, null_v)
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            comparison = f"{key} vs {null_key}"
            print(f"{comparison:<45} {stat:>8.1f} {p:>10.4f} {sig:>6}")

    # -----------------------------------------------------------------------
    # Analysis 3: Budget degradation (4K vs 2K)
    # -----------------------------------------------------------------------
    print(f"\n{'=' * 75}")
    print("ANALYSIS 3: Budget Degradation (4K -> 2K per adapter)")
    print(f"{'=' * 75}")
    print(f"{'Adapter':<30} {'delta Mean':>10} {'Stat':>8} {'p-value':>10} {'Sig?':>6}")
    print("-" * 68)

    adapter_set = set()
    for key in vectors:
        adapter, budget = get_adapter_budget(key)
        adapter_set.add(adapter)

    for adapter in sorted(adapter_set):
        k4 = f"{adapter}/4k"
        k2 = f"{adapter}/2k"
        if k4 not in vectors or k2 not in vectors:
            continue
        v4, v2 = vectors[k4], vectors[k2]
        if len(v4) != len(v2):
            continue
        delta = float(v4.mean() - v2.mean())
        stat, p = wilcoxon_test(v4, v2)
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"{adapter:<30} {delta:>10.4f} {stat:>8.1f} {p:>10.4f} {sig:>6}")

    # -----------------------------------------------------------------------
    # Analysis 4: Differential degradation
    # -----------------------------------------------------------------------
    print(f"\n{'=' * 75}")
    print("ANALYSIS 4: Differential Degradation (adapter vs null)")
    print(f"{'=' * 75}")
    print(f"{'Adapter':<30} {'Interaction':>12} {'95% CI Low':>12} {'95% CI High':>12}")
    print("-" * 70)

    null_4k = vectors.get("null/4k")
    null_2k = vectors.get("null/2k")

    if null_4k is not None and null_2k is not None:
        for adapter in sorted(adapter_set):
            if adapter == "null":
                continue
            k4, k2 = f"{adapter}/4k", f"{adapter}/2k"
            if k4 not in vectors or k2 not in vectors:
                continue
            a4, a2 = vectors[k4], vectors[k2]
            n = min(len(a4), len(a2), len(null_4k), len(null_2k))
            mean, lo, hi = differential_degradation_ci(a4[:n], a2[:n], null_4k[:n], null_2k[:n])
            direction = "adapter holds better" if mean > 0 else "null holds better"
            print(f"{adapter:<30} {mean:>12.4f} {lo:>12.4f} {hi:>12.4f}  ({direction})")
    else:
        print("  (Need both null/4k and null/2k to compute interaction)")

    # -----------------------------------------------------------------------
    # Go/No-Go Verdict
    # -----------------------------------------------------------------------
    print(f"\n{'=' * 75}")
    print("GO/NO-GO VERDICT")
    print(f"{'=' * 75}")

    max_nba_2k = None
    best_adapter_2k = None
    for key, (mean, lo, hi) in cis.items():
        adapter, budget = get_adapter_budget(key)
        if budget == "2k" and adapter != "null":
            if max_nba_2k is None or mean > max_nba_2k:
                max_nba_2k = mean
                best_adapter_2k = key

    null_ci_2k = cis.get("null/2k", (None, None, None))

    if max_nba_2k is not None:
        print(f"  Best adapter NBA at 2K: {best_adapter_2k} = {max_nba_2k:.4f}")
        if null_ci_2k[0] is not None:
            print(f"  Null baseline NBA at 2K: {null_ci_2k[0]:.4f}")
            print(f"  Adapter advantage: {max_nba_2k - null_ci_2k[0]:+.4f}")

        if max_nba_2k > 0.45:
            print("\n  VERDICT: VALIDATE")
            print("  At least one adapter shows NBA > 0.45 at 2K budget.")
        elif max_nba_2k < 0.30:
            print("\n  VERDICT: INVALIDATE")
            print("  No adapter achieves NBA > 0.30 at 2K budget.")
        else:
            print("\n  VERDICT: INCONCLUSIVE")
            print(f"  Best adapter NBA at 2K = {max_nba_2k:.4f} (between 0.30 and 0.45).")
    else:
        print("  No adapter data at 2K budget.")

    # -----------------------------------------------------------------------
    # Plot
    # -----------------------------------------------------------------------
    if not args.no_plot:
        RESULTS_DIR.mkdir(exist_ok=True)
        suffix = f"_scope{args.scope}" if args.scope else ("_aggregate" if use_aggregate else "")
        plot_path = RESULTS_DIR / f"constrained_degradation{suffix}.png"
        plot_degradation(vectors, cis, plot_path)

    # -----------------------------------------------------------------------
    # Export
    # -----------------------------------------------------------------------
    RESULTS_DIR.mkdir(exist_ok=True)
    analysis_out = {
        "bootstrap_cis": {k: {"mean": m, "ci_low": lo, "ci_high": hi} for k, (m, lo, hi) in cis.items()},
        "verdict": {
            "best_adapter_2k": best_adapter_2k,
            "best_nba_2k": max_nba_2k,
            "null_nba_2k": null_ci_2k[0],
        },
    }
    analysis_path = RESULTS_DIR / "constrained_analysis.json"
    analysis_path.write_text(json.dumps(analysis_out, indent=2) + "\n")
    log.info("Analysis exported to %s", analysis_path)


if __name__ == "__main__":
    main()

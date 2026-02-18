#!/usr/bin/env python3
"""Multi-judge agreement analysis for LENS benchmark scoring.

Scores a subset of questions with two different judge models and computes
inter-rater reliability metrics (percent agreement, Cohen's kappa, tie rates).

Usage:
    uv run python scripts/judge_reliability.py \
        --run output/<run_id> \
        --judge-a gpt-4o-mini \
        --judge-b gpt-4o \
        --scopes 2 \
        --out output/judge_reliability/
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lens.artifacts.bundle import load_run_result
from lens.core.models import QuestionResult, RunResult
from lens.scorer.judge import pairwise_fact_judge
from lens.scorer.tier1 import _all_question_results


def make_openai_judge(model: str, api_key: str | None = None):
    """Create a judge_fn callable that uses OpenAI chat completions."""
    import openai

    key = api_key or os.environ.get("OPENAI_API_KEY") or os.environ.get("LENS_LLM_API_KEY")
    if not key:
        print("ERROR: Judge requires an API key. Set OPENAI_API_KEY or LENS_LLM_API_KEY.")
        sys.exit(1)
    client = openai.OpenAI(api_key=key)

    def judge_fn(prompt: str) -> str:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=10,
        )
        return resp.choices[0].message.content or "TIE"

    return judge_fn


def score_question_with_judge(
    qr: QuestionResult, judge_fn, seed: int = 42
) -> list[dict]:
    """Score a single question's key facts with the given judge.

    Returns per-fact verdicts: [{fact, winner, verdict_raw, fact_score}]
    """
    key_facts = qr.question.ground_truth.key_facts
    if not key_facts:
        return []

    _, details = pairwise_fact_judge(
        candidate_answer=qr.answer.answer_text,
        reference_answer=qr.question.ground_truth.canonical_answer,
        key_facts=key_facts,
        question=qr.question.prompt,
        judge_fn=judge_fn,
        seed=seed,
    )
    return details


def compute_agreement(
    verdicts_a: list[dict], verdicts_b: list[dict]
) -> dict:
    """Compute agreement metrics between two sets of per-fact verdicts.

    Returns:
        {
            "total_facts": int,
            "agree_count": int,
            "percent_agreement": float,
            "cohens_kappa": float,
            "tie_rate_a": float,
            "tie_rate_b": float,
        }
    """
    assert len(verdicts_a) == len(verdicts_b), "Verdict lists must be same length"

    if not verdicts_a:
        return {
            "total_facts": 0,
            "agree_count": 0,
            "percent_agreement": 0.0,
            "cohens_kappa": 0.0,
            "tie_rate_a": 0.0,
            "tie_rate_b": 0.0,
        }

    n = len(verdicts_a)
    agree_count = 0
    tie_count_a = 0
    tie_count_b = 0

    # Category counts for kappa: candidate, reference, tie
    categories = ["candidate", "reference", "tie"]
    count_a = {c: 0 for c in categories}
    count_b = {c: 0 for c in categories}

    for va, vb in zip(verdicts_a, verdicts_b):
        winner_a = va["winner"]
        winner_b = vb["winner"]

        if winner_a == winner_b:
            agree_count += 1

        if winner_a == "tie":
            tie_count_a += 1
        if winner_b == "tie":
            tie_count_b += 1

        count_a[winner_a] = count_a.get(winner_a, 0) + 1
        count_b[winner_b] = count_b.get(winner_b, 0) + 1

    p_o = agree_count / n  # observed agreement

    # Expected agreement by chance (Cohen's kappa)
    p_e = 0.0
    for c in categories:
        p_a = count_a.get(c, 0) / n
        p_b = count_b.get(c, 0) / n
        p_e += p_a * p_b

    if p_e == 1.0:
        kappa = 1.0
    else:
        kappa = (p_o - p_e) / (1.0 - p_e)

    return {
        "total_facts": n,
        "agree_count": agree_count,
        "percent_agreement": p_o,
        "cohens_kappa": kappa,
        "tie_rate_a": tie_count_a / n,
        "tie_rate_b": tie_count_b / n,
    }


def run_reliability_analysis(
    result: RunResult,
    judge_a_fn,
    judge_b_fn,
    judge_a_name: str,
    judge_b_name: str,
    max_scopes: int = 2,
    seed: int = 42,
) -> dict:
    """Run full reliability analysis across scopes.

    Returns structured results with per-question and aggregate metrics.
    """
    qrs = _all_question_results(result)

    # Limit to first N scopes
    scope_ids = []
    for scope in result.scopes:
        if len(scope_ids) >= max_scopes:
            break
        scope_ids.append(scope.scope_id)
    scope_set = set(scope_ids)

    filtered_qrs = [qr for qr in qrs if qr.question.scope_id in scope_set]
    print(f"Analyzing {len(filtered_qrs)} questions across {len(scope_ids)} scopes")

    per_question_results = []
    all_verdicts_a: list[dict] = []
    all_verdicts_b: list[dict] = []
    per_type_verdicts: dict[str, tuple[list[dict], list[dict]]] = {}

    for i, qr in enumerate(filtered_qrs):
        key_facts = qr.question.ground_truth.key_facts
        if not key_facts:
            continue

        qtype = qr.question.question_type
        print(f"  [{i+1}/{len(filtered_qrs)}] {qr.question.question_id} ({qtype}, {len(key_facts)} facts)")

        verdicts_a = score_question_with_judge(qr, judge_a_fn, seed=seed)
        verdicts_b = score_question_with_judge(qr, judge_b_fn, seed=seed)

        agreement = compute_agreement(verdicts_a, verdicts_b)

        per_question_results.append({
            "question_id": qr.question.question_id,
            "question_type": qtype,
            "num_facts": len(key_facts),
            "agreement": agreement,
        })

        all_verdicts_a.extend(verdicts_a)
        all_verdicts_b.extend(verdicts_b)

        if qtype not in per_type_verdicts:
            per_type_verdicts[qtype] = ([], [])
        per_type_verdicts[qtype][0].extend(verdicts_a)
        per_type_verdicts[qtype][1].extend(verdicts_b)

    # Aggregate agreement
    aggregate = compute_agreement(all_verdicts_a, all_verdicts_b)

    # Per-type agreement
    per_type_agreement = {}
    for qtype, (va, vb) in per_type_verdicts.items():
        per_type_agreement[qtype] = compute_agreement(va, vb)

    return {
        "judge_a": judge_a_name,
        "judge_b": judge_b_name,
        "scopes_analyzed": scope_ids,
        "questions_analyzed": len(per_question_results),
        "total_facts_compared": len(all_verdicts_a),
        "aggregate_agreement": aggregate,
        "per_type_agreement": per_type_agreement,
        "per_question": per_question_results,
    }


def format_summary(results: dict) -> str:
    """Format results as human-readable summary."""
    lines = [
        "=" * 60,
        "JUDGE RELIABILITY ANALYSIS",
        "=" * 60,
        f"Judge A: {results['judge_a']}",
        f"Judge B: {results['judge_b']}",
        f"Scopes: {', '.join(results['scopes_analyzed'])}",
        f"Questions: {results['questions_analyzed']}",
        f"Total facts compared: {results['total_facts_compared']}",
        "",
        "--- Aggregate Agreement ---",
    ]

    agg = results["aggregate_agreement"]
    lines.extend([
        f"  Percent agreement: {agg['percent_agreement']:.1%}",
        f"  Cohen's kappa:     {agg['cohens_kappa']:.3f}",
        f"  Tie rate (A):      {agg['tie_rate_a']:.1%}",
        f"  Tie rate (B):      {agg['tie_rate_b']:.1%}",
        "",
    ])

    # Kappa interpretation
    kappa = agg["cohens_kappa"]
    if kappa >= 0.8:
        interp = "EXCELLENT (almost perfect)"
    elif kappa >= 0.6:
        interp = "GOOD (substantial)"
    elif kappa >= 0.4:
        interp = "MODERATE (fair)"
    elif kappa >= 0.2:
        interp = "WEAK (slight)"
    else:
        interp = "POOR (no agreement)"
    lines.append(f"  Interpretation:    {interp}")
    lines.append("")

    # Per-type breakdown
    lines.append("--- Per-Type Agreement ---")
    for qtype, metrics in sorted(results["per_type_agreement"].items()):
        lines.append(
            f"  {qtype:30s}  agree={metrics['percent_agreement']:.1%}  "
            f"kappa={metrics['cohens_kappa']:.3f}  "
            f"n={metrics['total_facts']}"
        )

    lines.extend(["", "=" * 60])

    if kappa >= 0.6:
        lines.append("PASS: Kappa >= 0.6 — judge agreement is sufficient.")
    else:
        lines.append("FAIL: Kappa < 0.6 — judge agreement is insufficient.")
        lines.append("Consider: stronger judge prompt, better model, or binary A/B (no TIE).")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Multi-judge reliability analysis")
    parser.add_argument("--run", required=True, help="Run output directory")
    parser.add_argument("--judge-a", default="gpt-4o-mini", help="First judge model")
    parser.add_argument("--judge-b", default="gpt-4o", help="Second judge model")
    parser.add_argument("--scopes", type=int, default=2, help="Number of scopes to analyze")
    parser.add_argument("--out", default="output/judge_reliability", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    print(f"Loading run from {args.run}")
    result = load_run_result(args.run)

    print(f"Creating judges: {args.judge_a} vs {args.judge_b}")
    judge_a = make_openai_judge(args.judge_a)
    judge_b = make_openai_judge(args.judge_b)

    results = run_reliability_analysis(
        result=result,
        judge_a_fn=judge_a,
        judge_b_fn=judge_b,
        judge_a_name=args.judge_a,
        judge_b_name=args.judge_b,
        max_scopes=args.scopes,
        seed=args.seed,
    )

    # Write outputs
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "reliability.json"
    json_path.write_text(json.dumps(results, indent=2))
    print(f"\nJSON results: {json_path}")

    summary = format_summary(results)
    summary_path = out_dir / "reliability_summary.txt"
    summary_path.write_text(summary)
    print(f"Summary: {summary_path}")

    print()
    print(summary)


if __name__ == "__main__":
    main()

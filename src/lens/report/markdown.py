from __future__ import annotations

from lens.core.models import ScoreCard


def generate_markdown_report(scorecard: ScoreCard) -> str:
    """Generate a Markdown report from a ScoreCard."""
    lines: list[str] = []

    lines.append(f"# LENS Benchmark Report")
    lines.append("")
    lines.append(f"**Run ID**: {scorecard.run_id}")
    lines.append(f"**Adapter**: {scorecard.adapter}")
    lines.append(f"**Dataset**: {scorecard.dataset_version}")
    lines.append(f"**Budget Preset**: {scorecard.budget_preset}")
    lines.append(f"**Composite Score**: {scorecard.composite_score:.4f}")
    lines.append("")

    # Metrics table
    lines.append("## Metrics")
    lines.append("")
    lines.append("| Metric | Tier | Score |")
    lines.append("|--------|------|-------|")

    for metric in sorted(scorecard.metrics, key=lambda m: (m.tier, m.name)):
        lines.append(f"| {metric.name} | {metric.tier} | {metric.value:.4f} |")

    lines.append("")

    # Per-question timing table
    budget_metric = next(
        (m for m in scorecard.metrics if m.name == "budget_compliance"), None
    )
    if budget_metric and budget_metric.details.get("per_question_timing"):
        lines.append("## Per-Question Timing")
        lines.append("")
        lines.append("| Question | Type | Checkpoint | Wall Time | Tokens | Tool Calls |")
        lines.append("|----------|------|------------|-----------|--------|------------|")
        for t in budget_metric.details["per_question_timing"]:
            wall_s = t["wall_time_ms"] / 1000
            lines.append(
                f"| {t['question_id']} | {t['question_type']} | {t['checkpoint']} "
                f"| {wall_s:.1f}s | {t['total_tokens']} | {t['tool_calls']} |"
            )
        lines.append("")

    # Details
    lines.append("## Details")
    lines.append("")
    for metric in sorted(scorecard.metrics, key=lambda m: (m.tier, m.name)):
        if metric.details:
            lines.append(f"### {metric.name}")
            lines.append("")
            for k, v in metric.details.items():
                if k == "per_question_timing":
                    continue  # Already rendered as a table above
                lines.append(f"- **{k}**: {v}")
            lines.append("")

    return "\n".join(lines)


def generate_comparison_report(scorecards: list[ScoreCard]) -> str:
    """Generate a Markdown comparison of multiple runs."""
    if not scorecards:
        return "# LENS Comparison Report\n\nNo runs to compare."

    lines: list[str] = []
    lines.append("# LENS Benchmark Comparison")
    lines.append("")

    # Summary table
    header = "| Metric |"
    sep = "|--------|"
    for sc in scorecards:
        header += f" {sc.adapter} ({sc.budget_preset}) |"
        sep += "------|"

    lines.append(header)
    lines.append(sep)

    # Composite
    row = "| **Composite** |"
    for sc in scorecards:
        row += f" {sc.composite_score:.4f} |"
    lines.append(row)

    # Individual metrics
    all_metric_names = sorted(
        {m.name for sc in scorecards for m in sc.metrics}
    )

    for metric_name in all_metric_names:
        row = f"| {metric_name} |"
        for sc in scorecards:
            value = next(
                (m.value for m in sc.metrics if m.name == metric_name),
                None,
            )
            row += f" {value:.4f} |" if value is not None else " — |"
        lines.append(row)

    # Timing comparison
    has_timing = any(
        m.name == "budget_compliance" and m.details.get("per_question_timing")
        for sc in scorecards
        for m in sc.metrics
    )
    if has_timing:
        lines.append("")
        lines.append("## Timing Comparison")
        lines.append("")
        t_header = "| Stat |"
        t_sep = "|------|"
        for sc in scorecards:
            t_header += f" {sc.adapter} |"
            t_sep += "------|"
        lines.append(t_header)
        lines.append(t_sep)

        for stat_key, stat_label in [
            ("total_wall_time_minutes", "Total (min)"),
            ("avg_wall_time_ms", "Avg/question (ms)"),
            ("max_wall_time_ms", "Max/question (ms)"),
        ]:
            row = f"| {stat_label} |"
            for sc in scorecards:
                bm = next((m for m in sc.metrics if m.name == "budget_compliance"), None)
                val = bm.details.get(stat_key) if bm else None
                row += f" {val} |" if val is not None else " -- |"
            lines.append(row)

    lines.append("")
    return "\n".join(lines)

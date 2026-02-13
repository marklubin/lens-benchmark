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

    # Details
    lines.append("## Details")
    lines.append("")
    for metric in sorted(scorecard.metrics, key=lambda m: (m.tier, m.name)):
        if metric.details:
            lines.append(f"### {metric.name}")
            lines.append("")
            for k, v in metric.details.items():
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

    lines.append("")
    return "\n".join(lines)

from __future__ import annotations

from lens.core.models import ScoreCard


def generate_html_report(scorecard: ScoreCard) -> str:
    """Generate a self-contained HTML report from a ScoreCard."""
    metrics_sorted = sorted(scorecard.metrics, key=lambda m: (m.tier, m.name))

    metric_rows = ""
    for m in metrics_sorted:
        pct = m.value * 100
        if pct >= 70:
            bar_color = "#22c55e"
        elif pct >= 40:
            bar_color = "#eab308"
        else:
            bar_color = "#ef4444"
        metric_rows += f"""
        <tr>
          <td>T{m.tier}</td>
          <td>{_esc(m.name)}</td>
          <td class="score">{m.value:.4f}</td>
          <td>
            <div class="bar-bg"><div class="bar" style="width:{pct:.1f}%;background:{bar_color}"></div></div>
          </td>
        </tr>"""

    details_sections = ""
    for m in metrics_sorted:
        if m.details:
            items = "".join(
                f"<li><strong>{_esc(str(k))}</strong>: {_esc(str(v))}</li>"
                for k, v in m.details.items()
            )
            details_sections += f"""
      <details>
        <summary>{_esc(m.name)} (T{m.tier})</summary>
        <ul>{items}</ul>
      </details>"""

    composite_pct = scorecard.composite_score * 100
    if composite_pct >= 70:
        composite_color = "#22c55e"
    elif composite_pct >= 40:
        composite_color = "#eab308"
    else:
        composite_color = "#ef4444"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LENS Report &mdash; {_esc(scorecard.run_id)}</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
         max-width: 900px; margin: 2rem auto; padding: 0 1rem; color: #1e293b; background: #f8fafc; }}
  h1 {{ font-size: 1.6rem; margin-bottom: .5rem; }}
  .meta {{ color: #64748b; font-size: .9rem; margin-bottom: 1.5rem; }}
  .meta span {{ margin-right: 1.5rem; }}
  .composite {{ display: inline-block; font-size: 2rem; font-weight: 700;
                padding: .3rem .8rem; border-radius: .5rem; margin-bottom: 1.5rem;
                color: white; background: {composite_color}; }}
  table {{ width: 100%; border-collapse: collapse; margin-bottom: 1.5rem; }}
  th, td {{ text-align: left; padding: .5rem .75rem; border-bottom: 1px solid #e2e8f0; }}
  th {{ background: #f1f5f9; font-weight: 600; font-size: .85rem; text-transform: uppercase; color: #475569; }}
  .score {{ font-family: monospace; }}
  .bar-bg {{ width: 100%; height: 1rem; background: #e2e8f0; border-radius: .25rem; overflow: hidden; }}
  .bar {{ height: 100%; border-radius: .25rem; transition: width .3s; }}
  details {{ margin-bottom: .5rem; }}
  summary {{ cursor: pointer; font-weight: 600; padding: .4rem 0; }}
  details ul {{ margin: .5rem 0 .5rem 1.5rem; }}
  h2 {{ font-size: 1.2rem; margin: 1.5rem 0 .75rem; }}
</style>
</head>
<body>
<h1>LENS Benchmark Report</h1>
<div class="meta">
  <span><strong>Run:</strong> {_esc(scorecard.run_id)}</span>
  <span><strong>Adapter:</strong> {_esc(scorecard.adapter)}</span>
  <span><strong>Dataset:</strong> {_esc(scorecard.dataset_version)}</span>
  <span><strong>Budget:</strong> {_esc(scorecard.budget_preset)}</span>
</div>
<div class="composite">{scorecard.composite_score:.4f}</div>
<h2>Metrics</h2>
<table>
  <thead><tr><th>Tier</th><th>Metric</th><th>Score</th><th>Bar</th></tr></thead>
  <tbody>{metric_rows}
  </tbody>
</table>
<h2>Details</h2>
{details_sections}
</body>
</html>
"""


def _esc(text: str) -> str:
    """Minimal HTML escaping."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")

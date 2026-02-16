from __future__ import annotations


def generate_verification_report(
    results: dict, spec: object, manifest: dict
) -> str:
    """Generate a self-contained HTML verification report.

    Args:
        results: Full verification results dict (from verify_scope).
        spec: ScopeSpec dataclass instance.
        manifest: Parsed manifest.json dict.

    Returns:
        Complete HTML string.
    """
    meta = results.get("_meta", {})

    # --- Scope overview ---
    overview_html = _build_overview(meta, manifest, spec)

    # --- Episode timeline ---
    timeline_html = _build_timeline(spec)

    # --- Key fact coverage ---
    keyfact_html = _build_key_fact_coverage(manifest, spec)

    # --- Contamination check ---
    contamination_html = _build_contamination(results.get("contamination"))

    # --- Distractor purity ---
    distractor_html = _build_distractor_purity(results.get("distractor_purity"))

    # --- Naive baseline ---
    baseline_html = _build_naive_baseline(results.get("naive_baseline"))

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LENS Verification Report &mdash; {_esc(meta.get("scope_id", ""))}</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
         max-width: 960px; margin: 2rem auto; padding: 0 1rem; color: #1e293b; background: #f8fafc; }}
  h1 {{ font-size: 1.6rem; margin-bottom: .5rem; }}
  h2 {{ font-size: 1.2rem; margin: 1.5rem 0 .75rem; border-bottom: 2px solid #e2e8f0; padding-bottom: .25rem; }}
  .meta {{ color: #64748b; font-size: .9rem; margin-bottom: 1.5rem; }}
  .meta span {{ display: inline-block; margin-right: 1.5rem; margin-bottom: .25rem; }}
  table {{ width: 100%; border-collapse: collapse; margin-bottom: 1.5rem; }}
  th, td {{ text-align: left; padding: .5rem .75rem; border-bottom: 1px solid #e2e8f0; font-size: .9rem; }}
  th {{ background: #f1f5f9; font-weight: 600; font-size: .85rem; text-transform: uppercase; color: #475569; }}
  .bar-bg {{ width: 100%; height: 1rem; background: #e2e8f0; border-radius: .25rem; overflow: hidden; }}
  .bar {{ height: 100%; border-radius: .25rem; }}
  .badge {{ display: inline-block; padding: .15rem .5rem; border-radius: .25rem; font-size: .8rem; font-weight: 600; color: white; }}
  .badge-pass {{ background: #22c55e; }}
  .badge-fail {{ background: #ef4444; }}
  .badge-na {{ background: #94a3b8; }}
  .check {{ color: #22c55e; font-weight: bold; }}
  .cross {{ color: #ef4444; font-weight: bold; }}
  details {{ margin-bottom: .5rem; }}
  summary {{ cursor: pointer; font-weight: 600; padding: .4rem 0; }}
  details ul {{ margin: .5rem 0 .5rem 1.5rem; }}
  .timeline {{ display: flex; margin-bottom: 1.5rem; border-radius: .25rem; overflow: hidden; }}
  .phase-bar {{ height: 2rem; display: flex; align-items: center; justify-content: center;
                font-size: .75rem; font-weight: 600; color: white; min-width: 40px; }}
  .density-none {{ background: #94a3b8; }}
  .density-low {{ background: #3b82f6; }}
  .density-medium {{ background: #f97316; }}
  .density-high {{ background: #ef4444; }}
  .not-run {{ color: #94a3b8; font-style: italic; }}
  .answer-text {{ background: #f1f5f9; padding: .5rem; border-radius: .25rem; font-size: .85rem;
                  white-space: pre-wrap; word-break: break-word; max-height: 200px; overflow-y: auto; }}
</style>
</head>
<body>
<h1>LENS Verification Report</h1>
{overview_html}
{timeline_html}
{keyfact_html}
{contamination_html}
{distractor_html}
{baseline_html}
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------


def _build_overview(meta: dict, manifest: dict, spec: object) -> str:
    scope_id = _esc(meta.get("scope_id", getattr(spec, "scope_id", "")))
    domain = _esc(meta.get("domain", getattr(spec, "domain", "")))
    ep_count = meta.get("episode_count", "?")
    q_count = meta.get("question_count", "?")
    verified_at = _esc(meta.get("verified_at", ""))
    model = _esc(meta.get("model", ""))
    generated_at = _esc(manifest.get("generated_at", ""))

    return f"""<div class="meta">
  <span><strong>Scope:</strong> {scope_id}</span>
  <span><strong>Domain:</strong> {domain}</span>
  <span><strong>Episodes:</strong> {ep_count}</span>
  <span><strong>Questions:</strong> {q_count}</span>
  <span><strong>Model:</strong> {model}</span>
  <span><strong>Verified:</strong> {verified_at}</span>
  <span><strong>Generated:</strong> {generated_at}</span>
</div>"""


def _build_timeline(spec: object) -> str:
    arc = getattr(spec, "arc", [])
    if not arc:
        return ""

    total_eps = getattr(spec, "episodes", None)
    total_count = getattr(total_eps, "count", 30) if total_eps else 30

    bars = ""
    for phase in arc:
        start, end = phase.episode_range()
        span = end - start + 1
        pct = (span / total_count) * 100
        density_cls = f"density-{phase.signal_density}"
        bars += (
            f'<div class="phase-bar {density_cls}" '
            f'style="width:{pct:.1f}%" '
            f'title="{_esc(phase.id)}: eps {phase.episodes}, {phase.signal_density}">'
            f"{_esc(phase.id)}</div>"
        )

    return f"""<h2>Episode Timeline</h2>
<div class="timeline">{bars}</div>"""


def _build_key_fact_coverage(manifest: dict, spec: object) -> str:
    kf_coverage = manifest.get("key_fact_coverage")
    if not kf_coverage:
        return ""

    # Build lookup from spec key_facts
    spec_facts = {kf.id: kf.fact for kf in getattr(spec, "key_facts", [])}

    rows = ""
    for fact_id, info in kf_coverage.items():
        target = info.get("target_episodes", [])
        found = info.get("found_in", [])
        total = len(target)
        hit = len(found)
        missing = sorted(set(target) - set(found))
        rate = (hit / total * 100) if total else 0
        color = _score_color(rate)
        fact_text = _esc(spec_facts.get(fact_id, fact_id))
        missing_str = _esc(", ".join(missing)) if missing else "-"

        rows += f"""
        <tr>
          <td>{_esc(fact_id)}</td>
          <td>{fact_text}</td>
          <td>{hit}/{total}</td>
          <td>{missing_str}</td>
          <td><div class="bar-bg"><div class="bar" style="width:{rate:.0f}%;background:{color}"></div></div></td>
        </tr>"""

    return f"""<h2>Key Fact Coverage</h2>
<table>
  <thead><tr><th>Fact ID</th><th>Fact</th><th>Found</th><th>Missing</th><th>Hit Rate</th></tr></thead>
  <tbody>{rows}
  </tbody>
</table>"""


def _build_contamination(contamination: dict | None) -> str:
    if contamination is None:
        return '<h2>Contamination Check</h2>\n<p class="not-run">Not run</p>'

    summary = contamination.get("summary", "")
    badge_cls = "badge-pass" if summary == "pass" else "badge-fail"
    questions = contamination.get("questions", [])

    rows = ""
    for q in questions:
        q_id = _esc(q.get("question_id", ""))
        max_cov = q.get("max_single_episode_coverage", 0)
        pct = max_cov * 100
        color = _score_color_inverse(pct)  # lower is better for contamination
        is_clean = not q.get("contaminated", False)
        status_badge = (
            '<span class="badge badge-pass">PASS</span>'
            if is_clean
            else '<span class="badge badge-fail">FAIL</span>'
        )
        worst = _esc(q.get("worst_episode", "-"))

        # Expandable per-episode detail
        episode_scores = q.get("episode_scores", [])
        ep_detail = ""
        if episode_scores:
            ep_items = ""
            for es in episode_scores:
                ep_items += (
                    f"<li><strong>{_esc(es.get('episode_id', ''))}</strong>: "
                    f"{es.get('coverage', 0):.1%}</li>"
                )
            ep_detail = f"""
        <details>
          <summary>Per-episode scores ({len(episode_scores)} episodes)</summary>
          <ul>{ep_items}</ul>
        </details>"""

        rows += f"""
        <tr>
          <td>{q_id}</td>
          <td>{status_badge}</td>
          <td>
            <div class="bar-bg"><div class="bar" style="width:{pct:.0f}%;background:{color}"></div></div>
            {pct:.1f}%
          </td>
          <td>{worst}</td>
        </tr>
        <tr><td colspan="4">{ep_detail}</td></tr>"""

    return f"""<h2>Contamination Check <span class="badge {badge_cls}">{_esc(summary).upper()}</span></h2>
<table>
  <thead><tr><th>Question</th><th>Status</th><th>Max Single-Episode Coverage</th><th>Worst Episode</th></tr></thead>
  <tbody>{rows}
  </tbody>
</table>"""


def _build_distractor_purity(purity: dict | None) -> str:
    if purity is None:
        return ""

    summary = purity.get("summary", "")
    badge_cls = "badge-pass" if summary == "pass" else "badge-fail"
    threshold = purity.get("threshold", 0.3)
    total = purity.get("total_distractors", 0)
    flagged = purity.get("flagged_count", 0)
    avg_sim = purity.get("avg_similarity", 0.0)
    max_sim = purity.get("max_similarity", 0.0)

    stats = (
        f'<span><strong>Total:</strong> {total}</span> '
        f'<span><strong>Flagged:</strong> {flagged}</span> '
        f'<span><strong>Threshold:</strong> {threshold:.2f}</span> '
        f'<span><strong>Avg Similarity:</strong> {avg_sim:.3f}</span> '
        f'<span><strong>Max Similarity:</strong> {max_sim:.3f}</span>'
    )

    rows = ""
    for ep in purity.get("episodes", []):
        eid = _esc(ep.get("episode_id", ""))
        theme = _esc(ep.get("theme", ""))
        sim = ep.get("max_similarity", 0)
        pct = sim * 100
        color = _score_color_inverse(pct)
        is_flagged = ep.get("flagged", False)
        status = (
            '<span class="badge badge-fail">FLAGGED</span>'
            if is_flagged
            else '<span class="badge badge-pass">OK</span>'
        )
        rows += f"""
        <tr>
          <td>{eid}</td>
          <td>{theme}</td>
          <td>{status}</td>
          <td>
            <div class="bar-bg"><div class="bar" style="width:{pct:.0f}%;background:{color}"></div></div>
            {sim:.3f}
          </td>
        </tr>"""

    return f"""<h2>Distractor Purity <span class="badge {badge_cls}">{_esc(summary).upper()}</span></h2>
<div class="meta">{stats}</div>
<table>
  <thead><tr><th>Episode</th><th>Theme</th><th>Status</th><th>Max Key-Fact Similarity</th></tr></thead>
  <tbody>{rows}
  </tbody>
</table>"""


def _build_naive_baseline(baseline: dict | None) -> str:
    if baseline is None:
        return '<h2>Naive Baseline</h2>\n<p class="not-run">Not run</p>'

    summary = baseline.get("summary", {})
    questions = baseline.get("questions", [])

    # Summary stats
    summary_items = ""
    if isinstance(summary, dict):
        for qt, avg in summary.items():
            summary_items += f"<span><strong>{_esc(qt)}:</strong> {avg:.1%}</span> "

    rows = ""
    for q in questions:
        q_id = _esc(q.get("question_id", ""))
        q_type = _esc(q.get("question_type", ""))
        coverage = q.get("fact_coverage", 0)
        pct = coverage * 100
        color = _score_color(pct)
        answer = q.get("answer", q.get("answer_preview", ""))

        # Per-fact matches
        per_fact = q.get("per_fact_matches", [])
        fact_list = ""
        if per_fact:
            items = ""
            for pf in per_fact:
                icon = '<span class="check">&#10003;</span>' if pf.get("matched") else '<span class="cross">&#10007;</span>'
                items += f"<li>{icon} {_esc(pf.get('fact', ''))} ({pf.get('overlap_ratio', 0):.0%})</li>"
            fact_list = f"<ul>{items}</ul>"

        rows += f"""
        <tr>
          <td>{q_id}</td>
          <td>{q_type}</td>
          <td>
            <div class="bar-bg"><div class="bar" style="width:{pct:.0f}%;background:{color}"></div></div>
            {pct:.1f}%
          </td>
        </tr>
        <tr><td colspan="3">
          {fact_list}
          <details><summary>Full answer</summary><div class="answer-text">{_esc(answer)}</div></details>
        </td></tr>"""

    return f"""<h2>Naive Baseline</h2>
<div class="meta">{summary_items}</div>
<table>
  <thead><tr><th>Question</th><th>Type</th><th>Fact Coverage</th></tr></thead>
  <tbody>{rows}
  </tbody>
</table>"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _score_color(pct: float) -> str:
    """Return color for a percentage score (higher = better)."""
    if pct >= 70:
        return "#22c55e"
    elif pct >= 40:
        return "#eab308"
    else:
        return "#ef4444"


def _score_color_inverse(pct: float) -> str:
    """Return color for contamination (lower = better)."""
    if pct <= 30:
        return "#22c55e"
    elif pct <= 50:
        return "#eab308"
    else:
        return "#ef4444"


def _esc(text: str) -> str:
    """Minimal HTML escaping."""
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )

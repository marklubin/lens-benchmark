"""Standalone prompt building for the synix pipeline.

All functions work with plain dicts (from spec_utils) — no lens.* imports.
"""
from __future__ import annotations

import json

from spec_utils import episode_range, get_key_facts_for_phase, make_episode_timestamp


SYSTEM_PROMPT = (
    "You generate realistic synthetic data for a benchmark evaluation. "
    "Output valid JSON only. No markdown fences, no commentary."
)

RENDER_SYSTEM = (
    "You are a technical writer formatting raw data sheets into terse log entries. "
    "Output ONLY the formatted log entry text. No JSON, no markdown fences, no commentary."
)


# ---------------------------------------------------------------------------
# Progressive-expansion prompts (plan → render)
# ---------------------------------------------------------------------------


def build_plan_signal_prompt(spec: dict) -> str:
    """Build planner prompt for signal episode data sheets.

    The planner sees the full spec and produces per-episode data sheets
    with concrete metric values. Signal is encoded as metric progressions
    (numbers only), never as text commentary.
    """
    ep_count = spec["episodes"]["count"]
    sections: list[str] = []

    # Scenario context
    setting = spec["scenario"]["setting"]
    if setting:
        sections.append(f"## Scenario\n{setting.strip()}")

    # Arc with signal density
    arc_lines: list[str] = []
    for phase in spec["arc"]:
        start, end = episode_range(phase)
        arc_lines.append(
            f"- **{phase['id']}** (episodes {start}-{end}): "
            f"{phase['description']} [signal_density: {phase['signal_density']}]"
        )
    sections.append("## Arc Phases\n" + "\n".join(arc_lines))

    # Key facts — the planner needs these to encode signal as metric values
    kf_lines: list[str] = []
    for kf in spec["key_facts"]:
        placements = [kf["first_appears"]] + kf["reinforced_in"]
        kf_lines.append(
            f'- {kf["id"]}: "{kf["fact"]}" '
            f'(appears: {", ".join(p for p in placements if p)})'
        )
    if kf_lines:
        sections.append("## Key Facts to Encode as Metric Progressions\n" + "\n".join(kf_lines))

    # Noise guidance
    noise = spec.get("noise", {})
    if noise.get("description") or noise.get("examples"):
        noise_parts: list[str] = []
        if noise.get("description"):
            noise_parts.append(noise["description"].strip())
        if noise.get("examples"):
            noise_parts.append("Examples of noise content:")
            for ex in noise["examples"]:
                noise_parts.append(f"  - {ex}")
        sections.append("## Noise / Routine Data\n" + "\n".join(noise_parts))

    # Timeline
    timeline = spec["episodes"]["timeline"]
    sections.append(
        f"## Timeline\n"
        f"Start date: {timeline['start']}, interval: {timeline['interval']}, "
        f"total episodes: {ep_count}"
    )

    # Data sheet richness requirements
    sections.append(
        "## DATA SHEET RICHNESS\n"
        "Each data sheet must contain DENSE structured data — enough raw material for "
        "the renderer to produce a 400-500 word log entry. Include ALL of the following "
        "categories in EVERY episode:\n\n"
        "1. **endpoint_metrics**: At least 5-8 API endpoints with full stats each "
        "(requests, p50, p95, p99, error_count, error_pct, success_rate)\n"
        "2. **infrastructure**: Host-level stats for 3-4 servers "
        "(cpu_pct, memory_pct, disk_pct, network_in_mbps, network_out_mbps, "
        "open_connections, thread_count)\n"
        "3. **connection_pools**: Per-pool stats "
        "(active, idle, waiting, exhaustion_events, max_size, avg_wait_ms)\n"
        "4. **cdn**: Cache hit rate, bandwidth, origin_requests, edge_locations_reporting\n"
        "5. **alerts**: List of alerts that fired (alert name, severity, host, value). "
        "Baseline episodes have routine alerts (disk, cert expiry). "
        "Later episodes have escalating alerts.\n"
        "6. **deployments**: Any deployments, config changes, feature flag toggles\n"
        "7. **events**: 3-6 operational events per episode "
        "(maintenance windows, rotation, scaling events, A/B test updates, cert renewals, "
        "monitoring changes)\n"
        "8. **on_call**: Shift handoff note with: engineer name, pages received (count), "
        "tickets opened (count), status. Example: "
        '"Shift: J. Martinez. 0 pages. 1 ticket (DISK-4421 cleanup). Quiet shift."'
    )

    # Signal encoding with explicit numeric ranges
    sections.append(
        "## SIGNAL ENCODING (METRIC PROGRESSIONS)\n"
        "Encode each key fact as a DRAMATIC numeric progression across the arc:\n\n"
        "- geo_latency_degradation: geo-lookup p99 should progress from ~180ms (baseline) "
        "→ ~250ms (early_signal) → ~400ms (red_herring) → ~600ms (escalation) → ~850ms (root_cause)\n"
        "- service_b_retries: service-B retry_count should progress from ~50 (baseline) "
        "→ ~120 (early_signal) → ~300 (escalation) → ~800 (root_cause). "
        "Also include retry_rate_pct as a metric.\n"
        "- pool_exhaustion: connection pool exhaustion_events from 0 (baseline) "
        "→ 0-1 (early_signal) → 5-15 (escalation) → 30-80 (root_cause). "
        "Also show pool waiting count climbing.\n"
        "- deploy_red_herring: In episodes 14-16, include a service-C deploy event. "
        "In episode 15, add a service-C rollback event. These are factual events, "
        "not commentary.\n\n"
        "The progressions must be LARGE enough to be detectable by comparing numbers "
        "across episodes. A 5ms change is invisible — use 50-200ms jumps."
    )

    # Critical instructions
    sections.append(
        "## LANGUAGE RULES (CRITICAL)\n"
        "ALL text fields (on_call, events, notes) must contain ONLY factual statements.\n\n"
        "FORBIDDEN in ALL fields — these words MUST NOT appear anywhere in any data sheet:\n"
        "increasing, decreasing, degrading, elevated, concerning, anomalous, notable, "
        "rising, higher, lower, growing, worsening, spiking, climbing, dropping, "
        "unusual, abnormal, started to, continues to, trending\n\n"
        "GOOD on_call examples:\n"
        '- "Shift: J. Martinez. 0 pages. 1 ticket (DISK-4421). Quiet shift."\n'
        '- "Shift: A. Chen. 2 pages (PG-CONN-5501, CDN-MISS-2203). Both resolved."\n'
        '- "Shift: K. Okafor. 4 pages. Checkout error rate at 2.1%. Investigating."\n\n'
        "GOOD event examples:\n"
        '- "Deployed service-C v3.2.1"\n'
        '- "SSL cert renewed for api.example.com, expires 2025-01-14"\n'
        '- "Scaled checkout-worker pool from 8 to 12 instances"\n\n'
        "BAD (FORBIDDEN) examples:\n"
        '- "Geo-lookup latency started to increase" ← uses "started to increase"\n'
        '- "Latency higher than normal" ← uses "higher"\n'
        '- "Pool exhaustion events rising" ← uses "rising"'
    )

    # Output format
    sections.append(
        f"## Output\n"
        f"Produce exactly {ep_count} episode data sheets.\n"
        f"Return JSON:\n"
        f'{{"episodes": [\n'
        f'  {{\n'
        f'    "index": 1,\n'
        f'    "date": "YYYY-MM-DD",\n'
        f'    "endpoint_metrics": {{\n'
        f'      "checkout": {{"requests": N, "p50": N, "p95": N, "p99": N, "error_count": N, "error_pct": N, "success_rate": N}},\n'
        f'      "fraud_check": {{...}},\n'
        f'      "geo_lookup": {{...}},\n'
        f'      "auth": {{...}},\n'
        f'      "product_catalog": {{...}},\n'
        f'      "search": {{...}},\n'
        f'      "recommendations": {{...}}\n'
        f'    }},\n'
        f'    "infrastructure": {{\n'
        f'      "gateway-01": {{"cpu_pct": N, "memory_pct": N, "disk_pct": N, "open_connections": N}},\n'
        f'      "gateway-02": {{...}},\n'
        f'      "metrics-db-01": {{...}}\n'
        f'    }},\n'
        f'    "connection_pools": {{\n'
        f'      "primary": {{"active": N, "idle": N, "waiting": N, "exhaustion_events": N, "max_size": N}},\n'
        f'      "replica": {{...}}\n'
        f'    }},\n'
        f'    "cdn": {{"hit_rate_pct": N, "bandwidth_gbps": N, "origin_requests": N}},\n'
        f'    "alerts": [{{"name": "...", "severity": "...", "host": "...", "value": "..."}}],\n'
        f'    "deployments": ["Deployed X vN.N.N"],\n'
        f'    "events": ["event1", "event2", "event3"],\n'
        f'    "on_call": "Shift: Name. N pages. N tickets. Status."\n'
        f'  }},\n'
        f'  ...\n'
        f"]}}"
    )

    return "\n\n".join(sections)


def build_plan_distractor_prompt(spec: dict, theme: dict, count: int) -> str:
    """Build planner prompt for distractor episode data sheets.

    The planner produces data sheets for a specific distractor theme.
    It knows the key facts so it can avoid them.
    """
    sections: list[str] = []

    # Theme scenario
    sections.append(f"## Distractor Theme: {theme['id']}\n{theme['scenario'].strip()}")

    # Key facts to AVOID
    kf_lines: list[str] = []
    for kf in spec["key_facts"]:
        kf_lines.append(f'- "{kf["fact"]}"')
    if kf_lines:
        sections.append(
            "## Key Facts to AVOID (do NOT include data related to these)\n"
            + "\n".join(kf_lines)
        )

    # Theme excluded terms
    excluded = theme.get("excluded_terms", [])
    if excluded:
        terms = ", ".join(f'"{t}"' for t in excluded)
        sections.append(f"## Excluded Terms\nDo NOT use these terms: {terms}")

    # Instructions
    sections.append(
        "## CRITICAL INSTRUCTIONS\n"
        "You are producing DATA SHEETS for distractor episodes — these are about a "
        "completely different topic from the signal episodes.\n\n"
        "RULES:\n"
        "1. Use domain-appropriate metrics for this theme (NOT the signal scenario metrics).\n"
        "2. Do NOT include any data that relates to the key facts listed above.\n"
        "3. Each data sheet has theme-specific metrics, events, and notes.\n"
        "4. Data sheets must be self-contained and coherent within the theme.\n"
        "5. Each data sheet must be DENSE — include 8-15 different metric categories, "
        "3-5 events, infrastructure stats, and detailed on-call notes. "
        "The renderer needs enough raw data to produce a 400-500 word log entry."
    )

    # Output format
    sections.append(
        f"## Output\n"
        f"Produce exactly {count} episode data sheets for theme '{theme['id']}'.\n"
        f"Return JSON:\n"
        f'{{"episodes": [\n'
        f'  {{\n'
        f'    "index": 1,\n'
        f'    "theme": "{theme["id"]}",\n'
        f'    "date": "YYYY-MM-DD",\n'
        f'    "metrics": {{"category1": {{"metric": N, ...}}, "category2": {{...}}, ...}},\n'
        f'    "infrastructure": {{"host1": {{"cpu_pct": N, "memory_pct": N, ...}}, ...}},\n'
        f'    "events": ["event1", "event2", "event3"],\n'
        f'    "on_call": "Shift: Name. N pages. N tickets. Status."\n'
        f'  }},\n'
        f'  ...\n'
        f"]}}"
    )

    return "\n\n".join(sections)


def build_render_prompt(spec: dict, brief: dict, episode_type: str = "signal") -> str:
    """Build renderer prompt for formatting a single data sheet into a log entry.

    The renderer sees ONLY shared context (voice, format, entity names) and
    one data sheet. It does NOT see the arc, key facts, signal placements,
    questions, or any other episodes.
    """
    target_words = spec["episodes"]["target_words"]
    if episode_type == "distractor":
        dc = spec.get("distractors")
        if dc and dc.get("target_words", 0) > 0:
            target_words = dc["target_words"]

    sections: list[str] = []

    sections.append("You are writing a single daily operational log entry.")

    # Format
    fmt = spec["episodes"]["format"]
    if fmt:
        sections.append(f"## Format\n{fmt}")

    # Voice
    voice = spec["scenario"]["voice"]
    if voice:
        sections.append(f"## Voice\n{voice}")

    # Structure template
    date_str = brief.get("date", "YYYY-MM-DD")
    sections.append(
        "## REQUIRED STRUCTURE\n"
        "The log entry MUST follow this structure with section headers:\n\n"
        f"```\n"
        f"## {date_str} Daily Operations Summary\n\n"
        f"### Endpoint Performance\n"
        f"- /endpoint-name: REQUESTS req | p50: Xms p95: Xms p99: Xms | err: X% (N errors) | success: X%\n"
        f"  (one line per endpoint, include ALL endpoints from data sheet)\n\n"
        f"### Infrastructure\n"
        f"- hostname: CPU X% | Mem X% | Disk X% | Conns: N | Net: X/X Mbps\n"
        f"  (one line per host)\n\n"
        f"### Connection Pools\n"
        f"- pool-name: active N | idle N | waiting N | exhaustion: N | max: N | avg_wait: Xms\n\n"
        f"### CDN & Caching\n"
        f"- Hit rate: X% | Bandwidth: X Gbps | Origin requests: N\n\n"
        f"### Alerts\n"
        f"- [SEVERITY] alert-name on hostname: value\n\n"
        f"### Deployments & Changes\n"
        f"- description of each deployment or change\n\n"
        f"### Events\n"
        f"- description of each operational event\n\n"
        f"### On-Call\n"
        f"- shift handoff note\n"
        f"```"
    )

    # Strict formatting rules
    sections.append(
        "## STRICT RULES\n"
        "- Include EVERY metric from the data sheet. Do not skip any.\n"
        "- Each endpoint gets its own line with ALL its stats.\n"
        "- Each host gets its own line with ALL its stats.\n"
        "- Bullet points and metrics ONLY. No prose paragraphs.\n"
        "- Do NOT add interpretation, trends, analysis, or commentary.\n"
        "- Do NOT mention anything not in the data sheet.\n"
        "- Do NOT use words like 'increasing', 'degrading', 'elevated', "
        "'anomalous', 'concerning', 'notable', 'higher', 'rising', 'unusual'.\n"
        "- Do NOT compare to previous days or expected values.\n"
        f"- MINIMUM {target_words} words. Include every data point from the sheet."
    )

    # The data sheet itself
    sections.append(f"## Data Sheet\n```json\n{json.dumps(brief, indent=2)}\n```")

    sections.append(
        "## Output\n"
        "Return ONLY the formatted log entry text. No JSON wrapping, "
        "no markdown fences around the output. "
        "Include ALL section headers and ALL metrics from the data sheet."
    )

    return "\n\n".join(sections)


def build_phase_prompt(
    spec: dict,
    phase: dict,
    prior_summaries: list[str],
) -> str:
    """Build the user prompt for generating episodes in a single arc phase."""
    start, end = _episode_range(phase)
    episode_count = end - start + 1
    target_words = spec["episodes"]["target_words"]

    sections: list[str] = []

    # Scenario
    setting = spec["scenario"]["setting"]
    if setting:
        sections.append(f"## Scenario\n{setting.strip()}")

    # Voice & format
    voice_parts: list[str] = []
    voice = spec["scenario"]["voice"]
    if voice:
        voice_parts.append(voice.strip())
    fmt = spec["episodes"]["format"]
    if fmt:
        voice_parts.append(f"Format: {fmt}")
    voice_parts.append(
        f"MINIMUM {target_words} words per episode. "
        f"Each episode MUST be at least {target_words} words. "
        f"Include detailed metrics, extended notes, multiple subsections, "
        f"and contextual commentary to reach this length."
    )
    sections.append("## Voice & Format\n" + "\n".join(voice_parts))

    # Prior context
    if prior_summaries:
        context = "\n\n".join(
            f"Phase {i + 1}: {s}" for i, s in enumerate(prior_summaries)
        )
        sections.append(f"## Prior Context\n{context}")

    # Current phase
    sections.append(
        f"## Current Phase: {phase['id']} (episodes {start}-{end})\n"
        f"{phase['description']}\n"
        f"Signal density: {phase['signal_density']}"
    )

    # Signal requirements from key facts
    kf_placements = get_key_facts_for_phase(spec, phase)
    if kf_placements:
        lines = []
        for local_idx, kf in kf_placements:
            lines.append(
                f'- Episode {local_idx} (of this phase): '
                f'Must naturally contain "{kf["fact"]}"'
            )
        sections.append("## Signal Requirements\n" + "\n".join(lines))

    # Noise guidance
    noise = spec.get("noise", {})
    if noise.get("description") or noise.get("examples"):
        noise_parts: list[str] = []
        if noise.get("description"):
            noise_parts.append(noise["description"].strip())
        if noise.get("examples"):
            noise_parts.append("Examples of noise content:")
            for ex in noise["examples"]:
                noise_parts.append(f"  - {ex}")
        sections.append("## Noise\n" + "\n".join(noise_parts))

    # Output format
    sections.append(
        f"## Output\n"
        f"Generate exactly {episode_count} episodes for this phase.\n"
        f"Return JSON: "
        f'{{"episodes": [{{"index": 1, "text": "...", "meta": {{}}}}, ...], '
        f'"phase_summary": "2-3 sentence summary of what happened in this phase"}}'
    )

    return "\n\n".join(sections)


def build_distractor_prompt(
    spec: dict,
    theme: dict,
    count: int,
    prior_summaries: list[str],
) -> str:
    """Build the user prompt for generating distractor episodes."""
    dc = spec["distractors"]
    target_words = dc["target_words"] if dc["target_words"] > 0 else spec["episodes"]["target_words"]

    sections: list[str] = []

    # Role framing — explain what distractors are and why neutrality matters
    sections.append(
        "## Purpose\n"
        "You are generating DISTRACTOR episodes for a benchmark evaluation.\n\n"
        "The benchmark tests whether an AI can identify specific patterns (the "
        "\"ground truth\") across a longitudinal dataset. Distractor episodes must "
        "be **entirely neutral** with respect to the ground truth — they must not "
        "provide evidence FOR or AGAINST the hypothesis being tested.\n\n"
        "A good distractor:\n"
        "- Touches on a completely different aspect of the broader topic\n"
        "- Involves different concerns, different actors, different subject matter\n"
        "- Maintains a coherent tone and format consistent with the rest of the dataset\n"
        "- Contains NO information that would materially sway a reader toward or "
        "away from the ground truth answer\n\n"
        "Think of distractors as realistic background noise — plausible, "
        "well-written content that happens to be about something else entirely."
    )

    # Scenario — theme's own scenario
    sections.append(f"## Scenario\n{theme['scenario'].strip()}")

    # Voice & format — same as signal for style matching
    voice_parts: list[str] = []
    voice = spec["scenario"]["voice"]
    if voice:
        voice_parts.append(voice.strip())
    fmt = spec["episodes"]["format"]
    if fmt:
        voice_parts.append(f"Format: {fmt}")
    voice_parts.append(
        f"MINIMUM {target_words} words per episode. "
        f"Each episode MUST be at least {target_words} words. "
        f"Include detailed metrics, extended notes, multiple subsections, "
        f"and contextual commentary to reach this length."
    )
    sections.append("## Voice & Format\n" + "\n".join(voice_parts))

    # Prior distractor context
    if prior_summaries:
        context = "\n\n".join(
            f"Batch {i + 1}: {s}" for i, s in enumerate(prior_summaries)
        )
        sections.append(f"## Prior Batches\n{context}")

    # Build the full exclusion list: theme excluded_terms + ALL key facts + question answers
    all_excluded: list[str] = list(theme.get("excluded_terms", []))

    # Add key fact texts
    key_facts = spec.get("key_facts", [])
    key_fact_texts = [kf["fact"] for kf in key_facts]
    all_excluded.extend(key_fact_texts)

    # Add question ground truth answers and key fact references
    for q in spec.get("questions", []):
        gt = q.get("ground_truth", {})
        if gt.get("canonical_answer"):
            all_excluded.append(gt["canonical_answer"])

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique_excluded: list[str] = []
    for term in all_excluded:
        lower = term.lower().strip()
        if lower and lower not in seen:
            seen.add(lower)
            unique_excluded.append(term)

    # Neutrality constraint — domain-agnostic
    constraint_lines = [
        "## NEUTRALITY CONSTRAINT",
        "",
        "Your distractor episodes must have NO material impact on the ground truth "
        "hypothesis. They must not sway a reader toward or away from the correct "
        "answer. This means:",
        "",
        "1. Do not include information that SUPPORTS the ground truth",
        "2. Do not include information that CONTRADICTS the ground truth",
        "3. Stay on a completely orthogonal axis — different subject matter, "
        "different concerns, different aspects of the broader topic",
        "",
    ]

    # Show the ground truth so the LLM knows what to be neutral about
    if key_fact_texts:
        constraint_lines.extend([
            "### Ground truth facts (your episodes must be neutral on ALL of these):",
            "",
        ])
        for kf in key_facts:
            constraint_lines.append(f'- "{kf["fact"]}"')
        constraint_lines.append("")

    # Show question answers if available
    gt_answers = []
    for q in spec.get("questions", []):
        gt = q.get("ground_truth", {})
        if gt.get("canonical_answer"):
            gt_answers.append(gt["canonical_answer"])
    if gt_answers:
        constraint_lines.extend([
            "### Ground truth answers (your episodes must not help or hinder these):",
            "",
        ])
        for ans in gt_answers:
            constraint_lines.append(f'- "{ans}"')
        constraint_lines.append("")

    # Theme-specific excluded terms (from spec)
    theme_excluded = theme.get("excluded_terms", [])
    if theme_excluded:
        terms = ", ".join(f'"{t}"' for t in theme_excluded)
        constraint_lines.extend([
            f"### Additionally excluded terms: {terms}",
            "",
        ])

    constraint_lines.extend([
        "### How to stay neutral",
        "- Focus your episodes entirely on the theme's own subject matter",
        "- The people, systems, processes, or events in your episodes should be "
        "distinct from those involved in the ground truth",
        "- If the ground truth is about a trend or pattern, your episodes should "
        "concern a completely unrelated trend or pattern",
        "- Avoid using the same key terms, even in a different context — "
        "a reader scanning for those terms should not find them in your output",
    ])

    sections.append("\n".join(constraint_lines))

    # Output format
    sections.append(
        f"## Output\n"
        f"Generate exactly {count} episodes for theme '{theme['id']}'.\n"
        f"Return JSON: "
        f'{{"episodes": [{{"index": 1, "text": "...", "meta": {{}}}}, ...], '
        f'"batch_summary": "2-3 sentence summary of this batch"}}'
    )

    return "\n\n".join(sections)


def build_expand_prompt(text: str, target_words: int) -> str:
    """Build prompt for expanding a short episode."""
    return (
        f"Expand the following log entry to at least {target_words} words "
        f"while preserving all facts, format, and voice. "
        f"Add detailed metrics, extended notes, additional subsections, "
        f"and contextual commentary. Return ONLY the expanded text, no JSON.\n\n"
        f"{text}"
    )

EXPAND_SYSTEM = "You expand text while preserving its style and content. Return only the expanded text."


def build_contamination_prompt(episode_text: str, question_prompt: str) -> str:
    """Build prompt for contamination check."""
    return (
        f"You are given a single record from a longitudinal dataset. "
        f"Answer the question below using ONLY the information in this record. "
        f"If you cannot answer fully from this single record, say so.\n\n"
        f"## Record\n{episode_text}\n\n"
        f"## Question\n{question_prompt}\n\n"
        f"## Answer"
    )


def build_naive_baseline_prompt(
    episode_texts: list[str],
    question_prompt: str,
) -> str:
    """Build prompt for naive full-context baseline."""
    records = "\n\n---\n\n".join(
        f"Record {i + 1}:\n{text}" for i, text in enumerate(episode_texts)
    )
    return (
        f"You are given all records from a longitudinal dataset. "
        f"Answer the question below using the information provided.\n\n"
        f"## Records\n{records}\n\n"
        f"## Question\n{question_prompt}\n\n"
        f"## Answer"
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _episode_range(phase: dict) -> tuple[int, int]:
    parts = phase["episodes"].split("-")
    return int(parts[0]), int(parts[1])

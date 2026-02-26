"""Prompt builders for narrative scope generation via Claude Code agents.

Usage:
    From Claude Code, read a spec and call these functions to get prompts
    for the planner agent (sees full spec) and renderer agents (blind).

    The planner produces per-episode FACT SHEETS (entities, actions, quotes,
    observations — no causal interpretation).

    Each renderer receives ONE fact sheet + format guide and produces a
    ~5,000-word narrative episode.

    This module is intentionally separate from the synix pipeline's
    prompt_utils.py — narrative scopes are generated via Claude Code
    agents, not the synix LLM pipeline.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "lens" / "datagen" / "synix"))
import spec_utils


# ---------------------------------------------------------------------------
# Forbidden words for narrative mode
# ---------------------------------------------------------------------------

NARRATIVE_FORBIDDEN_WORDS = [
    "this suggests", "this indicates", "evidence of", "pattern of",
    "linked to", "caused by", "confirms", "alarming", "suspicious",
    "raises concerns", "connection between", "correlated with",
    "consistent with", "points to", "reveals", "exposes",
    "the pattern suggests", "which means", "it appears that",
    "clearly shows", "unmistakable", "telltale sign",
]


# ---------------------------------------------------------------------------
# Signal planner prompt (sees FULL spec)
# ---------------------------------------------------------------------------


def build_narrative_planner_prompt(spec: dict) -> str:
    """Build the planner prompt for narrative signal episodes.

    The planner sees the full spec (arc, key facts, questions) and produces
    per-episode FACT SHEETS with: entities, actions, observations, quotes,
    document metadata. NO causal interpretation.
    """
    ep_count = spec["episodes"]["count"]
    sections: list[str] = []

    # Role
    sections.append(
        "# Role\n"
        "You are a data architect designing fact sheets for a benchmark dataset.\n"
        "You will produce one FACT SHEET per episode. Each fact sheet lists:\n"
        "- **Entities**: People, systems, organizations, documents that appear\n"
        "- **Actions**: What happens (who does what, what is observed)\n"
        "- **Quotes/Excerpts**: Specific text that should appear verbatim\n"
        "- **Document metadata**: Document type, date, author, recipients\n"
        "- **Observable details**: Concrete, factual details (numbers, names, timestamps)\n\n"
        "CRITICAL: Fact sheets must contain NO causal interpretation. "
        "Do not explain WHY things happen or what they MEAN. "
        "Only describe WHAT is observable."
    )

    # Scenario
    sections.append(f"# Scenario\n{spec['scenario']['setting'].strip()}")

    # Arc
    arc_lines: list[str] = []
    for phase in spec["arc"]:
        start, end = spec_utils.episode_range(phase)
        arc_lines.append(
            f"- **{phase['id']}** (episodes {start}-{end}): "
            f"{phase['description'].strip()} [signal_density: {phase['signal_density']}]"
        )
    sections.append("# Arc Phases\n" + "\n".join(arc_lines))

    # Key facts with placement
    kf_lines: list[str] = []
    for kf in spec["key_facts"]:
        placements = [kf["first_appears"]] + kf["reinforced_in"]
        kf_lines.append(
            f'- **{kf["id"]}**: "{kf["fact"]}" '
            f'(appears: {", ".join(p for p in placements if p)})'
        )
    sections.append("# Key Facts to Encode\n" + "\n".join(kf_lines))

    # Episode format
    sections.append(f"# Episode Format\n{spec['episodes']['format'].strip()}")

    # Voice
    sections.append(f"# Voice\n{spec['scenario']['voice'].strip()}")

    # Noise guidance
    noise = spec.get("noise", {})
    if noise.get("description"):
        noise_parts = [noise["description"].strip()]
        if noise.get("examples"):
            noise_parts.append("Examples of routine/noise content:")
            for ex in noise["examples"]:
                noise_parts.append(f"  - {ex}")
        sections.append("# Noise / Routine Content\n" + "\n".join(noise_parts))

    # Critical rules
    sections.append(
        "# CRITICAL RULES\n\n"
        "1. **No causal interpretation**: Fact sheets describe WHAT happens, "
        "not WHY or what it MEANS. The renderer must not be able to infer "
        "the storyline from a single fact sheet.\n\n"
        "2. **Encode signal as entity appearances and actions**: Instead of "
        "'the student is cheating,' write 'Student mchen_2026 asks: can you "
        "show me what a strong answer looks like for this prompt?'\n\n"
        "3. **Each fact sheet is self-contained**: Do not reference other "
        "episodes. Each should read as an independent set of observations.\n\n"
        "4. **Baseline episodes need RICH detail**: Don't skimp on baseline. "
        "Include full entity lists, normal actions, routine quotes. These "
        "establish what 'normal' looks like.\n\n"
        "5. **Include enough material for ~5,000 words**: Each fact sheet "
        "should have 15-25 distinct items (entities, actions, quotes, details) "
        "to give the renderer enough raw material.\n\n"
        "6. **Forbidden language in fact sheets**:\n"
        + "\n".join(f"   - \"{fw}\"" for fw in NARRATIVE_FORBIDDEN_WORDS[:10])
    )

    # Timeline
    tl = spec["episodes"]["timeline"]
    sections.append(
        f"# Timeline\n"
        f"Start: {tl['start']}, Interval: {tl['interval']}, "
        f"Total signal episodes: {ep_count}"
    )

    # Output format
    sections.append(
        f"# Output Format\n"
        f"Produce exactly {ep_count} fact sheets.\n"
        f"Return JSON:\n"
        f"```json\n"
        f'{{"episodes": [\n'
        f'  {{\n'
        f'    "index": 1,\n'
        f'    "date": "YYYY-MM-DD",\n'
        f'    "phase": "baseline",\n'
        f'    "documents": [\n'
        f'      {{\n'
        f'        "type": "chat_session | board_minutes | slack_thread | ...",\n'
        f'        "metadata": {{"author": "...", "recipients": "...", "subject": "...", ...}},\n'
        f'        "entities": ["entity1", "entity2", ...],\n'
        f'        "actions": ["action1", "action2", ...],\n'
        f'        "quotes": ["verbatim quote 1", "verbatim quote 2", ...],\n'
        f'        "details": ["observable detail 1", "observable detail 2", ...]\n'
        f'      }}\n'
        f'    ]\n'
        f'  }},\n'
        f'  ...\n'
        f"]}}\n"
        f"```"
    )

    return "\n\n".join(sections)


# ---------------------------------------------------------------------------
# Signal renderer prompt (BLIND — one episode at a time)
# ---------------------------------------------------------------------------


def build_narrative_renderer_prompt(spec: dict, fact_sheet: dict) -> str:
    """Build the renderer prompt for a single narrative episode.

    The renderer sees ONLY:
    - The episode format description
    - The voice guide
    - ONE fact sheet (entities, actions, quotes, details)

    It does NOT see: the arc, key facts, questions, or other episodes.
    """
    target_words = spec["episodes"]["target_words"]
    sections: list[str] = []

    # Role
    sections.append(
        "# Role\n"
        "You are a writer producing a single episode for a benchmark dataset. "
        "You will receive a FACT SHEET containing entities, actions, quotes, "
        "and observable details. Your job is to write a realistic, "
        f"~{target_words}-word document that naturally incorporates ALL "
        "items from the fact sheet."
    )

    # Format
    sections.append(f"# Document Format\n{spec['episodes']['format'].strip()}")

    # Voice
    sections.append(f"# Voice\n{spec['scenario']['voice'].strip()}")

    # Rules
    sections.append(
        "# STRICT RULES\n\n"
        "1. Include ALL entities, actions, quotes, and details from the fact sheet.\n"
        "2. Do NOT add causal interpretation or analysis.\n"
        "3. Do NOT explain what observations mean or why they matter.\n"
        "4. Do NOT reference other episodes or suggest a timeline.\n"
        "5. Write the document as a standalone piece — it should read as a "
        "routine document of its type.\n"
        "6. Use natural language appropriate to the document format.\n"
        "7. Pad with realistic but mundane detail to reach the word count — "
        "routine items, normal business, background context.\n"
        f"8. MINIMUM {target_words} words. Aim for {target_words}-{target_words + 1000}.\n"
        "9. Do NOT use any of these phrases: "
        + ", ".join(f'"{fw}"' for fw in NARRATIVE_FORBIDDEN_WORDS[:8])
        + "\n10. Output ONLY the episode text. No JSON, no markdown fences around the output."
    )

    # The fact sheet
    sections.append(
        f"# Fact Sheet\n"
        f"Date: {fact_sheet.get('date', 'unknown')}\n"
        f"```json\n{json.dumps(fact_sheet, indent=2)}\n```"
    )

    return "\n\n".join(sections)


# ---------------------------------------------------------------------------
# Distractor planner prompt
# ---------------------------------------------------------------------------


def build_narrative_distractor_planner_prompt(
    spec: dict, theme: dict, count: int
) -> str:
    """Build the planner prompt for distractor episodes of a single theme."""
    sections: list[str] = []

    sections.append(
        "# Role\n"
        "You are producing fact sheets for DISTRACTOR episodes. These must "
        "be completely unrelated to the main signal but use the same document "
        "format. They are background noise in a benchmark dataset."
    )

    sections.append(f"# Distractor Theme: {theme['id']}\n{theme['scenario'].strip()}")

    # Key facts to AVOID
    kf_lines = [f'- "{kf["fact"]}"' for kf in spec["key_facts"]]
    sections.append(
        "# Key Facts to AVOID\n"
        "Your distractor episodes must contain NOTHING related to these:\n"
        + "\n".join(kf_lines)
    )

    # Excluded terms
    excluded = theme.get("excluded_terms", [])
    if excluded:
        terms = ", ".join(f'"{t}"' for t in excluded)
        sections.append(f"# Excluded Terms\nDo NOT use: {terms}")

    # Format — same as signal for style matching
    sections.append(f"# Episode Format (match this exactly)\n{spec['episodes']['format'].strip()}")

    # Output
    sections.append(
        f"# Output\n"
        f"Produce exactly {count} fact sheets for theme '{theme['id']}'.\n"
        f"Same JSON structure as signal fact sheets but with theme-appropriate "
        f"content that is completely orthogonal to the key facts above."
    )

    return "\n\n".join(sections)


# ---------------------------------------------------------------------------
# CLI helper — dump prompts to files for review
# ---------------------------------------------------------------------------


def dump_prompts(scope_dir: str) -> None:
    """Read a spec and dump all prompts to the generated/ directory."""
    scope_path = Path(scope_dir)
    spec = spec_utils.load_spec(scope_path / "spec.yaml")
    spec_utils.validate_spec_or_raise(spec)

    out_dir = scope_path / "generated" / "prompts"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Signal planner
    planner = build_narrative_planner_prompt(spec)
    (out_dir / "signal_planner.md").write_text(planner)
    print(f"  Wrote signal planner prompt ({len(planner)} chars)")

    # Distractor planners
    dc = spec.get("distractors")
    if dc and dc["count"] > 0:
        themes = dc["themes"]
        per_theme = [dc["count"] // len(themes)] * len(themes)
        for i in range(dc["count"] % len(themes)):
            per_theme[i] += 1
        for idx, theme in enumerate(themes):
            if per_theme[idx] > 0:
                prompt = build_narrative_distractor_planner_prompt(
                    spec, theme, per_theme[idx]
                )
                (out_dir / f"distractor_planner_{theme['id']}.md").write_text(prompt)
                print(f"  Wrote distractor planner for {theme['id']} ({len(prompt)} chars)")

    # Renderer template (with placeholder fact sheet)
    placeholder = {
        "index": 1,
        "date": "2025-01-01",
        "phase": "baseline",
        "documents": [{"type": "placeholder", "entities": [], "actions": [], "quotes": [], "details": []}],
    }
    renderer = build_narrative_renderer_prompt(spec, placeholder)
    (out_dir / "renderer_template.md").write_text(renderer)
    print(f"  Wrote renderer template ({len(renderer)} chars)")

    print(f"\nAll prompts written to {out_dir}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python narrative_prompts.py <scope_dir> [<scope_dir> ...]")
        print("Example: python narrative_prompts.py datasets/scopes/07_tutoring_jailbreak")
        sys.exit(1)

    for scope_dir in sys.argv[1:]:
        print(f"\n=== {scope_dir} ===")
        dump_prompts(scope_dir)

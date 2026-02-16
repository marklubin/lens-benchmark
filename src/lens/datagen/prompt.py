from __future__ import annotations

from lens.datagen.spec import (
    DistractorTheme,
    KeyFact,
    PhaseArc,
    ScopeSpec,
    get_key_facts_for_phase,
)


SYSTEM_PROMPT = (
    "You generate realistic synthetic data for a benchmark evaluation. "
    "Output valid JSON only. No markdown fences, no commentary."
)


def build_phase_prompt(
    spec: ScopeSpec,
    phase: PhaseArc,
    prior_summaries: list[str],
) -> str:
    """Build the user prompt for generating episodes in a single phase.

    Args:
        spec: The scope specification.
        phase: The current phase to generate.
        prior_summaries: Summaries of previously generated phases (for coherence).

    Returns:
        The user prompt string.
    """
    start, end = phase.episode_range()
    episode_count = end - start + 1

    sections: list[str] = []

    # Scenario setting
    if spec.scenario.setting:
        sections.append(f"## Scenario\n{spec.scenario.setting.strip()}")

    # Voice & format
    voice_parts: list[str] = []
    if spec.scenario.voice:
        voice_parts.append(spec.scenario.voice.strip())
    if spec.episodes.format:
        voice_parts.append(f"Format: {spec.episodes.format}")
    voice_parts.append(
        f"MINIMUM {spec.episodes.target_words} words per episode. "
        f"Each episode MUST be at least {spec.episodes.target_words} words. "
        f"Include detailed metrics, extended notes, multiple subsections, "
        f"and contextual commentary to reach this length."
    )
    sections.append(f"## Voice & Format\n" + "\n".join(voice_parts))

    # Prior context
    if prior_summaries:
        context = "\n\n".join(
            f"Phase {i + 1}: {s}" for i, s in enumerate(prior_summaries)
        )
        sections.append(f"## Prior Context\n{context}")

    # Current phase
    sections.append(
        f"## Current Phase: {phase.id} (episodes {start}-{end})\n"
        f"{phase.description}\n"
        f"Signal density: {phase.signal_density}"
    )

    # Signal requirements from key facts
    kf_placements = get_key_facts_for_phase(spec, phase)
    if kf_placements:
        lines = []
        for local_idx, kf in kf_placements:
            lines.append(
                f"- Episode {local_idx} (of this phase): "
                f"Must naturally contain \"{kf.fact}\""
            )
        sections.append(f"## Signal Requirements\n" + "\n".join(lines))

    # Noise guidance
    if spec.noise.description or spec.noise.examples:
        noise_parts = []
        if spec.noise.description:
            noise_parts.append(spec.noise.description.strip())
        if spec.noise.examples:
            noise_parts.append("Examples of noise content:")
            for ex in spec.noise.examples:
                noise_parts.append(f"  - {ex}")
        sections.append(f"## Noise\n" + "\n".join(noise_parts))

    # Output spec
    sections.append(
        f"## Output\n"
        f"Generate exactly {episode_count} episodes for this phase.\n"
        f"Return JSON: "
        f'{{"episodes": [{{"index": 1, "text": "...", "meta": {{}}}}, ...], '
        f'"phase_summary": "2-3 sentence summary of what happened in this phase"}}'
    )

    return "\n\n".join(sections)


def build_distractor_prompt(
    spec: ScopeSpec,
    theme: DistractorTheme,
    count: int,
    prior_summaries: list[str],
) -> str:
    """Build the user prompt for generating distractor episodes.

    Distractors are format-matched but topically orthogonal episodes
    that share the same voice/style but describe unrelated systems.

    Args:
        spec: The scope specification.
        theme: The distractor theme to use.
        count: Number of distractor episodes to generate for this theme.
        prior_summaries: Summaries of previously generated distractor batches.

    Returns:
        The user prompt string.
    """
    target_words = spec.distractors.target_words if (spec.distractors and spec.distractors.target_words) else spec.episodes.target_words

    sections: list[str] = []

    # Scenario — use the theme's scenario, NOT the main one
    sections.append(f"## Scenario\n{theme.scenario.strip()}")

    # Voice & format — same as signal episodes for style matching
    voice_parts: list[str] = []
    if spec.scenario.voice:
        voice_parts.append(spec.scenario.voice.strip())
    if spec.episodes.format:
        voice_parts.append(f"Format: {spec.episodes.format}")
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

    # Negative constraint — excluded terms
    if theme.excluded_terms:
        terms = ", ".join(f'"{t}"' for t in theme.excluded_terms)
        sections.append(
            f"## CRITICAL CONSTRAINT\n"
            f"These episodes must NOT mention or reference any of the following "
            f"terms or concepts: {terms}.\n"
            f"These terms belong to a different system and must not appear."
        )

    # Output spec
    sections.append(
        f"## Output\n"
        f"Generate exactly {count} episodes for theme '{theme.id}'.\n"
        f"Return JSON: "
        f'{{"episodes": [{{"index": 1, "text": "...", "meta": {{}}}}, ...], '
        f'"batch_summary": "2-3 sentence summary of this batch"}}'
    )

    return "\n\n".join(sections)


def build_contamination_prompt(episode_text: str, question_prompt: str) -> str:
    """Build prompt for contamination check — answer a question from a single episode."""
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
    """Build prompt for naive full-context baseline — all episodes in one prompt."""
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

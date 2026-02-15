from __future__ import annotations

from lens.datagen.spec import (
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
    voice_parts.append(f"Each episode: approximately {spec.episodes.target_words} words.")
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

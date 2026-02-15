from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from lens.core.errors import DatasetError, atomic_write
from lens.datagen.prompt import SYSTEM_PROMPT, build_phase_prompt
from lens.datagen.spec import (
    ScopeSpec,
    compute_spec_hash,
    get_key_facts_for_phase,
    load_spec,
    make_episode_id,
    make_episode_timestamp,
    validate_spec_or_raise,
)


MAX_RETRIES = 3


# ---------------------------------------------------------------------------
# Phase result dataclass (plain dict to keep it simple)
# ---------------------------------------------------------------------------


def _empty_phase_result(phase_id: str) -> dict:
    return {
        "status": "pending",
        "attempts": 0,
        "warnings": [],
        "tokens": {"prompt": 0, "completion": 0},
    }


# ---------------------------------------------------------------------------
# Core generation pipeline
# ---------------------------------------------------------------------------


def generate_scope(
    spec_path: str | Path,
    provider: str = "openai",
    model: str = "gpt-4o",
    api_key: str | None = None,
    api_base: str | None = None,
    verbose: bool = False,
    log_fn: object | None = None,
) -> Path:
    """Generate episodes for a scope from its spec.

    Args:
        spec_path: Path to the spec.yaml file.
        provider: LLM provider (only "openai" supported for generation).
        model: Model name.
        api_key: API key (falls back to LENS_LLM_API_KEY / OPENAI_API_KEY env vars).
        api_base: Optional API base URL.
        verbose: Print progress to console.
        log_fn: Optional callable(str) for logging.

    Returns:
        Path to the generated/ directory.
    """
    spec_path = Path(spec_path)
    spec = load_spec(spec_path)
    validate_spec_or_raise(spec)

    # Resolve output directory
    scope_dir = spec_path.parent
    gen_dir = scope_dir / "generated"
    phases_dir = gen_dir / "phases"
    phases_dir.mkdir(parents=True, exist_ok=True)

    def _log(msg: str) -> None:
        if log_fn:
            log_fn(msg)  # type: ignore[operator]
        elif verbose:
            print(msg)

    # Create OpenAI client
    client = _create_client(provider, api_key, api_base)

    # Track generation metadata
    spec_hash = compute_spec_hash(spec_path)
    manifest: dict = {
        "scope_id": spec.scope_id,
        "spec_version": "1.0.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generator": {
            "provider": provider,
            "model": model,
            "temperature": spec.generation.temperature,
            "seed": spec.generation.seed,
        },
        "spec_hash": spec_hash,
        "phases": {},
        "total_tokens": {"prompt": 0, "completion": 0},
        "key_fact_coverage": {},
        "validation": {
            "key_fact_hit_rate": 0.0,
            "contamination_check": "pending",
            "naive_baseline": "pending",
        },
    }

    # Generate phase by phase
    # Use a dict keyed by episode_id so overlapping phases (e.g. early_signal
    # 9-15 and red_herring 14-16) deduplicate — later phase wins.
    episode_map: dict[str, dict] = {}
    prior_summaries: list[str] = []

    for phase in spec.arc:
        _log(f"Generating phase: {phase.id}")
        phase_result = _empty_phase_result(phase.id)
        manifest["phases"][phase.id] = phase_result

        start, end = phase.episode_range()
        phase_episodes = None

        for attempt in range(1, MAX_RETRIES + 1):
            phase_result["attempts"] = attempt
            _log(f"  Attempt {attempt}/{MAX_RETRIES}")

            try:
                prompt = build_phase_prompt(spec, phase, prior_summaries)
                response = _call_llm(
                    client,
                    model=model,
                    system=SYSTEM_PROMPT,
                    user=prompt,
                    temperature=spec.generation.temperature,
                    seed=spec.generation.seed,
                )

                # Track tokens
                phase_result["tokens"]["prompt"] += response["usage"]["prompt"]
                phase_result["tokens"]["completion"] += response["usage"]["completion"]

                # Parse response
                parsed = _parse_phase_response(response["content"])

                # Validate
                warnings = _validate_phase_output(parsed, phase, spec)
                phase_result["warnings"].extend(warnings)

                phase_episodes = parsed["episodes"]
                phase_summary = parsed.get("phase_summary", "")
                phase_result["status"] = "success"

                # Save raw phase output
                with atomic_write(phases_dir / f"phase_{phase.id}.json") as tmp:
                    tmp.write_text(json.dumps(parsed, indent=2))

                prior_summaries.append(f"[{phase.id}] {phase_summary}")
                break

            except (DatasetError, json.JSONDecodeError, KeyError) as e:
                _log(f"  Phase {phase.id} attempt {attempt} failed: {e}")
                if attempt == MAX_RETRIES:
                    phase_result["status"] = "failed"
                    phase_result["warnings"].append(f"Failed after {MAX_RETRIES} attempts: {e}")

        # Build structured episodes with IDs and timestamps
        if phase_episodes:
            for i, ep in enumerate(phase_episodes):
                global_idx = start + i
                eid = make_episode_id(spec.scope_id, global_idx)
                episode = {
                    "episode_id": eid,
                    "scope_id": spec.scope_id,
                    "timestamp": make_episode_timestamp(spec, global_idx),
                    "text": ep.get("text", ""),
                    "meta": ep.get("meta", {}),
                }
                episode_map[eid] = episode

    # Flatten episode map to sorted list (by episode_id for stable ordering)
    all_episodes = [episode_map[eid] for eid in sorted(episode_map)]

    # Build questions with resolved refs
    all_questions = _resolve_questions(spec)

    # Compute key fact coverage
    _compute_key_fact_coverage(spec, all_episodes, manifest)

    # Tally total tokens
    for phase_id, pr in manifest["phases"].items():
        manifest["total_tokens"]["prompt"] += pr["tokens"]["prompt"]
        manifest["total_tokens"]["completion"] += pr["tokens"]["completion"]

    # Write outputs
    with atomic_write(gen_dir / "episodes.json") as tmp:
        tmp.write_text(json.dumps(all_episodes, indent=2))

    with atomic_write(gen_dir / "questions.json") as tmp:
        tmp.write_text(json.dumps(all_questions, indent=2))

    with atomic_write(gen_dir / "manifest.json") as tmp:
        tmp.write_text(json.dumps(manifest, indent=2))

    _log(f"Generation complete: {len(all_episodes)} episodes, {len(all_questions)} questions")
    return gen_dir


# ---------------------------------------------------------------------------
# LLM client helpers
# ---------------------------------------------------------------------------


def _create_client(provider: str, api_key: str | None, api_base: str | None) -> object:
    """Create an OpenAI client for generation."""
    if provider != "openai":
        raise DatasetError(f"Generation only supports 'openai' provider, got {provider!r}")

    try:
        from openai import OpenAI
    except ImportError:
        raise DatasetError(
            "OpenAI package required for generation. Install with: pip install 'lens-bench[datagen]'"
        )

    key = api_key or os.environ.get("LENS_LLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise DatasetError(
            "API key required. Set LENS_LLM_API_KEY, OPENAI_API_KEY, or pass --api-key."
        )

    kwargs: dict = {"api_key": key}
    base = api_base or os.environ.get("LENS_LLM_API_BASE")
    if base:
        kwargs["base_url"] = base

    return OpenAI(**kwargs)


def _call_llm(
    client: object,
    model: str,
    system: str,
    user: str,
    temperature: float,
    seed: int,
) -> dict:
    """Call the LLM and return content + usage."""
    response = client.chat.completions.create(  # type: ignore[union-attr]
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=temperature,
        seed=seed,
        response_format={"type": "json_object"},
    )
    choice = response.choices[0]
    usage = response.usage
    return {
        "content": choice.message.content or "",
        "usage": {
            "prompt": usage.prompt_tokens if usage else 0,
            "completion": usage.completion_tokens if usage else 0,
        },
    }


# ---------------------------------------------------------------------------
# Response parsing & validation
# ---------------------------------------------------------------------------


def _parse_phase_response(content: str) -> dict:
    """Parse JSON response from LLM."""
    try:
        data = json.loads(content)
    except json.JSONDecodeError as e:
        raise DatasetError(f"LLM returned invalid JSON: {e}")

    if "episodes" not in data:
        raise DatasetError("LLM response missing 'episodes' key")
    if not isinstance(data["episodes"], list):
        raise DatasetError("LLM response 'episodes' must be a list")

    return data


def _validate_phase_output(
    parsed: dict, phase: PhaseArc, spec: ScopeSpec
) -> list[str]:
    """Validate phase output. Returns warnings (non-fatal issues)."""
    warnings: list[str] = []
    start, end = phase.episode_range()
    expected_count = end - start + 1
    actual_count = len(parsed["episodes"])

    if actual_count != expected_count:
        warnings.append(
            f"Phase {phase.id}: expected {expected_count} episodes, got {actual_count}"
        )

    for i, ep in enumerate(parsed["episodes"]):
        text = ep.get("text", "")
        if not text.strip():
            warnings.append(f"Phase {phase.id}, episode {i + 1}: empty text")
            continue

        word_count = len(text.split())
        target = spec.episodes.target_words
        if word_count < target * 0.3:
            warnings.append(
                f"Phase {phase.id}, episode {i + 1}: "
                f"only {word_count} words (target: {target})"
            )

    # Check key fact presence (fuzzy: case-insensitive substring)
    kf_placements = get_key_facts_for_phase(spec, phase)
    for local_idx, kf in kf_placements:
        ep_idx = local_idx - 1  # 0-based
        if ep_idx < len(parsed["episodes"]):
            text = parsed["episodes"][ep_idx].get("text", "").lower()
            fact_lower = kf.fact.lower()
            # Fuzzy: check if key words from the fact appear in the text
            fact_words = set(fact_lower.split())
            text_words = set(text.split())
            overlap = fact_words & text_words
            if len(overlap) < len(fact_words) * 0.5:
                warnings.append(
                    f"Phase {phase.id}, episode {local_idx}: "
                    f"key fact '{kf.id}' may not be present (fuzzy match)"
                )

    return warnings


# ---------------------------------------------------------------------------
# Question resolution
# ---------------------------------------------------------------------------


def _resolve_questions(spec: ScopeSpec) -> list[dict]:
    """Resolve spec questions into the dataset question format."""
    from lens.datagen.spec import resolve_phase_ref

    kf_map = {kf.id: kf.fact for kf in spec.key_facts}
    questions = []

    for q in spec.questions:
        # Resolve evidence refs to episode IDs
        evidence_refs = []
        for ref in q.ground_truth.evidence:
            try:
                evidence_refs.append(resolve_phase_ref(ref, spec))
            except DatasetError:
                pass  # Skip unresolvable refs (already validated)

        # Map key fact IDs to their text
        key_fact_texts = [kf_map[fid] for fid in q.ground_truth.key_facts if fid in kf_map]

        questions.append({
            "question_id": q.id,
            "scope_id": spec.scope_id,
            "checkpoint_after": q.checkpoint_after,
            "question_type": q.type,
            "prompt": q.prompt,
            "ground_truth": {
                "canonical_answer": q.ground_truth.canonical_answer,
                "required_evidence_refs": evidence_refs,
                "key_facts": key_fact_texts,
            },
        })

    return questions


# ---------------------------------------------------------------------------
# Key fact coverage analysis
# ---------------------------------------------------------------------------


def _compute_key_fact_coverage(
    spec: ScopeSpec, episodes: list[dict], manifest: dict
) -> None:
    """Compute key fact coverage and store in manifest."""
    from lens.datagen.spec import resolve_phase_ref

    coverage: dict = {}
    total_targets = 0
    total_found = 0

    for kf in spec.key_facts:
        # Collect all target episode IDs for this fact
        target_ids = []
        all_refs = []
        if kf.first_appears:
            all_refs.append(kf.first_appears)
        all_refs.extend(kf.reinforced_in)

        for ref in all_refs:
            try:
                eid = resolve_phase_ref(ref, spec)
                target_ids.append(eid)
            except DatasetError:
                pass

        # Check which episodes actually contain the fact (fuzzy)
        found_in = []
        fact_words = set(kf.fact.lower().split())
        for eid in target_ids:
            for ep in episodes:
                if ep["episode_id"] == eid:
                    text_words = set(ep["text"].lower().split())
                    overlap = fact_words & text_words
                    if len(overlap) >= len(fact_words) * 0.5:
                        found_in.append(eid)
                    break

        coverage[kf.id] = {
            "target_episodes": target_ids,
            "found_in": found_in,
        }
        total_targets += len(target_ids)
        total_found += len(found_in)

    manifest["key_fact_coverage"] = coverage
    manifest["validation"]["key_fact_hit_rate"] = (
        total_found / total_targets if total_targets > 0 else 1.0
    )

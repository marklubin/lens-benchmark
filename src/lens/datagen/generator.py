from __future__ import annotations

import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from lens.core.errors import DatasetError, atomic_write
from lens.datagen.prompt import SYSTEM_PROMPT, build_distractor_prompt, build_phase_prompt
from lens.datagen.spec import (
    DistractorTheme,
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
    model: str = "gpt-5.2",
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

    # Kick off distractor generation concurrently (if configured)
    distractor_future = None
    distractor_executor = None
    if spec.distractors and spec.distractors.count > 0:
        _log("Starting distractor generation (concurrent)...")
        # Distractors use a separate client to avoid contention
        distractor_client = _create_client(provider, api_key, api_base)
        distractor_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="distractor")
        distractor_future = distractor_executor.submit(
            _generate_distractors, spec, distractor_client, model, _log,
        )

    # Generate signal phases sequentially (they chain on prior_summaries)
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
            # Word count enforcement: expand short episodes
            target = spec.episodes.target_words
            for i, ep in enumerate(phase_episodes):
                text = ep.get("text", "")
                word_count = len(text.split())
                if text.strip() and word_count < target * 0.7:
                    try:
                        ep["text"] = _expand_episode(client, model, text, target, spec)
                    except Exception:
                        pass  # Keep original if expansion fails

            for i, ep in enumerate(phase_episodes):
                global_idx = start + i
                eid = make_episode_id(spec.scope_id, global_idx)
                meta = ep.get("meta", {})
                meta["episode_type"] = "signal"
                episode = {
                    "episode_id": eid,
                    "scope_id": spec.scope_id,
                    "timestamp": make_episode_timestamp(spec, global_idx),
                    "text": ep.get("text", ""),
                    "meta": meta,
                }
                episode_map[eid] = episode

    # Collect distractor results
    distractor_episodes: list[dict] = []
    if distractor_future is not None:
        _log("Waiting for distractor generation to complete...")
        distractor_result = distractor_future.result()
        distractor_episodes = distractor_result["episodes"]
        manifest["distractors"] = distractor_result["manifest"]
        manifest["total_tokens"]["prompt"] += distractor_result["tokens"]["prompt"]
        manifest["total_tokens"]["completion"] += distractor_result["tokens"]["completion"]
        _log(
            f"  Distractors: {len(distractor_episodes)} generated, "
            f"{distractor_result['manifest']['rejected_count']} rejected"
        )
        if distractor_executor:
            distractor_executor.shutdown(wait=False)

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

    if distractor_episodes:
        with atomic_write(gen_dir / "distractors.json") as tmp:
            tmp.write_text(json.dumps(distractor_episodes, indent=2))

    with atomic_write(gen_dir / "questions.json") as tmp:
        tmp.write_text(json.dumps(all_questions, indent=2))

    with atomic_write(gen_dir / "manifest.json") as tmp:
        tmp.write_text(json.dumps(manifest, indent=2))

    _log(
        f"Generation complete: {len(all_episodes)} signal episodes"
        + (f", {len(distractor_episodes)} distractors" if distractor_episodes else "")
        + f", {len(all_questions)} questions"
    )
    return gen_dir


# ---------------------------------------------------------------------------
# Distractor generation
# ---------------------------------------------------------------------------


def _generate_distractors(
    spec: ScopeSpec,
    client: object,
    model: str,
    log_fn: object,
) -> dict:
    """Generate distractor episodes across all configured themes in parallel.

    Themes are independent and run concurrently via ThreadPoolExecutor.
    Returns a result dict with episodes, manifest stats, and token counts.
    """
    dc = spec.distractors
    if not dc or dc.count <= 0:
        return {"episodes": [], "manifest": {}, "tokens": {"prompt": 0, "completion": 0}}

    all_key_fact_strings = [kf.fact for kf in spec.key_facts]

    # Divide count across themes (round-robin)
    themes = dc.themes
    per_theme = [dc.count // len(themes)] * len(themes)
    for i in range(dc.count % len(themes)):
        per_theme[i] += 1

    # Generate each theme in parallel
    theme_results: list[dict] = []
    max_workers = min(len(themes), 8)
    log_fn(f"  Generating {len(themes)} themes with {max_workers} workers")  # type: ignore[operator]

    with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="theme") as pool:
        futures = {}
        for theme_idx, theme in enumerate(themes):
            count = per_theme[theme_idx]
            if count <= 0:
                continue
            fut = pool.submit(
                _generate_theme_batch,
                spec, client, model, theme, theme_idx, count,
                all_key_fact_strings, log_fn,
            )
            futures[fut] = theme.id

        for fut in as_completed(futures):
            theme_id = futures[fut]
            try:
                theme_results.append(fut.result())
            except Exception as e:
                log_fn(f"    Theme {theme_id} failed: {e}")  # type: ignore[operator]

    # Merge results from all themes
    all_episodes: list[dict] = []
    total_tokens = {"prompt": 0, "completion": 0}
    rejected_count = 0
    similarity_scores: list[float] = []

    for tr in theme_results:
        all_episodes.extend(tr["episodes"])
        total_tokens["prompt"] += tr["tokens"]["prompt"]
        total_tokens["completion"] += tr["tokens"]["completion"]
        rejected_count += tr["rejected_count"]
        similarity_scores.extend(tr["similarity_scores"])

    # Assign IDs and interleave timestamps
    _assign_distractor_ids_and_timestamps(spec, all_episodes)

    manifest_stats = {
        "count": len(all_episodes),
        "themes": [t.id for t in themes],
        "rejected_count": rejected_count,
        "avg_similarity": (
            round(sum(similarity_scores) / len(similarity_scores), 3)
            if similarity_scores else 0.0
        ),
    }

    return {
        "episodes": all_episodes,
        "manifest": manifest_stats,
        "tokens": total_tokens,
    }


def _generate_theme_batch(
    spec: ScopeSpec,
    client: object,
    model: str,
    theme: DistractorTheme,
    theme_idx: int,
    count: int,
    all_key_fact_strings: list[str],
    log_fn: object,
) -> dict:
    """Generate a single theme's distractor batch. Thread-safe."""
    dc = spec.distractors
    assert dc is not None
    target_words = dc.target_words if dc.target_words > 0 else spec.episodes.target_words

    episodes: list[dict] = []
    rejected_count = 0
    similarity_scores: list[float] = []
    tokens = {"prompt": 0, "completion": 0}

    log_fn(f"  Theme '{theme.id}': generating {count} episodes")  # type: ignore[operator]

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            prompt = build_distractor_prompt(spec, theme, count, [])
            response = _call_llm(
                client,
                model=model,
                system=SYSTEM_PROMPT,
                user=prompt,
                temperature=spec.generation.temperature,
                seed=dc.seed + theme_idx,
            )

            tokens["prompt"] += response["usage"]["prompt"]
            tokens["completion"] += response["usage"]["completion"]

            parsed = _parse_phase_response(response["content"])

            for ep in parsed["episodes"]:
                text = ep.get("text", "")
                if not text.strip():
                    continue

                # Word count enforcement
                word_count = len(text.split())
                if word_count < target_words * 0.7:
                    try:
                        text = _expand_episode(client, model, text, target_words, spec)
                    except Exception:
                        pass

                # Similarity check
                sim = _compute_distractor_similarity(text, all_key_fact_strings)
                similarity_scores.append(sim)

                if sim > dc.max_similarity:
                    rejected_count += 1
                    log_fn(  # type: ignore[operator]
                        f"    [{theme.id}] Rejected distractor (similarity {sim:.2f} > {dc.max_similarity})"
                    )
                    continue

                episodes.append({
                    "text": text,
                    "meta": {
                        "episode_type": "distractor",
                        "theme": theme.id,
                        **(ep.get("meta", {})),
                    },
                })

            log_fn(f"  Theme '{theme.id}': done ({len(episodes)} episodes)")  # type: ignore[operator]
            break

        except (DatasetError, json.JSONDecodeError, KeyError) as e:
            log_fn(f"    Theme {theme.id} attempt {attempt} failed: {e}")  # type: ignore[operator]
            if attempt == MAX_RETRIES:
                log_fn(f"    Theme {theme.id} failed after {MAX_RETRIES} attempts")  # type: ignore[operator]

    return {
        "episodes": episodes,
        "tokens": tokens,
        "rejected_count": rejected_count,
        "similarity_scores": similarity_scores,
    }


def _assign_distractor_ids_and_timestamps(
    spec: ScopeSpec, distractor_episodes: list[dict]
) -> None:
    """Assign dx_ IDs and interleaved timestamps to distractor episodes."""
    if not distractor_episodes:
        return

    from datetime import timedelta

    start_date = spec.episodes.timeline.start_date()
    interval_days = spec.episodes.timeline.interval_days()
    total_signal = spec.episodes.count
    # Total timeline span in days
    total_days = interval_days * (total_signal - 1)

    for i, ep in enumerate(distractor_episodes):
        ep["episode_id"] = f"{spec.scope_id}_dx_{i + 1:03d}"
        ep["scope_id"] = spec.scope_id
        # Interleave uniformly across the signal timeline
        if len(distractor_episodes) > 1:
            offset_days = (total_days * i) / (len(distractor_episodes) - 1)
        else:
            offset_days = total_days / 2
        dt = start_date + timedelta(days=offset_days)
        ep["timestamp"] = f"{dt.isoformat()}T10:30:00"


def _compute_distractor_similarity(text: str, key_facts: list[str]) -> float:
    """Compute max word-overlap similarity between a distractor and any key fact.

    Returns a value between 0.0 and 1.0 representing the highest overlap
    ratio across all key facts.
    """
    if not key_facts:
        return 0.0

    text_words = set(text.lower().split())
    max_sim = 0.0

    for fact in key_facts:
        fact_words = set(fact.lower().split())
        if not fact_words:
            continue
        overlap = fact_words & text_words
        sim = len(overlap) / len(fact_words)
        if sim > max_sim:
            max_sim = sim

    return max_sim


def _expand_episode(
    client: object,
    model: str,
    text: str,
    target_words: int,
    spec: ScopeSpec,
) -> str:
    """Expand a short episode to reach the target word count.

    Sends the short episode back to the LLM with instructions to expand.
    Retries up to 2 times.
    """
    for _ in range(2):
        prompt = (
            f"Expand the following log entry to at least {target_words} words "
            f"while preserving all facts, format, and voice. "
            f"Add detailed metrics, extended notes, additional subsections, "
            f"and contextual commentary. Return ONLY the expanded text, no JSON.\n\n"
            f"{text}"
        )
        response = _call_llm(
            client,
            model=model,
            system="You expand text while preserving its style and content. Return only the expanded text.",
            user=prompt,
            temperature=spec.generation.temperature,
            seed=spec.generation.seed,
        )
        expanded = response["content"].strip()
        if len(expanded.split()) >= target_words * 0.7:
            return expanded
        text = expanded  # Try again with the partially expanded version

    return text  # Return best effort


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

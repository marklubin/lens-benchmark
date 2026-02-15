from __future__ import annotations

import json
import os
from pathlib import Path

from lens.core.errors import DatasetError, atomic_write
from lens.datagen.prompt import build_contamination_prompt, build_naive_baseline_prompt
from lens.datagen.spec import ScopeSpec, load_spec, resolve_phase_ref


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def verify_scope(
    scope_dir: str | Path,
    contamination: bool = True,
    naive_baseline: bool = True,
    provider: str = "openai",
    model: str = "gpt-4o",
    api_key: str | None = None,
    api_base: str | None = None,
    verbose: bool = False,
    log_fn: object | None = None,
) -> dict:
    """Run verification checks on a generated scope.

    Args:
        scope_dir: Path to the scope directory (containing spec.yaml + generated/).
        contamination: Run contamination check.
        naive_baseline: Run naive full-context baseline.
        provider: LLM provider.
        model: Model name.
        api_key: API key.
        api_base: Optional API base URL.
        verbose: Print progress.
        log_fn: Optional logging callback.

    Returns:
        Verification results dict.
    """
    scope_dir = Path(scope_dir)
    gen_dir = scope_dir / "generated"

    def _log(msg: str) -> None:
        if log_fn:
            log_fn(msg)  # type: ignore[operator]
        elif verbose:
            print(msg)

    # Load spec and generated data
    spec_path = scope_dir / "spec.yaml"
    if not spec_path.exists():
        raise DatasetError(f"No spec.yaml found in {scope_dir}")
    spec = load_spec(spec_path)

    episodes_path = gen_dir / "episodes.json"
    questions_path = gen_dir / "questions.json"
    if not episodes_path.exists():
        raise DatasetError(f"No generated episodes found. Run 'lens generate' first.")

    episodes = json.loads(episodes_path.read_text())
    questions = json.loads(questions_path.read_text()) if questions_path.exists() else []

    results: dict = {}

    # Create LLM client if needed
    client = None
    if contamination or naive_baseline:
        client = _create_client(provider, api_key, api_base)

    if contamination:
        _log("Running contamination check...")
        results["contamination"] = _run_contamination_check(
            spec, episodes, questions, client, model, _log
        )

    if naive_baseline:
        _log("Running naive full-context baseline...")
        results["naive_baseline"] = _run_naive_baseline(
            spec, episodes, questions, client, model, _log
        )

    # Update manifest with verification results
    manifest_path = gen_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if contamination and "contamination" in results:
            manifest["validation"]["contamination_check"] = results["contamination"]["summary"]
        if naive_baseline and "naive_baseline" in results:
            manifest["validation"]["naive_baseline"] = results["naive_baseline"]["summary"]
        with atomic_write(manifest_path) as tmp:
            tmp.write_text(json.dumps(manifest, indent=2))

    _log("Verification complete.")
    return results


# ---------------------------------------------------------------------------
# Contamination check
# ---------------------------------------------------------------------------


def _run_contamination_check(
    spec: ScopeSpec,
    episodes: list[dict],
    questions: list[dict],
    client: object,
    model: str,
    log_fn: object,
) -> dict:
    """Check if longitudinal questions can be answered from single episodes."""
    longitudinal_qs = [q for q in questions if q.get("question_type") == "longitudinal"]

    if not longitudinal_qs:
        return {"summary": "pass", "detail": "No longitudinal questions to check"}

    results: list[dict] = []
    any_contaminated = False

    for q in longitudinal_qs:
        q_id = q["question_id"]
        q_prompt = q["prompt"]
        key_facts = q.get("ground_truth", {}).get("key_facts", [])
        checkpoint = q.get("checkpoint_after", len(episodes))

        log_fn(f"  Checking question: {q_id}")  # type: ignore[operator]

        # Test each episode individually up to the checkpoint
        max_single_coverage = 0.0
        worst_episode = None

        relevant_episodes = [ep for ep in episodes[:checkpoint]]

        for ep in relevant_episodes:
            prompt = build_contamination_prompt(ep["text"], q_prompt)
            answer = _call_completion(client, model, prompt)

            # Score: what fraction of key facts appear in the answer?
            coverage = _compute_fact_coverage(answer, key_facts)
            if coverage > max_single_coverage:
                max_single_coverage = coverage
                worst_episode = ep["episode_id"]

        contaminated = max_single_coverage > 0.5
        if contaminated:
            any_contaminated = True

        results.append({
            "question_id": q_id,
            "max_single_episode_coverage": round(max_single_coverage, 3),
            "worst_episode": worst_episode,
            "contaminated": contaminated,
        })

    return {
        "summary": "fail" if any_contaminated else "pass",
        "questions": results,
    }


# ---------------------------------------------------------------------------
# Naive full-context baseline
# ---------------------------------------------------------------------------


def _run_naive_baseline(
    spec: ScopeSpec,
    episodes: list[dict],
    questions: list[dict],
    client: object,
    model: str,
    log_fn: object,
) -> dict:
    """Run naive baseline: all episodes in context, no memory system."""
    results: list[dict] = []

    for q in questions:
        q_id = q["question_id"]
        q_prompt = q["prompt"]
        key_facts = q.get("ground_truth", {}).get("key_facts", [])
        checkpoint = q.get("checkpoint_after", len(episodes))

        log_fn(f"  Baseline for question: {q_id}")  # type: ignore[operator]

        # Concatenate all episodes up to checkpoint
        episode_texts = [ep["text"] for ep in episodes[:checkpoint]]
        prompt = build_naive_baseline_prompt(episode_texts, q_prompt)

        answer = _call_completion(client, model, prompt)
        coverage = _compute_fact_coverage(answer, key_facts)

        results.append({
            "question_id": q_id,
            "question_type": q.get("question_type", ""),
            "fact_coverage": round(coverage, 3),
            "answer_preview": answer[:200],
        })

    # Compute summary stats by question type
    by_type: dict[str, list[float]] = {}
    for r in results:
        qt = r["question_type"]
        by_type.setdefault(qt, []).append(r["fact_coverage"])

    summary_stats = {}
    for qt, coverages in by_type.items():
        avg = sum(coverages) / len(coverages) if coverages else 0.0
        summary_stats[qt] = round(avg, 3)

    return {
        "summary": summary_stats,
        "questions": results,
    }


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------


def _compute_fact_coverage(answer: str, key_facts: list[str]) -> float:
    """Compute what fraction of key facts appear in an answer (fuzzy matching)."""
    if not key_facts:
        return 1.0

    answer_lower = answer.lower()
    answer_words = set(answer_lower.split())
    hits = 0

    for fact in key_facts:
        fact_words = set(fact.lower().split())
        # Require at least 50% of fact words to appear in the answer
        overlap = fact_words & answer_words
        if len(overlap) >= max(1, len(fact_words) * 0.5):
            hits += 1

    return hits / len(key_facts)


# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------


def _create_client(provider: str, api_key: str | None, api_base: str | None) -> object:
    """Create an OpenAI client for verification."""
    if provider != "openai":
        raise DatasetError(f"Verification only supports 'openai' provider, got {provider!r}")

    try:
        from openai import OpenAI
    except ImportError:
        raise DatasetError(
            "OpenAI package required for verification. "
            "Install with: pip install 'lens-bench[datagen]'"
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


def _call_completion(client: object, model: str, prompt: str) -> str:
    """Simple completion call — no tools, no structured output."""
    response = client.chat.completions.create(  # type: ignore[union-attr]
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    return response.choices[0].message.content or ""

"""Automated narrative episode generation for LENS benchmark scopes.

Generates signal and distractor episodes by calling an LLM API:
1. Builds planner prompts from spec (signal + distractor themes)
2. Calls LLM to produce fact sheet outlines (JSON)
3. Renders each episode from its fact sheet (parallel)
4. Validates word counts

Usage:
    python scripts/generate_narrative_episodes.py datasets/scopes/10_clinical_trial \
        [--api-base URL] [--model MODEL] [--api-key KEY] \
        [--workers 4] [--skip-existing]

Env var fallbacks: LENS_LLM_API_BASE, LENS_LLM_MODEL, LENS_LLM_API_KEY
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Ensure scripts/ and synix are importable
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "lens" / "datagen" / "synix"))

import spec_utils
from narrative_prompts import (
    build_narrative_distractor_planner_prompt,
    build_narrative_planner_prompt,
    build_narrative_renderer_prompt,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LLM client
# ---------------------------------------------------------------------------

def _get_client(api_base: str, api_key: str):
    """Create an OpenAI-compatible client."""
    from openai import OpenAI
    return OpenAI(base_url=api_base, api_key=api_key)


def _call_llm(
    client,
    model: str,
    prompt: str,
    *,
    max_tokens: int = 32768,
    temperature: float = 0.7,
    retries: int = 3,
) -> str:
    """Call LLM with retry on transient failures."""
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=max_tokens,
                temperature=temperature,
            )
            content = resp.choices[0].message.content
            if not content:
                raise ValueError("Empty response from LLM")
            return content
        except Exception as e:
            if attempt < retries - 1:
                wait = 2 ** (attempt + 1)
                log.warning("LLM call failed (attempt %d/%d): %s. Retrying in %ds...",
                            attempt + 1, retries, e, wait)
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("Unreachable")


def _extract_json(text: str) -> dict:
    """Extract JSON from LLM response, handling markdown fences and malformed output."""
    # Try to find JSON in code blocks first
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass  # Fall through to other methods

    # Try parsing the whole response
    text_stripped = text.strip()
    try:
        return json.loads(text_stripped)
    except json.JSONDecodeError:
        pass

    # Find the first { and try to parse from there
    start = text.find("{")
    if start < 0:
        raise ValueError(f"No JSON object found in response (length={len(text)})")

    # Use incremental parsing to find the correct closing brace
    # (handles cases where LLM adds commentary after JSON)
    decoder = json.JSONDecoder()
    try:
        obj, _ = decoder.raw_decode(text, start)
        return obj
    except json.JSONDecodeError:
        pass

    # Last resort: find last } and try
    end = text.rfind("}")
    if end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not extract JSON from response (length={len(text)})")


def _normalize_outline(raw: dict, expected_count: int) -> dict:
    """Normalize LLM outline output to standard format with 'episodes' list.

    The LLM may return:
    - {"episodes": [...]} — standard format
    - {"fact_sheets": [...]} — alternate key
    - A top-level list
    - Nested dicts with "data" sub-keys

    Each episode must have an "index" field (1-based).
    """
    # Find the episode list
    if isinstance(raw, list):
        items = raw
    elif "episodes" in raw:
        items = raw["episodes"]
    elif "fact_sheets" in raw:
        items = raw["fact_sheets"]
    else:
        # Try the first list value found
        for v in raw.values():
            if isinstance(v, list):
                items = v
                break
        else:
            log.warning("Could not find episode list in outline. Keys: %s", list(raw.keys()))
            return {"episodes": []}

    # Normalize each episode
    normalized = []
    for i, item in enumerate(items):
        ep = dict(item)
        # Ensure index exists
        if "index" not in ep:
            ep["index"] = i + 1
        # If fact sheet data is nested under a "data" key, flatten it
        if "data" in ep and isinstance(ep["data"], dict) and "documents" not in ep:
            data = ep.pop("data")
            if "documents" in data:
                ep["documents"] = data["documents"]
            else:
                # Wrap the data dict as a single document
                ep["documents"] = [data]
        # Ensure documents list exists (renderer needs it)
        if "documents" not in ep:
            ep["documents"] = []
        normalized.append(ep)

    if len(normalized) < expected_count:
        log.warning("Outline has %d episodes, expected %d", len(normalized), expected_count)

    return {"episodes": normalized}


# ---------------------------------------------------------------------------
# Generation steps
# ---------------------------------------------------------------------------

def generate_signal_outline(
    client, model: str, spec: dict, gen_dir: Path, *, skip_existing: bool
) -> dict:
    """Generate signal episode fact sheets."""
    outline_path = gen_dir / "signal_outline.json"
    if skip_existing and outline_path.exists():
        log.info("Signal outline already exists, loading")
        raw = json.loads(outline_path.read_text())
        return _normalize_outline(raw, spec["episodes"]["count"]) if "episodes" not in raw else raw

    prompt = build_narrative_planner_prompt(spec)
    # Save prompt for reproducibility
    (gen_dir / "prompts").mkdir(parents=True, exist_ok=True)
    (gen_dir / "prompts" / "signal_planner.md").write_text(prompt)

    log.info("Calling LLM for signal outline (%d episodes)...", spec["episodes"]["count"])
    response = _call_llm(client, model, prompt, max_tokens=32768, temperature=0.7)
    raw = _extract_json(response)
    outline = _normalize_outline(raw, spec["episodes"]["count"])

    outline_path.write_text(json.dumps(outline, indent=2))
    log.info("Wrote signal outline: %d episodes", len(outline["episodes"]))
    return outline


def generate_distractor_outlines(
    client, model: str, spec: dict, gen_dir: Path, *, skip_existing: bool
) -> dict[str, dict]:
    """Generate distractor episode fact sheets for each theme."""
    dc = spec.get("distractors")
    if not dc or dc["count"] <= 0:
        return {}

    themes = dc["themes"]
    per_theme = [dc["count"] // len(themes)] * len(themes)
    for i in range(dc["count"] % len(themes)):
        per_theme[i] += 1

    outlines = {}
    for idx, theme in enumerate(themes):
        count = per_theme[idx]
        if count <= 0:
            continue

        theme_id = theme["id"]
        outline_path = gen_dir / f"distractor_outline_{theme_id}.json"

        if skip_existing and outline_path.exists():
            log.info("Distractor outline for %s already exists, loading", theme_id)
            raw = json.loads(outline_path.read_text())
            outline = _normalize_outline(raw, count) if "episodes" not in raw else raw
            outlines[theme_id] = outline
            continue

        prompt = build_narrative_distractor_planner_prompt(spec, theme, count)
        (gen_dir / "prompts").mkdir(parents=True, exist_ok=True)
        (gen_dir / "prompts" / f"distractor_planner_{theme_id}.md").write_text(prompt)

        log.info("Calling LLM for distractor outline: %s (%d episodes)...", theme_id, count)
        response = _call_llm(client, model, prompt, max_tokens=16384, temperature=0.7)
        raw = _extract_json(response)
        outline = _normalize_outline(raw, count)

        outline_path.write_text(json.dumps(outline, indent=2))
        log.info("Wrote distractor outline for %s: %d episodes", theme_id, len(outline["episodes"]))
        outlines[theme_id] = outline

    return outlines


def render_episode(
    client, model: str, spec: dict, fact_sheet: dict,
    output_path: Path, label: str
) -> tuple[str, int]:
    """Render a single episode from its fact sheet. Returns (label, word_count)."""
    prompt = build_narrative_renderer_prompt(spec, fact_sheet)

    log.info("Rendering %s...", label)
    text = _call_llm(client, model, prompt, max_tokens=16384, temperature=0.7)

    # Strip any markdown fences from the output
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first and last fence lines
        if lines[-1].strip() == "```":
            lines = lines[1:-1]
        else:
            lines = lines[1:]
        text = "\n".join(lines).strip()

    output_path.write_text(text)
    wc = len(text.split())
    log.info("Wrote %s: %d words", label, wc)
    return label, wc


def render_all_episodes(
    client, model: str, spec: dict,
    signal_outline: dict, distractor_outlines: dict[str, dict],
    gen_dir: Path, *, workers: int = 4, skip_existing: bool = False,
) -> dict[str, int]:
    """Render all signal and distractor episodes in parallel."""
    ep_dir = gen_dir / "episodes"
    ep_dir.mkdir(parents=True, exist_ok=True)

    tasks: list[tuple] = []  # (fact_sheet, output_path, label)

    # Signal episodes
    for ep in signal_outline.get("episodes", []):
        idx = ep["index"]
        path = ep_dir / f"signal_{idx:03d}.txt"
        label = f"signal_{idx:03d}"
        if skip_existing and path.exists():
            log.info("Episode %s already exists, skipping", label)
            continue
        tasks.append((ep, path, label))

    # Distractor episodes
    dc = spec.get("distractors", {})
    themes = dc.get("themes", [])
    per_theme = [dc["count"] // len(themes)] * len(themes) if themes else []
    for i in range(dc.get("count", 0) % max(len(themes), 1)):
        per_theme[i] += 1

    for tidx, theme in enumerate(themes):
        theme_id = theme["id"]
        outline = distractor_outlines.get(theme_id, {})
        for ep in outline.get("episodes", []):
            idx = ep["index"]
            path = ep_dir / f"distractor_{theme_id}_{idx:03d}.txt"
            label = f"distractor_{theme_id}_{idx:03d}"
            if skip_existing and path.exists():
                log.info("Episode %s already exists, skipping", label)
                continue
            tasks.append((ep, path, label))

    if not tasks:
        log.info("All episodes already exist, nothing to render")
        return {}

    log.info("Rendering %d episodes with %d workers...", len(tasks), workers)

    results: dict[str, int] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(render_episode, client, model, spec, fs, path, label): label
            for fs, path, label in tasks
        }
        for future in as_completed(futures):
            label = futures[future]
            try:
                name, wc = future.result()
                results[name] = wc
            except Exception as e:
                log.error("Failed to render %s: %s", label, e)
                results[label] = -1

    return results


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_episodes(gen_dir: Path, spec: dict) -> bool:
    """Check that all expected episode files exist and meet word count minimums."""
    ep_dir = gen_dir / "episodes"
    ep_count = spec["episodes"]["count"]
    issues: list[str] = []

    # Signal episodes
    for i in range(1, ep_count + 1):
        path = ep_dir / f"signal_{i:03d}.txt"
        if not path.exists():
            issues.append(f"Missing: {path.name}")
        else:
            wc = len(path.read_text().split())
            if wc < 3000:
                issues.append(f"{path.name}: only {wc} words (min 3000)")

    # Distractor episodes
    dc = spec.get("distractors", {})
    themes = dc.get("themes", [])
    per_theme = [dc["count"] // len(themes)] * len(themes) if themes else []
    for i in range(dc.get("count", 0) % max(len(themes), 1)):
        per_theme[i] += 1

    for tidx, theme in enumerate(themes):
        for j in range(1, per_theme[tidx] + 1):
            path = ep_dir / f"distractor_{theme['id']}_{j:03d}.txt"
            if not path.exists():
                issues.append(f"Missing: {path.name}")
            else:
                wc = len(path.read_text().split())
                if wc < 3000:
                    issues.append(f"{path.name}: only {wc} words (min 3000)")

    if issues:
        log.warning("Validation issues:")
        for issue in issues:
            log.warning("  - %s", issue)
        return False

    log.info("Validation passed: all episodes present and >= 3000 words")
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate narrative episodes for a LENS benchmark scope"
    )
    parser.add_argument("scope_dir", help="Path to scope directory (e.g., datasets/scopes/10_clinical_trial)")
    parser.add_argument("--api-base", default=os.environ.get("LENS_LLM_API_BASE", ""),
                        help="LLM API base URL")
    parser.add_argument("--model", default=os.environ.get("LENS_LLM_MODEL", "Qwen/Qwen3.5-35B-A3B"),
                        help="LLM model name")
    parser.add_argument("--api-key", default=os.environ.get("LENS_LLM_API_KEY", "dummy"),
                        help="LLM API key")
    parser.add_argument("--workers", type=int, default=4, help="Parallel rendering workers")
    parser.add_argument("--skip-existing", action="store_true", help="Skip files that already exist")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    if not args.api_base:
        log.error("No API base URL. Set --api-base or LENS_LLM_API_BASE env var.")
        sys.exit(1)

    scope_path = Path(args.scope_dir)
    spec = spec_utils.load_spec(scope_path / "spec.yaml")
    spec_utils.validate_spec_or_raise(spec)

    gen_dir = scope_path / "generated"
    gen_dir.mkdir(parents=True, exist_ok=True)

    client = _get_client(args.api_base, args.api_key)

    log.info("=== Generating episodes for %s ===", spec["scope_id"])
    log.info("Model: %s, API: %s", args.model, args.api_base)
    log.info("Signal episodes: %d, Distractor episodes: %d",
             spec["episodes"]["count"],
             spec.get("distractors", {}).get("count", 0))

    t0 = time.time()

    # Step 1: Generate outlines
    signal_outline = generate_signal_outline(
        client, args.model, spec, gen_dir, skip_existing=args.skip_existing
    )
    distractor_outlines = generate_distractor_outlines(
        client, args.model, spec, gen_dir, skip_existing=args.skip_existing
    )

    # Step 2: Render episodes
    results = render_all_episodes(
        client, args.model, spec, signal_outline, distractor_outlines,
        gen_dir, workers=args.workers, skip_existing=args.skip_existing,
    )

    # Step 3: Validate
    valid = validate_episodes(gen_dir, spec)

    elapsed = time.time() - t0
    log.info("=== Done in %.1f minutes ===", elapsed / 60)
    log.info("Rendered %d episodes, %d failures",
             sum(1 for wc in results.values() if wc > 0),
             sum(1 for wc in results.values() if wc <= 0))

    if not valid:
        log.warning("Some validation issues detected. Review logs above.")
        sys.exit(1)


if __name__ == "__main__":
    main()

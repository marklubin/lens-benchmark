#!/usr/bin/env python3
"""Generate query plans for the static driver.

For each question in a dataset, generates targeted search queries that use
evidence vocabulary (entity names, metric names, timestamps) rather than
question vocabulary. This bridges the semantic gap between how questions
are phrased and how evidence is stored.

Usage:
    uv run python scripts/generate_query_plans.py \
        --dataset datasets/scope_07_with_distractors.json \
        --output query_plans/scope_07.json \
        [--model casperhansen/Meta-Llama-3.3-70B-Instruct-AWQ-INT4] \
        [--api-key ...] [--api-base ...]
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a search query expert. Given a benchmark question, its ground truth \
answer, and the required evidence episodes, generate 3-5 search queries that \
would find the right evidence in a memory system.

CRITICAL: Your queries must use the VOCABULARY of the evidence (entity names, \
metric values, timestamps, specific phrases) — NOT the vocabulary of the \
question. The whole point is to bridge the semantic gap between how questions \
are phrased and how evidence is stored.

Output valid JSON only, no markdown fences."""

_USER_TEMPLATE = """\
QUESTION:
{prompt}

GROUND TRUTH ANSWER:
{canonical_answer}

REQUIRED EVIDENCE EPISODE IDS:
{evidence_refs}

{episode_context}

Generate 3-5 search queries as JSON:
{{"searches": ["query1", "query2", ...], "retrieve_top_k": 3}}"""


def load_dataset(path: str) -> dict:
    """Load dataset JSON file."""
    with open(path) as f:
        return json.load(f)


def get_episode_text(dataset: dict, episode_ids: list[str]) -> str:
    """Get episode text for context, truncated."""
    ep_map: dict[str, str] = {}
    for scope in dataset.get("scopes", []):
        for ep in scope.get("episodes", []):
            ep_map[ep["episode_id"]] = ep["text"]

    lines = []
    for eid in episode_ids:
        text = ep_map.get(eid, "")
        if text:
            # Take first 500 chars for context
            lines.append(f"[{eid}] (first 500 chars):\n{text[:500]}")
    if lines:
        return "EVIDENCE EPISODE EXCERPTS:\n" + "\n\n".join(lines)
    return ""


def generate_plan_for_question(
    client, model: str, question: dict, episode_context: str,
) -> dict:
    """Generate a query plan for a single question via LLM."""
    prompt = question["prompt"]
    gt = question.get("ground_truth", {})
    canonical = gt.get("canonical_answer", "")
    refs = gt.get("required_evidence_refs", [])

    user_msg = _USER_TEMPLATE.format(
        prompt=prompt,
        canonical_answer=canonical,
        evidence_refs=json.dumps(refs),
        episode_context=episode_context,
    )

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.0,
            max_tokens=1024,
        )
        content = resp.choices[0].message.content or ""
        # Strip markdown fences if present
        content = content.strip()
        if content.startswith("```"):
            content = content.split("\n", 1)[1] if "\n" in content else content
        if content.endswith("```"):
            content = content.rsplit("```", 1)[0]
        content = content.strip()

        plan = json.loads(content)
        if "searches" not in plan:
            log.warning("LLM response missing 'searches' for: %.60s", prompt)
            plan = {"searches": [prompt], "retrieve_top_k": 3}
        return plan
    except (json.JSONDecodeError, Exception) as e:
        log.error("Failed to generate plan for '%.60s': %s", prompt, e)
        return {"searches": [prompt], "retrieve_top_k": 3}


def main():
    parser = argparse.ArgumentParser(description="Generate query plans for static driver")
    parser.add_argument("--dataset", required=True, help="Path to dataset JSON")
    parser.add_argument("--output", required=True, help="Output path for query plans JSON")
    parser.add_argument(
        "--model", default=os.environ.get("LENS_LLM_MODEL", "gpt-4o-mini"),
        help="LLM model for generating plans",
    )
    parser.add_argument("--api-key", default=None, help="API key (or LENS_LLM_API_KEY)")
    parser.add_argument("--api-base", default=None, help="API base URL (or LENS_LLM_API_BASE)")
    parser.add_argument(
        "--no-episode-context", action="store_true",
        help="Don't include episode text in LLM prompt (faster, less accurate)",
    )
    args = parser.parse_args()

    try:
        from openai import OpenAI
    except ImportError:
        log.error("openai package required. Install with: pip install openai")
        sys.exit(1)

    api_key = (
        args.api_key
        or os.environ.get("LENS_LLM_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
    )
    api_base = (
        args.api_base
        or os.environ.get("LENS_LLM_API_BASE")
        or os.environ.get("OPENAI_BASE_URL")
    )
    if not api_key:
        log.error("API key required: set LENS_LLM_API_KEY or pass --api-key")
        sys.exit(1)

    client_kwargs: dict = {"api_key": api_key}
    if api_base:
        client_kwargs["base_url"] = api_base
    client = OpenAI(**client_kwargs)

    # Strip provider prefix for model
    model = args.model
    if "/" in model and model.startswith(("together/", "openai/")):
        model = model.split("/", 1)[1]

    dataset = load_dataset(args.dataset)
    questions = dataset.get("questions", [])
    if not questions:
        log.error("No questions found in dataset: %s", args.dataset)
        sys.exit(1)

    log.info("Generating query plans for %d questions from %s", len(questions), args.dataset)

    plans: dict[str, dict] = {}
    for i, q in enumerate(questions):
        prompt = q["prompt"]
        refs = q.get("ground_truth", {}).get("required_evidence_refs", [])

        episode_context = ""
        if not args.no_episode_context:
            episode_context = get_episode_text(dataset, refs)

        log.info("[%d/%d] %.60s", i + 1, len(questions), prompt)
        plan = generate_plan_for_question(client, model, q, episode_context)
        plans[prompt] = plan
        log.info("  → %d searches, retrieve_top_k=%d", len(plan["searches"]), plan.get("retrieve_top_k", 3))

    output = {"plans": plans}
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2))
    log.info("Wrote query plans to %s", args.output)


if __name__ == "__main__":
    main()

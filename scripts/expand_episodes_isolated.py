"""Expand episode text files to ~5000 words using isolated LLM calls.

Each episode is expanded in a completely independent API call — the LLM never
sees other episodes, the spec, or the storyline. This maintains information
isolation: the expanding model cannot editorialize or leak cross-episode signal
because it doesn't know what's signal.

Usage:
    python scripts/expand_episodes_isolated.py datasets/scopes/13_implicit_decision
    python scripts/expand_episodes_isolated.py --all  # all 3 new scopes
    python scripts/expand_episodes_isolated.py --dry-run --all  # preview without calling API
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import anthropic

TARGET_WORDS = 5000
MIN_WORDS = 4000  # Don't expand if already above this
MAX_WORKERS = 8  # Parallel API calls

EXPANSION_PROMPT = """\
You are expanding a document to approximately {target} words. You will be given \
a single document — a meeting transcript, chat log, incident report, or similar \
team communication artifact.

Your job: add MORE realistic detail to make the document longer and denser, \
while preserving the EXACT same events, decisions, and information. Do not \
change what happens. Do not add new plot points or conclusions. Just add:

- More participants speaking (with realistic names and roles)
- More technical detail in discussions (code snippets, config examples, metrics)
- More side conversations and tangential topics within the meeting
- More action items, follow-ups, and logistical details
- More realistic formatting (timestamps, headers, agenda items)
- More natural conversational filler (greetings, transitions, clarifications)

CRITICAL RULES:
- Do NOT add any analysis, commentary, or conclusions that aren't in the original
- Do NOT reference events or information from other documents
- Do NOT add foreshadowing or hindsight
- Preserve the exact tone and style of the original
- If the document contains technical decisions, keep them exactly as stated
- Output ONLY the expanded document, no preamble or explanation

Current length: {current_words} words. Target: ~{target} words.
"""

REGEN_PROMPT = """\
Write a realistic team communication document (~{target} words) based on this brief description:

{description}

The document should be a {doc_type} with:
- Realistic names, roles, dates, and details
- Natural conversational tone appropriate to the format
- Enough technical/business detail to feel authentic
- Multiple participants and topics discussed

CRITICAL RULES:
- Write ONLY the document, no preamble or explanation
- Do not reference any external context or storyline
- Make it feel like a real artifact from a real team
"""


def expand_episode(
    filepath: Path,
    client: anthropic.Anthropic,
    dry_run: bool = False,
    regen_info: dict | None = None,
) -> dict:
    """Expand a single episode file. Returns stats dict."""
    text = filepath.read_text().strip()
    current_words = len(text.split())

    if current_words >= MIN_WORDS and regen_info is None:
        return {
            "file": filepath.name,
            "status": "skipped",
            "words_before": current_words,
            "words_after": current_words,
            "reason": f"already {current_words} words (>= {MIN_WORDS})",
        }

    if dry_run:
        return {
            "file": filepath.name,
            "status": "dry_run",
            "words_before": current_words,
            "words_after": "~5000",
            "reason": "would expand" if regen_info is None else "would regenerate",
        }

    if regen_info is not None:
        # Regenerate from description (for contaminated files)
        prompt = REGEN_PROMPT.format(
            target=TARGET_WORDS,
            description=regen_info["description"],
            doc_type=regen_info["doc_type"],
        )
        messages = [{"role": "user", "content": prompt}]
    else:
        # Expand existing content
        prompt = EXPANSION_PROMPT.format(
            target=TARGET_WORDS,
            current_words=current_words,
        )
        messages = [
            {"role": "user", "content": f"{prompt}\n\n---\n\n{text}"},
        ]

    try:
        response = client.messages.create(
            model="claude-sonnet-4-5-20250514",
            max_tokens=8192,
            messages=messages,
        )
        expanded = response.content[0].text.strip()
        new_words = len(expanded.split())

        filepath.write_text(expanded + "\n")

        return {
            "file": filepath.name,
            "status": "expanded" if regen_info is None else "regenerated",
            "words_before": current_words,
            "words_after": new_words,
        }
    except Exception as e:
        return {
            "file": filepath.name,
            "status": "error",
            "words_before": current_words,
            "words_after": current_words,
            "reason": str(e),
        }


# Descriptions for scope 14 signal_001-004 (contaminated, need regeneration)
SCOPE_14_REGEN = {
    "signal_001.txt": {
        "description": (
            "A startup engineering team Slack chat (around April 2025). The team "
            "culture is 'ship fast, fix later.' A developer built a document preview "
            "feature over the weekend and wants to push it to production immediately. "
            "No tests, no code review, no staging environment. The team discusses "
            "deploying it, someone asks about feature flags, another says 'just push "
            "it.' Include other sprint topics, standup updates, casual banter. "
            "The energy is fast, casual, and optimistic. Company is a small SaaS "
            "startup called Foldly (document management). Team members: Priya Kaur "
            "(CTO), Marcus Webb (backend), Jess Nakamura (frontend), Danny Okafor "
            "(junior dev), Lina Torres (DevOps). No mention of incidents, testing "
            "strategy, or process concerns — those come later."
        ),
        "doc_type": "Slack channel log (#engineering)",
    },
    "signal_002.txt": {
        "description": (
            "A Foldly engineering Slack chat. Marcus Webb deploys the entire billing "
            "and invoicing system — written as a single Python file (billing.py). "
            "He built it over a few days. No tests, no PR review. The team's first "
            "enterprise customer (Nara Logistics) is being onboarded and billing "
            "needs to work. Marcus casually mentions the file is ~800 lines. Someone "
            "notices some edge cases but doesn't push back hard. The tone is 'move "
            "fast, we have a customer waiting.' Include related chat about the "
            "enterprise onboarding, feature requests, and general dev work. "
            "Team: Priya, Marcus, Jess, Danny, Lina, Ravi Mehta (sales/success)."
        ),
        "doc_type": "Slack channel log (#engineering)",
    },
    "signal_003.txt": {
        "description": (
            "A Foldly engineering Slack chat. Danny Okafor (junior dev) asks about "
            "writing tests for a module he's working on. The team's response: "
            "'We'll add tests when we have time' and 'let's focus on shipping for "
            "now.' There's a brief discussion about a 'testing sprint' that gets "
            "deferred to 'next month' or 'after launch.' Marcus mentions he added "
            "a cron job for some maintenance task without telling anyone. Someone "
            "mentions that routes.py is getting very long. The overall vibe is "
            "that testing and code quality are acknowledged as important but "
            "always deprioritized in favor of shipping features. Include other "
            "dev chat topics: debugging, deployments, feature work."
        ),
        "doc_type": "Slack channel log (#engineering)",
    },
    "signal_004.txt": {
        "description": (
            "A Foldly engineering Slack chat and ship announcement. The team "
            "celebrates launching 3 features in one week (or similar rapid pace). "
            "Celebratory tone — emojis, congratulations, 'we're crushing it.' "
            "A second enterprise customer signs up. Brief mention of tech debt "
            "and 'things we should clean up eventually' but framed as low priority. "
            "Lina Torres flags zero test coverage, zero code reviews, no staging "
            "environment, no redundancy — but it's treated as 'we know, we'll get "
            "to it.' Speed is explicitly valued: 'this is our advantage.' Include "
            "product metrics, user growth numbers, and roadmap excitement."
        ),
        "doc_type": "Slack channel log (#engineering) + #general ship announcement",
    },
}


def main():
    parser = argparse.ArgumentParser(description="Expand episodes with information isolation")
    parser.add_argument("scope_dirs", nargs="*", help="Scope directories to process")
    parser.add_argument("--all", action="store_true", help="Process all 3 new scopes (13-15)")
    parser.add_argument("--dry-run", action="store_true", help="Preview without API calls")
    parser.add_argument("--workers", type=int, default=MAX_WORKERS, help="Parallel workers")
    args = parser.parse_args()

    if args.all:
        scope_dirs = [
            Path("datasets/scopes/13_implicit_decision"),
            Path("datasets/scopes/14_epoch_classification"),
            Path("datasets/scopes/15_value_inversion"),
        ]
    else:
        scope_dirs = [Path(d) for d in args.scope_dirs]

    if not scope_dirs:
        print("Usage: python scripts/expand_episodes_isolated.py --all")
        sys.exit(1)

    client = anthropic.Anthropic()

    # Collect all files to process
    tasks: list[tuple[Path, dict | None]] = []
    for scope_dir in scope_dirs:
        ep_dir = scope_dir / "generated" / "episodes"
        if not ep_dir.exists():
            print(f"WARNING: {ep_dir} does not exist, skipping")
            continue

        scope_name = scope_dir.name
        for f in sorted(ep_dir.glob("*.txt")):
            regen_info = None
            # Check if this file needs regeneration (contaminated scope 14 signal_001-004)
            if scope_name == "14_epoch_classification" and f.name in SCOPE_14_REGEN:
                regen_info = SCOPE_14_REGEN[f.name]
            tasks.append((f, regen_info))

    print(f"Processing {len(tasks)} episode files across {len(scope_dirs)} scopes")
    print(f"Target: ~{TARGET_WORDS} words per episode")
    print(f"Workers: {args.workers}")
    print(f"{'DRY RUN — no API calls' if args.dry_run else 'LIVE — calling Anthropic API'}")
    print()

    results = []
    start = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(expand_episode, f, client, args.dry_run, regen): (f, regen)
            for f, regen in tasks
        }
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            status = result["status"]
            before = result["words_before"]
            after = result["words_after"]
            marker = ""
            if status == "error":
                marker = f" ERROR: {result.get('reason', 'unknown')}"
            elif status == "skipped":
                marker = f" (skipped: {result.get('reason', '')})"
            print(f"  [{status:>11}] {result['file']:>45} {before:>5} → {after:>5}{marker}")

    elapsed = time.time() - start
    expanded = sum(1 for r in results if r["status"] in ("expanded", "regenerated"))
    skipped = sum(1 for r in results if r["status"] == "skipped")
    errors = sum(1 for r in results if r["status"] == "error")
    print(f"\nDone in {elapsed:.1f}s: {expanded} expanded, {skipped} skipped, {errors} errors")

    # Summary word counts
    print("\n=== Final Word Counts ===")
    for scope_dir in scope_dirs:
        ep_dir = scope_dir / "generated" / "episodes"
        total = 0
        for f in sorted(ep_dir.glob("*.txt")):
            wc = len(f.read_text().split())
            total += wc
        print(f"  {scope_dir.name}: {total:,} words")


if __name__ == "__main__":
    main()

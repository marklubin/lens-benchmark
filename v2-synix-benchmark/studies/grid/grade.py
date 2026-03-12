"""Claude grading script for grid run answers.

Reads grading tasks from grading_tasks.jsonl, evaluates each answer
against its key facts using the rubric in grading_rubric.md, and writes
structured scores to claude_scores.jsonl.

Usage:
  uv run python studies/grid/grade.py [--dry-run] [--limit N]
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from bench.broker import ModalBroker

RESULTS_DIR = Path(__file__).resolve().parent / "results"
GRADING_TASKS_FILE = RESULTS_DIR / "grading_tasks.jsonl"
CLAUDE_SCORES_FILE = RESULTS_DIR / "claude_scores.jsonl"

LLM_BASE_URL = "https://synix--lens-llm-llm-serve.modal.run/v1"
EMBED_BASE_URL = "https://synix--lens-embed-serve.modal.run"
JUDGE_MODEL = "Qwen/Qwen3.5-35B-A3B"

RUBRIC_VERSION = "v1.0"

logger = logging.getLogger("grade")


def build_grading_messages(task: dict) -> list[dict]:
    """Build few-shot grading messages for one answer.

    Uses few-shot examples to suppress Qwen's thinking mode and ensure
    clean JSON output.
    """
    key_facts_text = ""
    for i, fact in enumerate(task["key_facts"], 1):
        key_facts_text += f"  {i}. {fact}\n"

    if not key_facts_text.strip():
        key_facts_text = "  (no key facts defined)\n"

    system = (
        "You are a benchmark grader. For each key fact, score whether the answer states or implies it. "
        "1.0=present, 0.5=partial, 0.0=absent. Accept paraphrases. Reject vague allusions. "
        "Output ONLY valid JSON."
    )

    # Few-shot example to anchor output format
    example_user = (
        "Key facts:\n"
        "  1. Dogs are domesticated mammals\n"
        "  2. Dogs descended from wolves\n"
        "Answer: Canines are domesticated animals in the mammal family.\n"
        "Score each fact."
    )
    example_assistant = (
        '{"fact_scores":[1.0,0.0],'
        '"fact_verdicts":["Paraphrase: domesticated mammals","Wolf ancestry not mentioned"],'
        '"fact_f1":0.5}'
    )

    # Actual grading task
    user_msg = (
        f"Question: {task['question_prompt']}\n"
        f"Canonical answer: {task['canonical_answer']}\n\n"
        f"Key facts:\n{key_facts_text}\n"
        f"Answer: {task['answer_text']}\n\n"
        f"Score each fact."
    )

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": example_user},
        {"role": "assistant", "content": example_assistant},
        {"role": "user", "content": user_msg},
    ]


def parse_grade_response(response_text: str) -> dict | None:
    """Parse the JSON response from the grading LLM.

    Handles Qwen's thinking mode leak: strips everything before the first '{'.
    Also handles markdown code blocks and other preamble.
    """
    text = response_text.strip()

    # Remove markdown code blocks if present
    if "```" in text:
        code_blocks = re.findall(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
        for block in code_blocks:
            try:
                result = json.loads(block.strip())
                if "fact_scores" in result:
                    return result
            except json.JSONDecodeError:
                continue

    # Find the last complete JSON object (skip thinking preamble)
    # Search for balanced braces from the end
    brace_end = text.rfind("}")
    if brace_end < 0:
        return None

    # Find the matching opening brace by counting from the end
    depth = 0
    brace_start = -1
    for i in range(brace_end, -1, -1):
        if text[i] == "}":
            depth += 1
        elif text[i] == "{":
            depth -= 1
            if depth == 0:
                brace_start = i
                break

    if brace_start >= 0:
        try:
            result = json.loads(text[brace_start : brace_end + 1])
            if "fact_scores" in result:
                return result
        except json.JSONDecodeError:
            pass

    # Last resort: find any JSON object with fact_scores
    for match in re.finditer(r"\{[^{}]*\"fact_scores\"[^{}]*\}", text):
        try:
            result = json.loads(match.group())
            if "fact_scores" in result:
                return result
        except json.JSONDecodeError:
            continue

    return None


def grade_task(broker: ModalBroker, task: dict) -> dict | None:
    """Grade a single task and return the score record."""
    messages = build_grading_messages(task)

    try:
        response = broker.chat_completion(
            model=JUDGE_MODEL,
            messages=messages,
            max_tokens=2048,
            temperature=0.0,
        )

        response_text = response.choices[0].message.content
        grade = parse_grade_response(response_text)

        if grade is None:
            logger.warning(
                "Failed to parse grade for %s: %s",
                task["grade_id"],
                response_text[:200],
            )
            return None

        # Recompute fact_f1 from fact_scores for safety
        fact_scores = grade.get("fact_scores", [])
        if fact_scores:
            fact_f1 = sum(fact_scores) / len(fact_scores)
        else:
            fact_f1 = 0.0

        return {
            "grade_id": task["grade_id"],
            "scope_id": task["scope_id"],
            "policy_id": task["policy_id"],
            "question_id": task["question_id"],
            "checkpoint_id": task["checkpoint_id"],
            "run_id": task["run_id"],
            "rubric_version": RUBRIC_VERSION,
            "judge_model": JUDGE_MODEL,
            "fact_scores": fact_scores,
            "fact_verdicts": grade.get("fact_verdicts", []),
            "fact_f1": round(fact_f1, 4),
            "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
            "completion_tokens": response.usage.completion_tokens if response.usage else 0,
        }

    except Exception as e:
        logger.error("Error grading %s: %s", task["grade_id"], e)
        return None


def main():
    parser = argparse.ArgumentParser(description="Claude grading for grid run")
    parser.add_argument("--dry-run", action="store_true", help="Print prompts without calling LLM")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of tasks to grade")
    parser.add_argument("--resume", action="store_true", help="Skip already-graded tasks")
    parser.add_argument("--input", type=str, default=None, help="Input grading tasks JSONL file")
    parser.add_argument("--output", type=str, default=None, help="Output scores JSONL file")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    input_file = Path(args.input) if args.input else GRADING_TASKS_FILE
    output_file = Path(args.output) if args.output else CLAUDE_SCORES_FILE

    # Load tasks
    tasks = []
    with open(input_file) as f:
        for line in f:
            tasks.append(json.loads(line))
    logger.info("Loaded %d grading tasks from %s", len(tasks), input_file)

    # Resume support
    already_graded = set()
    if args.resume and output_file.exists():
        with open(output_file) as f:
            for line in f:
                rec = json.loads(line)
                already_graded.add(rec["grade_id"])
        logger.info("Resuming: %d already graded", len(already_graded))
        tasks = [t for t in tasks if t.get("grade_id") not in already_graded]

    if args.limit:
        tasks = tasks[: args.limit]
    logger.info("Grading %d tasks", len(tasks))

    if args.dry_run:
        for task in tasks[:3]:
            print("=" * 70)
            print(build_grading_prompt(task))
            print("=" * 70)
        print(f"\nDry run: would grade {len(tasks)} tasks")
        return 0

    broker = ModalBroker(
        llm_base_url=LLM_BASE_URL,
        embed_base_url=EMBED_BASE_URL,
        llm_api_key="unused",
        cache_enabled=False,  # Don't cache grading
        default_extra_body={
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )

    scored = 0
    failed = 0
    start_time = time.time()

    # Open in append mode for resume safety
    mode = "a" if args.resume else "w"
    with open(output_file, mode) as f:
        for i, task in enumerate(tasks):
            result = grade_task(broker, task)
            if result:
                f.write(json.dumps(result) + "\n")
                f.flush()
                scored += 1
                if scored % 10 == 0:
                    elapsed = time.time() - start_time
                    rate = scored / elapsed * 60
                    logger.info(
                        "Progress: %d/%d scored, %d failed, %.1f/min",
                        scored,
                        len(tasks),
                        failed,
                        rate,
                    )
            else:
                failed += 1

    elapsed = time.time() - start_time
    logger.info("=" * 70)
    logger.info("GRADING COMPLETE")
    logger.info("  Scored: %d, Failed: %d, Total: %d", scored, failed, len(tasks))
    logger.info("  Time: %.1fs (%.1f answers/min)", elapsed, scored / elapsed * 60 if elapsed > 0 else 0)
    logger.info("  Output: %s", output_file)
    logger.info("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())

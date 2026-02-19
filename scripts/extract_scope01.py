#!/usr/bin/env python3
"""Extract scope 01 (cascading_failure_01) from the full benchmark dataset
into a standalone file at datasets/scope_01_only.json.

Output schema mirrors the full dataset:
  {
    "version": "...",
    "scopes": [ <single scope object> ],
    "questions": [ <questions for this scope only> ]
  }
"""

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = REPO_ROOT / "datasets" / "benchmark_dataset.json"
OUTPUT_PATH = REPO_ROOT / "datasets" / "scope_01_only.json"
TARGET_SCOPE = "cascading_failure_01"


def main() -> None:
    with open(INPUT_PATH) as f:
        data = json.load(f)

    # Extract the single scope
    matching_scopes = [s for s in data["scopes"] if s["scope_id"] == TARGET_SCOPE]
    if not matching_scopes:
        raise ValueError(f"Scope {TARGET_SCOPE!r} not found in {INPUT_PATH}")
    scope = matching_scopes[0]

    # Extract questions belonging to this scope
    scope_questions = [q for q in data["questions"] if q["scope_id"] == TARGET_SCOPE]

    # Build output with the same top-level schema
    output = {
        "version": data["version"],
        "scopes": [scope],
        "questions": scope_questions,
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)

    # Print summary
    print(f"Wrote {OUTPUT_PATH}")
    print(f"  Version:   {output['version']}")
    print(f"  Scopes:    {len(output['scopes'])} (scope_id={TARGET_SCOPE})")
    print(f"  Episodes:  {len(scope['episodes'])}")
    print(f"  Questions: {len(scope_questions)}")


if __name__ == "__main__":
    main()

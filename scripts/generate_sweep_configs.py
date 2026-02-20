#!/usr/bin/env python3
"""Generate missing config files for 8 adapters × 3 budgets × 6 scopes sweep.

Usage:
    python3 scripts/generate_sweep_configs.py [--dry-run]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

CONFIGS_DIR = Path("configs")

# Model used for agent answering on RunPod vLLM (H200 — 32k context)
MODEL = "Qwen/Qwen3-32B"

CHECKPOINTS = [5, 10, 12, 15, 20, 25, 30]

# Adapter name → config adapter field
ADAPTERS = {
    "chunked-hybrid": "sqlite-chunked-hybrid",
    "compaction": "compaction",
    "cognee": "cognee",
    "graphiti": "graphiti",
    "letta": "letta",
    "letta-sleepy": "letta-sleepy",
    "mem0-raw": "mem0-raw",
    "hindsight": "hindsight",
}

SCOPES = ["01", "02", "03", "04", "05", "06"]

BUDGETS = {
    "standard": {
        "preset": "standard",
    },
    "4k": {
        "preset": "constrained-4k",
        "max_cumulative_result_tokens": 4096,
    },
    "2k": {
        "preset": "constrained-2k",
        "max_cumulative_result_tokens": 2048,
    },
}


def config_filename(adapter_key: str, scope: str, budget: str) -> str:
    """Generate config filename matching existing naming conventions."""
    # Match existing naming: chunked-hybrid → chunked_hybrid, letta-sleepy → letta_sleepy
    base = adapter_key.replace("-", "_")
    if budget == "standard":
        return f"{base}_scope{scope}.json"
    else:
        return f"{base}_scope{scope}_{budget}.json"


def make_config(adapter_key: str, scope: str, budget: str) -> dict:
    """Generate a config dict."""
    adapter_name = ADAPTERS[adapter_key]
    return {
        "adapter": adapter_name,
        "dataset": f"datasets/scope_{scope}_only.json",
        "output_dir": "output",
        "agent_budget": BUDGETS[budget],
        "llm": {
            "provider": "openai",
            "model": MODEL,
            "seed": 42,
            "temperature": 0.0,
        },
        "checkpoints": CHECKPOINTS,
        "seed": 42,
    }


def main():
    dry_run = "--dry-run" in sys.argv

    created = 0
    skipped = 0

    for adapter_key in ADAPTERS:
        for scope in SCOPES:
            for budget in BUDGETS:
                fname = config_filename(adapter_key, scope, budget)
                path = CONFIGS_DIR / fname

                if path.exists():
                    skipped += 1
                    continue

                config = make_config(adapter_key, scope, budget)

                if dry_run:
                    print(f"  [DRY RUN] Would create: {path}")
                else:
                    path.write_text(json.dumps(config, indent=2) + "\n")
                    print(f"  Created: {path}")
                created += 1

    print(f"\nDone: {created} created, {skipped} already existed")
    print(f"Total configs: {created + skipped}")


if __name__ == "__main__":
    main()

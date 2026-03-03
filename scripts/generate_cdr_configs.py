"""Generate static benchmark configs for CDR scopes (10-12).

Creates configs/static_{adapter}_scope{NN}d.json for each adapter × scope.

Usage:
    python scripts/generate_cdr_configs.py
"""
from __future__ import annotations

import json
from pathlib import Path

CONFIGS_DIR = Path("configs")

# Self-contained adapters (no external service needed)
# Names must match the adapter registry exactly (hyphens, not underscores)
ADAPTERS = [
    "null",
    "sqlite-chunked-hybrid",
    "graphrag-light",
    "hierarchical",
    "hopping",
    "triadv1-pairs",
]

# CDR scopes
SCOPES = [10, 11, 12]

# Narrative scope checkpoints (with distractors interleaved)
# Signal ep N is at position 2N-1; checkpoint fires after 2N
CHECKPOINTS = [12, 24, 32, 40]

CONFIG_TEMPLATE = {
    "llm": {
        "provider": "static",
        "model": "Qwen/Qwen3.5-35B-A3B",
        "seed": 42,
        "temperature": 0.0,
    },
    "agent_budget": {
        "preset": "constrained-8k",
        "max_tool_calls": 25,
        "max_turns": 30,
    },
}


def main():
    CONFIGS_DIR.mkdir(exist_ok=True)
    created = 0

    for scope_num in SCOPES:
        for adapter in ADAPTERS:
            config = {
                "adapter": adapter,
                "dataset": f"datasets/scope_{scope_num}_with_distractors.json",
                "llm": {
                    **CONFIG_TEMPLATE["llm"],
                    "query_plan": f"query_plans/scope_{scope_num}.json",
                },
                "agent_budget": dict(CONFIG_TEMPLATE["agent_budget"]),
                "checkpoints": CHECKPOINTS,
            }

            # Adapter name normalization for filename
            adapter_file = adapter.replace("-", "_")
            filename = f"static_{adapter_file}_scope{scope_num}d.json"
            path = CONFIGS_DIR / filename
            path.write_text(json.dumps(config, indent=2) + "\n")
            created += 1

    print(f"Created {created} config files in {CONFIGS_DIR}/")


if __name__ == "__main__":
    main()

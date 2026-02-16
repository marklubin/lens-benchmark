"""End-to-end test for the generation pipeline.

Requires OPENAI_API_KEY or LENS_LLM_API_KEY to be set.
Mark with pytest.mark.integration so it's skipped in CI by default.
"""
from __future__ import annotations

import json
import os
import textwrap
from pathlib import Path

import pytest


needs_api_key = pytest.mark.skipif(
    not (os.environ.get("LENS_LLM_API_KEY") or os.environ.get("OPENAI_API_KEY")),
    reason="No API key available (set LENS_LLM_API_KEY or OPENAI_API_KEY)",
)


@pytest.fixture
def mini_spec_dir(tmp_path: Path) -> Path:
    """Create a tiny spec for fast e2e testing (3 episodes, 2 phases)."""
    spec_content = textwrap.dedent("""\
        scope_id: e2e_test_01
        domain: testing
        description: "Minimal e2e test scope"

        generation:
          temperature: 0.3
          seed: 42

        episodes:
          count: 3
          timeline:
            start: "2024-01-01"
            interval: "1d"
          format: "Brief daily note"
          target_words: 50

        scenario:
          setting: "A small office with two employees tracking daily tasks."
          voice: "Brief, informal notes."

        arc:
          - id: normal
            episodes: "1-2"
            description: "Normal daily activities"
            signal_density: none
          - id: event
            episodes: "3-3"
            description: "An unusual event occurs"
            signal_density: high

        noise:
          description: "Weather, lunch plans"
          examples:
            - "Sunny day, grabbed coffee"

        key_facts:
          - id: unusual_event
            fact: "printer broke down"
            first_appears: "event:1"

        questions:
          - id: e2e_q01
            checkpoint_after: 3
            type: longitudinal
            prompt: "What notable event happened?"
            ground_truth:
              canonical_answer: "The printer broke down."
              key_facts: [unusual_event]
              evidence: ["event:1"]
    """)

    scope_dir = tmp_path / "e2e_test_01"
    scope_dir.mkdir()
    (scope_dir / "spec.yaml").write_text(spec_content)
    return scope_dir


@needs_api_key
def test_generate_and_compile(mini_spec_dir: Path, tmp_path: Path) -> None:
    """Full pipeline: generate -> compile -> validate."""
    from lens.datagen.generator import generate_scope
    from lens.datagen.compiler import compile_dataset
    from lens.datasets.schema import validate_dataset

    # Generate
    gen_dir = generate_scope(
        spec_path=mini_spec_dir / "spec.yaml",
        provider="openai",
        model="gpt-4o-mini",
        verbose=True,
    )

    # Verify generated files exist
    assert (gen_dir / "episodes.json").exists()
    assert (gen_dir / "questions.json").exists()
    assert (gen_dir / "manifest.json").exists()

    # Check episodes (LLM may generate ±1 episode per phase, so check >= expected)
    episodes = json.loads((gen_dir / "episodes.json").read_text())
    assert len(episodes) >= 3, f"Expected at least 3 episodes, got {len(episodes)}"
    assert all(ep["scope_id"] == "e2e_test_01" for ep in episodes)
    assert episodes[0]["episode_id"] == "e2e_test_01_ep_001"

    # Check manifest
    manifest = json.loads((gen_dir / "manifest.json").read_text())
    assert manifest["scope_id"] == "e2e_test_01"
    assert "normal" in manifest["phases"]
    assert "event" in manifest["phases"]

    # Compile
    out_path = tmp_path / "suite.json"
    compile_dataset([mini_spec_dir], "0.1.0-test", out_path)

    dataset = json.loads(out_path.read_text())
    errors = validate_dataset(dataset)
    assert errors == [], f"Validation errors: {errors}"
    assert len(dataset["scopes"]) == 1
    assert len(dataset["questions"]) == 1

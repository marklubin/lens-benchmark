"""Synix pipeline definition for the LENS datagen pipeline.

Invoked via:
    uvx --with pyyaml synix build src/lens/datagen/synix/pipeline.py ...
    uvx --with pyyaml synix validate src/lens/datagen/synix/pipeline.py ...

This module is loaded by synix's pipeline loader which imports it and extracts
the `pipeline` variable.
"""
from __future__ import annotations

import os
import sys

# Ensure this directory is on sys.path so sibling modules (transforms, validators,
# spec_utils, prompt_utils, scoring) can be imported by name.
sys.path.insert(0, os.path.dirname(__file__))

from synix import Pipeline, SearchIndex  # noqa: E402

from transforms import (  # noqa: E402
    LoadSpec,
    PlanOutline,
    RenderSignalEpisodes,
    RenderDistractorEpisodes,
    ResolveQuestions,
    AuditKeyFacts,
)
from validators import (  # noqa: E402
    WordCount,
    ContaminationCheck,
    NaiveBaseline,
)

# ---------------------------------------------------------------------------
# Pipeline definition
# ---------------------------------------------------------------------------

pipeline = Pipeline("lens-datagen")
pipeline.source_dir = os.environ.get("LENS_SCOPE_DIR", "./sources")
pipeline.build_dir = os.environ.get("LENS_BUILD_DIR", "./build")

pipeline.llm_config = {
    "provider": os.environ.get("LENS_LLM_PROVIDER", "openai"),
    "model": os.environ.get("LENS_LLM_MODEL", "gpt-4o-mini"),
    "temperature": float(os.environ.get("LENS_LLM_TEMPERATURE", "0.3")),
    "max_tokens": int(os.environ.get("LENS_LLM_MAX_TOKENS", "16384")),
}

# ---------------------------------------------------------------------------
# Layer instances (DAG via object references, levels auto-computed)
# ---------------------------------------------------------------------------

spec = LoadSpec("spec")
outline = PlanOutline("outline", depends_on=[spec], config={
    "llm_config": {
        "model": os.environ.get("LENS_PLANNER_MODEL", "gpt-5.2"),
        "max_completion_tokens": 32768,
        "temperature": None,  # let the model use its default
    },
})
signal = RenderSignalEpisodes("signal_episodes", depends_on=[spec, outline])
distractors = RenderDistractorEpisodes("distractor_episodes", depends_on=[spec, outline])
questions = ResolveQuestions("questions", depends_on=[spec, signal])
audit = AuditKeyFacts("key_fact_audit", depends_on=[spec, signal])

pipeline.add(spec, outline, signal, distractors, questions, audit)

# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------

_validator_llm_config = {
    "provider": os.environ.get("LENS_LLM_PROVIDER", "openai"),
    "model": os.environ.get("LENS_LLM_MODEL", "gpt-4o-mini"),
    "temperature": 0.0,
}

pipeline.add_validator(WordCount(layers=[signal, distractors], min_words=340))
pipeline.add_validator(ContaminationCheck(
    layers=[signal, questions],
    llm_config=_validator_llm_config,
    max_single_ep_coverage=0.80,
))
pipeline.add_validator(NaiveBaseline(
    layers=[signal, distractors, questions],
    llm_config=_validator_llm_config,
))

# ---------------------------------------------------------------------------
# Search projection
# ---------------------------------------------------------------------------

pipeline.add(SearchIndex(
    "episode-search",
    sources=[signal, distractors],
    search=["fulltext", "semantic"],
    embedding_config={
        "provider": "fastembed",
        "model": "BAAI/bge-small-en-v1.5",
    },
))

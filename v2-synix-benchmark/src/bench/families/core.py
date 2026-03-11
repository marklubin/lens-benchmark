"""Core memory family — FoldSynthesis integration.

Adds a progressive fold transform that accumulates key observations
as episodes are processed sequentially. The resulting artifact is
a single "working memory" blob that can be injected into the agent's
system prompt under policy_core.

Only the free-form fold variant is implemented for the smoke pilot.
Structured, maintained, and faceted variants are deferred to T011.
"""
from __future__ import annotations

from synix.core.models import Layer, Pipeline
from synix.ext.fold_synthesis import FoldSynthesis

CORE_FOLD_PROMPT = """\
You are maintaining a working memory of key observations from a sequence of \
documents. Your goal is to capture important details, metrics, anomalies, and \
factual observations that might be relevant for answering questions later.

Current working memory:
{accumulated}

New document (step {step} of {total}):
{artifact}

Update your working memory. Rules:
- Preserve specific numbers, dates, names, and metrics.
- Note anomalies, contradictions, or changes from prior observations.
- Remove information that has been superseded by newer data.
- Keep the memory concise — aim for key facts, not full summaries.
- Do not editorialize or draw conclusions. Record observations only."""


def add_core_memory(pipeline: Pipeline, *, depends_on: Layer) -> FoldSynthesis:
    """Add a core-memory FoldSynthesis to the pipeline.

    Args:
        pipeline: The Synix pipeline to extend.
        depends_on: The episodes source layer.

    Returns:
        The FoldSynthesis layer (already added to the pipeline).
    """
    fold = FoldSynthesis(
        "core-memory",
        depends_on=[depends_on],
        prompt=CORE_FOLD_PROMPT,
        initial="No observations yet.",
        sort_by="label",  # alphabetical by label → chronological for signal_001, signal_002, ...
        label="core-memory",
        artifact_type="core_memory",
    )
    pipeline.add(fold)
    return fold

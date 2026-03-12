"""Core maintained family — fold + refinement pass.

Adds a consolidation/refinement step after the FoldSynthesis completes.
The refinement pass resolves contradictions, prunes redundancy, and
sharpens the working memory. Models Letta sleep-time agents and Google
sleep-time compute.
"""
from __future__ import annotations

from synix.core.models import Layer, Pipeline
from synix.ext.fold_synthesis import FoldSynthesis
from synix.ext.map_synthesis import MapSynthesis

from bench.families.core import CORE_FOLD_PROMPT

REFINEMENT_PROMPT = """\
You are refining a working memory that was built incrementally from a sequence \
of documents. The fold process may have left contradictions, redundancies, or \
stale information.

Raw working memory:
{artifact}

Refine this working memory. Rules:
- Resolve contradictions: keep the latest observation, note what changed.
- Remove redundancy: merge duplicate observations into single entries.
- Prune stale information that has been fully superseded.
- Sharpen vague observations into concrete claims where the data supports it.
- Preserve ALL specific numbers, dates, names, and metrics.
- Preserve the temporal ordering of events — don't collapse timeline.
- Do NOT add new information or draw conclusions not supported by the entries.
- Aim for a concise, high-signal working memory with no noise."""


def add_core_maintained(pipeline: Pipeline, *, depends_on: Layer) -> MapSynthesis:
    """Add a FoldSynthesis + refinement MapSynthesis to the pipeline.

    The fold runs first (same as policy_core), then a MapSynthesis
    refinement pass cleans up the fold output.

    Args:
        pipeline: The Synix pipeline to extend.
        depends_on: The episodes source layer.

    Returns:
        The MapSynthesis refinement layer (already added to the pipeline).
    """
    fold = FoldSynthesis(
        "core-maintained-fold",
        depends_on=[depends_on],
        prompt=CORE_FOLD_PROMPT,
        initial="No observations yet.",
        sort_by="label",
        label="core-maintained-fold",
        artifact_type="core_memory_raw",
    )
    pipeline.add(fold)

    refine = MapSynthesis(
        "core-maintained",
        depends_on=[fold],
        prompt=REFINEMENT_PROMPT,
        label_fn=lambda _: "core-maintained",
        artifact_type="core_maintained",
    )
    pipeline.add(refine)
    return refine

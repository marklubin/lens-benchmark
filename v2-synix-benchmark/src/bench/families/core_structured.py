"""Core structured family — structured observation fold.

Mastra / ACE pattern: instead of free-form narrative, the fold emits
dated, prioritized, categorized observations. Each observation is a
discrete fact, not a paragraph of prose.

The accumulated state is a structured event log, not a narrative summary.
"""
from __future__ import annotations

from synix.core.models import Layer, Pipeline
from synix.ext.fold_synthesis import FoldSynthesis

CORE_STRUCTURED_FOLD_PROMPT = """\
You are maintaining a structured observation log from a sequence of documents. \
Each entry is a discrete, dated, categorized observation — not prose.

Current observation log:
{accumulated}

New document (step {step} of {total}):
{artifact}

Update the observation log. Format each entry as:

[STEP-{step}] <CATEGORY> | <observation> | priority: <high/medium/low>

Categories: METRIC, ENTITY, EVENT, ANOMALY, RELATIONSHIP, CHANGE

Rules:
- One entry per distinct observation. No paragraphs.
- Preserve exact numbers, dates, names, amounts.
- Mark contradictions with prior entries: [SUPERSEDES STEP-N]
- Mark anomalies or threshold violations as priority: high
- Remove entries that are fully superseded by newer data.
- Do NOT editorialize. Record what was observed, not what it means.
- Keep entries terse — aim for one line per observation."""


def add_core_structured(pipeline: Pipeline, *, depends_on: Layer) -> FoldSynthesis:
    """Add a structured-observation FoldSynthesis to the pipeline.

    Same architecture as policy_core, different prompt: emits structured
    dated observations instead of free-form narrative.

    Args:
        pipeline: The Synix pipeline to extend.
        depends_on: The episodes source layer.

    Returns:
        The FoldSynthesis layer (already added to the pipeline).
    """
    fold = FoldSynthesis(
        "core-structured",
        depends_on=[depends_on],
        prompt=CORE_STRUCTURED_FOLD_PROMPT,
        initial="No observations yet.",
        sort_by="label",
        label="core-structured",
        artifact_type="core_structured",
    )
    pipeline.add(fold)
    return fold

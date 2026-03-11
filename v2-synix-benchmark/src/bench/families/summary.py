"""Summary family — GroupSynthesis + ReduceSynthesis integration.

Groups episodes into windows, summarizes each group, then reduces
all group summaries into a single overview. The resulting artifact
can be injected into the agent's system prompt under policy_summary.
"""
from __future__ import annotations

import re
from collections.abc import Callable

from synix.core.models import Artifact, Layer, Pipeline
from synix.ext.group_synthesis import GroupSynthesis
from synix.ext.reduce_synthesis import ReduceSynthesis

GROUP_SUMMARY_PROMPT = """\
Summarize these documents, preserving key evidence, metrics, and factual details.

Documents:
{artifacts}

Write a concise summary that captures the most important observations. \
Preserve specific numbers, dates, names, and anomalies. \
Do not editorialize — report what was observed."""

REDUCE_SUMMARY_PROMPT = """\
Merge these group summaries into a single unified overview.

Group summaries:
{artifacts}

Total groups: {count}

Produce a comprehensive summary that preserves all key evidence and metrics. \
Remove redundancy but keep distinct observations. Be concise and factual."""

# Group size for episode windows
DEFAULT_GROUP_SIZE = 5


def _episode_group_fn(art: Artifact) -> str:
    """Group episodes into windows by ordinal parsed from label.

    Handles both Synix-munged labels (t-text-signal_001) and legacy
    labels (scope_id_ep_003, scope_id_distractor_theme_002).
    Ordinal is always the trailing digits.
    """
    m = re.search(r"(\d+)$", art.label)
    ordinal = int(m.group(1)) if m else 0
    group_idx = ordinal // DEFAULT_GROUP_SIZE
    return f"group_{group_idx}"


def add_summary(
    pipeline: Pipeline,
    *,
    depends_on: Layer,
    group_size: int = DEFAULT_GROUP_SIZE,
) -> ReduceSynthesis:
    """Add GroupSynthesis + ReduceSynthesis to the pipeline for summaries.

    Args:
        pipeline: The Synix pipeline to extend.
        depends_on: The episodes source layer.
        group_size: Number of episodes per group window.

    Returns:
        The ReduceSynthesis layer (already added to the pipeline).
    """
    # Update module-level group size if caller overrides
    group_fn: Callable[[Artifact], str]
    if group_size == DEFAULT_GROUP_SIZE:
        group_fn = _episode_group_fn
    else:
        def group_fn(art: Artifact) -> str:
            m = re.search(r"(\d+)$", art.label)
            ordinal = int(m.group(1)) if m else 0
            return f"group_{ordinal // group_size}"

    group = GroupSynthesis(
        "episode-groups",
        depends_on=[depends_on],
        group_by=group_fn,
        prompt=GROUP_SUMMARY_PROMPT,
        artifact_type="group_summary",
    )

    reduce = ReduceSynthesis(
        "summary",
        depends_on=[group],
        prompt=REDUCE_SUMMARY_PROMPT,
        label="summary",
        artifact_type="summary",
    )

    pipeline.add(group, reduce)
    return reduce

"""Core faceted family — 4 parallel cognitive facet folds + merge.

Triad / cognitive decomposition pattern: maintain 4 orthogonal facet
stores (entity, relation, event, cause) in parallel, then merge into
a unified context block for the agent.

Each facet fold processes the same episode stream but with a different
extraction lens.
"""
from __future__ import annotations

from synix.core.models import Layer, Pipeline
from synix.ext.fold_synthesis import FoldSynthesis
from synix.ext.reduce_synthesis import ReduceSynthesis

# --- Facet fold prompts ---

ENTITY_FOLD_PROMPT = """\
You are maintaining an ENTITY REGISTER — a catalog of people, organizations, \
systems, products, and other named entities observed across documents.

Current entity register:
{accumulated}

New document (step {step} of {total}):
{artifact}

Update the entity register. For each entity:
- Name / identifier
- Type (person, org, system, product, location, concept)
- Key attributes (role, status, metrics associated with them)
- First appearance (step number)

Rules:
- One entry per entity. Update existing entries, don't duplicate.
- Preserve exact names, titles, identifiers.
- Note attribute changes: "was X, now Y (step N)"
- Remove entities that are no longer relevant.
- Do NOT list generic concepts — only specific named entities."""

RELATION_FOLD_PROMPT = """\
You are maintaining a RELATIONSHIP MAP — tracking connections between entities \
observed across documents.

Current relationship map:
{accumulated}

New document (step {step} of {total}):
{artifact}

Update the relationship map. For each relationship:
- Entity A → Entity B
- Relationship type (works-for, owns, reports-to, depends-on, conflicts-with, etc.)
- Evidence (specific fact or metric)
- First observed (step number)

Rules:
- One entry per directional relationship.
- Update relationships that change — note the change.
- Remove relationships that are explicitly ended.
- Relationships must be grounded in specific evidence, not inferred."""

EVENT_FOLD_PROMPT = """\
You are maintaining an EVENT TIMELINE — a chronological log of significant \
events, actions, and state changes observed across documents.

Current event timeline:
{accumulated}

New document (step {step} of {total}):
{artifact}

Update the event timeline. For each event:
- Step/date when observed
- What happened (one sentence)
- Key metrics or quantities involved
- Entities involved

Rules:
- One entry per discrete event. Keep chronological order.
- Preserve exact timestamps, amounts, measurements.
- Mark events that contradict or supersede prior events.
- Remove events that are no longer significant given later developments.
- Do NOT include routine observations — only notable events."""

CAUSE_FOLD_PROMPT = """\
You are maintaining a CAUSAL ANALYSIS — tracking cause-effect relationships, \
patterns, and anomalies observed across documents.

Current causal analysis:
{accumulated}

New document (step {step} of {total}):
{artifact}

Update the causal analysis. For each causal observation:
- Pattern or anomaly observed
- Possible cause (if evidence supports it)
- Supporting evidence (specific data points)
- Confidence: confirmed / suspected / speculative

Rules:
- Only record causal links with specific evidence. No speculation without data.
- Update confidence as more evidence accumulates.
- Note when a suspected cause is confirmed or refuted.
- Remove causal hypotheses that have been clearly refuted.
- Preserve the specific metrics that support each claim."""

MERGE_PROMPT = """\
Merge these four cognitive facet analyses into a unified working memory.

Facet analyses:
{artifacts}

Total facets: {count}

Produce a unified working memory that integrates all facets. Structure:

## Entities & Relationships
Key actors, their roles, and how they connect.

## Timeline
Chronological sequence of significant events.

## Patterns & Causes
Observed patterns, anomalies, and their evidenced causes.

## Key Metrics
Specific numbers, dates, amounts that are important.

Rules:
- Cross-reference across facets — entities should link to events and causes.
- Resolve any contradictions between facets.
- Preserve ALL specific data points.
- Be concise — this is a reference, not a narrative."""


def add_core_faceted(pipeline: Pipeline, *, depends_on: Layer) -> ReduceSynthesis:
    """Add 4 parallel FoldSynthesis facets + ReduceSynthesis merge.

    Args:
        pipeline: The Synix pipeline to extend.
        depends_on: The episodes source layer.

    Returns:
        The ReduceSynthesis merge layer (already added to the pipeline).
    """
    facets = [
        ("entity", ENTITY_FOLD_PROMPT, "core_facet_entity"),
        ("relation", RELATION_FOLD_PROMPT, "core_facet_relation"),
        ("event", EVENT_FOLD_PROMPT, "core_facet_event"),
        ("cause", CAUSE_FOLD_PROMPT, "core_facet_cause"),
    ]

    fold_layers = []
    for facet_name, prompt, artifact_type in facets:
        fold = FoldSynthesis(
            f"core-faceted-{facet_name}",
            depends_on=[depends_on],
            prompt=prompt,
            initial="No observations yet.",
            sort_by="label",
            label=f"core-faceted-{facet_name}",
            artifact_type=artifact_type,
        )
        pipeline.add(fold)
        fold_layers.append(fold)

    merge = ReduceSynthesis(
        "core-faceted",
        depends_on=fold_layers,
        prompt=MERGE_PROMPT,
        label="core-faceted",
        artifact_type="core_faceted",
    )
    pipeline.add(merge)
    return merge

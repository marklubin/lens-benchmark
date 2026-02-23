"""Triad Memory Protocol v1 adapters for LENS.

v1: agents maintain an **object store** instead of plain text notebooks.
Every concept follows a fixed 5-field meta-schema:
  identity | schema | interface | state | lifecycle

Three operations per episode:
  INSTANTIATE — create a new object
  UPDATE      — modify an existing object's state/interface
  ACCOMMODATE — restructure the model itself (primary learning signal)

Two adapters:
- triadv1-panel:  4 agents (E/R/V/X), 4 parallel consults + synthesis
- triadv1-pairs:  4 agents (E/R/V/X), 6 parallel cross-reference pair consults + synthesis
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import combinations

from lens.adapters.base import SearchResult
from lens.adapters.registry import register_adapter
from lens.adapters.triad import (
    FACETS_4,
    _TriadBase,
    _complete,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# v1 recording system prompts (per-facet object store agents)
# ---------------------------------------------------------------------------

_V1_RECORD_ENTITY_SYSTEM = """\
You are the ENTITY object-store agent. You maintain a structured store of entity objects.

Each object in your store MUST follow this 5-field meta-schema:
- identity: unique name/label for this entity
- schema: what type of thing it is, its defining traits and roles
- interface: how this entity presents itself, observable behaviors
- state: current status, properties, values that change over time
- lifecycle: creation context, key transitions, developmental stage

Per episode, apply these operations:
- INSTANTIATE: create a new object when a genuinely new entity appears
- UPDATE: modify fields of an existing object when new information arrives
- ACCOMMODATE: restructure your model when observations contradict your schema \
(merge objects that are the same entity, split one that has diverged, rewrite schema)

Do NOT track: relationships between entities, temporal sequences, or causal explanations.
Focus ONLY on what entities exist and what makes each one distinct."""

_V1_RECORD_RELATION_SYSTEM = """\
You are the RELATION object-store agent. You maintain a structured store of relationship objects.

Each object in your store MUST follow this 5-field meta-schema:
- identity: unique label for this relationship (e.g. "Maya-Leo bond")
- schema: type of relationship, its nature and defining characteristics
- interface: how the relationship manifests — observable interactions, communication patterns
- state: current status, strength, quality, direction of the relationship
- lifecycle: when it formed, key transitions, developmental stage

Per episode, apply these operations:
- INSTANTIATE: create a new object when a genuinely new relationship appears
- UPDATE: modify fields of an existing object when new information arrives
- ACCOMMODATE: restructure your model when observations contradict your schema \
(reclassify a relationship type, merge connections, rewrite when your model was wrong)

Do NOT track: entity traits or identities, event timelines, or causal explanations.
Focus ONLY on connections between entities."""

_V1_RECORD_EVENT_SYSTEM = """\
You are the EVENT object-store agent. You maintain a structured store of event objects.

Each object in your store MUST follow this 5-field meta-schema:
- identity: unique label for this event (e.g. "coffee-spill-incident")
- schema: type of event, its nature and what category it belongs to
- interface: observable details — what happened, who was involved, what changed
- state: outcome, resolution status, consequences that are visible
- lifecycle: when it occurred, duration, whether it is ongoing or resolved

Per episode, apply these operations:
- INSTANTIATE: create a new object when a genuinely new event or state change occurs
- UPDATE: modify fields of an existing object when follow-up information arrives
- ACCOMMODATE: restructure your model when observations contradict your schema \
(reinterpret what actually happened, merge events that are one, reclassify)

Do NOT track: entity traits, relationship dynamics, or causal explanations.
Focus ONLY on what happened and when."""

_V1_RECORD_CAUSE_SYSTEM = """\
You are the CAUSE object-store agent. You maintain a structured store of causal-link objects.

Each object in your store MUST follow this 5-field meta-schema:
- identity: unique label for this causal link (e.g. "fear-drove-dropout")
- schema: type of causal mechanism — motivation, consequence, enablement, prevention
- interface: how this cause manifests — observable evidence linking cause to effect
- state: current confidence level, whether the link is confirmed, suspected, or revised
- lifecycle: when the link was identified, whether it has been validated or overturned

Per episode, apply these operations:
- INSTANTIATE: create a new object when a genuinely new causal connection is revealed
- UPDATE: modify fields when new evidence strengthens, weakens, or refines a link
- ACCOMMODATE: restructure your model when observations contradict your causal schema \
(the real cause turns out different, merge redundant links, reclassify mechanism)

Do NOT track: entity traits, relationship dynamics, or event timelines.
Focus ONLY on why things happened — the explanatory links between events."""

_V1_RECORD_SYSTEMS: dict[str, str] = {
    "entity": _V1_RECORD_ENTITY_SYSTEM,
    "relation": _V1_RECORD_RELATION_SYSTEM,
    "event": _V1_RECORD_EVENT_SYSTEM,
    "cause": _V1_RECORD_CAUSE_SYSTEM,
}

_V1_RECORD_USER = """\
CURRENT OBJECT STORE:
{store}

NEW EPISODE [{episode_id}] ({timestamp}):
{text}

Update the object store. Apply INSTANTIATE, UPDATE, or ACCOMMODATE operations as needed.
Return the complete updated object store."""

# ---------------------------------------------------------------------------
# v1 consult system prompts
# ---------------------------------------------------------------------------

_V1_CONSULT_ENTITY_SYSTEM = """\
You are the ENTITY object-store specialist. You have a structured store of entity objects.
Answer the question using ONLY information from your object store.
Reference specific objects by their identity field.
Highlight any ACCOMMODATE events as key insights where your model was restructured."""

_V1_CONSULT_RELATION_SYSTEM = """\
You are the RELATION object-store specialist. You have a structured store of relationship objects.
Answer the question using ONLY information from your object store.
Reference specific objects by their identity field.
Highlight any ACCOMMODATE events as key insights where your model was restructured."""

_V1_CONSULT_EVENT_SYSTEM = """\
You are the EVENT object-store specialist. You have a structured store of event objects.
Answer the question using ONLY information from your object store.
Reference specific objects by their identity field.
Highlight any ACCOMMODATE events as key insights where your model was restructured."""

_V1_CONSULT_CAUSE_SYSTEM = """\
You are the CAUSE object-store specialist. You have a structured store of causal-link objects.
Answer the question using ONLY information from your object store.
Reference specific objects by their identity field.
Highlight any ACCOMMODATE events as key insights where your model was restructured."""

_V1_CONSULT_SYSTEMS: dict[str, str] = {
    "entity": _V1_CONSULT_ENTITY_SYSTEM,
    "relation": _V1_CONSULT_RELATION_SYSTEM,
    "event": _V1_CONSULT_EVENT_SYSTEM,
    "cause": _V1_CONSULT_CAUSE_SYSTEM,
}

_V1_CONSULT_USER = """\
OBJECT STORE:
{store}

QUESTION: {question}

Answer based on your object store. Reference specific objects by identity.
Highlight any ACCOMMODATE events as key insights."""

# ---------------------------------------------------------------------------
# v1 pair cross-reference prompts
# ---------------------------------------------------------------------------

_V1_PAIR_DESCRIPTIONS: dict[tuple[str, str], str] = {
    ("entity", "relation"): "how entities are connected — who these entities are to each other",
    ("entity", "event"): "what entities did and experienced — actor-action connections",
    ("entity", "cause"): "what role entities played in causing outcomes — agent-cause attribution",
    ("relation", "event"): "how relationships manifested in events — relational dynamics in action",
    ("relation", "cause"): "why relationships formed, changed, or broke — relational causation",
    ("event", "cause"): "causal chains between events — why things happened when they did",
}

_V1_PAIR_CONSULT_SYSTEM = """\
You are a cross-reference specialist analyzing the intersection of {facet_a} and {facet_b} \
object stores: {pair_desc}.
Answer using information from BOTH stores. Find connections and patterns \
that emerge only when both perspectives are combined.
Highlight any ACCOMMODATE events from either store as key insights."""

_V1_PAIR_CONSULT_USER = """\
{facet_a_upper} OBJECT STORE:
{store_a}

{facet_b_upper} OBJECT STORE:
{store_b}

QUESTION: {question}

Analyze the intersection of these two perspectives. What insights emerge from combining them?"""

_V1_PAIR_SYNTHESIS_SYSTEM = """\
You are a synthesis agent. You receive analyses from 6 cross-reference specialists, \
each examining a different pairing of object stores (entity, relation, event, cause).

Combine into a single coherent answer. Highlight convergent findings across multiple pairings.
Highlight ACCOMMODATE events as key insights."""


def _build_v1_pair_synthesis_user(
    question: str,
    pairs: list[tuple[str, str]],
    pair_responses: dict[tuple[str, str], str],
) -> str:
    parts = [f"QUESTION: {question}"]
    for pair in pairs:
        label = f"{pair[0].upper()}\u00d7{pair[1].upper()}"
        response = pair_responses.get(pair, "(no response)")
        parts.append(f"\n{label} SPECIALIST:\n{response}")
    parts.append(
        "\nSynthesize these cross-reference analyses into a single coherent answer. "
        "Highlight findings that converge across multiple pairings."
    )
    return "\n".join(parts)

# ---------------------------------------------------------------------------
# v1 synthesis (used by panel)
# ---------------------------------------------------------------------------

_V1_SYNTHESIS_SYSTEM = """\
You are a synthesis agent. You receive specialist responses from 4 object-store agents \
(entity, relation, event, cause) and must combine them into a single coherent answer.

Follow cross-references between objects across stores.
Highlight ACCOMMODATE events — these are key insights where agents restructured their understanding.
Resolve contradictions using the most recently accommodated model."""


def _build_v1_synthesis_user(
    question: str,
    facets: tuple[str, ...],
    facet_responses: dict[str, str],
) -> str:
    parts = [f"QUESTION: {question}"]
    for facet in facets:
        response = facet_responses.get(facet, "(no response)")
        parts.append(f"\n{facet.upper()} STORE SPECIALIST:\n{response}")
    parts.append(
        "\nSynthesize these perspectives into a single, coherent answer. "
        "Follow cross-references between objects. "
        "Highlight ACCOMMODATE events as key insights."
    )
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Shared recording helper
# ---------------------------------------------------------------------------

def _v1_record_episode_parallel(
    adapter: _TriadBase,
    ep: dict,
    max_tokens: int = 3000,
) -> None:
    """Update all facet object stores in parallel for a single episode."""
    facets = adapter._notebook_keys

    def update_facet(facet: str) -> tuple[str, str]:
        result = _complete(
            adapter._oai,
            adapter._model,
            system=_V1_RECORD_SYSTEMS[facet],
            user=_V1_RECORD_USER.format(
                store=adapter._notebooks[facet],
                episode_id=ep["episode_id"],
                timestamp=ep["timestamp"],
                text=ep["text"],
            ),
            max_tokens=max_tokens,
        )
        return facet, result

    with ThreadPoolExecutor(max_workers=len(facets)) as pool:
        futures = {pool.submit(update_facet, f): f for f in facets}
        for fut in as_completed(futures):
            facet = futures[fut]
            try:
                _, updated = fut.result()
                adapter._notebooks[facet] = updated
            except Exception as e:
                logger.error(
                    "%s record failed for facet=%s episode=%s: %s",
                    adapter._adapter_label, facet, ep["episode_id"], e,
                )


# ---------------------------------------------------------------------------
# Shared parallel consult + synthesis (used by panel)
# ---------------------------------------------------------------------------

def _v1_parallel_search(
    adapter: _TriadBase,
    query: str,
    limit: int | None = None,
) -> list[SearchResult]:
    """Consult all facets in parallel, then synthesize."""
    facets = adapter._notebook_keys
    if not adapter._oai or all(
        adapter._notebooks.get(f) == "(empty)" for f in facets
    ):
        return adapter._fallback_search(limit)

    facet_responses: dict[str, str] = {}

    def consult_facet(facet: str) -> tuple[str, str]:
        result = _complete(
            adapter._oai,
            adapter._model,
            system=_V1_CONSULT_SYSTEMS[facet],
            user=_V1_CONSULT_USER.format(
                store=adapter._notebooks[facet],
                question=query,
            ),
            max_tokens=2000,
        )
        return facet, result

    try:
        with ThreadPoolExecutor(max_workers=len(facets)) as pool:
            futures = {pool.submit(consult_facet, f): f for f in facets}
            for fut in as_completed(futures):
                facet = futures[fut]
                try:
                    _, response = fut.result()
                    facet_responses[facet] = response
                except Exception as e:
                    logger.error(
                        "%s consult failed for facet=%s: %s",
                        adapter._adapter_label, facet, e,
                    )
                    facet_responses[facet] = "(no response)"

        answer = _complete(
            adapter._oai,
            adapter._model,
            system=_V1_SYNTHESIS_SYSTEM,
            user=_build_v1_synthesis_user(query, facets, facet_responses),
            max_tokens=2500,
        )
        return [SearchResult(
            ref_id=f"{adapter._adapter_label}-answer",
            text=answer[:500],
            score=1.0,
            metadata={
                "type": adapter._adapter_label.replace("-", "_"),
                "full_answer": answer,
            },
        )]
    except Exception as e:
        logger.error("%s search failed: %s", adapter._adapter_label, e)
        return adapter._fallback_search(limit)


# ---------------------------------------------------------------------------
# Panel v1 (4 parallel specialist agents)
# ---------------------------------------------------------------------------

@register_adapter("triadv1-panel")
class TriadV1PanelAdapter(_TriadBase):
    """Four parallel object-store agents (entity, relation, event, cause)."""

    _notebook_keys = FACETS_4
    _adapter_label = "triadv1-panel"

    def prepare(self, scope_id: str, checkpoint: int) -> None:
        if not self._episodes:
            return
        try:
            self._init_client()
        except RuntimeError as e:
            logger.error("%s init failed: %s", self._adapter_label, e)
            return

        for ep in self._episodes:
            _v1_record_episode_parallel(self, ep)

    def search(
        self,
        query: str,
        filters: dict | None = None,
        limit: int | None = None,
    ) -> list[SearchResult]:
        return _v1_parallel_search(self, query, limit)


# ---------------------------------------------------------------------------
# Pairs v1 (6 parallel cross-reference pair agents)
# ---------------------------------------------------------------------------

@register_adapter("triadv1-pairs")
class TriadV1PairsAdapter(_TriadBase):
    """Six parallel cross-reference agents — one per C(4,2) facet pairing."""

    _notebook_keys = FACETS_4
    _adapter_label = "triadv1-pairs"

    def prepare(self, scope_id: str, checkpoint: int) -> None:
        if not self._episodes:
            return
        try:
            self._init_client()
        except RuntimeError as e:
            logger.error("%s init failed: %s", self._adapter_label, e)
            return

        for ep in self._episodes:
            _v1_record_episode_parallel(self, ep)

    def search(
        self,
        query: str,
        filters: dict | None = None,
        limit: int | None = None,
    ) -> list[SearchResult]:
        facets = self._notebook_keys
        if not self._oai or all(
            self._notebooks.get(f) == "(empty)" for f in facets
        ):
            return self._fallback_search(limit)

        pairs = list(combinations(facets, 2))
        pair_responses: dict[tuple[str, str], str] = {}

        def consult_pair(pair: tuple[str, str]) -> tuple[tuple[str, str], str]:
            fa, fb = pair
            result = _complete(
                self._oai,
                self._model,
                system=_V1_PAIR_CONSULT_SYSTEM.format(
                    facet_a=fa,
                    facet_b=fb,
                    pair_desc=_V1_PAIR_DESCRIPTIONS[pair],
                ),
                user=_V1_PAIR_CONSULT_USER.format(
                    facet_a_upper=fa.upper(),
                    store_a=self._notebooks[fa],
                    facet_b_upper=fb.upper(),
                    store_b=self._notebooks[fb],
                    question=query,
                ),
                max_tokens=2000,
            )
            return pair, result

        try:
            with ThreadPoolExecutor(max_workers=len(pairs)) as pool:
                futures = {pool.submit(consult_pair, p): p for p in pairs}
                for fut in as_completed(futures):
                    pair = futures[fut]
                    try:
                        _, response = fut.result()
                        pair_responses[pair] = response
                    except Exception as e:
                        logger.error(
                            "%s consult failed for pair=%s\u00d7%s: %s",
                            self._adapter_label, pair[0], pair[1], e,
                        )
                        pair_responses[pair] = "(no response)"

            answer = _complete(
                self._oai,
                self._model,
                system=_V1_PAIR_SYNTHESIS_SYSTEM,
                user=_build_v1_pair_synthesis_user(query, pairs, pair_responses),
                max_tokens=2500,
            )
            return [SearchResult(
                ref_id=f"{self._adapter_label}-answer",
                text=answer[:500],
                score=1.0,
                metadata={
                    "type": self._adapter_label.replace("-", "_"),
                    "full_answer": answer,
                },
            )]
        except Exception as e:
            logger.error("%s search failed: %s", self._adapter_label, e)
            return self._fallback_search(limit)


# ---------------------------------------------------------------------------
# Pairs-fused v1 (all 6 pairings in a single call)
# ---------------------------------------------------------------------------

_V1_PAIRS_FUSED_SYSTEM = """\
You are a cross-reference analyst. You have 4 object stores (entity, relation, event, cause) \
and must analyze ALL 6 pairwise intersections between them in a single pass.

For each pairing, find connections and patterns that emerge only when both perspectives \
are combined. Then synthesize into a single coherent answer.
Highlight ACCOMMODATE events from any store as key insights."""

_V1_PAIRS_FUSED_USER = """\
ENTITY OBJECT STORE:
{entity_store}

RELATION OBJECT STORE:
{relation_store}

EVENT OBJECT STORE:
{event_store}

CAUSE OBJECT STORE:
{cause_store}

CROSS-REFERENCE PAIRINGS TO ANALYZE:
1. ENTITY\u00d7RELATION: how entities are connected \u2014 who these entities are to each other
2. ENTITY\u00d7EVENT: what entities did and experienced \u2014 actor-action connections
3. ENTITY\u00d7CAUSE: what role entities played in causing outcomes \u2014 agent-cause attribution
4. RELATION\u00d7EVENT: how relationships manifested in events \u2014 relational dynamics in action
5. RELATION\u00d7CAUSE: why relationships formed, changed, or broke \u2014 relational causation
6. EVENT\u00d7CAUSE: causal chains between events \u2014 why things happened when they did

QUESTION: {question}

Analyze all 6 pairings, then synthesize into a single coherent answer."""


@register_adapter("triadv1-pairs-fused")
class TriadV1PairsFusedAdapter(_TriadBase):
    """All 6 C(4,2) cross-reference pairings answered in a single LLM call."""

    _notebook_keys = FACETS_4
    _adapter_label = "triadv1-pairs-fused"

    def prepare(self, scope_id: str, checkpoint: int) -> None:
        if not self._episodes:
            return
        try:
            self._init_client()
        except RuntimeError as e:
            logger.error("%s init failed: %s", self._adapter_label, e)
            return

        for ep in self._episodes:
            _v1_record_episode_parallel(self, ep)

    def search(
        self,
        query: str,
        filters: dict | None = None,
        limit: int | None = None,
    ) -> list[SearchResult]:
        facets = self._notebook_keys
        if not self._oai or all(
            self._notebooks.get(f) == "(empty)" for f in facets
        ):
            return self._fallback_search(limit)

        try:
            answer = _complete(
                self._oai,
                self._model,
                system=_V1_PAIRS_FUSED_SYSTEM,
                user=_V1_PAIRS_FUSED_USER.format(
                    entity_store=self._notebooks["entity"],
                    relation_store=self._notebooks["relation"],
                    event_store=self._notebooks["event"],
                    cause_store=self._notebooks["cause"],
                    question=query,
                ),
                max_tokens=2500,
            )
            return [SearchResult(
                ref_id=f"{self._adapter_label}-answer",
                text=answer[:500],
                score=1.0,
                metadata={
                    "type": self._adapter_label.replace("-", "_"),
                    "full_answer": answer,
                },
            )]
        except Exception as e:
            logger.error("%s search failed: %s", self._adapter_label, e)
            return self._fallback_search(limit)

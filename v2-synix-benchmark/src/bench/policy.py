"""Policy registry — creates PolicyManifest instances for each benchmark policy.

Each policy defines:
  - Which artifact families are visible to the agent
  - Which search surfaces are queryable
  - Retrieval caps (max results, max context tokens)
  - Whether derived artifacts (core memory, summary) are injected into context

Policies consume a compiled bank — they never trigger rebuilds.
"""
from __future__ import annotations

from bench.schemas import FusionConfig, PolicyManifest, RetrievalCaps

# Default retrieval limits
_DEFAULT_MAX_RESULTS = 10
_DEFAULT_MAX_CONTEXT_TOKENS = 8192


def make_null_policy(version: str) -> PolicyManifest:
    """Null baseline — agent sees only the question, no tools, no context."""
    return PolicyManifest(
        policy_manifest_id=f"null-{version}",
        policy_id="null",
        visible_artifact_families=[],
        query_surfaces=[],
        fusion=FusionConfig(method="none"),
        retrieval_caps=RetrievalCaps(max_results=0, max_context_tokens=0),
        version=version,
    )


def make_base_policy(version: str) -> PolicyManifest:
    """Base policy — search over episodes + chunks, no derived artifacts."""
    return PolicyManifest(
        policy_manifest_id=f"policy_base-{version}",
        policy_id="policy_base",
        visible_artifact_families=["episodes", "chunks"],
        query_surfaces=["keyword", "semantic"],
        fusion=FusionConfig(method="rrf", parameters={"k": 60}),
        retrieval_caps=RetrievalCaps(
            max_results=_DEFAULT_MAX_RESULTS,
            max_context_tokens=_DEFAULT_MAX_CONTEXT_TOKENS,
        ),
        version=version,
    )


def make_core_policy(version: str) -> PolicyManifest:
    """Core memory policy — base search + core-memory blob in system prompt."""
    return PolicyManifest(
        policy_manifest_id=f"policy_core-{version}",
        policy_id="policy_core",
        visible_artifact_families=["episodes", "chunks", "core_memory"],
        query_surfaces=["keyword", "semantic"],
        fusion=FusionConfig(method="rrf", parameters={"k": 60}),
        retrieval_caps=RetrievalCaps(
            max_results=_DEFAULT_MAX_RESULTS,
            max_context_tokens=_DEFAULT_MAX_CONTEXT_TOKENS,
        ),
        version=version,
    )


def make_summary_policy(version: str) -> PolicyManifest:
    """Summary policy — base search + summary blob in system prompt."""
    return PolicyManifest(
        policy_manifest_id=f"policy_summary-{version}",
        policy_id="policy_summary",
        visible_artifact_families=["episodes", "chunks", "summary"],
        query_surfaces=["keyword", "semantic"],
        fusion=FusionConfig(method="rrf", parameters={"k": 60}),
        retrieval_caps=RetrievalCaps(
            max_results=_DEFAULT_MAX_RESULTS,
            max_context_tokens=_DEFAULT_MAX_CONTEXT_TOKENS,
        ),
        version=version,
    )


POLICY_REGISTRY: dict[str, type] = {
    "null": make_null_policy,
    "policy_base": make_base_policy,
    "policy_core": make_core_policy,
    "policy_summary": make_summary_policy,
}


def create_policy(policy_id: str, version: str) -> PolicyManifest:
    """Create a PolicyManifest by policy_id from the registry."""
    factory = POLICY_REGISTRY.get(policy_id)
    if factory is None:
        raise ValueError(f"Unknown policy_id: {policy_id!r}. Available: {list(POLICY_REGISTRY)}")
    return factory(version)

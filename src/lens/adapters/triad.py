"""Triad Memory Protocol — shared base infrastructure.

Provides _TriadBase (lifecycle, storage, fallback search), _complete (LLM helper),
and FACETS_4 (the 4-facet decomposition). Concrete adapters live in triad_v1.py.
"""
from __future__ import annotations

import logging
import os

try:
    from openai import OpenAI as _OpenAI
except ImportError:
    _OpenAI = None  # type: ignore[assignment,misc]

from lens.adapters.base import (
    CapabilityManifest,
    Document,
    MemoryAdapter,
    SearchResult,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Facet definitions
# ---------------------------------------------------------------------------

FACETS_4 = ("entity", "relation", "event", "cause")


# ---------------------------------------------------------------------------
# LLM helper
# ---------------------------------------------------------------------------

def _strip_provider_prefix(model: str) -> str:
    if "/" in model and model.startswith(("together/", "openai/")):
        return model.split("/", 1)[1]
    return model


def _complete(
    client: _OpenAI,  # type: ignore[valid-type]
    model: str,
    system: str,
    user: str,
    max_tokens: int = 1500,
) -> str:
    """Single chat completion call."""
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        max_tokens=max_tokens,
        temperature=0.0,
    )
    return resp.choices[0].message.content or ""


# ---------------------------------------------------------------------------
# Shared base
# ---------------------------------------------------------------------------

class _TriadBase(MemoryAdapter):
    """Shared logic for all triad variants."""

    _notebook_keys: tuple[str, ...] = ()
    _adapter_label: str = ""

    def __init__(self) -> None:
        self._episodes: list[dict] = []
        self._notebooks: dict[str, str] = {}
        self._oai: _OpenAI | None = None  # type: ignore[valid-type]
        self._model: str = ""

    def reset(self, scope_id: str) -> None:
        self._episodes = []
        self._notebooks = {k: "(empty)" for k in self._notebook_keys}
        self._oai = None
        self._model = ""

    def ingest(
        self,
        episode_id: str,
        scope_id: str,
        timestamp: str,
        text: str,
        meta: dict | None = None,
    ) -> None:
        self._episodes.append({
            "episode_id": episode_id,
            "scope_id": scope_id,
            "timestamp": timestamp,
            "text": text,
            "meta": meta or {},
        })

    def _init_client(self) -> None:
        if _OpenAI is None:
            raise RuntimeError("openai package required for triad adapters")
        api_key = (
            os.environ.get("LENS_LLM_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
            or "dummy"
        )
        base_url = (
            os.environ.get("LENS_LLM_API_BASE")
            or os.environ.get("OPENAI_BASE_URL")
        )
        model_raw = os.environ.get("LENS_LLM_MODEL", "gpt-4o-mini")
        self._model = _strip_provider_prefix(model_raw)

        kwargs: dict = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self._oai = _OpenAI(**kwargs)

    def retrieve(self, ref_id: str) -> Document | None:
        if ref_id.startswith("notebook-"):
            facet = ref_id[len("notebook-"):]
            if facet in self._notebooks:
                return Document(ref_id=ref_id, text=self._notebooks[facet])
            return None
        for ep in self._episodes:
            if ep["episode_id"] == ref_id:
                return Document(
                    ref_id=ref_id,
                    text=ep["text"],
                    metadata=ep.get("meta", {}),
                )
        return None

    def get_capabilities(self) -> CapabilityManifest:
        return CapabilityManifest(
            search_modes=["synthesis"],
            max_results_per_search=1,
        )

    def get_synthetic_refs(self) -> list[tuple[str, str]]:
        refs = []
        for key, content in self._notebooks.items():
            if content and content != "(empty)":
                refs.append((f"notebook-{key}", content))
        return refs

    def _fallback_search(self, limit: int | None) -> list[SearchResult]:
        cap = limit or 10
        return [
            SearchResult(
                ref_id=ep["episode_id"],
                text=ep["text"][:500],
                score=0.5,
                metadata=ep.get("meta", {}),
            )
            for ep in self._episodes[:cap]
        ]

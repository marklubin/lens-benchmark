from __future__ import annotations

from lens.adapters.base import (
    CapabilityManifest,
    Document,
    MemoryAdapter,
    SearchResult,
)
from lens.adapters.registry import register_adapter


@register_adapter("null")
class NullAdapter(MemoryAdapter):
    """Null baseline adapter. Returns empty results for all queries.

    Serves as the score floor — a system that stores nothing and returns nothing.
    All metrics should score ~0 against this adapter.
    """

    def reset(self, scope_id: str) -> None:
        pass

    def ingest(
        self,
        episode_id: str,
        scope_id: str,
        timestamp: str,
        text: str,
        meta: dict | None = None,
    ) -> None:
        pass

    def search(
        self,
        query: str,
        filters: dict | None = None,
        limit: int | None = None,
    ) -> list[SearchResult]:
        return []

    def retrieve(self, ref_id: str) -> Document | None:
        return None

    def get_capabilities(self) -> CapabilityManifest:
        return CapabilityManifest(
            search_modes=["semantic"],
            filter_fields=[],
            max_results_per_search=10,
            supports_date_range=False,
            extra_tools=[],
        )

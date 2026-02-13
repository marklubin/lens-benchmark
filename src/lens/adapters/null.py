from __future__ import annotations

from lens.adapters.base import MemoryAdapter
from lens.adapters.registry import register_adapter
from lens.core.models import Hit, Insight


@register_adapter("null")
class NullAdapter(MemoryAdapter):
    """Null baseline adapter. Returns empty results for all queries.

    Serves as the score floor — a system that stores nothing and returns nothing.
    All metrics should score 0 against this adapter.
    """

    def reset(self, persona_id: str) -> None:
        pass

    def ingest(
        self,
        episode_id: str,
        persona_id: str,
        timestamp: str,
        text: str,
        meta: dict | None = None,
    ) -> None:
        pass

    def refresh(self, persona_id: str, checkpoint: int) -> None:
        pass

    def core(self, persona_id: str, k: int, checkpoint: int) -> list[Insight]:
        return []

    def search(self, persona_id: str, query: str, k: int, checkpoint: int) -> list[Hit]:
        return []

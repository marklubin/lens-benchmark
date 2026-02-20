from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Adapter data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SearchResult:
    """A single search result returned by the adapter."""

    ref_id: str
    text: str
    score: float = 0.0
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "ref_id": self.ref_id,
            "text": self.text,
            "score": self.score,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict) -> SearchResult:
        return cls(
            ref_id=d["ref_id"],
            text=d["text"],
            score=d.get("score", 0.0),
            metadata=d.get("metadata", {}),
        )


@dataclass(frozen=True)
class Document:
    """A full document retrieved by reference ID."""

    ref_id: str
    text: str
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "ref_id": self.ref_id,
            "text": self.text,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Document:
        return cls(
            ref_id=d["ref_id"],
            text=d["text"],
            metadata=d.get("metadata", {}),
        )


@dataclass
class FilterField:
    """Describes a filterable field exposed by the adapter."""

    name: str
    field_type: str = "string"  # "string", "number", "date", "boolean"
    description: str = ""
    enum_values: list[str] | None = None


@dataclass
class ExtraTool:
    """An extended tool exposed by the adapter."""

    name: str
    description: str
    parameters: dict  # JSON Schema


@dataclass
class CapabilityManifest:
    """Describes what the adapter supports so the agent can adapt."""

    search_modes: list[str] = field(default_factory=lambda: ["semantic"])
    filter_fields: list[FilterField] = field(default_factory=list)
    max_results_per_search: int = 10
    supports_date_range: bool = False
    extra_tools: list[ExtraTool] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "search_modes": self.search_modes,
            "filter_fields": [
                {
                    "name": f.name,
                    "field_type": f.field_type,
                    "description": f.description,
                    "enum_values": f.enum_values,
                }
                for f in self.filter_fields
            ],
            "max_results_per_search": self.max_results_per_search,
            "supports_date_range": self.supports_date_range,
            "extra_tools": [
                {"name": t.name, "description": t.description, "parameters": t.parameters}
                for t in self.extra_tools
            ],
        }


# ---------------------------------------------------------------------------
# Abstract adapter
# ---------------------------------------------------------------------------


class MemoryAdapter(ABC):
    """Abstract base class for memory system adapters.

    Adapters expose data-loading methods (called by the runner) and
    tool methods (exposed to the agent via the tool bridge).
    """

    # --- Data loading (called by runner) ---

    @abstractmethod
    def reset(self, scope_id: str) -> None:
        """Clear all state for a scope. Called once before episode stream begins."""

    @abstractmethod
    def ingest(
        self,
        episode_id: str,
        scope_id: str,
        timestamp: str,
        text: str,
        meta: dict | None = None,
    ) -> None:
        """Ingest a single episode. Must complete within 200ms, no LLM calls allowed."""

    def prepare(self, scope_id: str, checkpoint: int) -> None:
        """Optional hook called before questions at a checkpoint.

        Replaces the old refresh(). Adapters may use this to build indices
        or consolidate memories.
        """

    # --- Mandatory tools (exposed to agent) ---

    @abstractmethod
    def search(
        self,
        query: str,
        filters: dict | None = None,
        limit: int | None = None,
    ) -> list[SearchResult]:
        """Search memory for relevant information."""

    @abstractmethod
    def retrieve(self, ref_id: str) -> Document | None:
        """Retrieve a full document by reference ID."""

    @abstractmethod
    def get_capabilities(self) -> CapabilityManifest:
        """Return the adapter's capability manifest."""

    # --- Synthetic refs (adapter-generated documents) ---

    def get_synthetic_refs(self) -> list[tuple[str, str]]:
        """Return (ref_id, text) pairs for adapter-generated documents.

        These are documents produced by the adapter (e.g. summaries,
        consolidations) that don't correspond to original episodes but
        should be treated as valid references for evidence grounding.
        Called by the runner after prepare() to register in the vault.
        """
        return []

    # --- Caching protocol (optional) ---

    def get_cache_state(self) -> dict | None:
        """Return serializable state for caching. None = not cacheable."""
        return None

    def restore_cache_state(self, state: dict) -> bool:
        """Restore from cached state. Return True if successful."""
        return False

    # --- Optional extended tools ---

    def call_extended_tool(self, tool_name: str, arguments: dict) -> object:
        """Dispatch a call to an adapter-defined extended tool."""
        msg = f"Unknown extended tool: {tool_name!r}"
        raise NotImplementedError(msg)

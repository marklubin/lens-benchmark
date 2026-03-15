# Writing a Memory Adapter

```
┌──────────────────────────────────────────────┐
│  LENS // Adapter Guide                       │
└──────────────────────────────────────────────┘
```

This guide walks you through implementing a LENS adapter — the bridge between your memory system and the benchmark.

---

## 1. The Interface

All adapters subclass `MemoryAdapter` from `src/lens/adapters/base.py`. There are five required methods and several optional hooks.

### Required Methods

```python
class MemoryAdapter(ABC):
    @abstractmethod
    def reset(self, scope_id: str) -> None:
        """Clear all state for a scope. Called once before episode stream begins."""

    @abstractmethod
    def ingest(self, episode_id: str, scope_id: str,
               timestamp: str, text: str, meta: dict | None = None) -> None:
        """Ingest a single episode. Must complete within 200ms, no LLM calls allowed."""

    @abstractmethod
    def search(self, query: str, filters: dict | None = None,
               limit: int | None = None) -> list[SearchResult]:
        """Search memory for relevant information."""

    @abstractmethod
    def retrieve(self, ref_id: str) -> Document | None:
        """Retrieve a full document by reference ID."""

    @abstractmethod
    def get_capabilities(self) -> CapabilityManifest:
        """Return the adapter's capability manifest."""
```

### Data Types

```python
@dataclass(frozen=True)
class SearchResult:
    ref_id: str      # Unique identifier for the result
    text: str        # Snippet or full text
    score: float     # Relevance score (higher = more relevant)
    metadata: dict   # Optional metadata

@dataclass(frozen=True)
class Document:
    ref_id: str      # Matches episode_id or synthetic ref
    text: str        # Full document text
    metadata: dict   # Optional metadata

@dataclass
class CapabilityManifest:
    search_modes: list[str]           # e.g., ["semantic", "keyword", "hybrid"]
    filter_fields: list[FilterField]  # Filterable metadata fields
    max_results_per_search: int       # Max results per query
    supports_date_range: bool         # Whether date-range filtering works
    extra_tools: list[ExtraTool]      # Additional tools exposed to agent
```

### How the Agent Uses Your Adapter

The agent discovers capabilities at runtime via `get_capabilities()` and adapts:
- If you declare `search_modes: ["semantic", "keyword"]`, the agent may choose between them
- If you expose `filter_fields`, the agent can filter by metadata
- If you expose `extra_tools`, the agent gets additional tool calls
- Systems with richer interfaces get a natural advantage

---

## 2. Minimal Example

The simplest working adapter — in-memory list with substring search:

```python
from lens.adapters.base import (
    CapabilityManifest,
    Document,
    MemoryAdapter,
    SearchResult,
)
from lens.adapters.registry import register_adapter


@register_adapter("my-memory")
class MyMemoryAdapter(MemoryAdapter):
    def __init__(self):
        self.store: dict[str, str] = {}

    def reset(self, scope_id: str) -> None:
        self.store.clear()

    def ingest(self, episode_id: str, scope_id: str,
               timestamp: str, text: str, meta: dict | None = None) -> None:
        self.store[episode_id] = text

    def search(self, query: str, filters: dict | None = None,
               limit: int | None = None) -> list[SearchResult]:
        limit = limit or 10
        results = []
        for eid, text in self.store.items():
            if query.lower() in text.lower():
                results.append(SearchResult(
                    ref_id=eid,
                    text=text[:200],
                    score=1.0,
                ))
        return results[:limit]

    def retrieve(self, ref_id: str) -> Document | None:
        text = self.store.get(ref_id)
        return Document(ref_id=ref_id, text=text) if text else None

    def get_capabilities(self) -> CapabilityManifest:
        return CapabilityManifest(
            search_modes=["keyword"],
            max_results_per_search=10,
        )
```

This beats null (which returns nothing) but won't score well — substring matching misses semantic connections. It's a starting point.

---

## 3. Reference Implementation: SQLite Adapter

The `sqlite` adapter (`src/lens/adapters/sqlite.py`) is the full-featured reference:

- **FTS5 full-text search** for keyword matching
- **Optional embedding similarity** for semantic search
- **Metadata filtering** via SQL WHERE clauses
- **Hybrid search** combining both signals

Study this adapter when building something more sophisticated. Key patterns:

- `ingest()` writes to both FTS and embedding tables in a single transaction
- `search()` dispatches to keyword/semantic/hybrid based on query analysis
- `get_capabilities()` dynamically reports available modes based on configuration
- `prepare()` hook builds any deferred indices before checkpoint questions

---

## 4. Registration

### Built-in Adapter (in the LENS repo)

Use the `@register_adapter` decorator:

```python
from lens.adapters.registry import register_adapter

@register_adapter("my-memory")
class MyMemoryAdapter(MemoryAdapter):
    ...
```

Then add the import to `_ensure_builtins()` in `src/lens/adapters/registry.py`:

```python
try:
    import lens.adapters.my_memory  # noqa: F401
except ImportError:
    pass  # optional dependency not installed
```

### External Adapter (separate package)

Register via the `lens.adapters` entry point group in your `pyproject.toml`:

```toml
[project.entry-points."lens.adapters"]
my-memory = "my_package.adapter:MyMemoryAdapter"
```

LENS discovers external adapters automatically via `importlib.metadata.entry_points`.

---

## 5. Testing Your Adapter

### Smoke Test

```bash
# Verify it registers
uv run lens adapters  # should list "my-memory"

# Run smoke test
uv run lens smoke --adapter my-memory
```

### Run Against S01

```bash
uv run lens compile --scope-dir datasets/scopes/01_cascading_failure \
  --output data_s01.json

uv run lens run --dataset data_s01.json --adapter my-memory --out output/test/
```

### Score and Compare

```bash
# Score your run
uv run lens score --run output/test/ --judge-model your-model

# Compare against null baseline
uv run lens compare output/test/ output/null_baseline/
```

Your adapter should beat null on most metrics. Key things to check:
- `evidence_grounding` > 0.5 (cited ref_ids actually exist)
- `fact_recall` > null baseline (you're finding relevant information)
- `evidence_coverage` > 0 (retrieved episodes include required evidence)

**Note**: These metrics apply to V1 adapter evaluation. V2 policy evaluation uses Fact F1 (per-fact binary grading, then F1 across all key facts per question). See the Scoring section in [README.md](../../README.md) for details on both systems.

### Unit Tests

Write tests for your adapter's specific logic:

```python
# tests/unit/test_my_adapter.py
from lens.adapters.my_memory import MyMemoryAdapter

def test_ingest_and_search():
    adapter = MyMemoryAdapter()
    adapter.reset("test")
    adapter.ingest("ep1", "test", "2024-01-01", "latency increased to 400ms")
    results = adapter.search("latency")
    assert len(results) > 0
    assert results[0].ref_id == "ep1"

def test_retrieve():
    adapter = MyMemoryAdapter()
    adapter.reset("test")
    adapter.ingest("ep1", "test", "2024-01-01", "test content")
    doc = adapter.retrieve("ep1")
    assert doc is not None
    assert doc.text == "test content"

def test_retrieve_missing():
    adapter = MyMemoryAdapter()
    adapter.reset("test")
    assert adapter.retrieve("nonexistent") is None
```

---

## 6. Optional Hooks

### `prepare(scope_id, checkpoint)`

Called before checkpoint questions. Use this for deferred work like building indices, consolidating memories, or running background processing:

```python
def prepare(self, scope_id: str, checkpoint: int) -> None:
    self.rebuild_index()  # e.g., re-rank, compress, summarize
```

### `get_synthetic_refs()`

Return adapter-generated documents (summaries, consolidations) that don't correspond to original episodes but should be valid evidence:

```python
def get_synthetic_refs(self) -> list[tuple[str, str]]:
    return [
        ("summary-checkpoint-10", self.generate_summary(checkpoint=10)),
    ]
```

### `call_extended_tool(tool_name, arguments)`

Handle calls to tools declared in your `CapabilityManifest.extra_tools`:

```python
def get_capabilities(self) -> CapabilityManifest:
    return CapabilityManifest(
        search_modes=["semantic"],
        extra_tools=[
            ExtraTool(
                name="timeline",
                description="Get events in chronological order for a topic",
                parameters={"type": "object", "properties": {
                    "topic": {"type": "string"},
                    "limit": {"type": "integer"},
                }},
            )
        ],
    )

def call_extended_tool(self, tool_name: str, arguments: dict) -> object:
    if tool_name == "timeline":
        return self.get_timeline(arguments["topic"], arguments.get("limit", 10))
    raise NotImplementedError(f"Unknown tool: {tool_name}")
```

### Caching Protocol

Implement `get_cache_state()` and `restore_cache_state()` to avoid re-ingesting episodes across runs:

```python
def get_cache_state(self) -> dict | None:
    return {"store": dict(self.store), "version": 1}

def restore_cache_state(self, state: dict) -> bool:
    if state.get("version") != 1:
        return False
    self.store = state["store"]
    return True
```

---

## 7. Constraints

- **`ingest()` must complete within 200ms** — no LLM calls during ingestion
- **`search()` and `retrieve()` are called by the agent** — they must be fast enough for interactive use
- **`ref_id` values must be consistent** — what you return from `search()` must work in `retrieve()`
- **No direct episode access** — the agent can only reach episodes through your adapter's search/retrieve interface. This is enforced by the `EpisodeVault` anticheat system.

---

## 8. Submitting Your Adapter

See [CONTRIBUTING.md](../../CONTRIBUTING.md) for the full PR process.

**PR checklist**:
- [ ] Subclasses `MemoryAdapter` with all required methods
- [ ] Registered via `@register_adapter` decorator
- [ ] Passes smoke test
- [ ] Runs successfully against S01
- [ ] Unit tests for adapter logic
- [ ] Added to `_ensure_builtins()` in `registry.py` (for built-ins)
- [ ] Dependencies added to `pyproject.toml` optional-dependencies (if any)

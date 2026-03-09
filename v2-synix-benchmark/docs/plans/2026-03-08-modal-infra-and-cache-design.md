# V2 Modal Infra + Response Cache Design

**Date**: 2026-03-08
**Status**: approved
**Covers**: T003 (Modal broker, cache, idempotent call layer)

## Objective

Establish rock-solid inference infrastructure and response caching as the foundation for all v2 benchmark work. No benchmark logic, scoring, or study execution until this layer is proven.

## Architecture

```
Benchmark Runner
    |
    +-- ModalBroker (single entry point for all inference)
    |   +-- LLMBroker  -> Modal vLLM (Qwen3.5-35B-A3B-FP8, H100 auto-scale)
    |   +-- EmbedBroker -> Modal embed (gte-modernbert-base, T4)
    |
    +-- ResponseCache (SQLite, content-addressed)
        +-- llm_responses table
        +-- embed_responses table
```

## Ownership Split

### Shared infrastructure (repo-root, already exists)

```
infra/modal/
  llm_server.py              # Qwen3.5-35B-A3B-FP8 on H100, vLLM nightly, 262K context
  embedding_server.py        # gte-modernbert-base on T4, sentence-transformers
  qwen3_permissive.jinja     # Qwen3-Coder XML tool-call template
```

These are model-agnostic OpenAI-compatible servers. V2 consumes them as endpoints. The only change: parameterize `min_containers` so run scripts can control scaling.

### V2 client-side code

```
v2-synix-benchmark/src/
  cache.py                   # ResponseCache (SQLite)
  broker.py                  # ModalBroker (LLM + embed)

v2-synix-benchmark/tests/
  test_cache.py
  test_broker.py
```

## Models

- **LLM**: Qwen3.5-35B-A3B-FP8 on H100 via vLLM nightly with qwen3_coder tool-call parser
- **Embedding**: Alibaba-NLP/gte-modernbert-base (768 dims) on T4 via sentence-transformers
- Both unchanged from v1. No confounds introduced.

## H100 Scaling

H100 count is a runtime parameter, not a deployment constant. Modal auto-scales containers based on load. Run scripts control concurrency to drive the desired parallelism. `min_containers` is set before study execution and zeroed after.

## ResponseCache

Single SQLite database, WAL mode, `check_same_thread=False`.

### Tables

**`llm_responses`**:

| Column | Type | Notes |
|--------|------|-------|
| key | TEXT PRIMARY KEY | sha256 of canonical request |
| model | TEXT | |
| request | TEXT (JSON) | full request kwargs |
| response | TEXT (JSON) | raw response |
| created_at | REAL | unix timestamp |
| latency_ms | REAL | |
| prompt_tokens | INTEGER | from response usage |
| completion_tokens | INTEGER | from response usage |
| hit_count | INTEGER DEFAULT 0 | |

**`embed_responses`**:

| Column | Type | Notes |
|--------|------|-------|
| key | TEXT PRIMARY KEY | sha256 of canonical request |
| model | TEXT | |
| request | TEXT (JSON) | |
| response | TEXT (JSON) | |
| created_at | REAL | |
| latency_ms | REAL | |
| token_count | INTEGER | estimated from input |
| hit_count | INTEGER DEFAULT 0 | |

### Cache Key Design

**LLM calls**:
```python
sha256(json.dumps({
    "model": ...,
    "messages": [...],
    "tools": [...],          # if present
    "tool_choice": ...,      # if present
    "temperature": ...,
    "seed": ...,
    "max_tokens": ...,
}, sort_keys=True))
```

**Embedding calls**:
```python
sha256(json.dumps({
    "model": ...,
    "input": [...],          # normalized to list[str]
}, sort_keys=True))
```

Excluded from LLM keys: `stream`, `timeout`, `top_p` (ephemeral, not correctness-affecting).

## ModalBroker

Single inference gate. All model calls in the entire v2 benchmark go through this.

### Responsibilities

- Cache lookup before every call
- On miss: call Modal endpoint, capture raw response + latency + tokens, store in cache
- Retry with exponential backoff on transient errors (5xx, rate limit, connection)
- Surface permanent errors immediately (4xx)
- Token and cost accounting via cache table queries
- Optional cache bypass flag for forcing fresh calls

### Interface

```python
class ModalBroker:
    def __init__(self, cache: ResponseCache, llm_base_url: str, embed_base_url: str, ...): ...
    def chat_completion(self, **kwargs) -> ChatCompletion: ...
    def embed(self, input: list[str], model: str = ...) -> list[list[float]]: ...
    def stats(self) -> BrokerStats: ...
```

### Cost Config

Cost-per-token rates passed at broker init. Accounting queries run against the cache SQLite DB directly. No separate accounting system.

## Testing Strategy

### test_cache.py

- Cache miss stores response, cache hit returns it
- Same request produces same key (deterministic)
- Different request produces different key
- Concurrent writes don't corrupt (WAL mode)
- Token/latency metadata preserved on store
- Hit count increments on repeated access
- Stats queries work (total tokens, cost, hit rate)
- Invalid/corrupt entries handled gracefully (logged, not crashed)

### test_broker.py

- LLM call routes through cache
- Embed call routes through cache
- Retry on transient errors (mock 503 then 200)
- Permanent errors surface immediately (mock 400)
- Token accounting matches response metadata
- Cache bypass forces fresh call even on cache hit

All tests use mocked HTTP. No Modal dependency for unit tests.

## Decisions

- D004 (Modal only for inference) is satisfied: both LLM and embed go through Modal
- Cache is client-side SQLite, not server-side Modal Volume: inspectable, portable, queryable
- Embedding model unchanged from v1 (gte-modernbert-base): no confound introduced
- H100 scaling is runtime, not deployment: run scripts control concurrency

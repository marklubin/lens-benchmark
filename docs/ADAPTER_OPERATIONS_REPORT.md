# Memory Adapter Operations Report

**Date**: 2026-02-23
**Context**: LENS benchmark constrained-budget validation (Phase 1 + Phase 2), 48 runs across 9 adapters
**Infrastructure**: Together AI serverless (Qwen3-235B judge, Llama-3.3-70B entity extraction, gte-modernbert-base embeddings)

---

## Executive Summary

Of the 6 "heavy" memory adapters tested (cognee, graphiti, mem0-raw, hindsight, letta, letta-sleepy), only **letta-sleepy** delivers consistent value over lightweight baselines. The rest suffer from some combination of: excessive setup complexity, brittle APIs requiring monkey-patches, slow entity extraction that blows past time budgets, and poor retrieval quality that fails to justify the operational overhead.

Meanwhile, **compaction** — a naive "re-summarize everything" baseline with zero external dependencies — leads all adapters at NBA 0.790. This is the uncomfortable finding: the sophisticated graph/vector/temporal systems tested here lose to a single LLM call.

---

## Adapter-by-Adapter Operational Assessment

### Tier 1: Operationally Sound

#### Compaction (NBA 0.790/0.787)
- **Setup**: Zero. No containers, no databases, no proxies. Just an OpenAI-compatible LLM endpoint.
- **How it works**: Buffers all episodes in memory, calls LLM once per checkpoint to produce a single summary. Agent searches the summary.
- **Failure modes**: Token overflow if episodes exceed context window. Never observed in practice (scope 01 = 30 episodes).
- **Verdict**: The baseline every system must beat. Embarrassingly simple, embarrassingly effective.

#### Chunked-Hybrid (NBA 0.466/0.479)
- **Setup**: None. In-process SQLite + embedding API calls. No containers.
- **How it works**: Splits episodes into ~150-word chunks, embeds independently, stores in SQLite. Search fuses BM25 (FTS5) + cosine similarity via RRF.
- **Failure modes**: Embedding API rate limits on large scopes. Never crashed.
- **Operational notes**: `batch_retrieve` tool is critical — without it the agent makes 40+ tool calls per question and blows its budget. With it, 3-4 calls.
- **Verdict**: Solid lightweight option. No operational headaches. The `batch_retrieve` optimization was the single biggest agent-efficiency win across the project.

---

### Tier 2: Functional But Questionable Value

#### Letta-Sleepy (NBA 0.667/0.693)
- **Setup**: Letta server container (8283) + embedding proxy (7878). Two processes to manage.
- **How it works**: Base Letta archival storage + "sleep consolidation" in prepare() that synthesizes cross-episode patterns via LLM. Synthesis is prepended to every search result.
- **Failure modes**: Embedding proxy routing (Ollama → Together AI fallback had a bug where `URLError` wasn't caught, only `HTTPError`). Letta server occasionally returns passages as objects vs. lists (adapter handles both).
- **Workarounds needed**: Custom embedding proxy (`letta_embed_proxy.py`) to route embeddings and chat completions to the right backend. Agent naming must be predictable for cleanup between runs.
- **Verdict**: Best heavy adapter. The delta/causal sleep variant (variant 3) produces a navigation document the agent actually uses. Worth the setup cost if you need graph-aware retrieval. But it's still 0.12 NBA behind compaction.

#### Letta (NBA 0.453/0.631)
- **Setup**: Same as letta-sleepy (container + proxy).
- **How it works**: Pure archival passage storage + semantic search. No consolidation.
- **Failure modes**: Same as letta-sleepy minus sleep-specific issues.
- **Verdict**: The sleep consolidation is what makes letta-sleepy competitive. Without it, base Letta is mid-pack. The 4K→2K degradation (0.631→0.453) shows it relies on raw retrieval volume rather than smart summarization.

#### Graphiti (NBA 0.270/0.517)
- **Setup**: FalkorDB container (Redis-compatible graph DB on port 6379) + Together AI for LLM + embeddings.
- **How it works**: LLM entity extraction in prepare() builds a knowledge graph in FalkorDB. Search uses hybrid BM25 + cosine on graph edges, reranked by episode mentions.
- **Failure modes**: Together AI rejects embedding batches >1MB (HTTP 413). Fixed with 20-item batch chunking via monkey-patched `_ChunkedEmbedder`. Thread-hosted event loop needed because graphiti is async-only.
- **Workarounds needed**: Monkey-patched embedder batching. Dedicated asyncio thread. Scope-isolated database naming.
- **Verdict**: Huge 2K→4K swing (0.270→0.517) suggests graph retrieval helps but needs token budget to express it. Entity extraction in prepare() is slow but doesn't crash. The monkey-patching is annoying but stable once done.

---

### Tier 3: Operationally Painful

#### Cognee (NBA ?/0.477)
- **Setup**: No container (embedded databases), but requires LanceDB + Kuzu + SQLite internally. Together AI for LLM (Llama-3.3-70B) + embeddings.
- **How it works**: `cognee.add()` ingests raw text, `cognee.cognify()` runs LLM entity extraction + graph construction. Search queries the CHUNKS collection in LanceDB.
- **Why it's painful**:
  1. **cognify() takes 30-60+ minutes per scope** on Together AI serverless. Our 2K run is at 28 minutes and counting. First attempt timed out at 900s. Second at 1800s.
  2. **ACL wrapping broke search results** — cognee 0.5.2+ wraps results in `{"dataset_id": UUID, "search_result": [...]}` instead of a flat list. Silently returned 0.0 scores until we debugged it.
  3. **Kuzu lock contention** — concurrent cognify + search operations deadlock the embedded graph DB. Had to disable ACL entirely.
  4. **LiteLLM `dimensions` parameter** — Together AI rejects it. Requires monkey-patching `litellm.aembedding()` to strip the kwarg.
  5. **tiktoken KeyError** — cognee's tokenizer doesn't recognize Together AI model names. Requires patching `tiktoken.encoding_for_model()` with a fallback.
  6. **Scopes 02-06 all timed out** in the full sweep. Cognee is scope-01-only in practice.
- **Workarounds needed**: 6 separate monkey-patches/env overrides. Aggressive database cleanup between runs (delete `.cognee_system/` and `.cognee_cache/`). Thread-hosted async event loop.
- **Verdict**: When it works (scope 01, 4K budget), cognee achieves 0.477 NBA with excellent evidence_coverage (0.6319). But getting it to work requires fighting the library at every turn. The 60-minute cognify time makes multi-scope evaluation impractical. Not production-ready.

#### Mem0-Raw (NBA 0.406/0.386)
- **Setup**: Qdrant container (port 6333) + Together AI for embeddings + LLM (even though raw mode doesn't use the LLM, mem0 instantiates it anyway).
- **How it works**: `Memory.add(text, infer=False)` stores episodes as vectors. Search is pure cosine similarity.
- **Why it's painful**:
  1. **Requires LLM config even when unused** — mem0 always initializes an LLM client, even with `infer=False`. Missing LLM env vars = crash on startup.
  2. **`dimensions` parameter rejection** — Same Together AI issue as cognee/graphiti. Requires monkey-patching the embedder to strip `dimensions` from OpenAI API calls. Set `MEM0_EMBED_NO_DIMS=1`.
  3. **Response shape instability** — `mem0.search()` returns `{"results": [...]}` in some versions and a bare list in others. Adapter must handle both.
  4. **Qdrant dependency for basic vector search** — Running a full vector DB container for what is essentially cosine similarity over a few hundred embeddings is overkill.
- **Verdict**: The simplest heavy adapter conceptually, but the mem0 library itself is fragile. Version updates break the API surface. The monkey-patch tax is high for what amounts to vanilla vector search that chunked-hybrid does better without any containers.

#### Hindsight (NBA 0.168/0.168)
- **Setup**: 17.3 GB container image. Largest of any adapter. Contains PostgreSQL + TEMPR engine + local processing. Requires 8 env vars with non-obvious naming (`HINDSIGHT_API_EMBEDDINGS_OPENAI_*` — the `_OPENAI_` suffix is required even when using Together AI).
- **How it works**: Episodes ingested via `client.retain()` (one at a time — batch endpoint triggers HTTP 413). TEMPR retrieval fuses semantic + BM25 + graph + temporal signals via RRF. Has a native `memory_reflect` tool for longitudinal synthesis.
- **Why it's painful**:
  1. **retain() = 20-100s per episode** — entity extraction runs on every ingest call. 30 episodes × 50s = 25 minutes of consolidation overhead per checkpoint.
  2. **Batch ingest is broken** — `retain_batch()` sends all episodes to the embedding API at once, triggering HTTP 413. Had to revert to sequential `retain()`.
  3. **Budget compliance collapse** — At constrained budgets, the consolidation overhead eats the agent's entire token/time budget. Budget compliance = 0.2083 (19/24 violations). The agent barely gets to answer questions.
  4. **Text reformatting destroys evidence** — Hindsight reformats ingested text internally, making it impossible for the agent to cite original episode content. Evidence_coverage = 0.1667.
  5. **RRF scores are unusably low** — TEMPR's fused scores come back in the ~0.03 range on operational metrics. The agent interprets this as low confidence, searches repeatedly, burns more tokens.
  6. **Agent ignores memory_reflect** — The native longitudinal synthesis tool exists but the agent rarely invokes it. Unclear if this is a prompt issue or tool-description issue.
  7. **Env var naming is a trap** — `HINDSIGHT_API_EMBEDDINGS_PROVIDER=openai` requires the corresponding keys under `HINDSIGHT_API_EMBEDDINGS_OPENAI_*`, not `HINDSIGHT_API_EMBEDDINGS_*`. This cost hours of debugging.
- **Verdict**: NBA 0.168 is statistically indistinguishable from the null adapter (0.150/0.168). You get null-adapter performance from a 17.3GB container that takes minutes to start, requires 8 env vars, and crashes on batch ingest. Hindsight's TEMPR retrieval may work in other contexts, but under constrained token budgets it is strictly worse than doing nothing. The entity extraction overhead is the core problem — it provides no retrievable benefit when the agent's budget is exhausted before it can query.

---

## Cross-Cutting Operational Issues

### Together AI Serverless Latency
Every heavy adapter is bottlenecked by Together AI's serverless inference. LLM calls take 30-128s when the model is cold. This compounds with entity extraction (cognee, graphiti, hindsight all do per-episode LLM calls in prepare()). Dedicated GPU hosting would cut these times 5-10x but costs ~$3/hr.

### The Monkey-Patch Tax
4 of 6 heavy adapters require runtime monkey-patches to work with Together AI:

| Adapter | Patches Required |
|---------|-----------------|
| Cognee | `litellm.aembedding` (strip dimensions), `tiktoken.encoding_for_model` (fallback), `max_tokens` injection |
| Graphiti | `OpenAIEmbedder` (batch chunking to avoid 413) |
| Mem0 | Embedder `dimensions` stripping (`MEM0_EMBED_NO_DIMS`) |
| Hindsight | None (but env var naming is a trap) |
| Letta | None (but requires custom embed proxy) |

This reflects a shared problem: these libraries assume direct OpenAI API access and break when routed through compatible-but-not-identical providers.

### Container Sprawl
Running the full adapter suite requires 4 containers + 1 proxy:

| Service | Port | Image Size | Purpose |
|---------|------|-----------|---------|
| FalkorDB | 6379 | ~200MB | Graphiti graph DB |
| Qdrant | 6333 | ~100MB | Mem0 vector DB |
| Hindsight | 8888 | 17.3GB | TEMPR engine + PostgreSQL |
| Letta | 8283 | ~2GB | MemGPT server |
| Embed proxy | 7878 | N/A | Routes Letta embeddings |

Plus Together AI API access for all of them. That's 5 processes + 1 external API to run a benchmark that compaction does better with zero infrastructure.

### Async/Threading Complexity
Cognee and Graphiti are async-only libraries being called from synchronous adapter code. Both require dedicated daemon threads hosting their own event loops (`_AsyncRunner` pattern). This adds ~50 lines of boilerplate per adapter and introduces subtle bugs around event loop lifecycle.

---

## Bug Log

Every bug below was encountered during benchmark runs. "Silent" means the adapter returned results (or zero results) without raising an error — we only discovered the problem by inspecting scores or output.

| # | Adapter | Bug | Symptom | Root Cause | Fix | Silent? |
|---|---------|-----|---------|------------|-----|---------|
| 1 | Cognee | ACL search result wrapping | All scores 0.0 — hard-gated on evidence_grounding | Cognee 0.5.2+ wraps search results in `{"dataset_id": UUID, "search_result": [...]}` instead of flat list. Adapter parsed `getattr()` on dicts, got nothing. | Check for dict key `"search_result"` in `_extract_chunks()` + disable ACL via `ENABLE_BACKEND_ACCESS_CONTROL=false` | **Yes** — returned empty results, no error raised |
| 2 | Cognee | LiteLLM `dimensions` rejection | `cognee.cognify()` crashes during entity extraction | Together AI embedding API rejects the `dimensions` kwarg. Cognee's `LiteLLMEmbeddingEngine` always sends it. | Monkey-patch `litellm.aembedding()` to strip `dimensions` from kwargs | No |
| 3 | Cognee | tiktoken KeyError | Crash on startup | `tiktoken.encoding_for_model()` doesn't recognize Together AI model names like `meta-llama/Llama-3.3-70B-Instruct-Turbo` | Patch `tiktoken.encoding_for_model()` with fallback to `cl100k_base` | No |
| 4 | Cognee | Kuzu database lock contention | Deadlock during concurrent cognify + search | Cognee's embedded Kuzu graph DB doesn't support concurrent writers. ACL backend triggers reads during writes. | Disable ACL entirely (`ENABLE_BACKEND_ACCESS_CONTROL=false`) | No — hangs indefinitely |
| 5 | Cognee | cognify() timeout | Run killed after 900s/1800s | Entity extraction via Llama-3.3-70B on Together AI serverless = 30-60+ min for scope 01 (30 episodes). Default timeout too short. | Increase timeout to 3600s. Scopes 02-06 remain infeasible. | No |
| 6 | Cognee | Embedding payload too large | HTTP 413 on late checkpoints (20+ episodes) | Together AI rejects embedding requests >1MB. Many chunks accumulated by checkpoint 20+. | Non-fatal — cognee retries/skips, earlier chunks still searchable | No |
| 7 | Cognee | LLM token limit during extraction | Incomplete entity graphs | LLM hits output token limit while extracting entities from dense operational logs | Inject `max_tokens=16384` into litellm calls | **Yes** — partial results, no error |
| 8 | Hindsight | `retain_batch()` HTTP 413 | Batch ingest crashes | Together AI embedding endpoint rejects large payloads when all episodes sent at once | Revert to sequential individual `retain()` calls (commit `0e33fdf`) | No |
| 9 | Hindsight | Env var naming trap | Container starts but LLM/embed calls fail silently | `HINDSIGHT_API_EMBEDDINGS_PROVIDER=openai` requires keys under `HINDSIGHT_API_EMBEDDINGS_OPENAI_*` (with `_OPENAI_` infix), not `HINDSIGHT_API_EMBEDDINGS_*` | Use correct env var prefix. Cost hours of debugging. | **Yes** — no error on startup, fails on first embed call |
| 10 | Hindsight | Text reformatting destroys evidence | evidence_coverage = 0.1667 | Hindsight internally reformats/paraphrases ingested text. Original operational metrics (`p99: 320ms`) become unrecognizable. | Pass `document_id=episode_id` in `retain()` for ID recovery. Text loss is unfixable without modifying Hindsight internals. | **Yes** — returns results, just wrong content |
| 11 | Hindsight | Budget compliance collapse | NBA 0.168 (= null adapter) | `retain()` entity extraction = 20-100s per episode. 30 episodes × 50s avg = 25 min consolidation. Agent's budget is exhausted before it can answer questions. | Structural — no adapter-level fix. Would need Hindsight to support deferred/async consolidation. | **Yes** — agent answers questions, just poorly |
| 12 | Hindsight | Low RRF scores cause search loops | Agent burns tokens on repeated searches | TEMPR's fused RRF scores come back ~0.03 on operational metrics. Agent interprets as low confidence, searches again. | No fix applied. Could threshold scores or limit search iterations in agent harness. | **Yes** |
| 13 | Graphiti | Embedding batch 413 | Entity extraction crashes on scopes with many entities | Together AI rejects embedding batches >1MB | Monkey-patch `OpenAIEmbedder` with `_ChunkedEmbedder` (20-item batches) | No |
| 14 | Mem0 | `dimensions` parameter rejection | Embedding calls fail | Together AI rejects `dimensions` kwarg for fixed-output models (gte-modernbert-base) | Monkey-patch embedder to strip `dimensions`. Set `MEM0_EMBED_NO_DIMS=1`. | No |
| 15 | Mem0 | LLM required even in raw mode | Crash on startup if LLM env vars missing | `mem0.Memory()` always instantiates an LLM client even when `infer=False` | Must provide valid `MEM0_LLM_*` env vars even though they're unused | No |
| 16 | Mem0 | Response shape instability | `search()` returns wrong type across versions | Some mem0 versions return `{"results": [...]}`, others return bare list | Handle both shapes in adapter | **Yes** — could silently return empty |
| 17 | Letta | Embed proxy URLError not caught | Embedding calls fail when Ollama not running | Proxy catches `HTTPError` (status 500+) for Together AI fallback, but connection refused raises `URLError` which wasn't caught | Catch `URLError`/`OSError` in addition to `HTTPError` in `letta_embed_proxy.py` | No |
| 18 | Letta | Passage search response shape | Search returns wrong type | `client.agents.passages.search()` returns object with `.results` in some Letta versions, bare list in others | Handle both shapes | **Yes** — could silently return empty |
| 19 | Letta | Stale agents from previous runs | Agent name collision on reset | Previous run's agent with same name still exists in Letta server | Scan and delete agents matching `lens-{scope_id}` pattern on `reset()` | No — Letta raises conflict error |

### Bug Severity Summary

| Category | Count | Examples |
|----------|-------|---------|
| **Silent data corruption** (returned wrong/empty results, no error) | 7 | #1, #7, #9, #10, #11, #12, #16 |
| **Provider compatibility** (Together AI ≠ OpenAI) | 6 | #2, #6, #8, #13, #14, #17 |
| **Library API instability** (response shapes change between versions) | 3 | #1, #16, #18 |
| **Configuration traps** (non-obvious env vars / required-but-unused config) | 3 | #9, #15, #19 |
| **Scalability** (works on small inputs, fails on benchmark-scale) | 3 | #4, #5, #6 |

The most dangerous category is **silent data corruption** — 7 of 19 bugs produced no error, just wrong results. Without a scoring pipeline that checks evidence grounding, these would go undetected.

---

## Conclusions

1. **Compaction is the adapter to beat.** NBA 0.790 with zero operational overhead. Any system that can't exceed this under constrained budgets isn't providing value — it's providing complexity.

2. **Letta-sleepy is the only heavy adapter worth running.** NBA 0.667/0.693 with manageable setup (1 container + 1 proxy). The sleep consolidation pattern — LLM synthesis in prepare(), prepended to search results — is the right architectural pattern for constrained budgets.

3. **Graph-based systems (cognee, graphiti, hindsight) pay entity extraction costs they can't recover.** Under token budgets, the agent doesn't have enough room to exploit the graph. The extraction overhead is pure waste.

4. **Hindsight is not functional for this workload.** NBA at null-adapter level from a 17.3GB container. The batch ingest bug, text reformatting, and budget compliance collapse make it unsuitable for constrained evaluation.

5. **The library ecosystem is immature.** 4/6 adapters need monkey-patches for basic OpenAI-compatible provider support. Response shapes change between versions. Error messages are unhelpful. These are research-grade libraries, not production tools.

6. **Dedicated inference would help but won't change the ranking.** Faster LLM calls reduce wall-clock time but don't fix the fundamental issue: graph extraction provides no benefit when the agent's retrieval budget is tiny.

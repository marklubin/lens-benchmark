# Phase 3 Execution Plan: Distractors + Realistic Budgets

## Motivation

Phase 1-2 tested memory systems on **30 signal-only episodes (~14K tokens)**. At that scale:
- Compaction summarizes everything in one LLM call and wins trivially (NBA 0.790)
- The constrained budget (2K/4K) limits retrieval, but the corpus is small enough that pre-summarization bypasses retrieval entirely
- Memory systems never face the actual retrieval challenge they're designed for

Phase 3 adds **90 distractor episodes** per scope (3 themes × 30 each), bringing the corpus to **120 episodes (~84K tokens)**. Now:
- 75% of episodes are noise — memory systems must separate signal from distractor
- Compaction's single-pass summary will be diluted (75% irrelevant content)
- Retrieval precision becomes the differentiating factor
- Budget presets of 8K/16K give the agent 10-20% of the corpus — realistic for production use

## What Changed

| Item | Before (Phase 1-2) | After (Phase 3) |
|------|-------------------|-----------------|
| Episodes per scope | 30 (all signal) | 120 (30 signal + 90 distractor) |
| Corpus size (scope 01) | ~14K tokens | ~84K tokens |
| Budget presets | 2K, 4K | 8K, 16K |
| Agent retrieval coverage | 14-28% of corpus | 10-20% of corpus |
| Signal-to-noise ratio | 100% signal | 25% signal |
| Dataset file | `scope_01_only.json` | `scope_01_with_distractors.json` |

## New Budget Presets

| Preset | max_cumulative_result_tokens | max_turns | max_tool_calls | Corpus coverage |
|--------|------------------------------|-----------|----------------|-----------------|
| constrained-16k | 16,384 | 10 | 20 | ~19.5% |
| constrained-8k | 8,192 | 8 | 16 | ~9.8% |

## Adapters to Test

**All 9 adapters**, scope 01 only, 2 budgets each = **18 runs**.

| Adapter | Infrastructure | Expected time/run |
|---------|---------------|-------------------|
| null | None | ~5 min |
| chunked-hybrid | None | ~10 min |
| compaction | None | ~10 min (4x more episodes to summarize) |
| letta | Letta container + embed proxy | ~15 min |
| letta-sleepy | Letta container + embed proxy | ~20 min |
| graphiti | FalkorDB container | ~45 min (120 episodes × entity extraction) |
| cognee | None (embedded DBs) | ~2+ hours (120 episodes × cognify) |
| mem0-raw | Qdrant container | ~15 min |
| hindsight | Hindsight container | ~1+ hour (120 episodes × retain) |

**Estimated total wall-clock**: ~8-10 hours on Together AI serverless. **~1-2 hours on dedicated GPU inference.**

## Infrastructure Requirements

### Containers (same as Phase 2)
```bash
podman run -d -p 6379:6379 --name falkordb falkordb/falkordb
podman run -d -p 6333:6333 --name qdrant qdrant/qdrant
podman run -d -p 8283:8283 --name letta -e TOGETHER_API_KEY=$TOGETHER_API_KEY letta/letta:latest
podman run -d -p 8888:8888 --name hindsight \
  -e HINDSIGHT_API_LLM_PROVIDER=openai \
  -e HINDSIGHT_API_LLM_API_KEY=$TOGETHER_API_KEY \
  -e HINDSIGHT_API_LLM_MODEL=Qwen/Qwen3-235B-A22B-Instruct-2507-tput \
  -e HINDSIGHT_API_LLM_BASE_URL=https://api.together.xyz/v1 \
  -e HINDSIGHT_API_EMBEDDINGS_PROVIDER=openai \
  -e HINDSIGHT_API_EMBEDDINGS_OPENAI_API_KEY=$TOGETHER_API_KEY \
  -e HINDSIGHT_API_EMBEDDINGS_OPENAI_MODEL=Alibaba-NLP/gte-modernbert-base \
  -e HINDSIGHT_API_EMBEDDINGS_OPENAI_BASE_URL=https://api.together.xyz/v1 \
  ghcr.io/vectorize-io/hindsight:latest
```

### Embed proxy (for Letta)
```bash
TOGETHER_API_KEY=$TOGETHER_API_KEY uv run python scripts/letta_embed_proxy.py &
```

### Inference provider
Together AI serverless works but is slow (30-120s cold starts on 235B). Dedicated GPU options:
- **Together Dedicated**: ~$3/hr for 235B, eliminates cold starts
- **RunPod H200**: Qwen3-32B via vLLM, ~$2-3/hr (but 32B judge is weaker)
- **Modal**: Serverless with warm pools, competitive pricing

**Recommendation**: Run on dedicated inference. 18 runs × entity extraction on 120 episodes × serverless cold starts = days of wall-clock time. Dedicated GPU cuts this to hours.

## Execution

### Option A: Manual
```bash
# Lightweight adapters (no containers needed)
for adapter in null chunked-hybrid compaction; do
  for budget in 8k 16k; do
    uv run lens run --config configs/${adapter}_scope01d_${budget}.json --parallel-questions 4 -v
  done
done

# Heavy adapters (start containers first)
for adapter in letta letta-sleepy mem0-raw graphiti; do
  for budget in 8k 16k; do
    uv run lens run --config configs/${adapter}_scope01d_${budget}.json --parallel-questions 4 -v
  done
done

# Slow adapters (long timeouts)
for adapter in cognee hindsight; do
  for budget in 8k 16k; do
    timeout 7200 uv run lens run --config configs/${adapter}_scope01d_${budget}.json --parallel-questions 4 -v
  done
done
```

### Option B: Orchestrator
Extend `scripts/run_constrained_validation.py` with a `--phase 3` flag that uses the distractor datasets and 8K/16K presets. Same state tracking, concurrent execution, and retry logic.

### Scoring
```bash
# Score each run with Qwen3-235B judge
for run_dir in output/*/; do
  OPENAI_API_KEY=$TOGETHER_API_KEY OPENAI_BASE_URL=https://api.together.xyz/v1 \
    uv run lens score --run "$run_dir" --judge-model Qwen/Qwen3-235B-A22B-Instruct-2507-tput -v
done
```

## What to Look For

1. **Does compaction's advantage shrink?** With 75% noise in the corpus, its single-pass summary should be diluted. If compaction still wins, the distractors may not be confusing enough.

2. **Do retrieval-based systems improve relatively?** Systems that can filter out distractors (chunked-hybrid, letta, graphiti) should gain relative to compaction.

3. **Does letta-sleepy's consolidation help with noise filtering?** Sleep synthesis might identify signal patterns that distinguish them from distractor themes.

4. **Does graphiti's graph structure help at 16K?** With more budget and more episodes, the knowledge graph should provide better navigation.

5. **Is hindsight still broken?** 120 episodes × 20-100s retain = potentially 60+ minutes of entity extraction before the agent can answer anything.

## Files Created

| File | Purpose |
|------|---------|
| `datasets/benchmark_dataset_with_distractors.json` | Full 6-scope dataset with distractors (720 episodes) |
| `datasets/scope_01_with_distractors.json` | Scope 01 only (120 episodes, 24 questions) |
| `datasets/scope_02_with_distractors.json` | Scope 02 only |
| `datasets/scope_03_with_distractors.json` | Scope 03 only |
| `datasets/scope_04_with_distractors.json` | Scope 04 only |
| `datasets/scope_05_with_distractors.json` | Scope 05 only |
| `datasets/scope_06_with_distractors.json` | Scope 06 only |
| `configs/*_scope01d_8k.json` | 9 adapter configs at 8K budget |
| `configs/*_scope01d_16k.json` | 9 adapter configs at 16K budget |

## Code Changes

| File | Change |
|------|--------|
| `scripts/compile_dataset.py` | Added `--include-distractors` flag to merge distractor episodes |
| `src/lens/core/config.py` | Added `constrained-8k` and `constrained-16k` budget presets |

# Summarization Adapters: compaction, hopping, hierarchical

Summarization-based memory adapters compress episodes via LLM calls before the agent queries them. They differ in *how* they compress: single-shot, rolling, or multi-level. All three share a fundamental limitation -- lossy compression destroys the episode-level evidence that LENS is designed to test. The interesting question is how much that matters, and where.

---

## Scope Categories

| Phase | Scopes | Episodes | Signal / Distractor | Words/Episode | Notes |
|-------|--------|----------|---------------------|---------------|-------|
| Phase 1: Numeric | S01--S06 | 120 | 30 + 90 | ~700 | Structured metrics, terse logs |
| Phase 2: Narrative | S07--S09 | 40 | 20 + 20 | ~5,000 | Conversational, long-form |
| Phase 3: Semantic Retrieval Stress | S10--S12 | 40 | 20 + 20 | ~5,000 | Structurally identical episodes to defeat embedding similarity |

---

## Architectures

### compaction

- **Pattern:** Offline Summarization (Pattern B) -- single-shot
- Buffer all episodes, produce one summary in a single LLM call at `prepare()`
- Search returns the summary verbatim regardless of query
- Zero infrastructure: no containers, databases, or embedding models
- Approximately 100 lines of code. The "naive memory baseline."

### hopping

- **Pattern:** Offline Summarization (Pattern B) -- incremental rolling
- At each checkpoint, batches of new episodes are merged into an evolving summary via K LLM calls
- Avoids the context-window ceiling of compaction by processing incrementally
- Search returns the rolling summary

### hierarchical

- **Pattern:** Offline Summarization (Pattern B) -- multi-level
- Three tiers: per-episode summary (L1), per-group summary (L2, groups of 5), global summary (L3)
- Search can return summaries from any level, offering variable granularity
- More LLM calls than compaction but preserves more structure

---

## Phase 1: Numeric Scopes (S01--S06)

### compaction -- Main evaluation (Phase 5, GPT-OSS-120B)

12/12 runs: 6 scopes x 2 budgets.

| Metric | Value |
|--------|-------|
| Overall Mean (8K+16K) | **0.272** [0.137, 0.328] |
| Rank | **#7 of 8** (only null worse) |
| 8K Mean | 0.245 |
| 16K Mean | 0.300 (+0.055) |
| Budget Compliance | 1.0 |

Per-scope at 8K budget:

| S01 | S02 | S03 | S04 | S05 | S06 |
|-----|-----|-----|-----|-----|-----|
| 0.339 | 0.000 | 0.198 | 0.364 | 0.309 | 0.259 |

S02 = 0.000 is a complete failure -- distractors overwhelmed the summary entirely.

### compaction -- Constrained validation (Qwen3-235B, 30 episodes, NO distractors)

Compaction **dominated** this small-corpus regime:

| Budget | NBA | 95% CI | Rank |
|--------|-----|--------|------|
| 4K | 0.711 | [0.652, 0.767] | **#1** |
| 2K | 0.735 | [0.676, 0.791] | **#1** |

Per-scope at 4K:

| S01 | S02 | S03 | S04 | S05 | S06 |
|-----|-----|-----|-----|-----|-----|
| 0.787 | 0.621 | 0.740 | 0.675 | 0.563 | 0.672 |

AnsQ: 0.974 / 0.990 -- near-perfect at small corpus.

Why: 30 episodes at ~700 words each is roughly 14K tokens. The entire corpus fits in a single summarization call. The summary IS the complete compressed knowledge.

### compaction -- The collapse

This is the central validation of the benchmark design:

| Condition | Episodes | Distractors | NBA | Rank |
|-----------|----------|-------------|-----|------|
| Constrained (Phase 1) | 30 | No | 0.735 | **1st** |
| Phase 3 (S01 only) | 120 | Yes (90) | 0.404 | 5th |
| Phase 5 (multi-scope) | 120 | Yes (90) | ~0.39 | **7th** |

Phase 3 run IDs: 8K = `6c55408270eb`, 16K = `e80bb2cc6d01`

Distractors dilute the summary, burying signal in noise. The adapter that looked best on a toy corpus collapses when the task gets real.

### compaction -- 7-Adapter sweep (Qwen3-32B, 30 episodes, no distractors)

| Mean | S01 | S02 | S03 | S04 | S05 | S06 |
|------|-----|-----|-----|-----|-----|-----|
| **0.399 (#1)** | 0.506 | 0.410 | 0.392 | 0.354 | 0.349 | 0.383 |

Again dominant in the small-corpus regime.

### hopping and hierarchical -- Not evaluated on numeric scopes in Phase 5

Both were added later for narrative and SRS evaluation.

---

## Phase 2: Narrative Scopes (S07--S09)

### compaction

3-rep run (Llama era):

| Metric | Value |
|--------|-------|
| Mean | 0.276 +/- 0.009 |
| S07 | 0.268 |
| S08 | 0.287 |
| S09 | 0.274 |
| Ingest time | 0s |
| Question time | 304s |
| Total | 304s (fastest real adapter) |
| Budget violations | 12 / 90 questions (lowest of real adapters) |

Perfectly deterministic across 3 reps.

**Hard limit:** 40 episodes x 5,500 words = ~220K words exceeds the 65K token context window. Compaction cannot run on full narrative datasets.

### hopping

Qwen3.5-35B-A3B (Modal), static driver.

| Scope | Score | Run ID |
|-------|-------|--------|
| S07 | 0.141 | `e5e4856b81b8` |
| S08 | 0.127 | `bccc9e5532ee` |
| S09 | 0.383 | `8b5b99440085` |
| **Mean** | **0.217** | |

Per-metric: EvGnd = 1.000 (all scopes), EvCov = 0.000 / 0.000 / 0.150, BudC = 0.00.

Extreme scope variance: S09 (0.383, structured HTTP logs) versus S07 (0.141, conversational chat). Rolling summaries work better for keyword-rich operational content than for indirect conversation.

### hierarchical -- Not separately evaluated on narrative scopes

---

## Phase 3: Semantic Retrieval Stress (S10--S12)

### compaction -- Not evaluated (context window limit)

### hopping -- SRS evaluation (Qwen3.5-35B-A3B, Modal)

**Modal driver:**

| Scope | Score | Run ID |
|-------|-------|--------|
| S10 | 0.446 | `808784df59cf` |
| S11 | 0.535 | `e6e0d1e21df7` |
| S12 | 0.597 | `4f06f28675d3` |
| **Mean** | **0.526 (#2 of 8)** | |

**Static driver:**

| Scope | Score | Run ID |
|-------|-------|--------|
| S10 | 0.372 | `7c87336f4eaa` |
| S11 | 0.509 | `e9c180d754d3` |
| S12 | 0.506 | `33068702c449` |
| **Mean** | **0.462** | |

Delta static to modal: **+0.064**. Hopping is the **only adapter to improve** with the modal driver versus static. The rolling-summary approach meshes better with iterative agent query-refine than with single-shot static queries.

### hierarchical -- SRS evaluation (Qwen3.5-35B-A3B, Modal)

**Modal driver:**

| Scope | Score | Run ID |
|-------|-------|--------|
| S10 | 0.471 | `ebf434b09f53` |
| S11 | 0.532 | `8c978f63b259` |
| S12 | 0.553 | `dd5bd1fb488d` |
| **Mean** | **0.519 (#3 of 8)** | |

**Static driver:**

| Scope | Score | Run ID |
|-------|-------|--------|
| S10 | 0.552 | `265bf4ba9abd` |
| S11 | 0.590 | `9df4e4d3f983` |
| S12 | 0.580 | `bb0d85ca45e8` |
| **Mean** | **0.574 (#4 of 7)** | |

Delta static to modal: **-0.055**. Unlike hopping, hierarchical does slightly worse with the modal driver. Multi-level summaries provide decent retrieval under both driver modes without favoring either.

---

## Cross-Phase Summary

**Comparability caveat:** Numeric uses GPT-OSS-120B (gated composites, 8K+16K combined). Narrative and SRS use Qwen3.5-35B-A3B (ungated). Per-run-ID scores above may differ slightly from Phase 2/3 report aggregates, which average over all runs per adapter (including duplicates from multiple scoring passes).

| Adapter | Numeric (GPT-OSS, gated) | Narrative (Qwen, ungated) | SRS Static (Qwen) | SRS Dynamic (Qwen) |
|---------|---------------------|----------------------|-----------------------|----------------------|
| compaction | 0.272 | 0.276 (static) | -- (context limit) | -- |
| hopping | -- | 0.217 (static) / 0.291 (dynamic) | 0.462 | 0.548 |
| hierarchical | -- | 0.408 (dynamic) | **0.574** | 0.554 |

---

## Key Findings

### 1. Compaction's collapse validates the benchmark

From #1 at 30 episodes (NBA = 0.735) to #7 at 120 episodes with distractors (NBA ~ 0.39). Distractors dilute the summary, burying signal in noise. This is the intended effect of the benchmark design -- if a simple summarizer can ace it, the benchmark is broken.

### 2. Hopping uniquely benefits from the agent driver

The only adapter that improved (+0.064) with modal versus static driver. Rolling summaries mesh well with iterative agent query-refine patterns. The summary evolves with the conversation rather than being a frozen artifact.

### 3. Hierarchical is solid but unremarkable

Competitive on SRS (#3--4 depending on driver) but without standout performance in any category. Multi-level granularity helps but does not beat simpler approaches decisively.

### 4. All summarization approaches are lossy

"Revenue showed irregular patterns" replaces specific quarterly figures. Evidence_coverage is near-zero for compaction because the summary discards episode-level attribution. This is not a bug in the adapters -- it is an inherent property of summarization as a memory strategy.

### 5. Compaction has a hard scaling limit

Cannot process corpora exceeding approximately 65K tokens. This rules out all narrative and SRS scopes, making it fundamentally unsuitable for real-world memory tasks where the corpus grows without bound.

### 6. Speed versus quality tradeoff

Compaction is the fastest adapter (304s for narrative, zero ingest overhead) but produces the lowest quality results at scale. The summarization overhead of hopping and hierarchical is modest, yet the lossy compression they all share destroys the episode-level evidence that LENS is designed to test.

# Baselines and Other Adapters: null, mem0-raw, hindsight

## Architectures

### null (no memory baseline)

- Returns nothing for all search/retrieve calls
- Agent must answer from its own context (system prompt + question only)
- Serves as the floor — any real memory system must beat this
- evidence_grounding = 0.0 always → hard gate → composite = 0.000 always
- High reasoning_quality (0.6-1.0) shows the model reasons well from nothing, but can't access evidence

### mem0-raw

- Pattern: Store-and-Retrieve (Pattern A)
- Uses Mem0 library with `infer=False` — bypasses Mem0's extraction LLM, stores raw text
- Storage: Qdrant vector database
- Search: pure semantic vector search (no keyword/BM25 component)
- NOTE: `mem0-extract` (with extraction enabled) scored 0.000 — Mem0's extraction prompt is hardcoded for personal chatbot memory ("Hi, my name is John" style examples), produces zero extractions from operational telemetry. The extraction layer is the problem, not storage.

### hindsight (REMOVED from evaluation)

- TEMPR architecture: semantic + BM25 + graph + temporal signals, RRF-fused
- 17.3GB container image with embedded PostgreSQL
- Entity extraction during ingest: 20-100s per episode
- Also exposes `memory_reflect` (native longitudinal synthesis) — agent rarely used it
- **Removed after evaluation**: NBA statistically indistinguishable from null, zero operational value

## Phase 1: Numeric Scopes (S01-S06)

120 episodes (30 signal + 90 distractor), ~700 words/episode.

### null — Phase 5 (GPT-OSS-120B)

- 12/12 runs scored
- Composite: 0.000 across all scopes and budgets (hard-gated by evidence_grounding=0)
- NBA: 0.231 (loses to naive baseline that reads all episodes)

### null — Constrained (Qwen3-235B)

- 4K NBA: 0.071 [0.053, 0.089], 2K NBA: 0.067 [0.048, 0.086]
- Per-scope: S01=0.168/0.150, S02=0.161/0.151, S03=0.166/0.150, S04=0.096/0.117, S05=0.128/0.115, S06=0.136/0.150

### mem0-raw — Phase 5 (GPT-OSS-120B)

- 12/12 runs scored
- Overall Mean: **0.349** [0.265, 0.388] — #4 of 8
- 8K Mean: 0.330, 16K Mean: 0.368 (+0.039)
- Per-scope (8K): S01=0.345, S02=0.320, S03=0.186, S04=0.419, S05=0.407, S06=0.299
- Statistically significantly worse than sqlite-chunked-hybrid (p<0.001)

### mem0-raw — Constrained (S01, Qwen3-235B)

- 2K NBA: 0.406, 4K NBA: 0.386
- AnsQ: 0.387/0.386, Budget compliance swings (0.375→0.000) without performance change
- Stable but unimpressive — pure vector search provides consistent but mediocre quality

### mem0-raw — Phase 3 (S01 with distractors)

- 8K: AnsQ=0.368, NBA=0.490, EvCov=0.094 (run `1754d32e84e2`)
- 16K: AnsQ=0.335, NBA=0.477, EvCov=0.083 (run `ab801d0bb064`)

### hindsight — Phase 3 (S01)

- Only 16K completed (8K failed: 413 batch embed too large for 120 episodes)
- 16K: AnsQ=0.213, NBA=0.358, EvCov=0.000 (run `4d77975d04fa`)
- 61 minutes for a single run

### hindsight — Constrained (S01, Qwen3-235B)

- 2K NBA: 0.168, 4K NBA: 0.168 — **identical to null** (0.150/0.168)
- evidence_coverage = 0.000 across all runs
- budget_compliance: 0.750/0.667 — high only because agent barely retrieves anything
- 17.3GB image provides zero value

### hindsight — 7-Adapter sweep

- Not included (already removed from evaluation by this point)

## Phase 2: Narrative Scopes (S07-S09)

40 episodes (20 signal + 20 distractor), ~5,000 words/episode.

### null — Static driver (Qwen3.5-35B-A3B)

- All scopes: composite=0.000 (evidence_grounding gate)
- AnsQ: 0.050-0.062, reasoning_quality: 0.6-1.0
- Budget compliance: 0.6-0.8 (no retrieval = no budget usage)
- Run IDs: S07=`c5cbb0c18421`, S08=`0e087e7ab7be`, S09=`32f22b512e26`

### mem0-raw — 3-rep run (Llama era)

- Mean: 0.254 +/- 0.007
- Per-scope: S07=0.249, S08=0.255, S09=0.259
- Ingest: 103.7s (2.6s/episode — extracts memories via LLM), Question time: 454.7s
- Tokens: 98,139, Tool calls: 17
- Only 12 token budget violations / 90 questions
- Narrowest score range of any adapter — consistently mediocre

### hindsight — Failed on narrative scopes

- Each 5,000-word episode triggered 10-15 `retain_extract_facts` LLM calls (30-90s each) plus consolidation
- Per-episode retain time exceeded 5 minutes — impractical for real workloads

## Phase 3: Semantic Retrieval Stress (S10-S12)

40 episodes (20 signal + 20 distractor), ~5,000 words/episode. Structurally identical to defeat embedding similarity.

### null — Both drivers

- Modal: 0.000 across all scopes
- Static: 0.000 across all scopes
- Run IDs (modal): S10=`0cb3e7bf841a`, S11=`b5220a50a1b1`, S12=`ddb25b2f6738`
- Run IDs (static): S10=`340477991de2`, S11=`d21e6de830a6`, S12=`f2dd4080d062`

### mem0-raw — Not evaluated on SRS scopes

### hindsight — Not evaluated (already removed)

## Cross-Phase Summary

**Comparability caveat:** Numeric column uses GPT-OSS-120B (gated composites). Narrative and SRS use Qwen3.5-35B-A3B (ungated composites). Cross-column comparisons are not controlled.

| Adapter | Numeric (GPT-OSS, gated) | Narrative (Qwen, ungated) | SRS (Qwen, ungated) | Status |
|---------|---------|-----------|-----|--------|
| null | 0.000 | 0.213 | 0.248 | Baseline (all phases) |
| mem0-raw | 0.349 | 0.322 | — | Active |
| hindsight | 0.213 (S01 only) | FAILED | — | **Removed** |

## Key Findings

1. **Null baseline validates the hard gate.** evidence_grounding=0 produces composite=0.000, guaranteeing the floor. But reasoning_quality of 0.6-1.0 shows the model can reason well without memory — it just cannot ground claims in evidence.

2. **mem0-raw is the pure vector search data point.** Consistent but mediocre. Similarity search finds dramatic episodes (Cr=132 ug/L) but misses early subtle signals (Cr=4, first exceedance). No keyword component means it cannot do precision lookups.

3. **mem0-extract is fundamentally broken for LENS.** The extraction prompt is hardcoded for personal preferences and names. Given operational telemetry, it finds zero personal facts and stores nothing. Would need a fork to fix.

4. **Hindsight provides zero value.** NBA indistinguishable from null (0.168 vs 0.150), evidence_coverage = 0.000, 17.3GB image, 20-100s per-episode ingest, 61 minutes for one run. The graph-based entity extraction is too slow and produces nothing the agent can use.

5. **The extraction quality spectrum.** Raw text storage (mem0-raw: 0.349) beats extraction-based storage (hindsight: 0.213) beats personal-memory extraction (mem0-extract: 0.000). Every extraction layer that throws away information hurts. Simpler storage consistently outperforms extraction-based approaches on LENS.

# Narrative Scope Performance Metrics

Performance data extracted from 63 benchmark runs across 7 adapters, 3 narrative scopes (S07-S09), 3 reps each. All runs use 8k budget with 40 episodes (20 signal + 20 distractor).

## Adapter Rankings (composite score, mean across scopes and reps)

| Rank | Adapter | Composite | Ingest (s) | Question Time (s) | Total Time (s) | Tokens (mean) | Tool Calls |
|------|---------|-----------|------------|-------------------|----------------|---------------|------------|
| 1 | sqlite-chunked-hybrid | 0.348 +/- 0.049 | 76.5 | 623.9 | 700.5 | 237,509 | 20 |
| 2 | letta | 0.342 +/- 0.007 | 60.1 | 518.4 | 578.5 | 260,992 | 20 |
| 3 | cognee | 0.327 +/- 0.029 | 0.0 | 530.7 | 530.7 | 222,312 | 20 |
| 4 | letta-sleepy | 0.301 +/- 0.013 | 56.6 | 809.5 | 866.0 | 150,936 | 17 |
| 5 | compaction | 0.276 +/- 0.009 | 0.0 | 304.0 | 304.0 | 114,346 | 22 |
| 6 | mem0-raw | 0.254 +/- 0.007 | 103.7 | 454.7 | 558.4 | 98,139 | 17 |
| 7 | null | 0.178 +/- 0.000 | 0.0 | 90.4 | 90.4 | 19,469 | 20 |

## Per-Scope Breakdown (mean across 3 reps)

### S07 (AI Tutoring Jailbreak)

| Adapter | Composite | Ingest (s) | QTime (s) | Tokens |
|---------|-----------|------------|-----------|--------|
| letta | 0.334 | 54.8 | 549.5 | 403,611 |
| letta-sleepy | 0.308 | 57.9 | 814.3 | 168,238 |
| cognee | 0.298 | 0.0 | 692.7 | 298,921 |
| sqlite-chunked-hybrid | 0.287 | 77.1 | 722.4 | 223,016 |
| compaction | 0.268 | 0.0 | 355.8 | 140,539 |
| mem0-raw | 0.249 | 104.1 | 557.0 | 87,684 |
| null | 0.179 | 0.0 | 97.4 | 19,846 |

### S08 (Corporate Acquisition)

| Adapter | Composite | Ingest (s) | QTime (s) | Tokens |
|---------|-----------|------------|-----------|--------|
| sqlite-chunked-hybrid | 0.398 | 66.3 | 544.5 | 189,057 |
| cognee | 0.364 | 0.0 | 468.9 | 178,899 |
| letta | 0.344 | 54.3 | 565.8 | 179,854 |
| letta-sleepy | 0.293 | 51.8 | 1,127.9 | 150,673 |
| compaction | 0.287 | 0.0 | 263.4 | 104,465 |
| mem0-raw | 0.255 | 97.6 | 529.7 | 124,555 |
| null | 0.179 | 0.0 | 82.1 | 18,212 |

### S09 (Shadow API Abuse)

| Adapter | Composite | Ingest (s) | QTime (s) | Tokens |
|---------|-----------|------------|-----------|--------|
| sqlite-chunked-hybrid | 0.359 | 86.3 | 604.9 | 300,455 |
| letta | 0.349 | 71.2 | 440.0 | 199,512 |
| cognee | 0.319 | 0.0 | 430.4 | 189,117 |
| letta-sleepy | 0.303 | 60.0 | 486.1 | 133,897 |
| compaction | 0.274 | 0.0 | 292.9 | 98,035 |
| mem0-raw | 0.259 | 109.4 | 277.4 | 82,179 |
| null | 0.179 | 0.0 | 91.9 | 20,349 |

## Per-Question Statistics (rep 1 only, 30 questions per adapter)

| Adapter | Avg QTime | Median QTime | Min QTime | Max QTime | Avg Tokens | Avg Tools | Avg Answer Len | Avg Refs |
|---------|-----------|-------------|-----------|-----------|------------|-----------|----------------|----------|
| letta-sleepy | 78.7s | 41.9s | 3.4s | 350.5s | 14,425 | 1.6 | 669 chars | 3.2 |
| sqlite-chunked-hybrid | 61.3s | 49.0s | 8.0s | 184.6s | 24,321 | 2.0 | 813 chars | 3.6 |
| mem0-raw | 43.1s | 31.4s | 7.3s | 193.8s | 9,273 | 1.6 | 692 chars | 0.6 |
| cognee | 33.8s | 31.1s | 6.6s | 91.0s | 22,230 | 2.0 | 850 chars | 3.0 |
| compaction | 30.8s | 29.6s | 7.3s | 84.6s | 11,430 | 2.2 | 593 chars | 1.2 |
| letta | 30.4s | 30.0s | 7.0s | 75.5s | 26,101 | 2.0 | 888 chars | 4.2 |
| null | 11.5s | 7.8s | 3.8s | 24.7s | 1,947 | 2.0 | 145 chars | 0.1 |

## Budget Violations (across all 9 runs per adapter = 90 questions)

| Adapter | Token Violations | Tool Violations | Latency Violations |
|---------|-----------------|-----------------|-------------------|
| letta | 96 | 6 | 1 |
| cognee | 87 | 0 | 0 |
| sqlite-chunked-hybrid | 82 | 0 | 0 |
| letta-sleepy | 60 | 0 | 1 |
| compaction | 12 | 0 | 0 |
| mem0-raw | 12 | 0 | 0 |
| null | 0 | 0 | 0 |

Token budget violations are pervasive: the 16,384-token agent budget is routinely exceeded because narrative episodes are ~5,000 words each, and search retrieval returns large chunks. The adapters that retrieve more context (letta, cognee, sqlite-chunked-hybrid) exceed it most often.

## Determinism

Most adapter-scope combinations are perfectly deterministic across 3 reps (stdev = 0). Notable exceptions:

| Config | R1 | R2 | R3 | Stdev |
|--------|------|------|------|-------|
| letta_sleepy_S07 | 0.329 | 0.304 | 0.291 | 0.019 |
| letta_sleepy_S08 | 0.293 | 0.283 | 0.302 | 0.009 |
| letta_sleepy_S09 | 0.306 | 0.306 | 0.298 | 0.005 |
| mem0_raw_S07 | 0.247 | 0.240 | 0.261 | 0.011 |
| sqlite_chunked_hybrid_S08 | 0.395 | 0.399 | 0.399 | 0.003 |
| letta_S08 | 0.346 | 0.346 | 0.339 | 0.004 |

letta-sleepy has the most variance (its sleep/wake cycle introduces non-determinism). Cognee, compaction, null, and most letta runs are fully deterministic.

## Ingest Time Patterns

Adapters with zero ingest time defer processing to prepare() or query time:
- **null**: no-op adapter, baseline
- **cognee**: stores raw text at ingest, processes in prepare() via knowledge graph construction
- **compaction**: stores raw, compacts in prepare()

Adapters with significant ingest time do LLM/embedding work during ingest:
- **mem0-raw**: ~104s (2.6s/episode) -- extracts memories via LLM during ingest
- **sqlite-chunked-hybrid**: ~77s (1.9s/episode) -- chunks + embeds during ingest
- **letta/letta-sleepy**: ~57s (1.4s/episode) -- Letta archival storage with embedding

## Key Findings

1. **sqlite-chunked-hybrid leads** on composite (0.348) but is second-slowest (700s total). Its chunking + hybrid search pays off in score but costs time.

2. **letta is the best efficiency tradeoff**: 0.342 composite in 579s, with very low variance. It processes quickly at ingest and maintains consistent quality.

3. **cognee achieves 0.327 with zero ingest overhead** (defers to prepare), making it the fastest high-performing adapter at 531s total.

4. **compaction is the fastest real adapter** (304s total) but scores lower (0.276). Its lossy summarization discards evidence needed for longitudinal questions.

5. **letta-sleepy is the slowest** (866s) due to high question-time variance. Its sleep/wake memory consolidation adds overhead without improving scores vs. regular letta.

6. **Token budgets are systematically exceeded** because narrative episodes (~5,000 words) are much larger than the numeric episodes the budget was calibrated for. This affects all adapters except null.

7. **S08 (Corporate Acquisition) produces the strongest differentiation** between adapters (sqlite 0.398 vs null 0.179 = 0.219 gap), while S07 (Tutoring Jailbreak) is hardest for all adapters.

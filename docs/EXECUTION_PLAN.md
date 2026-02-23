# Constrained Budget Validation — Full Execution Plan

**Goal**: Determine whether constrained token budgets (4K/2K) cause NBA to approach or exceed 0.5, validating the hypothesis that good retrieval matters when agents can't see all episodes.

**Full matrix**: 9 adapters x 6 scopes x 2 budgets = 108 runs + scoring + analysis

---

## Phased Execution

Each phase has a decision gate. Review results before proceeding.

### Phase 1: Scope 01 — Lightweight Adapters (6 runs)

**No external services needed.** All runs concurrent via Together AI serverless.

| Adapter | Budget | Config |
|---------|--------|--------|
| null | 4k, 2k | `configs/null_scope01_{4k,2k}.json` |
| chunked-hybrid | 4k, 2k | `configs/chunked_hybrid_scope01_{4k,2k}.json` |
| compaction | 4k, 2k | `configs/compaction_scope01_{4k,2k}.json` |

```bash
python3 scripts/run_constrained_validation.py --phase 1
```

**Wall time**: ~10-15 min (6 concurrent runs + inline scoring)

**Decision gate**:
- If best adapter NBA > 0.45 at 2K → strong signal, proceed to phase 2+3
- If best adapter NBA 0.25-0.45 → moderate, proceed to phase 2 for more data
- If ALL NBA < 0.25 at 2K → weak signal, proceed to phase 2 but temper expectations
- If ALL NBA < 0.15 at 2K → hypothesis likely invalid, phase 2 is still informative but consider pivoting

---

### Phase 2: Scope 01 — Heavy Infrastructure Adapters (12 runs)

**Requires external services.** Start services before running.

| Adapter | Budget | Needs | Serial Constraint |
|---------|--------|-------|-------------------|
| cognee | 4k, 2k | (embedded DBs, no container) | none |
| graphiti | 4k, 2k | FalkorDB on :6379 | none |
| mem0-raw | 4k, 2k | Qdrant on :6333 | none |
| hindsight | 4k, 2k | Hindsight server on :8888 | none |
| letta | 4k, 2k | Letta server on :8283 | letta group (serial) |
| letta-sleepy | 4k, 2k | Letta server on :8283 | letta group (serial) |

#### Service startup

```bash
# FalkorDB (for graphiti)
podman run -d -p 6379:6379 --name falkordb falkordb/falkordb

# Qdrant (for mem0-raw)
podman run -d -p 6333:6333 --name qdrant qdrant/qdrant

# Hindsight
podman run -d -p 8888:8888 --name hindsight \
  -e HINDSIGHT_API_LLM_PROVIDER=openai \
  -e HINDSIGHT_API_LLM_API_KEY=$TOGETHER_API_KEY \
  -e HINDSIGHT_API_LLM_MODEL=Qwen/Qwen3-235B-A22B-Instruct-2507-tput \
  -e HINDSIGHT_API_LLM_BASE_URL=https://api.together.xyz/v1 \
  -e HINDSIGHT_API_EMBEDDINGS_PROVIDER=openai \
  -e HINDSIGHT_API_EMBEDDINGS_API_KEY=$TOGETHER_API_KEY \
  -e HINDSIGHT_API_EMBEDDINGS_MODEL=Alibaba-NLP/gte-modernbert-base \
  -e HINDSIGHT_API_EMBEDDINGS_BASE_URL=https://api.together.xyz/v1 \
  ghcr.io/vectorize-io/hindsight:latest

# Letta (for letta + letta-sleepy)
podman run -d -p 8283:8283 --name letta \
  -e TOGETHER_API_KEY=$TOGETHER_API_KEY \
  letta/letta:latest
# Letta embedding proxy (separate terminal)
uv run python scripts/letta_embed_proxy.py
```

```bash
python3 scripts/run_constrained_validation.py --phase 2
```

**Concurrency**: cognee, graphiti, mem0-raw, hindsight run concurrently. letta + letta-sleepy run serially (shared server). Total: 4 concurrent tracks + 1 serial letta track.

**Wall time**: ~15-25 min

**Decision gate** (scope 01 complete, all 9 adapters x 2 budgets = 18 runs):
- Review NBA table across all adapters at both budgets
- Identify which adapters show NBA > 0.20 at 2K — these are **worth expanding**
- Adapters with NBA < 0.10 at 2K can be dropped from phase 3 to save cost
- null baseline must always expand (needed for paired tests)

```bash
# View interim results
python3 scripts/run_constrained_validation.py --status
```

---

### Phase 3: Scopes 02-06 — Expand Promising Adapters (up to 90 runs)

Run all adapters across remaining scopes, or filter to only promising ones.

```bash
# All adapters, all remaining scopes
python3 scripts/run_constrained_validation.py --phase 3

# Only specific adapters (based on phase 1+2 results)
python3 scripts/run_constrained_validation.py --phase 3 \
  --adapters null chunked-hybrid compaction letta-sleepy

# One scope at a time for gradual expansion
python3 scripts/run_constrained_validation.py --phase 3 --scopes 02
python3 scripts/run_constrained_validation.py --phase 3 --scopes 03
# ... review after each scope ...
python3 scripts/run_constrained_validation.py --phase 3 --scopes 04 05 06
```

**Strategy**: Run scopes incrementally. After each scope, check `--status` and run `--analyze` to see if results are stabilizing. If CIs are tight after 3 scopes, remaining scopes add statistical power but may not change conclusions.

**Wall time**: ~10-15 min per scope (adapters run concurrently), ~50-75 min total for all 5 scopes

**Decision gates per scope**:
- After scope 02: Do CIs narrow? Does adapter ranking change?
- After scope 03: Are results consistent across scopes? Any scope-specific anomalies?
- After scopes 04-06: Final statistical power for publication

---

### Phase 4: Full Analysis

```bash
python3 scripts/run_constrained_validation.py --phase 4
# or
python3 scripts/analyze_constrained.py --aggregate
```

Produces:
1. **Bootstrap CIs** on per-question NBA (10K resamples, pooled across scopes)
2. **Paired Wilcoxon**: each adapter vs null at each budget
3. **Budget degradation**: 4K vs 2K per adapter
4. **Differential degradation**: does adapter hold up better than null under tighter constraint?
5. **Degradation curve plot**: `results/constrained_degradation_aggregate.png`
6. **Go/no-go verdict**

---

## Adapter Inventory

| Adapter | Infrastructure | Concurrency | Notes |
|---------|---------------|-------------|-------|
| null | none | concurrent | Baseline — no memory, no retrieval |
| chunked-hybrid | none | concurrent | SQLite FTS + embedding search |
| compaction | none | concurrent | LLM-based summarization |
| cognee | embedded DBs | concurrent | GraphRAG, no container needed |
| graphiti | FalkorDB :6379 | concurrent | Temporal knowledge graph |
| mem0-raw | Qdrant :6333 | concurrent | Vector search |
| hindsight | Hindsight :8888 | concurrent | TEMPR retrieval |
| letta | Letta :8283 | serial (letta group) | Archival memory |
| letta-sleepy | Letta :8283 | serial (letta group) | Letta + sleep consolidation (variant 3) |

---

## Config Matrix

108 config files across `configs/`:
- `{adapter}_scope{01-06}_{4k,2k}.json`
- All use Qwen/Qwen3-32B, seed 42, temperature 0.0
- Checkpoints: [5, 10, 12, 15, 20, 25, 30]
- Budget presets: constrained-4k (4096 tokens), constrained-2k (2048 tokens)

---

## State Tracking & Resilience

**State file**: `constrained_validation_state.json`

Each run is tracked independently: `{label: {status, run_id, composite, nba, error}}`.

| Failure Mode | Recovery |
|---|---|
| Run fails | Marked as `failed`. Re-run with `--retry-failed` clears failed state. |
| Score fails | Run output intact. `--score-only` re-scores all unscored runs. |
| Script crash | State saved atomically per-operation. Resume skips completed runs. |
| Service dies | Runs against that service fail; others unaffected. Restart service and re-run. |
| Partial phase | Re-run same `--phase` command — completed runs skipped. |

**Useful commands**:
```bash
# Check progress
python3 scripts/run_constrained_validation.py --status

# Retry all failed runs
python3 scripts/run_constrained_validation.py --phase N --retry-failed

# Re-score completed but unscored runs
python3 scripts/run_constrained_validation.py --score-only

# Analyze single scope
python3 scripts/analyze_constrained.py --scope 01

# Analyze all scopes pooled
python3 scripts/analyze_constrained.py --aggregate
```

---

## Cost Estimate

Together AI serverless (Qwen3-235B MoE): $0.20/M input, $0.60/M output.

| Phase | Runs | Est. Cost |
|-------|------|-----------|
| Phase 1 (6 runs + scoring) | 6 | ~$0.50 |
| Phase 2 (12 runs + scoring) | 12 | ~$1.00 |
| Phase 3 all (90 runs + scoring) | 90 | ~$7.50 |
| Phase 3 filtered (4 adapters x 5 scopes x 2) | 40 | ~$3.30 |
| **Total (all phases, all adapters)** | **108** | **~$9.00** |
| **Total (filtered phase 3)** | **58** | **~$4.80** |

Even at 3x error margin: under $30 for full sweep.

---

## Go/No-Go Criteria

| Signal | Threshold | Action |
|--------|-----------|--------|
| Strong validate | Any adapter NBA > 0.45 at 2K | Constrained budgets work, retrieval matters |
| Moderate validate | Best adapter NBA 0.30-0.45 AND significantly > null | Retrieval helps, but not transformatively |
| Weak signal | Best adapter NBA 0.20-0.30 | Marginal, investigate per-question patterns |
| Invalidate | ALL adapter NBA < 0.20 at 2K | Constrained budget alone doesn't help |
| Differential | Adapter degrades less than null (interaction > 0, CI excludes 0) | Retrieval quality protects against budget cuts |

---

## Execution Checklist

- [ ] Phase 1: Run lightweight adapters on scope 01
- [ ] Phase 1: Review decision gate
- [ ] Phase 2: Start external services
- [ ] Phase 2: Run heavy adapters on scope 01
- [ ] Phase 2: Review scope 01 complete results, decide which adapters to expand
- [ ] Phase 3: Expand to scope 02, review
- [ ] Phase 3: Expand to scope 03, review
- [ ] Phase 3: Expand to scopes 04-06
- [ ] Phase 4: Run full analysis
- [ ] Phase 4: Review degradation curves, CIs, verdict
- [ ] Export results to `results/constrained_validation.json`
- [ ] Update `docs/STATUS_REPORT.md` with findings

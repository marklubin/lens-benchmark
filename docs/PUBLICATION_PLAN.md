# LENS Publication Plan: From Current State to Defensible Paper

**Date**: 2026-02-23
**Goal**: Minimum viable publication — a scientifically defensible result that demonstrates LENS's unique contribution to memory evaluation.

---

## Current State (What We Have)

### Data
- **Phase 1-2**: 48 scored runs (9 adapters × scope 01 × 2 budgets + 3 lightweight × 6 scopes × 2 budgets). 30 episodes, no distractors, 2K/4K budgets, Qwen3-235B agent.
- **Phase 3 (scope 01 only)**: 12/18 runs with distractors (120 episodes, ~84K tokens, 8K/16K budgets, GPT-OSS-120B agent). Missing: cognee (2), graphiti (2), letta-sleepy/8k (1), hindsight (1 — dropped).
- **Phase 5 infrastructure**: 96 configs ready (8 adapters × 6 scopes × 2 budgets), orchestrator updated.

### Analysis & Writing
- Benchmark landscape survey (10 benchmarks) ✓
- Deep methodology comparison (5 benchmarks × 8 dimensions) ✓
- Synthesis report with defensible thesis ✓
- Red team report with prioritized issues ✓
- Status report with all historical results ✓

### Code
- 5 parallelism optimizations implemented ✓
- Cognee embed fix applied ✓
- Graphiti adapter verified correct ✓
- 991 unit tests passing ✓

---

## What's Missing (The Gaps)

### Critical (paper cannot be published without these)
1. **N=1 scope with distractors** — all Phase 3 results are from scope 01 only. Can't generalize.
2. **Cognee + graphiti missing** — the two graph-based systems most likely to benefit from noise filtering never ran.
3. **No-key-fact questions score 1.0** — 4/24 questions per scope give free points, inflating all scores uniformly.
4. **fact_recall has zero discrimination** — 0.167 across all adapters. Dead metric in composite.
5. **No statistical significance testing** — rankings could be noise without CIs.

### Major (reviewers will ask about these)
6. **Budget enforcement narrative** — "constrained" framing is wrong; need to reframe as "unconstrained retrieval with noise."
7. **Judge reliability unknown** — no position-swap agreement data.
8. **Single agent LLM** — GPT-OSS-120B only. Rankings might change with different agent models.
9. **Compaction is privileged** — gets unconstrained LLM summarization in prepare() before budget clock.

### Paper-Level
10. **No written paper** — have synthesis report and analysis docs but no structured paper.
11. **No figures** — need clear visualizations for results.

---

## The Plan: 7 Phases, ~30 Hours Total

### Phase A: Scoring Fixes (2 hours, code changes)

Fix the two methodological issues that affect all subsequent data collection. Must happen BEFORE running Phase 5, or we'd have to re-score everything.

**A1. Fix no-key-fact scoring** (~30 min)
- `src/lens/scorer/tier1.py` line 75: `1.0` → `0.5`
- `src/lens/scorer/tier2.py` line 76: `return None` already handles this (appends 0.5 later) — verify
- `src/lens/scorer/judge.py` line 51: `1.0` → `0.5`
- `src/lens/scorer/tier3.py` line 18: `1.0` → `0.5`
- Run tests, verify 991 still pass

**A2. Remove fact_recall from composite** (~30 min)
- Zero-weight fact_recall in composite calculation OR remove entirely
- It provides zero discrimination (0.167 uniformly) — it's noise in the composite
- Keep it as a reported metric for transparency, just don't include in the composite score

**A3. Label compaction as "oracle upper bound"** (~30 min)
- Add a note in adapter metadata / scoring output that compaction has privileged prepare() access
- Does NOT need to be removed from results — just clearly labeled
- In the paper: "Compaction represents an oracle upper bound — it receives unconstrained LLM summarization of the full corpus before the agent's retrieval budget is applied."

**A4. Re-frame budget narrative** (~30 min)
- Update config preset names or documentation: "unconstrained retrieval with noise" not "constrained budget"
- The finding doesn't change — simple retrieval beats complex architectures even without budget pressure
- In the paper: "We initially designed a constrained-budget protocol but discovered that budget enforcement was non-binding (adapters consumed 4-5× the allocated tokens on the first retrieval call). We report results under unconstrained retrieval with signal-to-noise separation as the primary experimental manipulation."

### Phase B: Multi-Scope Data Collection — Lightweight (4-6 hours compute)

Run the three adapters that need no external services across all 6 scopes with distractors. This is the backbone of the paper — transforms N=1 into N=6.

**B1. Run Phase 5 lightweight** (~4 hrs compute, can run unattended)
```bash
python3 scripts/run_constrained_validation.py --phase 5 \
    --cerebras-key $CEREBRAS_API_KEY \
    --adapters null sqlite-chunked-hybrid compaction \
    --max-workers 6
```
- 36 runs: 3 adapters × 6 scopes × 2 budgets (8k, 16k)
- Scoring happens inline (Qwen3-235B on Together AI)
- Expect ~15-20 min per run × 6 concurrent = ~2 hrs wall time
- Scope 01 runs will repeat (that's fine — confirms reproducibility vs Phase 3)

**Delivers**: Cross-scope answer_quality for null, chunked-hybrid, compaction with CIs.

### Phase C: Fix and Run Graph-Based Adapters (4-6 hours)

Eliminate survivorship bias — cognee and graphiti are the systems most architecturally suited for noise filtering (knowledge graphs should theoretically extract signal entities while ignoring distractor entities).

**C1. Run cognee on scope 01 with distractors** (~2 hrs)
- Cognee embed fix is already applied
- Start containers: none needed (cognee is embedded)
- Run:
```bash
python3 scripts/run_constrained_validation.py --phase 5 \
    --cerebras-key $CEREBRAS_API_KEY \
    --adapters cognee --scopes 01
```
- If scope 01 succeeds, expand to 02-06

**C2. Run graphiti on scope 01 with distractors** (~2 hrs)
- Needs FalkorDB: `podman run -d -p 6379:6379 --name falkordb falkordb/falkordb`
- Graphiti adapter routes entity extraction to Together AI (Llama-3.3-70B), not Cerebras
- Run:
```bash
python3 scripts/run_constrained_validation.py --phase 5 \
    --cerebras-key $CEREBRAS_API_KEY \
    --adapters graphiti --scopes 01
```
- If scope 01 succeeds, expand to 02-06

**C3. Run mem0-raw multi-scope** (~2 hrs)
- Needs Qdrant: `podman run -d -p 6333:6333 --name qdrant qdrant/qdrant`
- Mem0-raw already works on scope 01 — expand to all 6
```bash
python3 scripts/run_constrained_validation.py --phase 5 \
    --cerebras-key $CEREBRAS_API_KEY \
    --adapters mem0-raw
```

**C4. Run letta + letta-sleepy multi-scope** (~3 hrs, serial)
- Needs Letta server + embed proxy
- These share a server so they must run serially (orchestrator handles this)
```bash
python3 scripts/run_constrained_validation.py --phase 5 \
    --cerebras-key $CEREBRAS_API_KEY \
    --adapters letta letta-sleepy
```

**Delivers**: Complete 8-adapter × 6-scope matrix. Or at minimum: cognee + graphiti on scope 01 to confirm they don't change the headline finding.

### Phase D: Statistical Analysis (2-3 hours)

**D1. Bootstrap confidence intervals** (~1 hr)
- Per-adapter answer_quality across 6 scopes: bootstrap 95% CI
- Key test: is chunked-hybrid > letta-sleepy statistically significant?
- Wilcoxon signed-rank tests for all adapter pairs
- Effect sizes (Cohen's d or rank-biserial correlation)

**D2. Cross-scope consistency analysis** (~1 hr)
- Heatmap: adapter × scope answer_quality matrix
- Identify if any adapter wins on specific domain types
- Compute Kendall's W (concordance) — do adapter rankings hold across scopes?

**D3. Distractor effect analysis** (~30 min)
- Paired comparison: Phase 1-2 (no distractors) vs Phase 5 (with distractors) for null, chunked-hybrid, compaction across all 6 scopes
- Quantifies how much distractors degrade each system
- Tests: is the degradation uniform or does it vary by architecture?

**D4. Question-type analysis** (~30 min)
- Which question types discriminate best? (longitudinal, distractor_resistance, temporal)
- Which question types show ceiling/floor effects?
- Per-question-type adapter ranking — does the hierarchy hold across question types?

**Delivers**: Statistical backing for all claims. Tables and figures for paper.

### Phase E: Judge Reliability Validation (2 hours)

**E1. Position-swap agreement** (~2 hrs compute)
- For a random sample of 50-100 judgment calls, run the judge twice with positions swapped
- Compute Cohen's kappa or % agreement
- If agreement > 0.8: judge is reliable, report it
- If agreement < 0.7: flag as limitation, consider multi-judge ensemble

**Implementation**: Add a `--position-swap-audit` flag to the scoring CLI that re-runs a subset of judgments with A↔B swapped and reports agreement.

**Delivers**: "Inter-rater reliability (position-swap): κ = X.XX" — one line in the paper that preempts the biggest methodological critique.

### Phase F: Paper Writing (8-12 hours)

Structure for a venue like NeurIPS Datasets & Benchmarks, COLM, or a workshop paper:

**F1. Abstract** (~30 min)
- The thesis in 250 words

**F2. Introduction** (2 hrs)
- Memory systems are proliferating, evaluated on benchmarks they're designed to beat
- The synthesis gap: no benchmark tests longitudinal pattern detection
- Our finding: no system > 50%, simple retrieval beats specialized architectures
- Contributions: (1) the LENS benchmark, (2) the landscape gap analysis, (3) the 8-adapter × 6-scope evaluation

**F3. Related Work / Benchmark Landscape** (2 hrs)
- Already written in `docs/BENCHMARK_LANDSCAPE_SURVEY.md` and `docs/BENCHMARK_METHODOLOGY_COMPARISON.md`
- Distill into 1.5 pages: the 5 patterns other benchmarks test, the 6th pattern LENS tests
- Table: benchmark × dimension comparison matrix

**F4. Benchmark Design** (2 hrs)
- Two-stage information isolation architecture
- Distractor generation and calibration
- Three-tier scoring with NBA
- Contamination and difficulty validation
- Question taxonomy (10 types)

**F5. Experimental Setup** (1 hr)
- 8 memory systems (null, sqlite-chunked-hybrid, compaction, cognee, graphiti, mem0-raw, letta, letta-sleepy)
- 6 domain-diverse scopes (120 episodes each, 3:1 noise ratio)
- Agent LLM: GPT-OSS-120B, Judge: Qwen3-235B
- Checkpoints, scoring, statistical methodology

**F6. Results** (2 hrs)
- Main finding: answer_quality table with CIs
- Cross-scope consistency (heatmap)
- Distractor effect (paired comparison)
- Question-type analysis
- Operational findings (3/8 systems failed to complete)
- Judge reliability

**F7. Discussion & Limitations** (1 hr)
- What this means for the field
- Limitations: single agent LLM, operational log domain, unconstrained retrieval
- What LENS doesn't test (action execution, cross-platform, selective forgetting)
- Future work: LENS-XL, agentic evaluation, semantic disconnect queries

**F8. Figures** (2 hrs)
- Fig 1: The two-stage information isolation pipeline (architecture diagram)
- Fig 2: Adapter × scope answer_quality heatmap
- Fig 3: Distractor effect (before/after bar chart)
- Fig 4: Question-type discrimination (radar or grouped bar)
- Fig 5: Compaction collapse (30 eps → 120 eps) — the money figure
- Table 1: Landscape comparison matrix
- Table 2: Main results with CIs

### Phase G: Second Agent LLM Ablation (4 hours, OPTIONAL but strengthens paper)

Run top 4 adapters (null, chunked-hybrid, letta, mem0-raw) on scope 01 with a second agent LLM (Claude Sonnet or Qwen3-235B) to show the finding is robust to agent model choice.

**Skip this if**: time-constrained. Instead, list as a limitation: "Rankings were established with a single agent LLM (GPT-OSS-120B). We note that prior phases using Qwen3-32B and Qwen3-235B showed qualitatively similar hierarchies, suggesting robustness, but formal ablation is future work."

---

## Dependency Graph

```
Phase A (scoring fixes, 2 hrs)
    ├──→ Phase B (lightweight multi-scope, 4-6 hrs) ──→ Phase D (analysis, 2-3 hrs)
    ├──→ Phase C (graph adapters + heavy, 4-6 hrs) ──→ Phase D
    └──→ Phase E (judge reliability, 2 hrs) ──────────→ Phase D
                                                            │
                                                            ▼
                                                     Phase F (paper, 8-12 hrs)
                                                            │
                                                            ▼ (optional)
                                                     Phase G (2nd LLM ablation, 4 hrs)
```

**Critical path**: A → B → D → F = ~16-23 hours
**With graph adapters**: A → B+C parallel → D → F = ~18-27 hours
**Full path with everything**: A → B+C+E parallel → D → F+G = ~22-33 hours

---

## Minimum Viable Paper (If We Had to Ship Tomorrow)

If we could only do ONE more thing before writing:

1. **Fix scoring (Phase A)** — 2 hours
2. **Run lightweight adapters across 6 scopes (Phase B)** — 4-6 hours
3. **Bootstrap CIs (Phase D1)** — 1 hour
4. **Write the paper (Phase F)** — 8-12 hours

**Total: ~15-21 hours → Paper with N=6 scopes, 3 core adapters, statistical significance.**

List cognee/graphiti/letta/mem0 scope-01 results as "preliminary single-scope" with note that full multi-scope evaluation is in progress. This is honest and defensible.

---

## What We Can Claim at Each Stage

### After Phase A+B (scoring fixes + lightweight multi-scope):
> "Across 6 domain-diverse scopes, simple FTS+embedding retrieval consistently outperforms LLM summarization and a no-memory baseline at longitudinal evidence synthesis. No configuration exceeds 50% answer quality."

### After Phase A+B+C (+ graph adapters):
> "Across 6 scopes and 8 memory architectures — including vector search, knowledge graphs, temporal graphs, and agent memory — no system meaningfully outperforms basic text retrieval at longitudinal synthesis."

### After Phase A+B+C+D (+ statistical analysis):
> "The ranking sqlite-chunked-hybrid > letta-sleepy > mem0-raw > letta > compaction > null is statistically significant (p < 0.05, Wilcoxon signed-rank) across 6 scopes with mean answer quality 0.XX (95% CI: [0.XX, 0.XX])."

### After Full Pipeline:
> The complete paper thesis with all qualifications addressed.

---

## Risk Factors

| Risk | Impact | Mitigation |
|------|--------|-----------|
| Cerebras rate limits during Phase B/C | Runs take 2-3× longer | Run off-peak, use Together as fallback |
| Cognee still fails after fix | Can't include in paper | Report as "operational failure" finding (3/8 systems couldn't complete) |
| Graphiti fails on multi-scope | Limited to scope 01 | Single-scope result is still publishable with qualification |
| Rankings change across scopes | Weakens headline finding | Actually strengthens paper — shows domain sensitivity |
| Judge reliability is poor | Undermines all T2/T3 scores | Fall back to T1 metrics only (programmatic) |
| Phase B shows chunked-hybrid doesn't consistently win | Changes narrative | The finding is still "no system > 50%" and "the gap exists" |

---

## Recommended Execution Order (Starting Now)

1. **Now**: Apply Phase A scoring fixes (2 hrs)
2. **Tonight/overnight**: Launch Phase B lightweight runs (unattended, 4-6 hrs)
3. **Tomorrow morning**: Start Phase C graph adapters (needs container babysitting)
4. **Tomorrow afternoon**: Phase D analysis on Phase B results
5. **Day 3**: Phase E judge reliability + start Phase F paper writing
6. **Day 4-5**: Complete Phase F paper draft
7. **Day 6**: Review, revise, finalize

**Calendar estimate: 5-6 working days to a complete paper draft.**

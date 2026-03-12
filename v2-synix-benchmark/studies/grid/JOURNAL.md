# Grid Run Work Journal

## Timeline

**Start**: 2026-03-11 22:39 PDT
**Target (M=3 decision)**: 2026-03-12 00:49 PDT

| Time (PDT) | Event |
|------------|-------|
| 22:39 | Plan approved, clock started |
| 22:42 | Code changes: temperature + replicate_id in agent.py, runner.py |
| 22:43 | Tests: 216 passed, 10 skipped |
| 22:44 | Grid runner script created: studies/grid/run_scope.py |
| 22:44 | Grading rubric v1.0 created |
| 22:44 | H100 x2 deploy initiated (min_containers=2) |
| 22:44 | H100 x2 deployed successfully |
| 22:42-22:43 | All 4 scope runners launched (S07, S11, S15, S16) |
| 22:49 | First banks released: S07-cp06 (489, 367s), S15-cp04 (299, 345s), S16-cp06 (118, 362s) |
| 22:50 | S11-cp06 released (627, 393s) |
| 22:53 | S15-cp08 (610, 216s), S16-cp12 (217, 227s) released |
| 23:01 | S15 all 3 banks done. Agent inference started |
| 23:02 | S15 null run completed (10 answers, 0 tool calls) |
| 23:04 | S15 policy_base completed (10 answers, 1-3 tool calls each) |
| 23:05 | S15 policy_core running. S07/S11/S16 still building banks (3/4, 2/4, 3/4) |
| 23:08 | S15: all 4 runs complete (40 answers). Scoring started |
| 23:10 | S16: all 4 banks done, all 4 runs complete |
| 23:15 | S07/S11: all 4 banks done, agent inference running |
| 23:18 | S15 COMPLETE: all scored. summary=0.657 > core=0.616 > base=0.468 > null=0.000 |
| 23:20 | S07: 4 runs done, scoring in progress |
| 23:26 | S07 COMPLETE: base=0.590 > summary=0.566 > core=0.466 > null=0.050 |
| 23:26 | S16 COMPLETE: core=0.663 > summary=0.644 > base=0.573 > null=0.050 |
| 23:30 | S11 COMPLETE: summary=0.573 > base=0.474 > core=0.457 > null=0.050 |
| 23:30 | **ALL 4 SCOPES COMPLETE** — 160 answers, 160 scores, 48 min total |
| 23:36 | Grading script created. First attempt failed (Qwen thinking leak) |
| 23:39 | Fixed: few-shot prompting anchors JSON output. Restarted grading |
| 23:50 | **GRADING COMPLETE** — 152/160 scored (95%), 8 parse failures, 9.4 min |
| 23:52 | Results collected. Few-shot grader shows higher fact_f1 than auto-scorer |
| 23:53 | **CHECKPOINT 3 READY** — 1h13m elapsed (target was 2h10m) |
| 23:55 | Decision: M=3 YES + add S08, S09, S12 (7 scopes total) |
| 00:36 | Fixed cache bypass for M>1 (--no-cache flag). Deleted cached r02/r03. Relaunched |
| 00:38 | All 7 scopes launched: 4 existing (r02+r03), 3 new (r01+r02+r03) |
| 01:17 | Existing 4 scopes: M=3 complete (480 answers, 480 scores) |
| 01:52 | New scopes M=3 complete. **84/84 runs, 840/840 answers, 840/840 scores** |
| 02:01 | Few-shot grading launched for all 840 answers (~84 min estimated) |
| 03:00 | **GRADING COMPLETE** — 807/840 scored (96%), 33 parse failures, 58 min |
| 03:04 | H100 scaled to min_containers=0 |
| 03:05 | **FULL GRID COMPLETE** — 7 scopes × 4 policies × 3 replicates |

## Phase 1 Final Results (M=3, 7 Scopes, 4 Policies)

See Full Results above for the combined 7-policy table. Phase 1 established:
1. **policy_core: 0.486** — core memory synthesis wins over base and summary
2. policy_summary: 0.457
3. policy_base: 0.412
4. null: 0.059

## Timing

| Phase | Planned | Actual |
|-------|---------|--------|
| M=1 grid (4 scopes) | 2h10m | 1h13m |
| M=3 expansion + 3 new scopes | — | 1h52m |
| Few-shot grading (840 answers) | — | 58m |
| Phase 1 total | — | 4h25m |
| New policy code + tests | — | 30m |
| Bank builds + inference (7×3×3) | — | 3h45m |
| Few-shot grading (630 answers) | — | 39m |
| Phase 2 total | — | ~5h |
| **Grand total** | — | **~9h25m** |

## Phase 2: New Policy Ablation (3 New Policies)

| Time (PDT) | Event |
|------------|-------|
| 07:30 | 3 new families implemented: core_structured, core_maintained, core_faceted |
| 08:00 | Tests: 216 passed. Smoke test confirmed pipeline/bank/runtime integration |
| 08:15 | H100 x2 redeployed. 7 scope runners launched (--work-prefix work_new) |
| 09:00 | Bank builds in progress (faceted = 4 folds + 1 reduce per checkpoint) |
| 11:02 | S07, S16, S08, S12 banks done. S15 r01 complete (90 answers) |
| 11:10 | Scaled to 4 H100s to accelerate remaining builds |
| 11:42 | 4/7 scopes fully done (S07, S15, S16, S08). Others finishing r03 |
| 11:57 | **63/63 runs, 630/630 answers, 630/630 Qwen auto-scores** |
| 12:01 | H100 scaled to 0. Grading launched (630 tasks) |
| 12:44 | **GRADING COMPLETE** — 600/630 scored (95.2%), 30 parse failures, 39 min |
| 12:45 | H100 scaled to 0. Full 7-policy results compiled |

## Full Results (M=3, 7 Scopes × 7 Policies)

### Fact F1 (Few-shot Qwen Grader, mean across 3 replicates, n=1407 graded)

| Policy | S07 | S08 | S09 | S11 | S12 | S15 | S16 | Mean |
|--------|-----|-----|-----|-----|-----|-----|-----|------|
| null | 0.069 | 0.052 | 0.050 | 0.083 | 0.050 | 0.034 | 0.074 | **0.059** |
| policy_base | 0.339 | 0.351 | 0.349 | 0.356 | 0.277 | 0.753 | 0.442 | **0.412** |
| policy_core | 0.458 | 0.301 | 0.643 | 0.456 | 0.201 | 0.808 | 0.511 | **0.486** |
| policy_summary | 0.376 | 0.388 | 0.561 | 0.395 | 0.162 | 0.722 | 0.604 | **0.457** |
| policy_core_structured | 0.529 | 0.426 | 0.538 | 0.410 | 0.156 | 0.750 | 0.475 | **0.472** |
| policy_core_maintained | 0.323 | 0.181 | 0.611 | 0.455 | 0.156 | 0.717 | 0.526 | **0.427** |
| policy_core_faceted | 0.547 | 0.373 | 0.598 | 0.343 | 0.233 | 0.828 | 0.679 | **0.511** |

### Overall Ranking (7 policies)
1. **policy_core_faceted: 0.511** — 4 parallel facets (entity/relation/event/cause) merged
2. policy_core: 0.486 — free-form fold synthesis
3. policy_core_structured: 0.472 — structured observation fold (Mastra/ACE pattern)
4. policy_summary: 0.457 — map-reduce summary
5. policy_core_maintained: 0.427 — fold + refinement pass
6. policy_base: 0.412 — search only, no derived context
7. null: 0.059 — no memory

### Ablation Key Findings
- **Faceted wins**: Decomposing into 4 cognitive facets then merging outperforms all other strategies (+0.025 over core)
- **Maintained hurts**: Refinement/consolidation pass _reduces_ performance (-0.059 vs core). Pruning loses useful signal
- **Structured is competitive**: Structured observations nearly match free-form core (0.472 vs 0.486)
- **Core memory family dominates**: 3 of top 4 policies are core memory variants
- **S12 therapy_chat hardest**: 0.156-0.277 for non-null policies across all variants
- **S15 value_inversion easiest**: 0.717-0.828 — faceted wins at 0.828

### Phase 1 M=1 Results (historical, 4 scopes only)

| Policy | S07 | S11 | S15 | S16 | Mean |
|--------|-----|-----|-----|-----|------|
| null | 0.111 | 0.100 | 0.056 | 0.056 | **0.081** |
| policy_base | 0.308 | 0.285 | 0.778 | 0.523 | **0.474** |
| policy_core | 0.400 | 0.512 | 0.881 | 0.615 | **0.602** |
| policy_summary | 0.333 | 0.404 | 0.813 | 0.648 | **0.550** |

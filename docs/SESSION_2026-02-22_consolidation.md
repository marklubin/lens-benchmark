# Session Summary: LENS Consolidation (2026-02-22)

## What Happened

LENS benchmark work was being maintained in two repositories — the canonical `lens-benchmark` and a unified platform fork in `synix-bench`. This session consolidated all LENS work back into `lens-benchmark` and removed all LENS code from `synix-bench`.

## Key Finding: The Constrained Budget Problem

The most important discovery this session was methodological:

**At standard budget (32K tokens), no memory adapter shows NBA > 0.5.** This means context stuffing — simply dumping all available text into the prompt — beats every memory system we've tested. The benchmark cannot demonstrate that memory systems add value under the conditions we've been running.

The constrained budget infrastructure (4K and 2K token caps) is **fully built** in `lens-benchmark`:
- `AgentBudgetConfig` presets: `constrained-4k` (4096 cumulative result tokens), `constrained-2k` (2048)
- Budget enforcement code in `src/lens/agent/budget.py`
- 72 config files ready in `configs/experiments/`
- Run scripts prepared

**Zero constrained-budget runs have been executed.** This is the P0 blocker before any dataset can be published.

A research directive was written to `docs/RESEARCH_DIRECTIVE.md` documenting the full experimental plan.

## Work Done

### Triad Memory Protocol Ported to lens-benchmark

The Triad adapters (4-facet memory decomposition with 3 operations: INSTANTIATE, UPDATE, ACCOMMODATE) were the only substantive LENS work unique to synix-bench. Ported:

| File | Lines | Description |
|------|-------|-------------|
| `src/lens/adapters/triad.py` | 159 | Shared base: FACETS_4, `_TriadBase`, `_complete()` helper |
| `src/lens/adapters/triad_v1.py` | 588 | 3 adapters: panel, pairs, pairs-fused |
| `tests/unit/test_triad_adapters.py` | 701 | 68 tests covering registry, lifecycle, LLM paths |
| `src/lens/adapters/registry.py` | +4 | Added triad_v1 to `_ensure_builtins()` |

Changes from synix-bench version:
- Imports: `synix.suites.lens.adapters.*` → `lens.adapters.*`
- Env vars: `SYNIX_LLM_*` → `LENS_LLM_*`
- Mock patch targets updated accordingly

**Verification**: 68/68 triad tests pass, 991 total tests pass (includes existing), all 3 adapters registered (`triadv1-panel`, `triadv1-pairs`, `triadv1-pairs-fused`).

### synix-bench Cleaned of All LENS References

Removed from synix-bench:
- `src/synix/suites/lens/` — entire directory (suite, models, anticheat, 17+ adapters)
- `src/synix/scorer/` — entirely LENS-based metrics
- `src/synix/agent/` — LENS agent harness
- All LENS test files, configs, results, scripts, datasets, docs
- LENS error classes (`AdapterError`, `BudgetExceededError`, `LatencyExceededError`, etc.)
- `AgentBudgetConfig` class
- All LENS branches in CLI commands (smoke, list, score, sweep)
- All LENS sections in `modal_harness.py`
- All LENS references in `CLAUDE.md` and docstrings

Post-cleanup: 161 unit tests pass in synix-bench. The repos are now cleanly separated:
- **synix-bench**: SWE-bench (10 context strategies) + Polyglot (Exercism exercises)
- **lens-benchmark**: LENS (21+ memory adapters, scoring, agent harness)

## Adapter Inventory (lens-benchmark, post-consolidation)

| Tier | Adapter | External Deps |
|------|---------|---------------|
| A | null, sqlite, sqlite-fts | None |
| B | sqlite-embedding-openai, sqlite-hybrid-openai, sqlite-chunked, sqlite-chunked-hybrid | Embeddings API |
| C | compaction, triadv1-panel, triadv1-pairs, triadv1-pairs-fused | LLM API |
| D | sqlite-embedding, sqlite-hybrid | Ollama |
| E | letta, letta-sleepy | Letta server |
| F | mem0-raw, mem0-extract | Qdrant |
| G | graphiti | FalkorDB + Together |
| H | cognee | Cognee + Together |
| I | hindsight | Hindsight server |

## What's Next (Priority Order)

1. **Run constrained budget experiments** — this is the validation experiment that determines whether the benchmark methodology works at all. See `docs/RESEARCH_DIRECTIVE.md`.

2. **Analyze results** — if adapters show NBA > 0.5 at constrained budgets, the methodology is validated and we can proceed to full dataset generation.

3. **If methodology fails** — diagnostic checklist in the research directive covers: reducing questions per scope, using harder questions only, testing with weaker models, narrowing to memory-dependent question types.

## Cross-Model Observations (from synix-bench, stack+heap context strategy)

These findings are from the SWE-bench context strategy work but are relevant to understanding model behavior with structured memory protocols:

- **gpt-5.2**: Follows structured protocols (push/pop, heap operations) at depth 5, achieves 3-5x context reduction
- **Claude Sonnet 4.6**: Stays at depth 0-1, doesn't decompose tasks into subtasks
- **Claude Haiku 4.5**: Same as Sonnet — flat execution, no benefit from protocol

Implication for LENS: Adapter effectiveness may be gated by model instruction-following capability. If testing Triad adapters, gpt-5.2 is likely to show the strongest differentiation.

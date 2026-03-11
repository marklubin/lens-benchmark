# Worklog

Append-only operational log for meaningful progress, blockers, and verification results.

## 2026-03-06

- Created V2 workspace scaffold.
- Added process guardrails in `CLAUDE.md`, `AGENTS.md`, and `AGENT.md`.
- Added benchmark spec, execution plan, and scoring v2 docs.
- Added workboard, task template, risk register, decision log, runbook, and initial task backlog.
- Reframed V2 around checkpoint-scoped artifact-bank compilation and runtime-policy evaluation.
- Updated the benchmark spec, execution plan, runbook, workboard, risk register, decision log, and task backlog to match the new model.
- No implementation code exists yet in `src/`.
- Froze the Synix versus LENS ownership split and converted downstream tasks from local artifact-family implementation to Synix integration work.
- Added `T013` for benchmark integration with the Synix runtime and tool API.
- Created `ops/SYNIX_UPSTREAM_TRACKER.md` with the sequential Synix-first DAG and issue mapping.
- Created Synix epic `#88` plus child issues `#81`, `#82`, `#83`, `#84`, `#85`, `#86`, and `#87`.
- Amended Synix issues `#34`, `#15`, `#10`, and `#60` to add sequencing, e2e, docs, and demo-follow-on requirements.
- Finalized the projection contract around sealed bank manifests and named projection handles, then reconciled downstream task dependencies with `T013` and the upstream tracker.

## 2026-03-08

- Eliminated Synix issue #81 (checkpoint projections + sealed bank manifests) as a platform requirement. Checkpoint isolation is handled entirely by LENS pipeline logic: one projection per checkpoint prefix, using label filtering over the existing projection/release model from PR #92.
- Recorded decision D014.
- Rewrote `ops/SYNIX_UPSTREAM_TRACKER.md` to reflect the simplified blocker DAG (#81 removed), current PR #92 status, and the checkpoint isolation design.
- Updated workboard to remove #81 from blocker notes and reflect PR #92 as the active critical-path item.
- Cleaned up stale git worktree (`worktree-new-scopes-10-12`) — branch was already merged into main.
- T003 complete: Implemented `ResponseCache` (SQLite WAL, content-addressed, thread-safe with lock) and `ModalBroker` (cache-through LLM + embedding calls, retry with backoff, token/cost accounting). Added cache config surface (enable/disable, per-type, namespace isolation via cache_dir). Added `wait_for_modal()` warmup poller. 42 unit tests + 8 E2E tests passing against live Modal. Parameterized `min_containers` in `infra/modal/llm_server.py` via `LENS_MIN_CONTAINERS` env var (default 0). Design doc at `docs/plans/2026-03-08-modal-infra-and-cache-design.md`.
- Memory strategy analysis: mapped chunks, core memory, summaries, and maintenance to existing Synix primitives (MapSynthesis, FoldSynthesis, GroupSynthesis, ReduceSynthesis). Graph (policy_graph) deferred from v2 first pass — open design questions on structured multi-stage pipelines in the artifact model. Requirements doc at `docs/plans/synix-capability-requirements.md`.
- V2 first pass policy set: null, policy_base, policy_core, policy_summary (4 policies, no graph). T009 deferred to P2.

## 2026-03-10

- T002 complete: Defined all 6 canonical schemas as Pydantic v2 models in `src/bench/schemas.py`: StudyManifest, PolicyManifest, BankManifest, RunManifest, Event, ScoreRecord. Added enums for BankStatus, RunStatus, EventType (16 event types covering bank build, run, checkpoint, question, model call, and scoring lifecycles). Nested models for FusionConfig, RetrievalCaps, BuildCost, RunCost, Diagnostics. 56 schema tests covering roundtrip serialization, required field validation, enum validation, defaults, checkpoint isolation fields, JSON Schema export, and backward compatibility with existing example files. Exported JSON Schemas to `schemas/*.schema.json`. Added pydantic>=2.0 to project dependencies. 98 total tests passing.
- Schema trace walkthrough: wrote executable pseudo-code (`docs/plans/schema-trace-walkthrough.py`) tracing full study → build → run → score → resume → replay → report chain. Found and fixed one high-severity gap (RunManifest.budget_tier — subsequently removed after confirming budget is not a v2 dimension). Removed StudyManifest.budget_configs. Confirmed 10 structural correctness properties.
- T004 complete: Implemented `StateStore` (SQLite WAL, 7 tables) and `EventWriter` in `src/bench/state.py`. CRUD for all manifest types, append-only event log with filtering, answer/score persistence, resume queries (completed questions, completed bank families), failure injection tests (run crash mid-checkpoint, bank crash mid-build). 44 state tests, 142 total tests passing.
- T005 complete: Implemented `BankBuilder` (`src/bench/bank.py`) and `dataset.py` loader. Pipeline DAG: Source → Chunk + SearchSurface + SynixSearch, optional FoldSynthesis (core memory), optional GroupSynthesis → ReduceSynthesis (summaries). Checkpoint isolation enforced at source level (prefix-valid episodes only). Resume (skips released banks), failure persistence, event emission. Dataset loader reads V1 spec.yaml + generated/episodes/ layout. 22 bank tests (+ 3 S08 real dataset tests) passing. Added `synix>=0.20.0` to dependencies.
- T007 + T008 complete: Implemented `src/bench/families/core.py` (FoldSynthesis with progressive fold prompt) and `src/bench/families/summary.py` (GroupSynthesis + ReduceSynthesis with 5-episode windows). Both integrate into bank pipeline via conditional `add_*` functions.
- T013 complete: Implemented `policy.py` (registry with null/base/core/summary factories), `runtime.py` (BenchmarkRuntime wrapping Synix Release with policy-gated search/context/tools), `agent.py` (tool-use agent loop adapted from V1 harness, with budget enforcement and citation extraction), `runner.py` (StudyRunner orchestrating scope × policy × checkpoint × question with resume). 32 runtime tests covering policy gating, search dispatch, context injection, tool enforcement, run lifecycle, resume, event emission, and bank reuse across policies.
- T006 complete: Implemented `scorer.py` (ScorerV2 with fact_f1 via LLM judge, citation_validity via mechanical artifact resolution, evidence_support via LLM judge). Composite formula: 0.5*fact_f1 + 0.3*evidence_support + 0.2*citation_validity. Supports rescore from saved answers. 20 scorer tests covering all metrics, edge cases, and schema validation.
- Full test suite: 216 passed, 10 skipped. All T005/T006/T007/T008/T013 blockers cleared. T010 (smoke pilot) is now ready.

## 2026-03-11

- T010 smoke pilot validated (gate 6 pass — S08 × 4 policies, 40 scored answers).
- Created gated pilot runner (`studies/pilot/run_pilot.py`) with 8 incremental gates (0=imports, 1=chunks-only bank, 2=synthesis bank, 3=null policy, 4=base policy, 5=scoring, 6=full S08, 7=full pilot).
- SDK validation findings:
  - Synix `ParseTransform` requires `.txt` extensions; prepends `t-text-` to stems. Changed episode ingestion to use `source.add(original_file)` instead of `add_text(content, label=id)`.
  - Synix `SearchSurface` uses `keyword` not `fulltext` for BM25 mode. Fixed in bank.py, policy.py, runtime.py.
  - Added `max_tokens=4096` to agent LLM calls — Qwen3.5 generates unlimited `<think>` chains without it (hung for 8+ min on single call).
  - Strip `<think>...</think>` blocks from agent answers and derived artifact context (core memory, summary).
  - Fixed `state.get_answer()` to include `checkpoint_id` for scorer's release_map lookup.
- S08 pilot results (initial run, scorer at max_tokens=10): null=0.050, base=0.212, core=0.192, summary=0.211.
- Fixed scorer: max_tokens 10→64→512→2048 to allow judge model sufficient thinking tokens. Added `/no_think` soft switch, paraphrase-tolerant prompts, `_resolve_ref`/`_resolve_artifact` for citation validation with prefix expansion.
- Fixed agent: split system prompts (tools vs no-tools), search result truncation (1500 chars), broadened citation extraction regex, `default_extra_body` for `enable_thinking: false`.
- **Final S08 results (validated)**: null=0.050, base=**0.446**, core=0.397, summary=0.393.
  - Base search provides 9× lift over null baseline.
  - Core memory slightly hurts vs base — derived context adds noise when agent has search.
  - Summary matches core — no incremental value from summary context.
  - Evidence support now producing real 0.0-1.0 scores. Citation validity works (base=1.000).
  - fact_f1 low across all policies — genuine finding: agent retrieves evidence but doesn't synthesize specific conclusions matching key facts.
  - Echoes V1 finding: simple retrieval beats complex preprocessing under agent-driven queries.
- Total cost (gate 6): 315K tokens, ~17 min wall time, ~54s cached bank build.
- Verification: rescore from saved answers, resume after kill, cache-hit on re-run, bank reuse across policies — all confirmed.

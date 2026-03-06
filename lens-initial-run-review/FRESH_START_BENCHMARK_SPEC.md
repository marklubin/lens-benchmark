# Fresh-Start Benchmark Spec

## 1. Purpose

This document defines the fresh-start version of the LENS memory benchmark.

The goal is to replace the current heterogeneous adapter evaluation with a controlled, reproducible benchmark that:

- evaluates memory mechanisms rather than third-party products
- uses a single implementation substrate
- produces scientifically auditable results
- can resume from failures without re-running the same inference
- keeps inference spend low enough to run a meaningful study on Modal credits
- simplifies scoring to focus on correctness and evidence

The underlying implementation substrate will be **Synix**. The benchmark story becomes:

1. Synix is used to generate the benchmark datasets.
2. Synix is used to implement multiple memory strategy replicas on one common substrate.
3. LENS evaluates those strategy replicas under one fixed agent and scoring stack.

This is a stronger scientific story than comparing external memory systems with different storage layers, tool interfaces, internal agents, citation formats, and operational failure modes.

## 2. Benchmark Principles

The new benchmark must satisfy the following principles.

### 2.1 Controlled Substrate

All memory strategies are implemented on one shared substrate with the same:

- raw episode store
- chunk store
- embedding model
- retrieval interface
- provenance model
- citation format
- agent loop
- scoring pipeline

Only the memory feature set changes between strategies.

### 2.2 Raw Evidence Is Immutable

Every episode is stored verbatim and never deleted from the canonical raw store.

Derived artifacts such as summaries, graph nodes, core memory blocks, or maintenance outputs may be added, but they must always point back to raw source episode ids.

### 2.3 No Internal Q&A Agents

The benchmark runner's external agent answers all questions.

Memory strategies may prepare derived artifacts and expose them through tools, but they do not run their own autonomous Q&A agent. This avoids the current confounds around internal reasoning, tool use, and citation formatting.

### 2.4 Append-Only Execution Records

Every run emits an append-only event log and a structured state database.

If a run fails halfway through, the system resumes from recorded state and cached model outputs. It does not repeat already completed inference unless explicitly forced.

### 2.5 Deterministic Where Possible

Primary runs should use deterministic or near-deterministic settings:

- temperature `0`
- fixed prompts
- fixed retrieval limits
- fixed preprocessing
- fixed endpoint versions

Where the serving stack is not fully deterministic, the actual outputs are still preserved and reused through caching.

### 2.6 Costs Are First-Class

All model calls must be pre-plannable, attributable, and replayable. Cost is treated as a tracked measurement, not an afterthought.

## 3. What We Are Benchmarking

The fresh-start benchmark measures whether an LLM agent can answer longitudinal questions by using different memory representations built over the same underlying raw evidence.

It is not a benchmark of:

- vendor product maturity
- container reliability
- one-off prompt engineering
- internal agent orchestration frameworks

## 4. Scope Set

The benchmark should be reduced to a coherent and cost-effective scope set.

### 4.1 Main Scope Family

Use `S07-S12` as the primary benchmark.

This gives six long-form scopes across two meaningful content families:

- `S07-S09`: narrative, heterogeneous, cross-document reasoning
- `S10-S12`: semantic retrieval stress, high topical overlap, adversarial retrieval

This is the strongest part of the benchmark and where memory representation matters most.

### 4.2 Optional Extension Family

Keep `S13-S15` as an extension set, but report them separately.

These scopes are structurally different:

- shorter
- no distractors
- different task shape

They should not be pooled into the main headline score unless they are regenerated under the same structural assumptions as `S07-S12`.

### 4.3 Optional Numeric Anchor

Keep only `S01` as a numeric smoke test if a numeric anchor is needed.

Do not keep the full numeric suite in the main benchmark. It adds cost and noise while mostly rewarding straightforward retrieval.

### 4.4 Headline Reporting Unit

The primary benchmark result should be reported on `S07-S12`.

Secondary tables:

- `S13-S15` extension
- `S01` numeric anchor, if retained

## 5. System Under Test: SynixMemory

The benchmark will evaluate a single system, `SynixMemory`, with feature-controlled configurations.

### 5.1 Core Architecture

`SynixMemory` consists of:

- `raw_episodes`: immutable canonical episode store
- `chunks`: chunked episode views
- `fts_index`: lexical retrieval over chunks or episodes
- `embedding_index`: semantic retrieval over chunks
- `provenance_graph`: source-to-derived lineage
- `derived_views`: summaries, core memory, graph projections, recency-weighted indexes
- `maintenance_jobs`: offline consolidation jobs executed at checkpoints

### 5.2 Invariants

All configurations must preserve:

- a shared raw evidence base
- the same chunk ids and episode ids
- the same embedding model
- the same retrieval API
- the same answering agent

### 5.3 Tools Exposed to the Agent

The external agent should receive a stable tool interface:

- `memory_search(query, view=?, limit=?)`
- `memory_retrieve(ref_id)`
- `memory_capabilities()`

Optional:

- `memory_batch_retrieve(ref_ids)`

The agent does not know whether a result came from raw retrieval, a graph projection, or a maintained summary unless the result metadata says so.

## 6. Feature Modules

The benchmark should replicate major memory strategy families using composable feature flags.

### 6.1 Base Features

- `chunking`
- `fts`
- `cosine`
- `rrf`

This is the baseline retrieval system.

### 6.2 Derived Memory Features

- `core_memory`
  - compact maintained state blocks intended for always-useful working memory
- `summaries`
  - checkpoint-level or hierarchical summaries with provenance
- `graph`
  - extracted entities, relations, and graph-neighbor retrieval
- `recency_weighting`
  - time-aware ranking or decay, not deletion
- `maintenance`
  - background or checkpoint consolidation jobs that update derived artifacts

### 6.3 Explicitly Excluded for V1

Do not include these in the first publishable version:

- hard expiration or deletion of raw evidence
- internal answering agents
- multi-agent memory coordination
- product-specific APIs or storage engines

## 7. Strategy Matrix

We should not run the full power set of features. The benchmark should compare a small set of hypothesis-driven replicas.

### 7.1 Recommended Main Configurations

1. `null`
   - no memory
2. `rag_base`
   - chunking + FTS + cosine + RRF
3. `rag_core`
   - `rag_base` + core memory
4. `rag_summary`
   - `rag_base` + summaries
5. `rag_graph`
   - `rag_base` + graph
6. `rag_core_maint`
   - `rag_core` + maintenance
7. `rag_summary_maint`
   - `rag_summary` + maintenance
8. `rag_graph_maint`
   - `rag_graph` + maintenance

### 7.2 Optional Cost-Reduced Main Set

If cost is too high, use:

1. `null`
2. `rag_base`
3. `rag_core_maint`
4. `rag_summary_maint`
5. `rag_graph_maint`

This still tests the major memory families while keeping the system count manageable.

### 7.3 What Each Configuration Represents

- `rag_base`: raw retrieval baseline
- `rag_core_*`: Letta-like maintained core memory without product confounds
- `rag_summary_*`: compaction/hierarchical-style consolidation
- `rag_graph_*`: graph memory strategy
- `*_maint`: sleep-time or checkpoint-time maintenance as a controlled ablation

## 8. Primary Hypotheses

The benchmark should be hypothesis-driven.

### 8.1 Main Hypotheses

1. `rag_base` is a strong baseline but not uniformly best on `S07-S12`.
2. `core_memory` helps on narrative continuity tasks but may add only modest value without maintenance.
3. `summaries` help more on semantically overlapping scopes by acting as relevance filtering.
4. `graph` helps more when cross-document entity linkage is the dominant challenge.
5. `maintenance` helps only when it updates a representation that the agent can exploit reliably.

### 8.2 Negative Hypotheses

1. Maintenance alone does not guarantee improvement.
2. Derived representations that lose provenance reduce evidence fidelity.
3. No single representation dominates across all content types.

## 9. Modal-Only Inference Policy

All inference must run on Modal because the available credits are there.

### 9.1 Modal Services

Use a small set of stable Modal services:

- one chat completion endpoint for the answering agent
- one chat completion endpoint for scoring, if a separate judge is used
- one embedding endpoint

If possible, use the same serving stack for all inference classes and pin:

- container image digest
- model identifier
- generation parameters
- endpoint version

### 9.2 Concurrency Policy

Inference orchestration should include:

- capped concurrency per endpoint
- micro-batching for embeddings
- exponential backoff on transient failures
- strict idempotency keys for all requests

### 9.3 Cost Policy

Every request must record:

- prompt token count
- completion token count
- latency
- estimated dollar cost
- endpoint identity

These values must be recorded in the run log and aggregated into per-run and per-study cost reports.

## 10. Run Manifest and Provenance

Every study must be driven by an immutable manifest.

### 10.1 Study Manifest

Each study manifest must include:

- study id
- benchmark version
- scope list
- strategy list
- prompt set version
- scoring version
- code git sha
- Synix pipeline hash
- model endpoint ids
- embedding model id
- retrieval parameters
- chunking parameters
- checkpoint policy
- random seed policy

### 10.2 Run Manifest

Each individual run must include:

- study id
- run id
- scope id
- strategy id
- replicate id
- configuration hash
- dataset artifact hash
- start time
- end time
- status

### 10.3 Artifact Hashes

All build artifacts should be content-addressed where practical. The benchmark should never depend on unnamed mutable intermediate state.

## 11. Event Log Requirements

The benchmark must emit a full event log for every run.

### 11.1 Storage Format

Use both:

- append-only `events.jsonl`
- indexed `state.sqlite`

The JSONL log is the forensic record. The SQLite database is the operational state and query surface.

### 11.2 Required Event Types

- `study_started`
- `run_started`
- `episode_ingested`
- `prepare_started`
- `prepare_completed`
- `question_started`
- `agent_turn_started`
- `tool_called`
- `tool_result`
- `model_call_requested`
- `model_call_cache_hit`
- `model_call_completed`
- `answer_emitted`
- `score_started`
- `score_completed`
- `run_failed`
- `run_completed`

### 11.3 Event Payload Requirements

Every event must include:

- timestamp
- study id
- run id
- scope id
- strategy id
- event type
- attempt number
- input artifact refs
- output artifact refs
- config hash

### 11.4 Failure Recording

Failures must be explicit and non-destructive:

- stack trace
- failing component
- upstream dependency refs
- retryable vs terminal classification

## 12. Idempotent Caching and Replay

The system must never pay twice for the same model call unless explicitly requested.

### 12.1 Model Call Cache Key

Each model call should be cached by a deterministic key derived from:

- provider
- endpoint id
- model id
- prompt hash
- full input digest
- decoding parameters
- transform code hash

### 12.2 Cached Response Payload

Store:

- raw provider response
- normalized response
- tokens
- latency
- cost estimate
- first-seen timestamp

### 12.3 Resume Semantics

On `resume`, the runner should:

1. read the state database
2. identify incomplete tasks
3. skip completed tasks
4. reuse cached model responses
5. continue downstream

### 12.4 Replay Semantics

Support:

- replaying a run without model calls
- rescoring from saved answers
- regenerating reports from saved run state

This is critical for scientific auditability.

## 13. Execution DAG

The execution pipeline should be explicit.

### 13.1 Study Build Phase

1. freeze manifest
2. validate dataset artifacts
3. materialize strategy configs
4. create task graph

### 13.2 Strategy Build Phase

For each strategy x scope x replicate:

1. ingest raw episodes
2. build derived views as required
3. run maintenance jobs at checkpoints
4. persist all artifacts and provenance

### 13.3 QA Phase

For each checkpoint question:

1. call agent
2. log all turns, tools, and retrievals
3. persist final answer and cited refs

### 13.4 Scoring Phase

1. score against canonical answer and key facts
2. validate citations
3. aggregate metrics
4. write immutable score artifacts

## 14. Statistical Design

The study must be statistically defensible without exploding cost.

### 14.1 Primary Unit of Analysis

The primary analysis unit should be the question-level outcome, clustered by:

- scope
- run
- strategy

Question-level scoring gives sufficient sample size, but all confidence intervals and tests must respect clustering to avoid pseudoreplication.

### 14.2 Recommended Design

Use deterministic primary runs with cluster-aware analysis rather than many expensive stochastic reruns.

Recommended main study:

- scopes: `S07-S12`
- strategies: `5` from the cost-reduced main set
- one deterministic run per scope x strategy cell
- question-level paired analysis across aligned checkpoints/questions

This yields:

- `6 scopes x 5 strategies = 30 primary cells`

Then add a smaller stability audit:

- re-run `10-20%` of cells
- compare answer identity, retrieval identity, and score drift

This is much cheaper than repeating every cell three times while still providing a reproducibility check.

Recommended default:

- rerun `6` randomly stratified cells
- total execution target: `36` cells

This should be the default publishable-low-cost design unless the pilot shows unexpectedly low variance and cheap enough full repeats.

### 14.3 If Full Repeats Are Affordable

If budget allows, use:

- `5 strategies x 6 scopes x 3 repeats = 90 runs`

That is the preferred confirmatory design.

### 14.4 Statistical Methods

Use:

- bootstrap confidence intervals
- paired cluster bootstrap for strategy comparisons
- Holm correction for multiple comparisons
- effect sizes, not only p-values

Avoid headline claims based solely on rank ordering with tiny margins.

## 15. Simplified Scoring

The current scoring system is too complex and has introduced avoidable confusion.

### 15.1 Remove Entirely

Remove these from the primary score:

- hard gating
- `budget_compliance` as a score component
- `reasoning_quality`
- `insight_depth`
- `longitudinal_advantage`
- `naive_baseline_advantage`
- `action_quality`

These can remain as optional diagnostics if useful, but they should not drive the headline ranking.

### 15.2 Primary Metrics

The main benchmark should focus on three things:

1. `fact_f1`
   - how well the answer captures the canonical key facts
2. `evidence_support`
   - how well the cited evidence actually supports the claims
3. `citation_validity`
   - how many cited ids resolve to real source evidence

### 15.3 Secondary Metrics

Track but do not include in the primary score:

- latency
- total tokens
- estimated cost
- retrieval count
- tool count
- budget overrun, if any

### 15.4 Recommended Composite

Use a simple primary composite:

`primary_score = 0.5 * fact_f1 + 0.3 * evidence_support + 0.2 * citation_validity`

No hard gate.

### 15.5 Human Audit

Audit a sampled subset of scored answers by hand to validate the automated scorer.

Recommended:

- `10%` sample of final answers
- stratified across strategies and scopes
- adjudicate disagreements and publish agreement rates

## 16. Scoring Implementation

### 16.1 Fact Matching

Each question should have:

- canonical answer
- canonical key fact set
- evidence clusters expected to support the answer

`fact_f1` should be computed against the key fact set using a stable scorer.

### 16.2 Evidence Support

`evidence_support` should answer:

"Given the cited evidence and the answer, are the claims actually supported?"

This can be computed with a single judge prompt over:

- question
- answer
- retrieved cited evidence
- canonical evidence clusters

This is much simpler than multiple judge dimensions and pairwise ranking.

### 16.3 Citation Validity

`citation_validity` is mechanical:

- cited ids exist
- cited ids resolve
- resolved objects point to canonical raw evidence

## 17. Cost-Control Strategy

Keeping cost low while preserving scientific credibility requires staged execution.

### 17.0 Cost Budget Formula

Before the main study, estimate cost with:

`total_cost = N_cells * (prep_cost + qa_cost + scoring_cost) + stability_audit_cost`

Where:

- `prep_cost`: ingest, embeddings, summaries, graph extraction, maintenance
- `qa_cost`: answer-generation across all checkpoint questions
- `scoring_cost`: fact scoring plus evidence-support scoring
- `stability_audit_cost`: extra replay or rerun budget for sampled cells

The pilot must measure each component separately so cost tradeoffs are visible.

### 17.1 Stage A: Implementation Smoke

Run:

- scopes: `S01`, `S08`, `S10`
- strategies: `null`, `rag_base`, one maintained strategy

Purpose:

- validate pipeline
- validate logs
- validate resume/replay
- validate scorer

### 17.2 Stage B: Feature Screening

Run:

- scopes: `S07`, `S08`, `S10`, `S11`
- all proposed strategies

Purpose:

- eliminate clearly dominated configurations
- estimate cost per run

### 17.3 Stage C: Confirmatory Main Study

Run:

- scopes: `S07-S12`
- only shortlisted strategies

Default confirmatory plan:

- `30` primary cells
- `6` stability-audit reruns
- `36` total run cells

Full confirmatory plan, if budget permits:

- `90` run cells from `5 strategies x 6 scopes x 3 repeats`

### 17.4 Cost Controls

- deterministic runs
- no duplicate inference due to idempotent caching
- embedding batching
- single judge call per answer
- optional human audit only on sample
- avoid static-driver secondary studies until the main study is complete

## 18. Implementation Plan

### 18.1 Phase 0: Design Freeze

Deliverables:

- benchmark manifest schema
- event schema
- scoring spec
- strategy config spec
- scope inclusion list

Exit criterion:

- all scoring and scope decisions frozen before implementation

### 18.2 Phase 1: Core Runtime

Build:

- manifest loader
- run state database
- event logger
- idempotent model broker
- Modal client wrapper
- replay and resume support

Exit criterion:

- a failed run can resume without duplicate inference

### 18.3 Phase 2: SynixMemory Base

Build:

- raw episode store
- chunking
- FTS
- embeddings
- RRF
- retrieval tools
- provenance mapping

Exit criterion:

- `rag_base` works end-to-end

### 18.4 Phase 3: Feature Modules

Build:

- core memory
- summaries
- graph layer
- maintenance jobs
- recency weighting if retained

Exit criterion:

- all shortlisted strategy configs materialize from one substrate

### 18.5 Phase 4: Scoring

Build:

- fact scorer
- evidence support scorer
- citation validator
- aggregation pipeline
- audit export tooling

Exit criterion:

- scoring can be rerun from saved answers without new inference

### 18.6 Phase 5: Pilot and Cost Calibration

Run Stage A and Stage B.

Exit criterion:

- cost per cell known
- runtime per cell known
- stability of logs and cache verified

### 18.7 Phase 6: Confirmatory Study

Run frozen main study under the final manifest.

Exit criterion:

- all primary tables regenerate from artifacts

## 19. Directory and Artifact Layout

Recommended study layout:

```text
studies/
  2026-xx-synix-memory-v1/
    study_manifest.json
    state.sqlite
    events.jsonl.zst
    cache/
      llm/
      embeddings/
      scoring/
    datasets/
      scope_manifests/
    runs/
      <run_id>/
        run_manifest.json
        answers.jsonl
        retrieval_trace.jsonl
        score.json
        artifacts/
    reports/
      summary.csv
      per_question.csv
      cost_report.csv
      audit_sample.csv
```

## 20. Commands to Support

The fresh-start benchmark should expose a minimal CLI.

- `bench study init`
- `bench study plan`
- `bench run start`
- `bench run resume`
- `bench run status`
- `bench score run`
- `bench score study`
- `bench replay run`
- `bench export report`

## 21. Publishability Criteria

The new benchmark should not be considered publishable until all of the following are true.

### 21.1 Reproducibility

- every published table regenerates from released artifacts
- every run has a manifest
- every model call is cached or logged
- rerunning scoring does not require rerunning inference

### 21.2 Scientific Clarity

- all systems are implemented on one controlled substrate
- claims are feature-level, not vendor-level
- primary score is simple and auditable
- no hard gates distort the ranking

### 21.3 Statistical Defensibility

- confidence intervals are reported
- multiple comparisons are corrected
- effect sizes are reported
- at least a sampled stability audit or true repeated cells exist

### 21.4 Practicality

- the full main study fits within a known Modal credit envelope
- failure recovery works
- a human can inspect any score back to the source evidence

## 22. Recommended Immediate Next Steps

1. Freeze the scope set: `S07-S12` main, `S13-S15` extension, optional `S01`.
2. Freeze the main strategy set.
3. Freeze the simplified scoring system.
4. Implement the execution substrate first: manifest, event log, cache, replay, resume.
5. Implement `rag_base` before any derived strategy.
6. Run the smallest possible smoke study on Modal to calibrate cost and runtime.

## 23. What Changes From the Current Benchmark

The fresh-start benchmark intentionally abandons several properties of the current implementation:

- no more product benchmark framing
- no more mixed scoring regimes
- no more hard gating
- no more internal agent-based adapters
- no more irreproducible cross-document summary tables

The benchmark becomes narrower, but much stronger.

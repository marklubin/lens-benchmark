# Execution Plan

## Summary

This plan assumes the following already exist:

- the datasets
- Synix
- Modal endpoints and credits
- the high-level benchmark direction

The fresh work is the checkpoint-scoped artifact-bank compiler, the runtime policy layer, the simplified scorer, and the operational process that prevents expensive rerun waste.

## Planning Assumption

The schedule below assumes one lead engineer with agentic parallel execution available for bounded tasks.

## Build-Time Estimate

### Fastest Credible MVP

Includes:

- schema freeze
- Modal broker and cache
- state store, resume, replay
- base artifact-bank compiler
- scoring v2
- `policy_base`, `policy_core`, `policy_summary`
- 2-scope smoke pilot

Estimated total: `5 to 7 working days`

### Minimal Publishable Package

Includes:

- the MVP above
- `policy_graph`
- 4-scope feature screening study
- stability reruns on a stratified subset
- reproducible report exports

Estimated total: `8 to 12 working days`

### Strong Main Study

Includes:

- full `S07-S12`
- main policy set
- pilot, screening, stability audit, and final reporting

Estimated total: `3 working weeks`

## Phases

### Phase 0: Program Freeze

Duration: `0.5 to 1 day`

Deliverables:

- final scope policy
- final runtime policy set
- final scoring v2 spec
- final artifact-bank model
- cost target for pilot and main study

Exit criteria:

- no open design churn on the benchmark fundamentals

### Phase 1: Schemas And Runtime Foundation

Duration: `1.5 to 2.5 days`

Deliverables:

- study manifest schema
- policy manifest schema
- artifact-bank manifest schema
- run manifest schema
- event schema
- score schema
- append-only event logger
- `state.sqlite`
- Modal broker
- idempotent cache
- resume support
- replay support

Exit criteria:

- failed build or run can resume without redoing completed model calls
- artifacts can be replayed without new inference

### Phase 2: Base Artifact-Bank Compiler

Duration: `1 to 2 days`

Deliverables:

- raw evidence ingestion
- checkpoint-scoped chunk artifacts
- FTS indexes
- embedding indexes
- RRF-ready hybrid retrieval views
- provenance-preserving ref resolution
- bank snapshot metadata

Exit criteria:

- the base bank compiles for a smoke-test scope
- checkpoint isolation is verified

### Phase 3: Scoring V2

Duration: `1 to 2 days`

Deliverables:

- fact scorer
- evidence-support scorer
- citation validator
- score export format
- re-score path from saved answers only

Exit criteria:

- scoring reruns without new inference
- scorer outputs stable JSON records

### Phase 4: Derived Artifact Families

Duration: `2 to 4 days`

Deliverables:

- core-memory artifact family
- summary artifact family
- graph artifact family
- retrieval exposure for each family back to source evidence

Exit criteria:

- all selected artifact families compile from one Synix-backed runtime
- policy execution uses compiled families without triggering rebuilds

### Phase 5: Runtime Policy Layer

Duration: `1 to 2 days`

Deliverables:

- `null` policy
- `policy_base`
- `policy_core`
- `policy_summary`
- `policy_graph`
- policy manifest loader
- policy-run accounting and logging

Exit criteria:

- the same compiled bank can be reused across multiple policies
- policy reruns hit cached compiled artifacts

### Phase 6: Pilot And Cost Calibration

Duration: `1 to 2 days`

Recommended pilot:

- scopes: `S08`, `S10`
- policies: `null`, `policy_base`, `policy_core`, `policy_summary`

Exit criteria:

- compile cost per checkpoint is known
- policy-run cost per cell is known
- resume and replay are validated under failure injection
- policy rerun does not recompile the bank

### Phase 7: Feature Screening Study

Duration: `2 to 3 days`

Recommended screening set:

- scopes: `S07`, `S08`, `S10`, `S11`
- main policy set

Exit criteria:

- primary policy comparisons exist on a publishable 4-scope slice
- cost and score deltas justify the final main-study matrix

### Phase 8: Confirmatory Main Study

Duration: depends on cost and runtime

Low-cost default:

- `4 scopes x 5 policies = 20 primary cells`
- rerun `4` stratified cells

Stronger default:

- `6 scopes x 5 policies = 30 primary cells`
- rerun `6` stratified cells

Full repeat design if budget allows:

- `6 scopes x 5 policies x 3 repeats = 90 cells`

Exit criteria:

- all reported tables regenerate from manifests and saved artifacts only

## Low-Cost Recommended Path

If the goal is fastest path to publishable-quality evidence:

1. freeze schemas and artifact-bank policy
2. implement Modal broker, cache, state store, resume, and replay
3. implement the base artifact-bank compiler
4. implement scoring v2
5. implement `policy_base`, `policy_core`, and `policy_summary`
6. run the 2-scope pilot
7. implement `policy_graph`
8. run the 4-scope screening study
9. expand to `S07-S12` only if cost and variance support it

## Work Sequencing Rules

1. Schemas before runtime code
2. Runtime foundation before expensive build work
3. Base artifact-bank compiler before policy comparison work
4. Scoring before serious studies
5. Pilot before screening and confirmatory runs
6. No run is valid before checkpoint-isolation tests pass

## Critical Checkpoints

### Checkpoint A

Can we fail and resume a build or run without redoing completed model calls?

### Checkpoint B

Can we prove that each bank snapshot was compiled only from the checkpoint prefix?

### Checkpoint C

Can we reproduce a score from saved answers only?

### Checkpoint D

Can we regenerate a summary table from manifests, bank metadata, run artifacts, and score records only?

If any answer is no, the system is not ready for serious runs.

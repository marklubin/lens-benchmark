# Execution Plan

## Summary

This plan now has two layers:

1. an upstream Synix platform milestone
2. the downstream LENS benchmark integration milestone

The benchmark should not reimplement core build or runtime primitives that belong in Synix.

The fresh work is to land the missing Synix platform features in sequence, then integrate the benchmark manifests, Modal broker, scoring, and study execution on top of those released contracts.

## Delivery Rule

For every Synix platform feature, definition of done is:

- design locked against actual Synix primitives and examples
- implementation complete
- unit tests complete
- at least one automated end-to-end test added
- documentation updated
- a demo or template extension note recorded, even if that demo work is deferred

Do not start the next Synix platform feature until the current one reaches that bar.

## Ownership Split

### Synix Platform Scope

Synix owns:

- immutable snapshots and aliases
- checkpoint projections and sealed bank manifests
- first-class projection dependencies
- retrieval APIs over named projections
- the Python-local runtime and tool API
- built-in chunk, summary, core-memory, and graph families
- typed schemas for those artifact families and tool payloads

### LENS Benchmark Scope

LENS owns:

- study and policy manifests
- the Modal broker and cache
- run state, replay, and cost accounting
- benchmark policy gating over Synix tools
- scoring, audit, and report generation
- study execution and result interpretation

## Sequential Work DAG

```text
[SYNIX EPIC]
    |
    v
[#34 Immutable snapshots]
    |
    v
[#81 Checkpoint projections + sealed bank manifests]
    |
    v
[#15 Projections as first-class DAG nodes]
    |
    v
[#10 Retrieval API over named projections]
    |
    v
[#82 Python-local runtime/tool API]
    |
    v
[#83 Built-in chunk family]
    |
    +-------------------+--------------------+
    |                   |                    |
    v                   v                    v
[#84 Summary family] [#85 Core-memory family] [#86 Graph family]
    |                   |                    |
    +-------------------+--------------------+
                        |
                        v
[#60 Typed schemas closeout]
                        |
                        v
[#87 Optional mesh/API parity]
                        |
                        v
[LENS integration milestone]
                        |
                        v
[T001] -> [T002] -> [T003] -> [T004] -> [T005] -> [T013]
                                                |        \
                                                v         v
                                              [T006]   [T007/T008/T009]
                                                 \        /
                                                  v      v
                                                   [T010]
                                                     |
                                                     v
                                                   [T011]
                                                     |
                                                     v
                                                   [T012]
```

## Synix Platform Milestone

### Phase S1: Immutable Snapshot Substrate

Land immutable build snapshots, shared object storage, and a movable `latest` or `HEAD` alias as generic Synix functionality.

Exit criteria:

- older snapshots never mutate
- clients mount immutable snapshot identifiers instead of a mutable build directory
- one automated e2e test verifies snapshot immutability and alias movement

### Phase S2: Checkpoint Projections And Sealed Banks

Tracked in: `#81`

Land prefix-valid checkpoint projections and sealed bank manifests on top of the snapshot substrate.

Exit criteria:

- one scope build can emit multiple checkpoint-valid banks
- later source data cannot mutate earlier checkpoint banks
- sealed manifests expose the named projections needed by downstream transforms and runtime mounts
- one automated e2e test proves no-future-leakage across checkpoints

### Phase S3: Explicit Projection Dependencies

Tracked in: `#15`

Make projections first-class DAG dependencies rather than relying on implicit `search.db` conventions.

Exit criteria:

- downstream transforms and runtime mounts depend on named projections explicitly
- one automated e2e test proves a projection dependency path works without direct file coupling

### Phase S4: Retrieval API Over Named Projections

Tracked in: `#10`

Expose a general retrieval contract for transforms over named projections and released retrieval modes.

Exit criteria:

- `keyword`, `semantic`, `hybrid`, and `layered` retrieval are callable through the registered projection path
- one automated e2e test verifies stable source-backed refs from that API

### Phase S5: Python-Local Runtime And Tool API

Tracked in: `#82`

Add the default local runtime mount for a sealed bank with standard benchmark-friendly tool calls.

Exit criteria:

- an external agent loop can use the runtime without opening Synix internals directly
- one automated e2e test mounts a bank and executes a tool-backed run

### Phase S6: Built-In Chunk Family

Tracked in: `#83`

Land chunking as a Synix built-in with stable IDs and provenance-safe citation anchors.

Exit criteria:

- chunk outputs are typed, stable, and citation-safe
- one automated e2e test runs source to chunk to retrieval

### Phase S7: Built-In Summary, Core-Memory, And Graph Families

Tracked in: `#84`, `#85`, and `#86`

Land the derived memory families as Synix built-ins, one family at a time.

Exit criteria per family:

- typed artifacts with provenance chains
- runtime exposure through the standard tool API
- at least one automated e2e test showing retrieval back to source evidence

### Phase S8: Typed Schema Closeout

Tracked in: `#60`

Close the platform milestone by making the built-in artifact schemas and tool payload schemas explicit.

Exit criteria:

- built-ins and runtime responses validate against declared schemas
- one automated e2e test exercises schema-valid artifacts and runtime payloads

### Phase S9: Demo And Mesh Follow-Ons

Tracked in: `#87` for Mesh parity, with demo follow-ons recorded on each feature issue.

These are follow-ons, not blockers for the first benchmark integration:

- Mesh and local runtime parity for supported retrieval modes
- demo or template extensions for each newly landed Synix feature

## LENS Integration Milestone

### Phase L0: Program Freeze

Freeze the benchmark scope, runtime policy set, artifact-bank model, scoring policy, and the Synix/LENS ownership boundary.

### Phase L1: Schemas And Runtime Foundation

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

### Phase L2: Synix Base-Bank Integration

Deliverables:

- benchmark-side selection of sealed Synix checkpoint bank manifests
- benchmark-side lookup of named projection handles from those manifests
- integration with Synix chunk and layered search artifacts
- provenance-preserving ref resolution
- bank manifest metadata and manifest wiring

Exit criteria:

- the benchmark mounts the released Synix base bank for a smoke-test scope
- checkpoint isolation is verified

### Phase L3: Runtime Policy Integration

Deliverables:

- benchmark runtime wrapper over the Synix tool surface
- policy access to named projections through the sealed manifest contract
- `null` policy
- `policy_base`
- policy gating and accounting

Exit criteria:

- the same sealed bank can be reused across multiple policies
- policy reruns do not trigger rebuilds

### Phase L4: Derived-Family Integration

Deliverables:

- `policy_core`
- `policy_summary`
- `policy_graph`
- benchmark-side configuration for released Synix built-in families

Exit criteria:

- all selected artifact families run from one Synix-backed runtime
- policy execution uses compiled families without triggering rebuilds

### Phase L5: Scoring V2

Deliverables:

- fact scorer
- evidence-support scorer
- citation validator
- score export format
- re-score path from saved answers only

Exit criteria:

- scoring reruns without new inference
- scorer outputs stable JSON records

### Phase L6: Pilot And Cost Calibration

Recommended pilot:

- scopes: `S08`, `S10`
- policies: `null`, `policy_base`, `policy_core`, `policy_summary`

Exit criteria:

- compile cost per checkpoint is known
- policy-run cost per cell is known
- resume and replay are validated under failure injection
- policy rerun does not recompile the bank

### Phase L7: Feature Screening Study

Recommended screening set:

- scopes: `S07`, `S08`, `S10`, `S11`
- main policy set

Exit criteria:

- primary policy comparisons exist on a publishable 4-scope slice
- cost and score deltas justify the final main-study matrix

### Phase L8: Confirmatory Main Study

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

1. finish the Synix platform milestone through the built-in summary and core-memory families
2. freeze schemas and artifact-bank policy in LENS
3. implement Modal broker, cache, state store, resume, and replay
4. integrate the base Synix bank path and the Synix runtime/tool API
5. integrate `policy_base`, `policy_core`, and `policy_summary`
6. implement scoring v2
7. run the 2-scope pilot
8. integrate `policy_graph`
9. run the 4-scope screening study
10. expand to `S07-S12` only if cost and variance support it

## Work Sequencing Rules

1. Synix platform features are delivered one at a time in dependency order.
2. Every Synix platform feature must land with unit coverage, at least one automated e2e test, docs updates, and a demo-extension note.
3. LENS does not reimplement Synix platform primitives locally while the upstream milestone is open.
4. Schemas before benchmark runtime code.
5. Runtime foundation before expensive build work.
6. Base bank integration before policy comparison work.
7. Scoring before serious studies.
8. Pilot before screening and confirmatory runs.
9. No run is valid before checkpoint-isolation tests pass.

## Critical Checkpoints

### Checkpoint A

Can we fail and resume a build or run without redoing completed model calls?

### Checkpoint B

Can we prove that each checkpoint bank was compiled only from the checkpoint prefix?

### Checkpoint C

Can we reproduce a score from saved answers only?

### Checkpoint D

Can we regenerate a summary table from manifests, bank metadata, run artifacts, and score records only?

If any answer is no, the system is not ready for serious runs.

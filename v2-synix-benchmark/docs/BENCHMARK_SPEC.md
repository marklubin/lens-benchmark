# Benchmark Spec

## Objective

Build a controlled benchmark of memory mechanisms for longitudinal question answering over the existing LENS datasets.

The benchmark compares runtime policies implemented on one shared Synix-backed artifact compiler.

## Benchmark Model

### Compile Once Per Checkpoint Prefix

For each `scope x checkpoint`, Synix compiles a checkpoint-scoped artifact bank from the episode prefix available at that checkpoint.

The artifact bank may include:

- raw episode artifacts
- chunk artifacts
- hybrid search indexes
- core-memory artifacts
- summary artifacts
- graph artifacts

This compilation is shared across policies.

### Run Policies Over The Compiled Bank

A benchmark policy does not rebuild memory.

A policy selects:

- which artifact families are visible
- which search surfaces are queried
- how hits are fused and ranked
- what retrieval caps apply
- what the agent can cite back to source evidence

This is the main cost and control advantage of the V2 design.

### Scientific Constraint

No artifact bank may contain information from episodes beyond the active checkpoint prefix.

That rule is mandatory. Violating it invalidates the run.

## Primary Scope Set

### Main Benchmark

Use `S07-S12`.

This gives two main content families:

- `S07-S09`: narrative, heterogeneous, cross-document reasoning
- `S10-S12`: semantic retrieval stress, high-overlap retrieval challenge

### Extension Set

Use `S13-S15` only as a separate extension table.

### Optional Smoke-Test Anchor

Use `S01` only if a simple numeric smoke test is needed.

## Minimal Scope Count

### Engineering Smoke Test

Use `S08` and `S10`.

### Minimal Credible Internal Benchmark

Use `S07`, `S08`, `S10`.

### Minimal Publishable Benchmark

Use `S07`, `S08`, `S10`, `S11`.

### Strong Main Study

Use all `S07-S12`.

## Runtime Policy Set

### Main Cost-Reduced Set

1. `null`
2. `policy_base`
3. `policy_core`
4. `policy_summary`
5. `policy_graph`

### Policy Definitions

#### `null`

No memory retrieval tools. This is the no-memory baseline.

#### `policy_base`

Uses only raw or chunk search surfaces:

- FTS
- cosine
- RRF over chunk hits

#### `policy_core`

Uses the `policy_base` surfaces plus core-memory artifacts.

#### `policy_summary`

Uses the `policy_base` surfaces plus summary artifacts.

#### `policy_graph`

Uses the `policy_base` surfaces plus graph artifacts and graph-informed expansion over source evidence.

### Expanded Diagnostic Set

If cost allows, add combined policies only after the main five are stable:

1. `policy_core_summary`
2. `policy_graph_summary`
3. `policy_all`

Do not start with combined policies.

## Memory Artifact Families To Build

### Shared Foundation

These are required for all non-null policies:

- immutable raw episode store
- chunking
- FTS retrieval
- cosine retrieval
- RRF fusion
- provenance mapping
- stable citation ids
- batch retrieval
- Modal call broker
- append-only event log
- cache, resume, replay

### Derived Artifact Families

#### `core_memory`

Maintained working-state memory blocks intended to surface durable cross-episode context.

#### `summaries`

Checkpoint-level or hierarchical summaries that preserve source provenance.

#### `graph`

Entity and relation artifacts plus graph-informed retrieval back to source evidence.

#### `maintenance`

Maintenance is not a separate runtime policy.

Maintenance is the checkpoint-scoped compilation process that updates derived artifacts before policy runs.

#### `recency_weighting`

Time-aware retrieval weighting. Optional for later variants, not required for the initial main study.

## Out Of Scope For V1

- raw evidence deletion or hard expiration
- internal answering agents
- multi-agent memory orchestration
- vendor or product comparisons
- multiple inference providers
- large sets of chunking variants as headline systems
- combinatorial policy sweeps over every feature subset

## Core Invariants

1. Raw evidence is stored verbatim and never deleted.
2. Every derived artifact points back to source episode ids.
3. The benchmark agent always produces the final answer.
4. All policies use the same agent loop and tool interface.
5. All inference flows through Modal.
6. All policies operate on checkpoint-scoped artifact banks.
7. Policy execution must not trigger artifact rebuilds unless the build manifest changed.

## Cost-Control Rules

1. Compile artifact families once per `scope x checkpoint` and reuse them across policies.
2. Use one canonical configuration per artifact family for the initial study.
3. Treat additional artifact variants as separate costed decisions.
4. Keep the initial main policy set to the five single-family policies above.
5. Record compilation cost separately from policy-run cost.

## Minimal Publishable Study Design

Recommended default:

- scopes: `S07`, `S08`, `S10`, `S11`
- runtime policies: `null`, `policy_base`, `policy_core`, `policy_summary`, `policy_graph`
- one deterministic run per cell
- targeted reruns on a stratified subset of cells for stability auditing

Expanded main study:

- scopes: `S07-S12`
- same runtime policy set

## Primary Hypotheses

1. `policy_base` is strong but not uniformly best across all long-form scope types.
2. `policy_core` helps more on narrative continuity than on adversarial semantic retrieval.
3. `policy_summary` helps more on high-overlap scopes where filtering matters.
4. `policy_graph` helps more when entity linkage across documents is the key difficulty.
5. Shared offline compilation plus cheap runtime policy comparison substantially lowers study cost and reduces experimental thrash.

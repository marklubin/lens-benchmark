# Decisions

Append-only design and policy decisions.

## D001 - Use Synix As The Sole Benchmark Substrate

Date: 2026-03-06
Status: accepted

Decision:

Use Synix as the implementation substrate for all V2 memory mechanism replicas.

Rationale:

- gives one controlled runtime
- aligns dataset generation and benchmark implementation story
- avoids third-party product confounds

## D002 - Remove Hard Gating From The Primary Score

Date: 2026-03-06
Status: accepted

Decision:

Do not use hard gates in the primary composite.

Rationale:

- reduces ranking distortion
- simplifies interpretation
- improves reproducibility and auditability

## D003 - Main Scope Family Is S07-S12

Date: 2026-03-06
Status: accepted

Decision:

Use `S07-S12` as the main benchmark, `S13-S15` as extension only, and optionally `S01` as a smoke-test anchor.

Rationale:

- focuses effort where memory representation matters most
- reduces numeric-scope overhead
- keeps the main study coherent

## D004 - Modal Only For Inference

Date: 2026-03-06
Status: accepted

Decision:

All inference for V2 must go through Modal.

Rationale:

- existing credits are there
- cost tracking is easier on one provider path
- avoids new infrastructure churn

## D005 - Runtime Foundation Before Expensive Study Work

Date: 2026-03-06
Status: accepted

Decision:

Do not run expensive compilation or study work before the manifest, event log, cache, resume, and replay substrate exists.

Rationale:

- prevents repeated inference waste
- reduces thrash during pilot execution

## D006 - V2 Uses Checkpoint-Scoped Artifact-Bank Compilation

Date: 2026-03-06
Status: accepted

Decision:

Compile memory artifacts once per `scope x checkpoint` from the episode prefix available at that checkpoint.

Rationale:

- prevents future-leakage
- shares expensive compilation across policies
- matches the scientific unit of available memory state

## D007 - Benchmark Comparisons Are Runtime Policies Over A Shared Bank

Date: 2026-03-06
Status: accepted

Decision:

Treat the main benchmark variants as runtime policies over a compiled artifact bank rather than as separately built systems.

Rationale:

- isolates policy differences from build-system churn
- makes new comparisons cheaper to add
- better demonstrates the value of Synix as a memory compiler and workbench

## D008 - One Canonical Configuration Per Artifact Family For The Initial Study

Date: 2026-03-06
Status: accepted

Decision:

Use one canonical configuration per artifact family in the initial main study.

Rationale:

- prevents combinatorial artifact explosion
- keeps cost and attribution tractable
- makes the first study interpretable

## D009 - Synix Owns The Snapshot, Bank, Artifact-Family, And Tooling Substrate

Date: 2026-03-06
Status: accepted

Decision:

Treat Synix as the upstream platform for immutable snapshots, checkpointed banks, built-in chunk and memory artifact families, and the default Python-local runtime and tool interface.

Rationale:

- keeps benchmark-specific logic out of the platform layer
- prevents duplicated compiler work in this repository
- gives the benchmark one stable upstream contract to consume

## D010 - Checkpoint Isolation Must Be Enforced During Build, Not By Query Masking

Date: 2026-03-06
Status: accepted

Decision:

Implement checkpoint isolation by prefix-valid snapshot or projection semantics inside Synix. Post-hoc query masking is not an acceptable substitute.

Rationale:

- derived summaries, graphs, and rankings can leak future information before query-time filtering
- build-time isolation is the only defensible scientific unit
- this aligns the benchmark with the intended Synix snapshot model

## D011 - Python-Local Synix Runtime Is The First Integration Path

Date: 2026-03-06
Status: accepted

Decision:

Use a Python-local Synix runtime and tool surface as the first benchmark integration path. Mesh or HTTP parity is a follow-on concern.

Rationale:

- reduces integration surface area for the first milestone
- matches current Synix strengths more closely than the mesh layer
- keeps benchmark policy work focused on one auditable runtime path

## D012 - Named Projections And Sealed Manifests Are The Benchmark Integration Contract

Date: 2026-03-06
Status: accepted

Decision:

Treat sealed bank manifests plus named projection handles as the only benchmark-facing contract for compiled Synix banks. LENS runtime code, policies, and scoring may consume projections and refs through that contract, but must not couple to internal file layouts or mutable build directories.

Rationale:

- makes checkpoint isolation auditable at the manifest boundary
- prevents downstream coupling to incidental Synix file structure
- gives chunk, summary, core-memory, and graph integrations one uniform access pattern

## D013 - Synix Build And Release Are Separate Lifecycle Boundaries

Date: 2026-03-06
Status: accepted

Decision:

Treat `synix build` as immutable snapshot construction only, and treat `synix release` as the only durable materialization path. Benchmark integration should rely on immutable build refs, release refs, and sealed manifests rather than mutable local build directories.

Rationale:

- removes ambiguity between immutable state and mutable realization
- makes CI or CD promotion and rollback ref-addressed operations instead of workspace mutations
- gives checkpoint banks and future runtime mounts a cleaner upstream contract

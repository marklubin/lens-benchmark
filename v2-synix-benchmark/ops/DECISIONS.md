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

# Runbook

## Purpose

This runbook defines the standard operating procedure for creating, compiling, running, resuming, replaying, and verifying benchmark studies.

## Before Starting A Study

1. Confirm the relevant task is `in_progress`.
2. Confirm a study manifest exists and is frozen.
3. Confirm the policy manifests are frozen for the study.
4. Confirm the artifact-family configs are frozen for the study.
5. Confirm the checkpoint mapping for each scope.
6. Confirm Modal endpoints are pinned and reachable.
7. Confirm cache location and state database path.
8. Confirm scoring version and prompt-set version.

## Starting A Study

1. Create `study_manifest.json`.
2. Materialize policy manifests.
3. Materialize artifact-bank manifests for each `scope x checkpoint`.
4. Materialize run manifests.
5. Initialize `state.sqlite`.
6. Start `events.jsonl`.
7. Record `study_started` event.
8. Launch compilation and runs through the orchestrator, not ad hoc scripts.

## During Execution

For every study:

1. record `bank_build_started`
2. ingest the checkpoint prefix episodes
3. compile required artifact families
4. persist bank snapshot metadata
5. record `bank_build_completed` or `bank_build_failed`
6. execute policy runs against the compiled bank
7. answer questions
8. score outputs
9. persist final artifacts
10. record `run_completed` or `run_failed`

## Compile-Once Rule

Policy execution should reuse compiled artifact banks.

A repeated run under a different policy should not trigger artifact recompilation unless the bank manifest hash changed.

## On Failure

1. Do not delete partial artifacts or partial bank snapshots.
2. Record the failure event with stack trace and failure classification.
3. Inspect `state.sqlite`, cache state, and bank snapshot status.
4. Resume from the orchestrator.
5. Verify that completed model calls and completed bank builds were not repeated.

## Resume Procedure

1. open the study state database
2. locate incomplete bank builds and incomplete question steps
3. verify cache integrity
4. restart only incomplete steps
5. append new events, do not rewrite old ones

## Replay Procedure

Replay is for audit and reporting, not new inference.

1. use stored bank manifests, run artifacts, and score artifacts
2. do not hit Modal
3. regenerate summary tables and exports from saved state
4. compare regenerated outputs to archived reports

## Verification Checklist

A study is considered valid only if all of the following are true:

- every run has a study manifest, policy manifest, and run manifest
- every compiled bank snapshot has a bank manifest
- every run has an event trail
- cached model responses exist for all completed calls
- score artifacts exist for all completed answers
- summary tables regenerate from saved artifacts
- no unexplained missing cells remain
- no evidence of future-leakage exists in any bank snapshot

## Cost Verification Checklist

For each study, export:

- total compilation calls
- total policy-run calls
- total prompt tokens
- total completion tokens
- total estimated cost
- compilation cost per scope and checkpoint
- policy-run cost per policy
- cost per completed cell

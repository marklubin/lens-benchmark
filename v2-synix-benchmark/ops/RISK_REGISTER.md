# Risk Register

| ID | Risk | Impact | Likelihood | Mitigation | Owner | Status |
|----|------|--------|------------|------------|-------|--------|
| R001 | Runtime lacks reliable resume and replays expensive calls | High | High | Build cache, state store, and replay before study work | unassigned | open |
| R002 | Artifact-bank builds leak future information across checkpoints | High | Medium | Add explicit checkpoint-isolation manifests and tests before pilot runs | unassigned | open |
| R003 | Scorer remains too complex or unstable | High | Medium | Keep scoring to 3 primary metrics and validate via audit | unassigned | open |
| R004 | Main study matrix is too large for Modal credit budget | High | Medium | Separate compilation cost from policy-run cost, then freeze smaller matrix | unassigned | open |
| R005 | Provenance leaks or broken citations make results unauditable | High | Medium | Test provenance and citation validity early in the base artifact bank | unassigned | open |
| R006 | Scope or policy set expands again and causes thrash | Medium | Medium | Require decision entry for any scope or policy change | unassigned | open |
| R007 | Multiple workers step on the same files and create churn | Medium | Medium | Strict task ownership and scope boundaries | unassigned | open |
| R008 | Artifact family variants explode and erase the cost advantage | Medium | Medium | Freeze one canonical config per family for the initial study | unassigned | open |
| R009 | Graph retrieval semantics underperform despite low infra risk | Medium | Medium | Keep graph policy simple and provenance-first in V1 | unassigned | open |

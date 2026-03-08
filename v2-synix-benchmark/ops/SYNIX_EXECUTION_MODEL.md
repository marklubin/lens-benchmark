# Synix Execution Model For LENS

This note records the benchmark-side view of the Synix snapshot, projection, and release model that should govern the remaining upstream execution work.

Status as of 2026-03-06:

- landed upstream: build-time search capability (`SearchSurface`), explicit transform search access (`TransformContext` and `ctx.search(...)`), and canonical search outputs (`SynixSearch`)
- still open upstream: snapshot or projection or release closeout, sealed checkpoint banks, and the mounted runtime or tool contract

## Core Position

Use the strict model:

- `synix build` only computes immutable state
- `synix release` is the only command that materializes projections anywhere
- `HEAD` is the default read surface for immutable inspection
- temporary realizations are an internal implementation detail, not a user-visible release
- mutable local directories are not part of the public contract

That means Synix should not preserve a special semantic `build/` directory. If a user wants a persistent local materialization, they should invoke `synix release` explicitly and choose a target root explicitly.

## Before And After Ergonomics

### Before

The old mental model mixed build state and released state:

- `synix build` both computed artifacts and refreshed mutable local outputs
- the current directory layout acted like an implicit source of truth
- `search`, `status`, and `verify` depended on whatever happened to be live locally
- revert semantics were fuzzy because there was no clean distinction between immutable history and mutable realization

Typical operator flow:

```bash
synix build
synix search "pricing"
synix verify
```

This was easy to type, but it left the important question ambiguous: did the user inspect immutable snapshot state or a mutable local realization that might have drifted?

### After

The new mental model separates commit from materialization:

- `synix build` produces immutable artifacts plus immutable projection build state in `.synix`
- `synix release` materializes one chosen snapshot to one chosen target
- `synix diff`, `list`, `show`, and `lineage` operate over immutable refs by default
- commands that need a realized projection may use an isolated scratch realization under the hood, but they do not mutate a named release unless the user explicitly asks for it

Typical operator flow:

```bash
synix build
synix list
synix show core
synix diff HEAD refs/runs/20260306T230000Z-prev
synix release HEAD --to refs/releases/local --target-root ./.synix/releases/local/current
synix search "pricing" --release refs/releases/local
```

This is slightly more explicit, but it is much easier to reason about:

- build creates immutable state
- release materializes immutable state
- revert re-releases older immutable state

## State Model

Synix should expose five kinds of state.

### 1. Immutable artifact state

Artifacts are committed into `.synix` and addressed by object identifiers. This is the canonical build substrate.

### 2. Immutable projection build state

Every projection should also emit immutable build state into `.synix`. This is the snapshot-time handoff to release adapters. It is not the same thing as a released file on disk.

### 3. Build refs

Build refs identify immutable build history:

- `HEAD`
- `refs/heads/*`
- `refs/runs/*`

These refs should move only when a build transaction commits successfully.

### 4. Release refs

Release refs identify durable materializations:

- `refs/releases/local`
- `refs/releases/staging`
- `refs/releases/prod`
- `refs/releases/ab-a`
- `refs/releases/ab-b`

Each release ref points at one immutable snapshot plus the receipts that describe how that snapshot was materialized at a target.

### 5. Ephemeral realizations

Some commands may need a realized projection to answer a query. They should use isolated scratch materialization rooted under `.synix/work/...` and discard it afterward. Scratch materialization is not a release and should never move a release ref.

## Command Semantics

### Build

`synix build` should:

1. compute incremental artifacts
2. compute immutable projection build state
3. write the snapshot manifest into `.synix`
4. advance `HEAD` and mint a new `refs/runs/*`

`synix build` should not:

- refresh a local materialized directory
- mutate `refs/releases/*`
- imply that any projection is currently live anywhere

### Inspect

These commands should default to immutable refs and read directly from `.synix`:

- `synix list`
- `synix show`
- `synix lineage`
- `synix diff`

`--ref` should select another immutable ref without requiring any release.

### Search

Search has two acceptable execution paths:

- query canonical immutable search state directly if the adapter supports it
- otherwise stage an isolated scratch realization for the requested ref, query it, then delete it

Search should not silently mutate a persistent release target just because the user asked a question.

### Release

`synix release` should:

1. resolve one source ref
2. load its immutable projection build state
3. apply one release plan to one target root
4. write per-projection receipts
5. move one destination release ref only after the entire release succeeds

Release is the only command that should create or update durable realized files.

### Revert

`synix revert <older-ref> --to <release-ref>` should just be a convenient spelling for releasing an older immutable snapshot to the same destination. Revert should not attempt inverse filesystem edits.

## Pipeline Management Flow

The intended day-to-day operator loop is:

1. build immutable state
2. inspect and verify that immutable state
3. release one chosen snapshot to one chosen environment
4. promote or revert by naming refs, not by rebuilding or copying files

Example:

```bash
synix build
synix list --ref HEAD
synix show core --ref HEAD
synix verify --ref HEAD
synix release HEAD --to refs/releases/local --target-root ./.synix/releases/local/current
synix refs show refs/releases/local
synix release HEAD --to refs/releases/staging --target-root /srv/synix/staging
synix revert refs/runs/20260306T230000Z-prev --to refs/releases/staging
```

What changes for the user:

- a build never implies a deployment
- a deployment always points back to one immutable snapshot
- promotion, rollback, and audit all use the same ref vocabulary

## Ref Management

Ref management should be explicit and inspectable:

- `synix refs list`
- `synix refs show <ref>`
- `synix runs list`

The important split is:

- build refs tell the user what immutable states exist
- release refs tell the user what is materially live where

That split is what makes promotion and rollback easy to reason about.

## Major User Stories

### Incremental Builds

Goal:
compile only what changed while keeping every checkpoint inspectable.

Operator flow:

```bash
synix build
synix diff HEAD refs/runs/20260306T230000Z-prev
synix list --ref HEAD
synix show monthly --ref HEAD
```

How it should work:

- incremental build logic reuses prior immutable objects wherever fingerprints match
- the new snapshot records both artifact state and projection build state
- the user can compare the new snapshot against an older run without ever touching a mutable live directory

Why this is better for LENS:

- checkpoint compilation becomes auditable
- cost and replay accounting can key off immutable refs
- resume semantics are simpler because build and release are separate concerns

### CI Or CD Promotion

Goal:
build once in CI, then promote that exact snapshot through staging and production.

Operator flow:

```bash
synix build
synix verify --ref HEAD
synix release HEAD --to refs/releases/staging --target-root /srv/synix/staging
synix release HEAD --to refs/releases/prod --target-root /srv/synix/prod
```

How it should work:

- CI creates an immutable run ref and records verification against that ref
- CD promotes the same ref rather than rebuilding
- release receipts tell operators exactly what snapshot each environment is serving

Why this is better for LENS:

- benchmark-side bank compilation can be separated from runtime deployment
- study manifests can cite immutable upstream refs directly
- failure analysis no longer depends on preserving a mutable workspace

### A Or B Testing

Goal:
serve two candidate releases side by side and compare behavior without rebuilding either one.

Operator flow:

```bash
synix release refs/runs/20260306T230000Z-a --to refs/releases/ab-a --target-root /srv/synix/ab/a
synix release refs/runs/20260306T231500Z-b --to refs/releases/ab-b --target-root /srv/synix/ab/b
synix refs show refs/releases/ab-a
synix refs show refs/releases/ab-b
```

How it should work:

- each release ref points to one immutable source snapshot
- each target root is isolated
- diff and verification can compare the underlying immutable refs rather than guessing from current files

Why this is better for LENS:

- runtime policy experiments can compare banks or projections without recompiling the source corpus
- provenance is preserved for each side of the test
- promotion from canary to prod becomes ref movement plus explicit release, not ad hoc copying

### Immutable Checkpoints

Goal:
treat each checkpoint-scoped memory bank as a sealed, auditable object that can later be mounted by the runtime.

Checkpoints are not a Synix platform concept. They are LENS pipeline logic (see D014 in `ops/DECISIONS.md`).

The pipeline defines one projection per checkpoint, each scoped to an episode prefix via label filtering. A single `synix build` compiles all artifacts from the full corpus. Each projection selects only the artifacts derived from the checkpoint's episode prefix.

Operator flow:

```bash
synix build                          # one build, all artifacts, all checkpoint projections
synix refs show HEAD
synix release HEAD --to cp01         # materializes cp01 projection (episodes 1-6)
synix release HEAD --to cp02         # materializes cp02 projection (episodes 1-12)
synix release HEAD --to cp04         # materializes cp04 projection (episodes 1-20)
```

How it works:

- the pipeline definition declares per-checkpoint projections with prefix-filtered input artifacts
- one build produces all artifacts and all projection declarations
- each release materializes one checkpoint's projection with its own receipt
- the release receipt is the sealed bank manifest

Why this is better for LENS:

- checkpoint isolation is enforced at projection declaration time
- no Synix platform concept needed — just label filtering over existing projection model
- runtime mounting consumes release receipts
- study replay and audit cite concrete refs and receipts

## Remaining Upstream Execution Critical Path

The design above changes what remains on the Synix side.

Already landed enough to stop reopening:

- `SearchSurface` for build-time search capability
- `TransformContext` and `ctx.search(...)` for explicit transform access
- `SynixSearch` as the canonical search output contract

Still open and sequential:

1. projection release v2 (PR #92 / #34) — in progress
2. Python-local runtime or tool API (#82), including retrieval over named search surfaces
3. built-in chunk family (#83) with stable IDs and provenance anchors
4. built-in summary, core-memory, and graph families (#84, #85, #86 — parallel)
5. typed schema closeout (#60) and optional mesh parity (#87) later

Checkpoint isolation (#81) has been eliminated as a Synix requirement — handled by LENS pipeline logic (D014).

For LENS planning, `T005` and `T013` should stay blocked until items 1 through 3 are materially landed.

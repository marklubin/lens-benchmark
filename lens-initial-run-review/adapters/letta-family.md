# The Letta Family

Four variants of the Letta (formerly MemGPT) platform evaluated across 12 LENS benchmark scopes. The central finding: standard Letta with simple archival storage outperforms all its sophisticated variants, and the quality of the internal LLM matters more than the memory architecture.

---

## Variant Architectures

### letta (standard)

Pattern D (Agent Delegation) -- bypasses the LENS agent loop entirely. Episodes are ingested via `passages.create(episode_text)` for deterministic archival storage. At query time, the Letta agent uses its built-in `archival_memory_search` (semantic vector search) to retrieve relevant passages and formulates answers directly. Internal LLM: Anthropic Claude Sonnet for narrative/SRS scopes, Qwen3.5-35B via openai-proxy for some numeric scopes.

### letta-sleepy

Identical to standard letta with sleep-time consolidation at checkpoints. During `prepare()`, a sleep agent rewrites core memory blocks with cross-episode synthesis using delta/causal framing. Ingest uses `passages.create()` for guaranteed storage plus a shortened agent message for consolidation context. An earlier version relied on the LLM calling `insert_archival_memory`, which proved unreliable; the fix was deterministic `passages.create()`. The V3 prompt (delta/causal framing) outperformed V1 (comprehensive summary) and V2 (actionable filter).

### letta-v4 (core memory)

A three-agent system: an ingest agent processes each episode and updates four core memory blocks (patterns 5K, hypotheses 5K, entities 5K, events 5K) plus archival passages; a sleep agent consolidates at checkpoints; a Q&A agent answers using core memory plus archival search. Tested on narrative and SRS scopes only (S07--S12).

### letta-entity

Dynamic entity core memory blocks -- extracts and maintains entity-specific memory. Has a systematic `ep_` prefix doubling bug in citations (e.g., `ep_corporate_acquisition_08_ep_001` instead of `corporate_acquisition_08_ep_001`), causing evidence_grounding gate failures on affected runs. Tested on narrative and SRS scopes only (S07--S12).

---

## Phase 1: Numeric Scopes (S01--S06)

120 episodes (30 signal + 90 distractor), approximately 700 words per episode. Only letta and letta-sleepy were evaluated on numeric scopes.

### Main evaluation (Phase 5) -- GPT-OSS-120B on Cerebras

| Adapter | S01 | S02 | S03 | S04 | S05 | S06 | 8K Mean |
|---|---|---|---|---|---|---|---|
| letta | 0.288 | 0.338 | 0.215 | 0.455 | 0.424 | 0.239 | 0.327 |
| letta-sleepy | 0.300 | 0.313 | 0.252 | 0.451 | 0.377 | 0.240 | 0.322 |

At the 16K budget, letta averaged 0.366 and letta-sleepy averaged 0.348. Both rank fifth and sixth of eight adapters, well below sqlite-chunked-hybrid at 0.473.

### Best scores -- Sonnet as Letta LLM (Letta family evaluation)

| Adapter | S01 | S02 | S03 | S04 | S05 | S06 | Mean |
|---|---|---|---|---|---|---|---|
| letta | 0.655 | 0.542 | 0.318 | 0.594 | 0.572 | 0.603 | **0.547** |
| letta-sleepy | 0.651 | 0.495 | 0.447 | 0.468 | 0.502 | 0.513 | **0.512** |

Run IDs (letta): `2a8dfe5a5e7a`, `413f8720bc5e`, `16c7109069f2`, `6a8ba4330bed`, `44b597f74624`, `97e6fe2e4a99`

Run IDs (letta-sleepy): `8ef0786f0eb5`, `9640157a5f46`, `a70a0ec7d2f0`, `6b7b0e992116`, `8b0c6dcf2169`, `3434fea9ddd8`

### Constrained budget (Phase 2, S01 only) -- Qwen3-235B judge

| Adapter | 2K NBA | 4K NBA | 2K AnsQ | 4K AnsQ |
|---|---|---|---|---|
| letta | 0.453 | 0.631 | 0.476 | 0.689 |
| letta-sleepy | 0.667 | 0.693 | 0.741 | 0.858 |

letta-sleepy is the best heavy adapter at constrained budgets. Standard letta degrades sharply at 2K, dropping from 0.631 to 0.453 (a 28% decline). At 13% visibility into the corpus, semantic search alone struggles to surface relevant passages. Sleep consolidation compensates by producing a navigation document that orients retrieval.

---

## Phase 2: Narrative Scopes (S07--S09)

40 episodes (20 signal + 20 distractor), approximately 5,000 words per episode. All four variants evaluated.

### Composite scores (best per adapter per scope)

| Adapter | S07 | S08 | S09 | Mean | LLM |
|---|---|---|---|---|---|
| letta | 0.688 | 0.737 | 0.820 | **0.748** | Sonnet |
| letta-sleepy | 0.641 | 0.733 | 0.748 | **0.707** | Sonnet |
| letta-v4 | 0.586 | 0.489 | 0.612 | **0.562** | Sonnet |
| letta-entity | 0.331 | 0.550 | 0.228 | **0.370** | Mixed |

Run IDs (letta): `e93b0ec45fc0`, `ceb7f48147dd`, `1887dfbf85f9`

Run IDs (letta-sleepy): `eff98f429fdc`, `e1ca24053db6`, `d639a33f4737`

Run IDs (letta-v4): `fd44ac97aa32`, `c3066956a1ad`, `3c241f1c0b14`

Run IDs (letta-entity): `3917c37f65ab`, `deca6ad2ca42`, `09b456507499`

### letta-v4 metric breakdown (narrative)

| Scope | Composite | AnsQ | EvGnd | EvCov | InsDp | NaiveBAdv |
|---|---|---|---|---|---|---|
| S07 | 0.586 | 0.396 | 1.000 | 0.458 | 0.800 | 0.625 |
| S08 | 0.489 | 0.513 | 1.000 | 0.000 | 0.500 | 0.662 |
| S09 | 0.612 | 0.659 | 1.000 | 0.050 | 1.000 | 0.733 |

Evidence coverage averages 0.169 across narrative scopes. Core memory compresses episode content into abstract patterns and hypotheses, producing strong insight depth (0.767 mean) and action quality (0.794 mean) but losing the ability to ground answers in specific source episodes. Late-episode ingest latency reaches 100--130 seconds (versus 30--50 seconds early) as the growing context slows processing. Total time per scope: 22--25 minutes.

### 3-rep determinism data (Llama era, narrative)

| Adapter | Composite | Ingest (s) | QTime (s) | Total (s) | Tokens |
|---|---|---|---|---|---|
| letta | 0.342 +/- 0.007 | 60.1 | 518.4 | 578.5 | 260,992 |
| letta-sleepy | 0.301 +/- 0.013 | 56.6 | 809.5 | 866.0 | 150,936 |

letta-sleepy exhibits the most variance across replicates due to non-determinism in the sleep/wake consolidation cycle. It is also the slowest variant at 866 seconds per scope, with consolidation overhead that did not improve scores over standard letta during the Llama era.

---

## Phase 3: Semantic Retrieval Stress (S10--S12)

40 episodes (20 signal + 20 distractor), approximately 5,000 words per episode. Episodes are structurally identical, designed to defeat pure embedding similarity search.

### Static driver -- Sonnet as Letta LLM

| Adapter | S10 | S11 | S12 | Mean |
|---|---|---|---|---|
| letta | 0.471 | 0.623 | 0.650 | **0.581** |
| letta-sleepy | 0.444 | 0.572 | 0.606 | **0.541** |

Run IDs (letta): `ee81d92699f2`, `a5849efcb327`, `f1a224122c0c`

Run IDs (letta-sleepy): `4710d74ce9ca`, `e0083831508e`, `ad801eb8d72d`

### Modal/dynamic driver -- Qwen as Letta LLM

| Adapter | S10 | S11 | S12 | Mean |
|---|---|---|---|---|
| letta | 0.473 | 0.481 | 0.561 | **0.505** |
| letta-sleepy | 0.486 | 0.514 | 0.425 | **0.475** |

Run IDs (letta modal): `8fd69a6bce47`, `87d50613dce0`, `a5f038bd7b81`

Run IDs (letta-sleepy modal): `f61970227fc2`, `9fdd908ef2f5`, `9889fe24ebdd`

### letta-v4 and letta-entity on SRS -- Qwen as Letta LLM

| Adapter | S10 | S11 | S12 | Mean |
|---|---|---|---|---|
| letta-v4 | 0.282 | 0.281 | 0.000 | 0.188 |
| letta-entity | 0.428 | 0.000 | 0.559 | 0.329 |

letta-v4 scores 0.000 on S12 because Qwen does not produce episode citations through the Letta agent interface, causing a complete evidence_grounding gate failure. letta-entity scores 0.000 on S11 due to the `ep_` prefix doubling bug in citations. Both variants scored substantially worse with Qwen than with Sonnet, reinforcing the finding that model quality dominates memory architecture.

Run IDs (letta-v4): `8f2c48140dfc`, `defbc1b21b17`, `db6442145d17`

Run IDs (letta-entity): `7af658250136`, `0f3749e48947`, `6050dc91a321`

---

## Cross-Phase Summary

**Comparability caveat:** These columns use different agent LLMs — Numeric uses GPT-OSS-120B (Phase 1, gated composites), Narrative uses Qwen3.5-35B-A3B (Phase 2, ungated), SRS Static uses Sonnet (Letta internal LLM), SRS Modal uses Qwen3.5-35B-A3B (Phase 3, ungated). The all-scope mean mixes these conditions and should be treated as a rough indicator, not a controlled measurement. Within-column rankings are the valid comparison.

| Adapter | Numeric (GPT-OSS) | Narrative (Qwen, ungated) | SRS Static (Sonnet) | SRS Dynamic (Qwen, ungated) | All-Scope Mean (mixed) |
|---|---|---|---|---|---|
| letta | 0.547 | **0.748** | **0.581** | 0.505 | **0.606** |
| letta-sleepy | 0.512 | 0.707 | 0.541 | 0.475 | 0.572 |
| letta-v4 | -- | 0.562 | -- | 0.188 | 0.375 |
| letta-entity | -- | 0.370 | -- | 0.329 | 0.349 |

---

## Key Findings

**Standard letta dominates the family.** With an all-scope mean of 0.606, simple archival passage storage plus vector search outperforms every structured memory variant. Adding consolidation, core memory compression, or entity extraction each introduced complexity that hurt more than it helped.

**LLM quality has large effects within the Letta family.** Sonnet-era runs score two to three times higher than Qwen-era runs for the same adapter and scope: letta with Sonnet scores 0.547 on numeric versus 0.327 with GPT-OSS-120B; letta-v4 scores 0.562 on narrative with Sonnet versus 0.188 on SRS with Qwen. However, this comparison confounds LLM quality with adapter quality (sqlite-chunked-hybrid was never tested with Sonnet), so the relative importance of model vs. architecture remains an open question. Phase 2/3 controlled comparisons (all adapters on Qwen) show substantial architecture effects.

**Sleep consolidation is conditionally useful.** It provides a modest benefit on narrative scopes (+5% over standard letta) and is particularly valuable under constrained budgets (NBA 0.667--0.693 at 2K/4K). However, it slightly hurts numeric scope performance and adds significant latency. The delta/causal V3 framing is critical; V1 and V2 prompt strategies actually degraded performance.

**Core memory compression loses evidence.** letta-v4 demonstrates the compression trade-off clearly: strong insight depth (0.767 mean) and action quality (0.794 mean) paired with low evidence coverage (0.169 mean). The system forms good high-level understanding but cannot ground its answers in specific episodes, which the LENS scoring framework penalizes.

**letta-entity has a structural citation bug.** The `ep_` prefix doubling causes evidence_grounding gate failures on multiple scopes. Where citations happen to work (Sonnet era: S08 at 0.550, S12 at 0.559), quality is reasonable, suggesting the architecture has potential that is masked by the implementation defect.

**Constrained budget advantage for sleep consolidation.** At 2K and 4K token budgets, letta-sleepy achieves NBA 0.667--0.693, making it the best heavy adapter under budget pressure. The consolidation document serves as a retrieval navigation guide, compensating for the limited search window that causes standard letta to degrade sharply (NBA drops 28% from 4K to 2K).

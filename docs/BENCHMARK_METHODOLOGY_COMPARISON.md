# Benchmark Methodology Comparison: LENS vs. Top Memory Evaluations (2025-2026)

**Date**: 2026-02-23
**Purpose**: Deep methodological comparison of how the top 5 memory benchmarks construct their tests, score results, and control for contamination, compared to LENS.

---

## Table of Contents

1. [MemoryArena](#1-memoryarena)
2. [MEMTRACK](#2-memtrack)
3. [MemoryAgentBench](#3-memoryagentbench)
4. [LongMemEval](#4-longmemeval)
5. [LoCoMo-Plus](#5-locomo-plus)
6. [LENS](#6-lens-methodology-summary)
7. [Comparison Matrix](#7-comparison-matrix)
8. [Gap Analysis: Where LENS Remains Unique](#8-gap-analysis-where-lens-remains-unique)
9. [Overlap Analysis: Shared Testing Ground](#9-overlap-analysis-shared-testing-ground)
10. [Recommendations: What LENS Should Adopt](#10-recommendations-what-lens-should-adopt)

---

## 1. MemoryArena

**Paper**: [MemoryArena: Benchmarking Agent Memory in Interdependent Multi-Session Agentic Tasks](https://arxiv.org/abs/2602.16313) (Feb 2026)

### Test Construction

MemoryArena uses **human-crafted agentic tasks** with explicitly interdependent subtasks across four domains:

| Domain | Tasks | Sessions/Task | Mechanism |
|--------|-------|---------------|-----------|
| Bundled Web Shopping | 150 | 6 | Compatibility tracking across product bundles |
| Group Travel Planning | 270 | 5-9 travelers | Preference constraints accumulate per traveler |
| Progressive Web Search | 256 | 2-16 subqueries | Compositional queries with strict causal ordering |
| Formal Reasoning | 60 | 2-16 lemmas | Sequential math/physics derivations |

**Construction method**: Human experts design each task. Shopping tasks use product catalogs with manually verified compatibility chains. Travel tasks define preference constraints that interact (e.g., "Alice is vegan" + "Bob wants a restaurant with steak" = conflict resolution needed). Search tasks underwent two-stage filtering to remove instances solvable without memory. Math/physics problems are PhD-level curated with strict causal consistency.

**Evidence structure**: Evidence is **procedural and cumulative** -- each session produces facts that constrain future sessions. A shopping task accumulates compatibility requirements across 6 sessions; a travel task accumulates preference constraints across 5-9 travelers. The key structure is **dependency chains**, not scattered facts.

**Distractors/noise**: Shopping tasks include "compatible distractors" (logically valid but not target items) and "hard negative samples" that violate constraints. Each shopping level includes 2 compatible and 2 incompatible items. This is **adversarial structural noise**, not topical noise.

### Scoring Methodology

- **Success Rate (SR)**: Binary -- did the task complete fully? (Harsh)
- **Task Progress Score (PS)**: Fraction of correctly completed subtasks (partial credit)
- **Soft Progress Score**: For travel, awards partial credit based on constraint satisfaction ratios

Evaluation is **programmatic** (constraint checking), not LLM-judged. Ground truth is defined by constraint satisfaction -- there may be multiple valid solutions, but all must satisfy all accumulated constraints.

### Contamination Controls

- Human annotators manually verify all compatibility chains
- Two-stage human annotation for search ensures semantic coherence
- Expert PhD-level curation for formal reasoning confirms causal consistency
- Two-stage filtering removes instances solvable without memory retention
- Tasks require real tool interaction (web browsing, search), not just text answering

### Key Results

- All methods achieve **near-zero SR** on travel planning and formal reasoning
- Agents that saturate LoCoMo (>90%) fail catastrophically in MemoryArena
- Performance decays with subtask depth -- agents cannot sustain execution across sessions
- External memory and RAG are **not universally beneficial** -- sometimes hurt performance
- Average 57 action steps per task, trace lengths >40,000 tokens

### What Makes It Hard

**Not retrieval, but action-grounded memory application**. The agent must execute actions based on memory, not just answer questions. Memory errors compound across sessions -- a wrong product selection in session 2 cascades to incompatible choices in sessions 3-6.

---

## 2. MEMTRACK

**Paper**: [MEMTRACK: Evaluating Long-Term Memory and State Tracking in Multi-Platform Dynamic Agent Environments](https://arxiv.org/abs/2510.01353) (NeurIPS SEA Workshop 2025)
**Blog**: [Patronus AI Blog](https://www.patronus.ai/blog/memtrack)

### Test Construction

MEMTRACK contains **47 carefully curated datapoints** simulating realistic enterprise SWE workflows across three integrated platforms: **Slack, Linear, and Git** (via Gitea).

**Three construction methodologies**:
1. **Bottom-Up**: Uses closed issues on popular open-source repositories, working backward from merged PRs. Adds real-world distractions to simulate typical SWE work settings.
2. **Top-Down**: In-house experts who previously worked in product/engineering organizations design scenarios from scratch.
3. **Hybrid**: Expert ideation combined with LLM-prompted iterative refinement -- human supervision + automated generation.

**Evidence structure**: Each instance is a **chronologically platform-interleaved timeline**. Information about a single task/decision is fragmented across Slack messages, Linear tickets, and Git commits. The agent must track state across platforms, resolve cross-references (e.g., "see the PR John mentioned in #backend-team"), and handle conflicting information.

**Noise/contradictions**: Scenarios contain **noisy, conflicting, cross-referring information** as well as codebase/file-system comprehension requirements. Contradictions are organic -- a Slack discussion might say "we decided X" but a later Linear ticket overrides with Y. The noise is realistic organizational chatter, not synthetic.

### Scoring Methodology

Three evaluation dimensions:
- **Correctness**: Whether the agent produces factually accurate answers. Evaluated via brief phrase outputs using direct matching + LLM-as-judge for approximate matching.
- **Efficiency**: How effectively the agent utilizes memory relative to task requirements (measures unnecessary work).
- **Redundancy**: Whether the agent stores or retrieves duplicate information unnecessarily.

Ground truth is **expert-defined brief phrases** to minimize non-deterministic behavior and avoid overfitting on multiple-choice QA.

### Contamination Controls

- Novel scenarios grounded in real-world processes but not copied from public data
- Expert curation ensures scenarios are not solvable from pre-training knowledge
- Brief phrase outputs (not multiple choice) reduce guessing success

### Key Results

- Best model (GPT-5) achieves only **60% Correctness**
- Zep and Mem0 memory backends provide **"no significant improvement"** over base LLM
- Models fail to effectively call memory tools -- when given memory tools, LLMs often don't use them
- Cross-platform dependency resolution is the hardest capability
- Contradiction resolution across platforms remains largely unsolved

### What Makes It Hard

**Cross-platform state tracking with organic contradictions**. Unlike conversational benchmarks, MEMTRACK requires understanding multiple information modalities (chat, project management, code) and resolving real organizational ambiguity where decisions evolve over time.

---

## 3. MemoryAgentBench

**Paper**: [Evaluating Memory in LLM Agents via Incremental Multi-Turn Interactions](https://arxiv.org/abs/2507.05257) (ICLR 2026)
**Code**: [GitHub](https://github.com/HUST-AI-HYZ/MemoryAgentBench)

### Test Construction

MemoryAgentBench tests **four core memory competencies** grounded in cognitive science:

| Competency | Datasets | Construction | Scale |
|------------|----------|-------------|-------|
| Accurate Retrieval (AR) | EventQA (novel) | NER on 5 books (390K+ tokens each), GPT-4o extracts 101 events/book, 6-way MCQ | 505 questions |
| Test-Time Learning (TTL) | Existing datasets transformed to multi-turn | Classification + recommendation tasks | Various |
| Long-Range Understanding (LRU) | Existing long-context datasets reformatted | Summarization + reasoning | 103K-1.44M tokens |
| Selective Forgetting (SF) | FactConsolidation (novel) | MQUAKE counterfactual edit pairs, chronological ordering | 6K-262K tokens |

**EventQA construction**: Uses five books (390K+ tokens each). NER identifies frequent characters, GPT-4o extracts 101 events per book, then creates 6-way multiple-choice questions with one true event and five distractor options generated by the model.

**FactConsolidation construction**: Built from MQUAKE counterfactual edit pairs. Contradictory fact versions are ordered chronologically to simulate realistic updates. Split into single-hop (direct recall) and multi-hop (inference) sub-tasks.

**Multi-turn simulation**: Rather than presenting entire contexts at once, datasets are segmented into chunks presented incrementally: c1, c2, ... cn --> q1, q2, ... qm. Each chunk is accompanied by instructions prompting the agent to memorize its contents. This simulates how real agents operate incrementally.

**Evidence structure**: For AR and TTL, evidence is **single-fact with distractors** (find the right event among alternatives). For LRU, evidence is **distributed across a long context** requiring integration. For SF, evidence is **temporal with superseding facts** -- the latest version of a fact is correct.

### Scoring Methodology

- **AR**: Substring exact match accuracy, GPT-4o judgment for open-ended responses
- **TTL**: Classification accuracy, Recall@5 for recommendations
- **LRU**: F1-score for summarization, accuracy for reasoning
- **SF**: Accuracy on factual judgment tasks

Scoring is **competency-specific** -- different metrics per task type. No unified score.

### Contamination Controls

- EventQA uses full-length books (unlikely to be memorized)
- FactConsolidation uses counterfactual edits (contradicts pre-training knowledge)
- Multi-turn delivery prevents context-window shortcuts
- 6-way MCQ format limits random guessing to 16.7% baseline

### Key Results

- Long-context models dominate TTL and LRU (GPT-4o: 50-87.6%)
- RAG methods excel at AR (BM25: 60.5% average)
- **All methods fail on multi-hop selective forgetting** (max 7% accuracy)
- O4-mini achieves 80% at 6K tokens for SF but drops to 14% at 32K tokens
- No single architecture dominates across all four competencies

### What Makes It Hard

**Selective forgetting at scale**. When contradictory facts are spread across 32K-262K tokens of context, no system reliably identifies that a later fact supersedes an earlier one and updates accordingly. Multi-hop forgetting (where the contradiction requires inference, not direct matching) is essentially unsolved.

---

## 4. LongMemEval

**Paper**: [LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory](https://arxiv.org/abs/2410.10813) (ICLR 2025)
**Code**: [GitHub](https://github.com/xiaowu0162/LongMemEval)
**Website**: [Project Page](https://xiaowu0162.github.io/long-mem-eval/)

### Test Construction

LongMemEval contains **500 manually created questions** testing five core memory abilities:

1. **Information Extraction** (200 questions): Recalling specific details from extensive histories
2. **Multi-Session Reasoning** (100 questions): Synthesizing across multiple sessions
3. **Temporal Reasoning** (80 questions): Understanding time references and metadata
4. **Knowledge Updates** (60 questions): Recognizing changed user information
5. **Abstention** (60 questions): Correctly refusing unanswerable questions

**Construction pipeline** (hybrid human-LLM):
1. Define 164 user attributes across 5 categories (lifestyle, demographics, life events, etc.)
2. LLMs generate initial background paragraphs and seed questions
3. Human experts **manually rewrote ~95% of questions** (5% yield rate from ~1,000 initial candidates per type)
4. LLM-simulated user-AI dialogues embed evidence indirectly (e.g., car purchase mentioned through insurance questions)
5. Expert annotators **manually edited ~70% of sessions** for naturalness and accuracy

**Evidence embedding**: Evidence is embedded **indirectly** within task-oriented dialogues. A user's lactose intolerance might surface during a recipe recommendation session, not as a direct statement. This requires the memory system to extract implicit facts from conversational context.

**Chat history composition** (for haystack):
- 50% simulated sessions
- 25% ShareGPT (real user-AI conversations)
- 25% UltraChat
- Non-evidence sessions provide realistic conversational noise

**Scale**:
- **LongMemEvalS**: ~115K tokens per question
- **LongMemEvalM**: 500 sessions (~1.5M tokens)

### Scoring Methodology

- **Primary metric**: GPT-4o as judge, achieving **>97% agreement with human experts**
- Separate evaluation prompts for each question type
- Secondary metrics: Recall@k and NDCG@k when retrieval results are exposed
- Binary correctness (correct/incorrect), not partial credit for most types

### Contamination Controls

- Questions use **fictional user identities** with synthesized life details
- Evidence is embedded indirectly, requiring inference from conversational context
- 95% human rewrite rate ensures questions are not trivially answerable
- Abstention category explicitly tests false positive control (answering when no evidence exists)

### Key Results

- Commercial systems (ChatGPT, Coze): **30-64% accuracy drop** in online settings
- Long-context LLMs: **30-60% performance decline** vs oracle retrieval
- Proposed optimizations (session decomposition, fact-augmented keys, time-aware queries) improve by 4-11%
- Even in simplified 3-6 session scenarios, human evaluation confirms significant difficulty
- Multi-session reasoning and temporal reasoning are hardest categories

### What Makes It Hard

**Scale and indirect evidence**. At 1.5M tokens with 500 sessions, the retrieval challenge is massive. Evidence is embedded indirectly (lactose intolerance from a recipe discussion), requiring the memory system to extract, index, and retrieve implicit facts -- not just store raw text.

---

## 5. LoCoMo-Plus

**Paper**: [LoCoMo-Plus: Beyond-Factual Cognitive Memory Evaluation Framework for LLM Agents](https://arxiv.org/abs/2602.10715) (Feb 2026)

### Test Construction

LoCoMo-Plus extends the original LoCoMo benchmark with **Level-2 Cognitive Memory** -- testing whether models can retain and apply **implicit behavioral constraints** (not just explicit facts).

**Cue-trigger construction pipeline**:
1. Generate short dialogue snippets containing **implicit constraints** (user state, goals, preferences, values)
2. Manual verification ensures cues express "persistent or behaviorally constraining information"
3. Generate trigger queries with **low surface-level semantic similarity** to corresponding cues
4. Apply **BM25 and MPNet-based similarity scoring** to filter out cases where cue is recoverable through shallow matching
5. Embed validated cue-trigger pairs into long LoCoMo dialogue trajectories with specified temporal gaps

**Four types of implicit constraints**:
- **Causal**: Earlier causes affecting later events
- **State**: Physical/emotional states influencing behavior
- **Goal**: Long-term intentions shaping choices
- **Value**: Beliefs shaping reactions

**Semantic disconnect**: The defining innovation. A cue like "I just got diagnosed with celiac disease" should constrain a later recommendation query about restaurants, but the query "Can you recommend a nice dinner spot?" has **zero lexical overlap** with the cue. BM25 and embedding similarity filtering ensures surface-level retrieval cannot solve the task.

**Scale**: ~100 representative cognitive memory cases per category, with long dialogue trajectories. Prioritizes "diagnostic coverage over scale" due to high annotation costs.

### Scoring Methodology

- **Constraint consistency**: Does the response satisfy the implicit constraint induced by the cue? Defined as "membership in a valid response space, allowing multiple acceptable realizations."
- **Three-level labels**: Correct/partial/wrong for factual questions
- **Binary labels**: For temporal, adversarial, and cognitive awareness questions
- **LLM-based judgment**: Replaces string matching. Agreement with human annotators: 0.80+
- **No task-type disclosure**: Unlike original LoCoMo, the evaluation prompt does not reveal whether the question tests factual recall vs cognitive awareness, preventing prompt-specific gaming

### Contamination Controls

- BM25 + MPNet similarity filtering ensures cues are not recoverable through shallow retrieval
- Low surface-level semantic similarity between cue and trigger prevents embedding-based shortcuts
- Removal of task-type disclosure prevents strategy gaming
- Constraint-based evaluation (not string matching) prevents surface-level parroting

### Key Results

- Average **10-26 percentage point drop** from LoCoMo to LoCoMo-Plus across all methods
- Performance shows **"rapid collapse as context length increases"** for cognitive cases, while factual memory remains stable
- Traditional metrics show **systematic length bias** (scores peak near average ground-truth length)
- RAG methods (embedding-based retrieval) fail when cue-trigger pairs are semantically disconnected
- A-Mem, Mem0, SeCom memory systems all struggle significantly
- GPT-4.1, Gemini-2.5 variants tested -- none achieve strong cognitive memory

### What Makes It Hard

**Semantic disconnect between evidence and query**. The benchmark is specifically designed to defeat retrieval systems that rely on lexical or embedding similarity. When a user's dietary restriction (the cue) must constrain a restaurant recommendation (the trigger), no surface-level matching will connect them. The model must maintain an internal representation of user state that influences downstream behavior -- true cognitive memory, not recall.

---

## 6. LENS Methodology Summary

For comparison, here is LENS's methodology in the same analytical framework.

### Test Construction

LENS uses a **two-stage information-isolated pipeline**:

1. **PlanOutline** (gpt-5.2, sees full spec): Produces per-episode structured data sheets with concrete metric values. Signal encoded as numeric progressions only -- never text commentary.
2. **RenderEpisodes** (gpt-4.1-nano, blind to storyline): Formats each data sheet into a terse log entry independently. Cannot editorialize because it doesn't know what's signal.

**Evidence structure**: Evidence is **numeric progressions across sequential episodes**. No single episode contains the answer. A cascading failure scenario embeds signal as gradually degrading p99 latency values (200ms --> 400ms --> 800ms) across 15-20 episodes. The answer (geo-lookup API degradation causing connection pool exhaustion) can only be concluded by observing the progression.

**Scale**: 6 domain-diverse scopes, 30 signal episodes + 90 distractor episodes per scope (~84K tokens), 24 questions per scope (144 total), 10 question types.

**Distractors**: Format-matched but topically orthogonal. Three distractor themes per scope with explicit **excluded terms** preventing any lexical overlap with signal. Each distractor theme has its own scenario/voice. Distractors are generated through the same two-stage pipeline. max_similarity threshold (0.3) enforced.

**Arc structure**: Each scope defines a 5-phase arc (baseline, early_signal, red_herring, escalation, root_cause) with controlled signal density. Red herrings are deliberately planted to test whether systems are misled by correlational evidence.

### Scoring Methodology

**Three-tier scoring** with gating:

| Tier | Metrics | Method |
|------|---------|--------|
| Tier 1 | evidence_grounding, fact_recall, evidence_coverage, citation_coverage, budget_compliance | Programmatic (set overlap, substring match) |
| Tier 2 | answer_quality, insight_depth, reasoning_quality | Pairwise LLM judge (position-debiased) |
| Tier 3 | longitudinal_advantage, action_quality, naive_baseline_advantage | Differential scoring (synthesis vs control) |

**Pairwise judging**: For each key fact, randomly assigns candidate and reference to positions A/B, asks judge which better demonstrates the finding, maps back. Position bias controlled via random assignment.

**Naive Baseline Advantage (NBA)**: Head-to-head comparison -- for each question, generates a naive answer by concatenating all episodes into context, then pairwise-judges adapter answer vs naive answer per key fact. NBA > 0.5 means the adapter beats context stuffing.

**Question types**: longitudinal, null_hypothesis, action_recommendation, negative, paraphrase, temporal, counterfactual, distractor_resistance, severity_assessment, evidence_sufficiency. Each tests a different reasoning capability.

### Contamination Controls

1. **Two-stage information isolation**: The rendering LLM (gpt-4.1-nano) never sees the storyline, key facts, or questions. It formats numbers without knowing their significance. This prevents the LLM from editorializing (writing "latency is concerning" instead of "p99: 600ms").

2. **ContaminationCheck validator**: For each synthesis question, tests every single episode individually. An LLM attempts to answer the question from each episode alone. If max single-episode fact coverage > 80%, the question is flagged as contaminated.

3. **NaiveBaseline validator**: Concatenates all episodes (signal + distractors) and asks an LLM to answer. If average fact coverage > 50% per question type, the benchmark is too easy. If < 5%, signal may be missing.

4. **Forbidden words list**: Banned commentary terms ("increasing", "decreasing", "elevated", "concerning") that would leak signal in individual episodes.

5. **Excluded terms in distractors**: Each distractor theme explicitly bans terms related to the signal storyline.

6. **EpisodeVault (anticheat)**: Adapters cannot access raw episode text at query time -- only whatever they stored during ingest. The vault verifies citations are exact substrings.

### Key Results

- No memory system achieves >50% answer quality on longitudinal synthesis
- sqlite-chunked-hybrid (simple FTS + embedding search) beats every dedicated memory system
- Compaction strategy collapses when corpus scales from 30 to 120 episodes (0.790 --> 0.294 NBA)
- Mem0 extraction layer hardcoded for personal facts, scores 0.0 on structured telemetry
- Budget enforcement reveals adapters blast through token limits (5x over budget on first retrieval)

### What Makes It Hard

**Pattern synthesis across progressive numerical data buried in noise**. No single episode contains the answer. The answer emerges only from observing numeric trends across 15-20 episodes, while 90 format-matched distractor episodes create a 3:1 noise ratio. Red herrings deliberately mislead. The two-stage construction ensures individual episodes are operationally bland -- the signal is invisible without longitudinal context.

---

## 7. Comparison Matrix

### Dimension 1: Evidence Type

| Benchmark | Evidence Type | Description |
|-----------|--------------|-------------|
| MemoryArena | Procedural constraints | Compatibility requirements, preference constraints, causal lemmas |
| MEMTRACK | Cross-platform state | Facts scattered across Slack/Linear/Git with contradictions |
| MemoryAgentBench | Mixed: events, facts, contradictions | Events from books, counterfactual fact edits |
| LongMemEval | Implicit conversational facts | User attributes embedded indirectly in task dialogues |
| LoCoMo-Plus | Implicit behavioral constraints | User state/goals/values that constrain future behavior |
| **LENS** | **Numeric progressions (emergent patterns)** | **Metric degradation across episodes; answer requires pattern synthesis** |

**LENS uniqueness**: LENS is the only benchmark where evidence is purely numeric/structured data and the conclusion is emergent from the progression, not stated anywhere. All others have evidence that is, at some level, a fact or constraint that can be stated in natural language.

### Dimension 2: Reasoning Depth

| Benchmark | Reasoning Depth | Hops Required |
|-----------|----------------|---------------|
| MemoryArena | Multi-step action chains | 2-16 dependent subtasks |
| MEMTRACK | Cross-reference resolution | 2-5 platform cross-references |
| MemoryAgentBench | Single-hop to multi-hop forgetting | 1-3 hops (SF multi-hop max 7% accuracy) |
| LongMemEval | Single-hop to multi-session | 1-4 sessions per question |
| LoCoMo-Plus | Implicit constraint application | 1-hop but with semantic disconnect |
| **LENS** | **Causal chain + trend synthesis** | **5-20 episodes for full causal chain (geo-lookup --> retries --> pool exhaustion --> checkout failure)** |

**LENS uniqueness**: LENS requires the longest reasoning chain -- 4-step causal chains backed by evidence from 5-20 episodes. Other benchmarks require fewer hops but test different reasoning types (action execution, constraint satisfaction, forgetting).

### Dimension 3: Temporal Complexity

| Benchmark | Temporal Model | Description |
|-----------|---------------|-------------|
| MemoryArena | Session ordering | Tasks must be completed in sequence |
| MEMTRACK | Chronological timeline with superseding | Later decisions override earlier ones |
| MemoryAgentBench | Chronological with contradictions | Later facts supersede earlier versions |
| LongMemEval | Timestamped sessions with updates | Knowledge updates test temporal awareness |
| LoCoMo-Plus | Temporal gap between cue and trigger | Distance matters for interference |
| **LENS** | **Progressive degradation with arc phases** | **5-phase arc: baseline, early_signal, red_herring, escalation, root_cause** |

**LENS uniqueness**: LENS is the only benchmark with an explicitly designed **narrative arc** with phases. The temporal structure is not just ordering -- it includes controlled signal density, deliberate red herrings, and progressive escalation. Other benchmarks use temporal ordering but don't engineer arc-level narrative structure.

### Dimension 4: Noise Model

| Benchmark | Noise Type | Noise-to-Signal Ratio |
|-----------|-----------|----------------------|
| MemoryArena | Hard negatives + compatible distractors | 2 compatible + 2 incompatible per level |
| MEMTRACK | Organic organizational noise | Realistic but uncontrolled ratio |
| MemoryAgentBench | Book-length surrounding text | Up to 390K tokens per 101 events |
| LongMemEval | Real conversation sessions (ShareGPT, UltraChat) | ~115K-1.5M tokens around few evidence sessions |
| LoCoMo-Plus | Long dialogue with intervening turns | Controlled temporal gap |
| **LENS** | **Format-matched topically orthogonal distractors** | **3:1 (90 distractors vs 30 signal) with excluded term enforcement** |

**LENS uniqueness**: LENS has the most controlled noise model. Distractors are format-matched (look like signal episodes), topically orthogonal (different domains with explicit term exclusion), and present at a controlled 3:1 ratio. This tests signal/noise discrimination, not just context length.

### Dimension 5: Evaluation Granularity

| Benchmark | Granularity | Method |
|-----------|-------------|--------|
| MemoryArena | Binary (SR) + partial (PS) | Programmatic constraint checking |
| MEMTRACK | Correctness + Efficiency + Redundancy | LLM judge + approximate matching |
| MemoryAgentBench | Per-competency metrics | Mixed: exact match, F1, accuracy, LLM judge |
| LongMemEval | Binary per question | GPT-4o judge (>97% human agreement) |
| LoCoMo-Plus | 3-level (correct/partial/wrong) | LLM judge (0.80+ human agreement) |
| **LENS** | **3-tier with 10 metrics** | **Programmatic (T1) + pairwise LLM judge (T2) + differential/head-to-head (T3)** |

**LENS uniqueness**: LENS has the most granular evaluation with three tiers measuring different aspects. The pairwise position-debiased judging per key fact and the NBA (head-to-head vs naive baseline) are methodologically distinctive. No other benchmark separates "did you get the right facts" (T1) from "was your answer good" (T2) from "did you actually benefit from having memory" (T3).

### Dimension 6: Construction Rigor

| Benchmark | Construction Method | Information Isolation |
|-----------|-------------------|---------------------|
| MemoryArena | Human-crafted + human-verified | Tasks require tool interaction (implicit isolation) |
| MEMTRACK | Expert-designed + agent-synthesized | Novel scenarios not in pre-training data |
| MemoryAgentBench | Existing datasets + LLM-generated (EventQA) | Books as source material; counterfactual edits |
| LongMemEval | LLM-generated + 95% human rewrite | Fictional user identities; indirect evidence embedding |
| LoCoMo-Plus | LLM + human + similarity filtering | BM25/MPNet filtering ensures semantic disconnect |
| **LENS** | **Two-stage LLM pipeline with information isolation** | **Planner sees full context, renderer is blind to storyline; contamination validator; naive baseline validator** |

**LENS uniqueness**: LENS is the only benchmark with a formally defined **information isolation architecture** in data generation. The two-stage pipeline (full-context planner + blind renderer) is a principled approach to preventing contamination. Other benchmarks use post-hoc filtering or human review. LENS validates at construction time with automated contamination and baseline checks.

### Dimension 7: Scale

| Benchmark | Tokens | Sessions/Episodes | Questions | Memory Systems Tested |
|-----------|--------|-------------------|-----------|----------------------|
| MemoryArena | >40K per task | 2-16 per task | 736 tasks | LoCoMo-tuned agents, RAG, external memory |
| MEMTRACK | Not specified (enterprise timelines) | Multi-platform timelines | 47 datapoints | Mem0, Zep, MemGPT, base LLM |
| MemoryAgentBench | 103K-1.44M | Multi-turn chunks | 3,000+ questions | Long-context, RAG, Agentic (MemGPT, MIRIX, Self-RAG) |
| LongMemEval | 115K-1.5M | 500 sessions | 500 questions | ChatGPT, Coze, Claude, long-context LLMs |
| LoCoMo-Plus | Long LoCoMo dialogues | ~100 per category | ~400 test instances | GPT-4.1, Gemini-2.5, A-Mem, Mem0, SeCom |
| **LENS** | **~84K per scope** | **120 episodes (30 signal + 90 distractor)** | **144 (24 per scope x 6 scopes)** | **7 adapters: null, sqlite, compaction, letta, letta-sleepy, mem0, cognee, graphiti** |

### Dimension 8: What Makes It Hard

| Benchmark | Primary Difficulty Source |
|-----------|------------------------|
| MemoryArena | Action-grounded memory application with compounding errors |
| MEMTRACK | Cross-platform state resolution with organic contradictions |
| MemoryAgentBench | Selective forgetting at scale (multi-hop) |
| LongMemEval | Scale (1.5M tokens) + indirect evidence embedding |
| LoCoMo-Plus | Semantic disconnect between evidence and query |
| **LENS** | **Pattern synthesis from numeric progressions in 3:1 noise** |

---

## 8. Gap Analysis: Where LENS Remains Unique

### 8.1 Emergent Conclusions from Numeric Progressions

No other benchmark tests whether a memory system can synthesize conclusions from **purely numeric/structured data** where the answer is never stated in any episode. In MemoryArena, tasks have explicit constraints. In LongMemEval, evidence is factual (even if indirect). In LoCoMo-Plus, constraints are implicit but behavioral. Only in LENS is the evidence a sequence of numbers (p99: 200ms, 320ms, 400ms, 600ms, 800ms) from which the system must independently conclude "progressive degradation."

**Why this matters**: Real-world operational intelligence (infrastructure monitoring, financial analysis, clinical decision-making) requires synthesizing trends from structured data, not recalling conversational facts. LENS is the only benchmark targeting this use case.

### 8.2 Controlled Signal Density Arc with Red Herrings

LENS is the only benchmark with a **5-phase narrative arc** (baseline --> early_signal --> red_herring --> escalation --> root_cause) with controlled signal density per phase. The red herring phase deliberately introduces a plausible alternative explanation (service-C deploy coincides with first errors) that later evidence disproves. No other benchmark plants deliberate red herrings in this structured way.

### 8.3 Two-Stage Information-Isolated Construction

LENS's datagen pipeline is architecturally unique. The planner sees the full storyline and encodes signal as numbers. The renderer sees only individual data sheets and formats them without knowing the storyline. This is enforced at the **model level** (different LLMs with different context). Other benchmarks use:
- Human curation (MemoryArena) -- expensive, doesn't scale
- Post-hoc filtering (LoCoMo-Plus) -- removes bad cases but doesn't prevent contamination
- Existing datasets (MemoryAgentBench) -- relies on source material quality

### 8.4 Automated Contamination and Difficulty Validation

LENS validates at build time with:
- **ContaminationCheck**: Every synthesis question tested against every individual episode
- **NaiveBaseline**: Full context stuffing baseline ensures benchmark isn't too easy
- **WordCount + Forbidden Words**: Structural quality gates

No other benchmark has automated contamination checking integrated into the data generation pipeline. MemoryArena uses human verification. LongMemEval uses human rewriting. LoCoMo-Plus uses similarity filtering. These are all pre-deployment checks, not continuous build-time validation.

### 8.5 Naive Baseline Advantage Metric

The NBA metric (adapter answer vs context-stuffed naive baseline, pairwise judged per key fact) is unique to LENS. It directly measures whether having a memory system provides value over simply stuffing all available text into the context window. This is the most operationally relevant metric for memory system evaluation -- if context stuffing works as well, why use a memory system?

### 8.6 Multi-Question-Type Taxonomy

LENS tests 10 question types per scope: longitudinal, null_hypothesis, action_recommendation, negative, paraphrase, temporal, counterfactual, distractor_resistance, severity_assessment, evidence_sufficiency. This is more fine-grained than any other benchmark. LongMemEval tests 5 abilities. MemoryAgentBench tests 4 competencies. MemoryArena tests 4 domains. LENS distinguishes between "can the system identify what's NOT happening" (negative), "can it resist being misled by distractors" (distractor_resistance), and "does it know when evidence is insufficient" (evidence_sufficiency).

### 8.7 Format-Matched Distractor Design

LENS distractors are not random noise -- they are format-matched (same log style, same episode structure) but topically orthogonal (DNS migration, storage capacity, auth audit vs. API gateway cascading failure), with explicit excluded terms to prevent any lexical leakage. This is the most controlled distractor design in the field. MemoryArena uses hard negatives within task domains. LongMemEval uses real conversations. LoCoMo-Plus uses intervening dialogue turns. Only LENS ensures distractors look structurally identical to signal but carry zero topical signal.

---

## 9. Overlap Analysis: Shared Testing Ground

### 9.1 Temporal Reasoning (LENS + LongMemEval + LoCoMo-Plus)

All three test temporal awareness, but differently:
- LongMemEval: "When did X happen?" (timestamp retrieval)
- LoCoMo-Plus: Temporal gap degrades performance (interference over time)
- LENS: "When did latency first start degrading, and over how many periods?" (trend timing)

LENS's temporal questions are progression-aware (not just "when" but "how quickly"), but LongMemEval has broader temporal reasoning coverage.

### 9.2 Knowledge Update / Selective Forgetting (MemoryAgentBench + MEMTRACK + LENS)

MemoryAgentBench directly tests selective forgetting with counterfactual edits. MEMTRACK tests contradiction resolution across platforms. LENS tests this implicitly through the red herring phase -- the system must eventually override the "service-C deploy is the cause" hypothesis with "geo-lookup API degradation is the actual cause." LENS does not explicitly test selective forgetting as a competency, but the red herring mechanism requires something similar.

### 9.3 Abstention / Null Hypothesis (LongMemEval + LENS)

Both test whether systems correctly refuse to answer when evidence doesn't support a conclusion:
- LongMemEval: "Abstention" questions where no evidence exists
- LENS: "null_hypothesis" questions at early checkpoints (5 episodes in, no patterns visible), "evidence_sufficiency" questions, and "negative" questions testing whether the system avoids false attribution

### 9.4 Distractor Resistance (MemoryArena + LENS)

Both include explicitly designed distractors:
- MemoryArena: Compatible and incompatible alternatives in shopping tasks
- LENS: Format-matched topically orthogonal episodes with excluded terms

### 9.5 Memory System Comparison (All)

Every benchmark tests multiple memory backends and finds similar results:
- MemoryArena: External memory and RAG are "not universally beneficial"
- MEMTRACK: Zep/Mem0 provide "no significant improvement"
- MemoryAgentBench: No single architecture dominates all competencies
- LongMemEval: Commercial systems lose 30-64% accuracy
- LoCoMo-Plus: All memory systems struggle with cognitive memory
- **LENS: "Existing memory systems do not meaningfully outperform basic text search at longitudinal synthesis"**

**The field-wide consensus is emerging: current memory systems provide marginal or zero benefit for complex memory tasks.**

### 9.6 Scale Challenge (LongMemEval + MemoryAgentBench)

At 115K-1.5M tokens, LongMemEval and MemoryAgentBench test context length challenges that LENS (84K tokens) doesn't push as hard on. LENS's difficulty comes from noise ratio and reasoning complexity, not raw scale.

---

## 10. Recommendations: What LENS Should Adopt

### 10.1 HIGH PRIORITY: Agentic Task Evaluation (from MemoryArena)

**The gap**: LENS evaluates memory through Q&A -- the agent retrieves evidence and answers a question. MemoryArena shows that agents with near-saturated Q&A performance (LoCoMo) fail in agentic settings where memory informs actions.

**Recommendation**: Add an **action execution mode** where the agent must take actions (write incident reports, create runbooks, escalate tickets) based on memory, not just answer questions. Scoring would check constraint satisfaction (did the runbook address the actual root cause?), not just fact recall.

**Effort**: High. Requires new infrastructure for action validation. Could start with a simplified "write an incident postmortem" task that's scored for completeness and accuracy.

### 10.2 HIGH PRIORITY: Cross-Platform Evidence (from MEMTRACK)

**The gap**: LENS episodes are all in one format (daily API gateway log summaries). Real memory challenges involve synthesizing across different information types.

**Recommendation**: Add **multi-format scopes** where evidence is spread across different document types: log entries, Slack messages, Jira tickets, git commit messages, monitoring dashboards. The same two-stage pipeline could generate these with format-specific renderers.

**Effort**: Medium. Requires new rendering templates and format-specific prompt builders, but the core pipeline architecture supports this.

### 10.3 MEDIUM PRIORITY: Selective Forgetting Tests (from MemoryAgentBench)

**The gap**: LENS tests red herring resistance (overriding a wrong hypothesis) but doesn't explicitly test **fact superseding** -- where a later episode directly contradicts an earlier one and the system must update.

**Recommendation**: Add explicit **fact superseding episodes** within the arc. For example, an early episode might state "service-B retry rate: 2% (within normal range)" and a later episode shows "service-B retry rate: 47% (critical)". Questions should test whether the system reports the latest state, not the historical one.

**Effort**: Low-Medium. Can be incorporated into existing spec structure by adding contradictory baseline facts that escalation episodes supersede.

### 10.4 MEDIUM PRIORITY: Semantic Disconnect Questions (from LoCoMo-Plus)

**The gap**: LENS questions are topically aligned with the evidence -- "What is the root cause of checkout failures?" directly relates to checkout-related log entries. LoCoMo-Plus shows that semantic disconnect between query and evidence is a fundamentally different (and harder) challenge.

**Recommendation**: Add **indirect query questions** that reference the situation without using signal terminology. Instead of "What is the root cause of checkout failures?", ask "A customer just complained that their purchase didn't go through -- what should we investigate and why?" This tests whether memory retrieval works without keyword overlap.

**Effort**: Low. These are new questions in existing spec files, not new infrastructure.

### 10.5 MEDIUM PRIORITY: Scale Configurations (from LongMemEval)

**The gap**: LENS currently operates at ~84K tokens per scope. LongMemEval shows that 1.5M tokens creates qualitatively different challenges.

**Recommendation**: Add a **LENS-XL** configuration with 200+ signal episodes and 600+ distractors (~500K tokens) to test how systems degrade with scale. This would reveal whether systems that work at 84K collapse at larger scales (as compaction did when going from 30 to 120 episodes).

**Effort**: Medium. The pipeline can generate more episodes, but contamination validation scales quadratically (episodes x questions).

### 10.6 LOW PRIORITY: Efficiency and Redundancy Metrics (from MEMTRACK)

**The gap**: LENS measures budget compliance (token usage) but doesn't explicitly measure retrieval **efficiency** (how much irrelevant context was retrieved) or storage **redundancy** (how much duplicate information is stored).

**Recommendation**: Add a **retrieval precision** metric (fraction of retrieved content that was actually relevant to the question) and a **storage efficiency** metric (unique information per stored byte). These would complement existing budget_compliance.

**Effort**: Low. Retrieval precision can be computed from existing retrieved_ref_ids vs required_evidence_refs. Storage efficiency would require adapter-level inspection.

### 10.7 LOW PRIORITY: Multi-Turn Incremental Delivery (from MemoryAgentBench)

**The gap**: LENS delivers episodes sequentially via ingest(), then asks questions at checkpoints. MemoryAgentBench interleaves questions between chunk deliveries more aggressively.

**Recommendation**: LENS already has checkpoint_after per question (asking questions at episode 5, 10, 15, 20, 25, 30). This is conceptually similar to MemoryAgentBench's multi-turn format but could be made more granular -- asking questions after every 2-3 episodes to test very early detection.

**Effort**: Low. Already supported by the checkpoint mechanism.

---

## Summary Comparison Table

| Dimension | MemoryArena | MEMTRACK | MemoryAgentBench | LongMemEval | LoCoMo-Plus | LENS |
|-----------|-------------|----------|------------------|-------------|-------------|------|
| **Evidence** | Procedural constraints | Cross-platform state | Mixed events/facts | Implicit conversational | Implicit behavioral | Numeric progressions |
| **Reasoning** | Action chains (2-16 steps) | Cross-ref resolution | 1-3 hops | 1-4 sessions | Semantic disconnect | Causal chain (5-20 eps) |
| **Temporal** | Session ordering | Chronological + superseding | Chronological | Timestamped | Temporal gap | Progressive arc (5 phases) |
| **Noise** | Hard negatives | Organic organizational | Book-length context | Real conversations | Intervening dialogue | Format-matched orthogonal (3:1) |
| **Eval granularity** | Binary + partial | 3 dimensions | Per-competency | Binary | 3-level + binary | 3-tier, 10 metrics |
| **Construction** | Human-crafted | Expert + agent | Existing + LLM-generated | LLM + 95% human rewrite | LLM + similarity filter | 2-stage isolated pipeline |
| **Contamination** | Human verification | Novel scenarios | Counterfactual edits | Fictional users | BM25/MPNet filtering | Automated contamination + baseline check |
| **Scale** | >40K/task, 736 tasks | 47 datapoints | 3K+ questions, 1.44M tokens | 500 questions, 1.5M tokens | ~400 instances | 144 questions, 84K tokens |
| **Hardest for** | Action execution | Cross-platform resolution | Multi-hop forgetting | Scale + indirect evidence | Semantic disconnect | Pattern synthesis in noise |
| **Field consensus** | Memory doesn't help actions | Zep/Mem0 don't help | Forgetting is unsolved | Systems lose 30-64% | Cognitive memory collapses | Memory doesn't beat text search |

---

## Key Takeaway

The five benchmarks and LENS collectively reveal that **current memory systems fail across every dimension tested**:

- They fail at **action-grounded** memory (MemoryArena)
- They fail at **cross-platform** state tracking (MEMTRACK)
- They fail at **selective forgetting** (MemoryAgentBench)
- They fail at **scale** (LongMemEval)
- They fail at **cognitive/implicit** memory (LoCoMo-Plus)
- They fail at **longitudinal pattern synthesis** (LENS)

LENS's distinctive contribution is testing the **synthesis dimension** -- can a system go beyond recalling stored facts to deriving new conclusions from patterns in data? This is the gap no other benchmark addresses. The two-stage information-isolated construction, automated contamination validation, and three-tier differential scoring make LENS methodologically rigorous in ways that complement the other benchmarks.

The highest-priority adoptions for LENS are agentic task evaluation (MemoryArena's insight), cross-platform evidence format diversity (MEMTRACK's insight), and semantic disconnect queries (LoCoMo-Plus's insight). These would extend LENS from "can the system synthesize patterns?" to "can the system synthesize patterns, act on them, across heterogeneous sources, even when queries don't lexically match the evidence?"

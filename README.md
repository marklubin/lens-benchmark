<p align="center">
  <img src="lens_benchmark_logo_transparent.svg" alt="LENS Benchmark" width="600">
</p>

![Tests](https://img.shields.io/badge/tests-1040_passing-22c55e?style=flat-square)
![Python](https://img.shields.io/badge/python-3.11+-3b82f6?style=flat-square)
![License](https://img.shields.io/badge/license-MIT-06b6d4?style=flat-square)
![Scopes](https://img.shields.io/badge/scopes-16-eab308?style=flat-square)

---

## > WHAT IS LENS //

Most retrieval benchmarks dump a static corpus and test search quality. Real memory systems receive information **incrementally** — a support ticket each day, a sensor reading each hour, a clinical note each week — and must surface patterns that only emerge after enough data accumulates.

LENS streams timestamped episodes into your memory system chronologically, pauses at checkpoints to ask questions requiring synthesis across many episodes, and scores whether your system enables longitudinal reasoning — not just keyword retrieval.

If a single episode can answer the question, the benchmark is broken. LENS ensures signal only emerges from the *progression*.

---

## > RESULTS //

### V2: Memory Strategy Ablation (Current)

**Setup**: 10 scopes, 7 consolidation policies, M=3 repetitions, Fact F1 scoring. 2,100 answers generated, 1,900 graded (90.5%).

V2 isolates the **memory consolidation strategy** from retrieval architecture. All policies use the same underlying storage, embeddings, search, and agent loop — only the memory management policy varies.

| Rank | Policy | Fact F1 | Description |
|-----:|--------|--------:|-------------|
| 1 | core_faceted | 0.466 | 4 parallel folds (entity/relation/event/cause) + merge |
| 2 | summary | 0.443 | Progressive map-reduce summarization |
| 3 | core | 0.441 | Single-fold working memory (Letta/MemGPT pattern) |
| 4 | core_structured | 0.432 | Schema-driven structured observations (Mastra/ACE pattern) |
| 5 | core_maintained | 0.398 | Core memory + iterative refinement |
| 6 | base | 0.381 | Raw BM25 + semantic retrieval, no synthesis |
| 7 | null | 0.055 | No memory (parametric knowledge only) |

**Key findings:**
1. **Any memory beats no memory** — the null→base gap (+0.326) accounts for 79% of the total improvement
2. **Faceted decomposition wins** — 4 parallel cognitive folds capture more signal than a single pass (+0.025 over core)
3. **Refinement is dangerous** — iterative consolidation prunes useful signal (-0.043 vs core)
4. **No universal best strategy** — Kendall's W = 0.145 (weak concordance); different scopes favor different policies
5. **Domain matters more than strategy** — scope difficulty spans 0.109 to 0.763, a 6.9x range

Full results, per-scope breakdown, and statistical analysis: [LEADERBOARD.md](LEADERBOARD.md)

Research brief (PDF with figures): [v2-synix-benchmark/studies/grid/brief/research_brief.pdf](v2-synix-benchmark/studies/grid/brief/research_brief.pdf)

### V1: Adapter Benchmark (Legacy)

V1 compared 11 memory system architectures (Letta, GraphRAG, SQLite variants, etc.) across 6 scopes. The headline finding was that agent query quality — not memory architecture — is the binding constraint. See [LEADERBOARD.md](LEADERBOARD.md) for V1 results and methodology.

---

## > HOW IT WORKS //

```
                    ┌─────────────┐
                    │  spec.yaml  │    Dataset definition
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │   Episodes  │    Signal + distractor episodes
                    └──────┬──────┘
                           │ stream chronologically
                    ┌──────▼──────┐
                    │  Bank Build │    Chunk → embed → search index
                    │  + Policy   │    + consolidation (fold/summary/faceted)
                    └──────┬──────┘
                           │ at checkpoints...
                    ┌──────▼──────┐
                    │    Agent    │    Tool-use LLM interrogates
                    │  + Tools   │    memory via search + context
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │   Scorer    │    Fact F1 via few-shot
                    │  (per-fact) │    LLM grading
                    └─────────────┘
```

1. **Episodes stream chronologically** — Signal episodes follow a 5-phase narrative arc (baseline → early signal → red herring → escalation → root cause), interleaved with format-matched distractors.
2. **Bank build applies consolidation policy** — Episodes are chunked and indexed. Depending on the policy, derived context is synthesized (fold, summary, faceted decomposition, etc.) and injected into the agent's system prompt.
3. **At checkpoints, an LLM agent interrogates memory** — The agent answers questions using `memory_search` and injected context. No direct access to raw episodes.
4. **Fact F1 scoring** — Each key fact is graded as present/partial/absent by a few-shot LLM judge. F1 is computed across all facts per question.

---

## > QUICK START //

```bash
# Clone and install
git clone https://github.com/synix-dev/lens-benchmark.git
cd lens-benchmark
uv sync --all-extras

# Run tests
uv run pytest tests/unit/ -v
```

See [QUICKSTART.md](docs/guides/QUICKSTART.md) for a full walkthrough.

---

## > CONTRIBUTE A SCOPE //

Scopes define benchmark scenarios. Each scope has:
- A domain (system logs, clinical notes, financial reports, ...)
- A 5-phase narrative arc with signal distributed across episodes
- Key facts that require multi-episode synthesis
- Questions at checkpoints testing longitudinal reasoning

Current scopes span: cascading failures, financial irregularity, clinical signals, environmental drift, insider threats, market regimes, jailbreak detection, corporate acquisition, shadow APIs, clinical trials, zoning corruption, therapy chat, implicit decisions, epoch classification, value inversion, and parking friction.

Full guide: [SCOPE_GUIDE.md](docs/guides/SCOPE_GUIDE.md)

---

## > PROJECT STRUCTURE //

```
src/lens/
  adapters/          MemoryAdapter ABC, null/sqlite builtins, registry
  agent/             Agent harness, tool bridge, budget enforcement
  cli/               Click CLI (run, score, report, smoke, ...)
  core/              Episode, Question, GroundTruth, ScoreCard
  datagen/synix/     Two-stage dataset generation pipeline
  datasets/          Dataset loading
  matcher/           Answer matching
  report/            Report generation
  runner/            Benchmark runner with EpisodeVault anticheat
  scorer/            3-tier scoring (mechanical, judge, differential)
datasets/scopes/     16 scope specifications + generated artifacts
tests/unit/          1040 unit tests
v2-synix-benchmark/  V2 ablation study workspace
  src/bench/         Bank builder, policies, agent, scorer, runtime
  studies/grid/      Full grid results, figures, research brief
docs/                Documentation and guides
```

---

## > DOCUMENTATION //

| Document | Description |
|----------|-------------|
| [Quick Start](docs/guides/QUICKSTART.md) | Install and run |
| [Scope Guide](docs/guides/SCOPE_GUIDE.md) | Design and build a benchmark scope |
| [Leaderboard](LEADERBOARD.md) | V1 + V2 results and methodology |
| [Research Brief](v2-synix-benchmark/studies/grid/brief/research_brief.pdf) | V2 ablation study (PDF) |
| [Architecture](docs/architecture.md) | Core data flow and scoring internals |
| [Methodology](docs/methodology.md) | Dataset generation and contamination prevention |
| [Contributing](CONTRIBUTING.md) | How to contribute |

---

## > CITATION //

```bibtex
@software{lens_benchmark,
  title  = {LENS: Longitudinal Evidence-backed Narrative Signals},
  author = {Mark Lubin},
  year   = {2025},
  url    = {https://github.com/synix-dev/lens-benchmark}
}
```

---

## > LICENSE //

[MIT](LICENSE)

# LENS Benchmark — Rerun Plan for Clean Modal/Agent Driver Data

**Created**: 2026-03-04
**Goal**: Get one clean, comparable modal/agent-driver run per adapter × scope so we can answer: "Does the memory architecture matter when the agent formulates its own queries?"

---

## The Problem

We have two sets of results, and neither answers the question we set out to test:

1. **Static driver** (valid, 42 scored runs): Pre-computed query plans bypass agent query formulation. Tells us which retrieval backend surfaces the right evidence *given identical queries*. Cannot compare memory architectures end-to-end.

2. **Modal/agent driver** (unreliable, ~100 scored runs): The Qwen3 chat template (`qwen3_permissive.jinja`) was broken — tool definitions weren't rendered in the format vLLM's `qwen3_coder` parser expected. Most runs scored 0.050 (floor) because the agent couldn't call search/retrieve tools. We picked "best-of-N" from broken runs and called it a comparison.

**We need**: One clean modal run per adapter × scope, all using the same fixed infrastructure, scored once.

---

## What Exists Today

### Static Driver Scores (answer_quality, best per adapter × scope)

| Adapter | S07 | S08 | S09 | S10 | S11 | S12 | Mean |
|---------|-----|-----|-----|-----|-----|-----|------|
| sqlite-chunked-hybrid | 0.746 | 0.702 | 0.773 | 0.277 | 0.551 | 0.450 | **0.583** |
| hopping-hybrid | 0.704 | 0.693 | 0.668 | -- | -- | -- | 0.689 |
| hierarchical-hybrid | 0.700 | 0.617 | 0.741 | -- | -- | -- | 0.686 |
| mem0-raw | 0.512 | 0.542 | 0.829 | -- | -- | -- | 0.628 |
| compaction | 0.625 | -- | -- | -- | -- | -- | 0.625 |
| letta-sleepy | 0.471 | 0.706 | 0.730 | 0.391 | 0.430 | 0.506 | 0.539 |
| cognee | 0.404 | 0.581 | 0.635 | -- | -- | -- | 0.540 |
| letta | 0.500 | 0.655 | 0.830 | 0.266 | 0.480 | 0.475 | 0.534 |
| hopping | 0.479 | 0.428 | 0.516 | 0.287 | 0.462 | 0.438 | 0.435 |
| graphrag-light | -- | -- | -- | 0.298 | 0.514 | 0.400 | 0.404 |
| hierarchical | 0.346 | 0.392 | 0.352 | 0.323 | 0.413 | 0.427 | 0.376 |
| null | 0.450 | 0.349 | 0.260 | 0.454 | 0.290 | 0.290 | 0.349 |

### Modal Driver Scores (answer_quality, best per adapter — UNRELIABLE)

| Adapter | S07 | S08 | S09 | S10 | S11 | S12 | Mean |
|---------|-----|-----|-----|-----|-----|-----|------|
| graphrag-light | 0.633 | 0.525 | 0.577 | 0.390 | 0.510 | 0.471 | **0.518** |
| letta-entity | 0.450 | 0.544 | 0.291 | 0.386 | 0.425 | 0.475 | 0.428 |
| letta-v4 | 0.396 | 0.513 | 0.659 | 0.360 | 0.233 | 0.402 | 0.427 |
| hopping-hybrid | 0.321 | 0.378 | 0.387 | -- | -- | -- | 0.362 |
| triadv1-pairs | 0.417 | 0.399 | 0.458 | 0.304 | 0.234 | 0.346 | 0.360 |
| sqlite-chunked-hybrid | 0.250 | 0.372 | 0.251 | 0.317 | 0.438 | 0.500 | 0.354 |
| hierarchical-hybrid | 0.367 | 0.332 | 0.344 | -- | -- | -- | 0.347 |
| null | 0.342 | 0.338 | 0.295 | 0.340 | 0.297 | 0.360 | 0.329 |
| hierarchical | 0.358 | 0.283 | 0.313 | 0.182 | 0.364 | 0.450 | 0.325 |
| hopping | 0.300 | 0.355 | 0.184 | 0.255 | 0.387 | 0.444 | 0.321 |
| letta | 0.296 | 0.322 | 0.266 | 0.216 | 0.334 | 0.463 | 0.316 |
| letta-sleepy | 0.429 | 0.210 | 0.180 | 0.271 | 0.358 | 0.331 | 0.297 |
| cognee | 0.287 | 0.291 | 0.305 | -- | -- | -- | 0.295 |
| mem0-raw | 0.287 | 0.340 | 0.241 | -- | -- | -- | 0.290 |

**WARNING**: These modal scores are best-of-N from unreliable runs. They cannot be trusted for comparison.

---

## Step-by-Step Rerun Plan

### Step 0: Verify the chat template is actually broken (5 min)

Before fixing anything, confirm the problem exists:

```bash
# Send a tool-call test to the Modal vLLM server
curl -s https://synix--lens-llm-llm-serve.modal.run/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3.5-35B-A3B",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Search for information about clinical trials."}
    ],
    "tools": [
      {"type": "function", "function": {"name": "search", "description": "Search the memory store", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}}
    ],
    "max_tokens": 256
  }'
```

Check if the response contains a `tool_calls` field with properly parsed function calls. If it does, the template is fine and the problem was something else. If it returns plain text with `<tool_call>` XML mixed in, the parser isn't matching.

### Step 1: Fix the chat template (30 min)

**File**: `infra/modal/qwen3_permissive.jinja`

The template already renders tool definitions in Qwen3-Coder XML format (lines 56-110). The issue may be more subtle — compare against the official Qwen3-Coder template from HuggingFace to find what's different. Common problems:
- Missing required attributes on `<function=...>` tags
- Wrong parameter format that the `qwen3_coder` parser regex doesn't match
- Tool response format not matching what the parser expects for multi-turn

**Verification**: After fixing, redeploy and confirm the curl test from Step 0 returns proper `tool_calls` JSON.

### Step 2: Validate with ONE adapter end-to-end (15 min)

Run the null adapter (fastest, no memory system) on S10:

```bash
export LENS_LLM_API_KEY=dummy
export LENS_LLM_API_BASE=https://synix--lens-llm-llm-serve.modal.run/v1
export LENS_LLM_MODEL=Qwen/Qwen3.5-35B-A3B
export LENS_EMBED_BASE_URL=https://synix--lens-embed-serve.modal.run
export LENS_EMBED_API_KEY=dummy

uv run lens run --config configs/null_scope10d.json -v
```

**Success criteria**:
- Agent makes tool calls (search, retrieve) — visible in verbose output
- All 10 questions get non-empty answers
- answer_quality > 0.15 after scoring

If this fails, do NOT proceed. Diagnose and fix.

### Step 3: Run self-contained adapters (2-3 hours)

These adapters need no external services (no Letta, no Redis, no Cognee):

| Adapter | Config pattern | Est. time |
|---------|---------------|-----------|
| null | `null_scope{07..12}d.json` | 30 min |
| sqlite-chunked-hybrid | `sqlite_chunked_hybrid_scope{07..12}d.json` | 90 min |
| graphrag-light | `graphrag_light_scope{07..12}d.json` | 90 min |
| hierarchical | `hierarchical_scope{07..12}d.json` | 90 min |
| hopping | `hopping_scope{07..12}d.json` | 90 min |
| triadv1-pairs | `triadv1_pairs_scope{07..12}d.json` | 120 min (may hang — see known issues) |

**Run each adapter sequentially through all 6 scopes** (not in parallel — avoid resource contention). Different adapters CAN run in parallel if Modal has enough capacity (`min_containers >= 4`).

**After each run**: Score immediately. If answer_quality < 0.10, something is wrong — stop and investigate.

```bash
uv run lens score --run output/<run_id> --judge-model Qwen/Qwen3.5-35B-A3B
```

### Step 4: Run hybrid variants (1-2 hours)

These are the same base adapters with FTS+embedding fusion. Only needed for S10-12 (S07-09 already have runs, but they're from the broken era — rerun everything for consistency):

| Adapter | Config pattern |
|---------|---------------|
| hierarchical-hybrid | `hierarchical_hybrid_scope{07..12}d.json` |
| hopping-hybrid | `hopping_hybrid_scope{07..12}d.json` |

**Need configs**: `hierarchical_hybrid_scope{10,11,12}d.json` and `hopping_hybrid_scope{10,11,12}d.json` may not exist yet. Create them if missing.

### Step 5: Run Letta variants (3-4 hours)

Requires Letta server. **Run ONE adapter at a time on the Letta server.**

| Adapter | Config pattern | Notes |
|---------|---------------|-------|
| letta | `letta_scope{07..12}d.json` | Standard Letta |
| letta-sleepy | `letta_sleepy_scope{07..12}d.json` | Sleep consolidation |
| letta-entity | needs configs | Dynamic entity blocks |
| letta-v4 | needs configs | Core memory blocks |

**Letta server setup**:
```bash
podman run -d --name letta -p 8283:8283 \
  -e HTTPS_PROXY="" -e HTTP_PROXY="" -e NO_PROXY="*" \
  letta/letta:latest
```
Then apply patches (ApproxTokenCounter, archival limit 65K, disable run tracking, Redis None guard, enforce_run_id_set=False in v2+v3), restart, create openai-proxy + embed-proxy providers.

**Env vars**:
```bash
export LETTA_BASE_URL=http://localhost:8283
export LETTA_LLM_MODEL=openai-proxy/Qwen/Qwen3.5-35B-A3B
export LETTA_EMBED_MODEL=embed-proxy/text-embedding-3-small
```

### Step 6: Run heavy adapters (2-3 hours)

| Adapter | Config pattern | Notes |
|---------|---------------|-------|
| cognee | needs S10-12 configs | Requires separate setup |
| mem0-raw | needs S10-12 configs | Requires separate setup |
| compaction | needs configs | May OOM on long episodes |

These may require adapter-specific infrastructure. Decide if they're worth the effort based on earlier results.

### Step 7: Score everything (1 hour)

```bash
for run_dir in output/*/; do
  if [ ! -d "$run_dir/scores" ]; then
    uv run lens score --run "$run_dir" --judge-model Qwen/Qwen3.5-35B-A3B
  fi
done
```

### Step 8: Compile results and scale down

1. Build final comparison table: adapter × scope × {answer_quality, composite, evidence_grounding}
2. Set `min_containers=0` in `infra/modal/llm_server.py` and redeploy
3. Update STATUS_REPORT.md

---

## Known Issues to Address

### triadv1-pairs hangs
- Reproducible on S10. `prepare()` sends growing notebook to LLM with no timeout.
- **Fix**: Add `httpx.Timeout(120.0)` to OpenAI client in `triad.py`, cap notebook at 8000 chars, cap episode text at 2000 chars in recording prompt.

### Evidence grounding gate distorts composite
- `evidence_grounding >= 0.5` gates composite to 0. Many runs with good answers get zeroed because citation format doesn't match.
- **For analysis**: Report answer_quality alongside composite. Consider dropping the gate or lowering threshold.

### Letta container instability
- NEVER run multiple Claude agents targeting the same Letta container.
- Use a single sequential runner for all Letta adapter runs.

### Modal cold starts
- Set `min_containers >= 2` before running benchmarks.
- After all runs complete, set `min_containers=0` to stop billing.

---

## Target Adapter List (priority order)

**Tier 1 — Must have** (self-contained, fast):
1. null (baseline)
2. sqlite-chunked-hybrid (retrieval leader)
3. graphrag-light (graph + retrieval)
4. hierarchical (summarization)
5. hopping (multi-hop)

**Tier 2 — Should have** (need Letta server):
6. letta
7. letta-sleepy

**Tier 3 — Nice to have** (complex setup or unstable):
8. triadv1-pairs (needs hang fix first)
9. letta-entity (needs Letta server + configs)
10. letta-v4 (needs Letta server + configs)
11. hierarchical-hybrid (needs configs)
12. hopping-hybrid (needs configs)

**Tier 4 — Skip unless time allows**:
13. cognee (complex setup, moderate value)
14. mem0-raw (complex setup, lower scores)
15. compaction (OOM risk on narrative episodes)

---

## Minimum Viable Dataset

For a publishable comparison, we need at minimum:
- **5 Tier 1 adapters × 6 scopes = 30 runs** (all self-contained, ~6 hours total)
- Plus null baseline makes 6 × 6 = 36 runs
- Score all 36
- This gives us a clean head-to-head of the core architectures with an agent formulating its own queries

Everything in Tiers 2-4 is gravy.

---

## Run Validation Protocol

This is the process that prevents us from generating hours of useless data. Follow it for EVERY phase of work.

### Principle: Validate narrow, then widen

Never run N things in parallel hoping they all work. Run 1, confirm it works, then scale. At every level of the stack.

### Level 1: Infrastructure check (before any runs)

Before launching any benchmark run, confirm the LLM endpoint actually works with tool calls:

```bash
# 1. Is the server responding?
curl -s https://synix--lens-llm-llm-serve.modal.run/v1/models | python3 -m json.tool

# 2. Can it do a basic completion?
curl -s https://synix--lens-llm-llm-serve.modal.run/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen3.5-35B-A3B","messages":[{"role":"user","content":"Say hello"}],"max_tokens":32}' \
  | python3 -c "import sys,json; r=json.load(sys.stdin); print(r['choices'][0]['message']['content'])"

# 3. Can it parse tool calls? THIS IS THE CRITICAL ONE.
curl -s https://synix--lens-llm-llm-serve.modal.run/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model":"Qwen/Qwen3.5-35B-A3B",
    "messages":[{"role":"user","content":"Search for clinical trial data"}],
    "tools":[{"type":"function","function":{"name":"search","description":"Search memory","parameters":{"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}}}],
    "max_tokens":256
  }' | python3 -c "
import sys, json
r = json.load(sys.stdin)
msg = r['choices'][0]['message']
if msg.get('tool_calls'):
    print('PASS: tool_calls parsed correctly')
    for tc in msg['tool_calls']:
        print(f'  {tc[\"function\"][\"name\"]}({tc[\"function\"][\"arguments\"]})')
elif '<tool_call>' in msg.get('content', ''):
    print('FAIL: tool_call XML in content — parser not matching')
else:
    print(f'UNCLEAR: no tool_calls, content={msg.get(\"content\", \"\")[:200]}')
"
```

**If step 3 says FAIL**: Fix the chat template. Do not proceed.
**If step 3 says PASS**: Infrastructure is ready.

Similarly for embeddings:
```bash
curl -s https://synix--lens-embed-serve.modal.run/embeddings \
  -H "Content-Type: application/json" \
  -d '{"input":"test query","model":"text-embedding-3-small"}' \
  | python3 -c "import sys,json; r=json.load(sys.stdin); print(f'PASS: {len(r[\"data\"][0][\"embedding\"])} dims')"
```

### Level 2: Single-run validation (before scaling to full matrix)

Run ONE adapter on ONE scope. Check the output manually.

```bash
uv run lens run --config configs/null_scope10d.json -v 2>&1 | tee /tmp/validation_run.log
```

**Check these things in the output**:
1. `grep "tool_calls" /tmp/validation_run.log | head -5` — agent is making tool calls
2. `grep "answer_text" /tmp/validation_run.log | head -3` — answers are non-empty
3. No repeating errors or timeouts in the log

**Then score and check**:
```bash
uv run lens score --run output/<run_id> --judge-model Qwen/Qwen3.5-35B-A3B
```

**Gate criteria** — ALL must pass:
- [ ] 8+ out of 10 questions have non-empty answers
- [ ] answer_quality > 0.15
- [ ] Agent made > 5 total tool calls across all questions
- [ ] No question took > 120 seconds

**If any gate fails**: Diagnose. Don't scale.

### Level 3: First-adapter validation (before running remaining adapters)

After null passes, run sqlite-chunked-hybrid on the same scope (S10). This validates that the memory adapter's search/retrieve tools also work through the pipeline.

Same gate criteria as Level 2, plus:
- [ ] answer_quality > answer_quality of null (memory should help, not hurt)
- [ ] Agent called both `search` and `retrieve` tools (not just one)

### Level 4: Per-run spot checks (during the full matrix)

After EACH run completes (not each batch — each individual run):

```bash
# Quick health check — run this immediately after each run
python3 -c "
import json, sys
qr_files = __import__('glob').glob(f'output/{sys.argv[1]}/scopes/*/checkpoint_*/question_results.json')
if not qr_files:
    print('NO RESULTS FILES'); sys.exit(1)
qr = json.load(open(sorted(qr_files)[-1]))  # last checkpoint
answered = sum(1 for q in qr if q.get('answer', {}).get('answer_text', '').strip())
total_tools = sum(q.get('answer', {}).get('tool_calls_made', 0) for q in qr)
print(f'Answers: {answered}/{len(qr)}, Tool calls: {total_tools}')
if answered < len(qr) * 0.7:
    print('WARNING: >30% empty answers — possible tool-call failure')
if total_tools < len(qr) * 2:
    print('WARNING: very few tool calls — agent may not be using memory')
" <run_id>
```

**If WARNING appears**: Score this run immediately and check answer_quality before continuing to the next scope. If answer_quality < 0.10, stop and investigate.

### Level 5: Cross-run consistency check (after completing an adapter)

After one adapter finishes all 6 scopes, compare scores:

```bash
python3 -c "
import json, glob, sys
adapter = sys.argv[1]
for cfg_path in sorted(glob.glob('output/*/config.json')):
    cfg = json.load(open(cfg_path))
    if cfg.get('adapter') != adapter or cfg.get('llm',{}).get('provider') != 'modal':
        continue
    run_id = cfg_path.split('/')[1]
    sc_path = f'output/{run_id}/scores/scorecard.json'
    if not __import__('os').path.exists(sc_path):
        continue
    sc = json.load(open(sc_path))
    aq = next((m['value'] for m in sc.get('metrics',[]) if m['name']=='answer_quality'), None)
    ds = cfg.get('dataset','')
    scope = [s for s in ['07','08','09','10','11','12'] if f'scope_{s}' in ds]
    scope = scope[0] if scope else '??'
    print(f'  S{scope}: aq={aq:.3f}  run={run_id}')
" <adapter-name>
```

**Red flags**:
- Any scope with answer_quality < 0.10 → that run is broken, redo it
- Variance > 0.4 between scopes → investigate the outlier
- answer_quality below null baseline → adapter may be hurting, not helping

---

## Debugging Decision Tree

When something goes wrong, follow this tree instead of retrying blindly.

```
Run failed or scored floor
├── Agent produced 0 answers
│   ├── Check: did the run even start? (look for output/<run_id>/scopes/)
│   │   └── No scopes dir → config error, dataset missing, or crash on init
│   └── Check: did questions get attempted? (look for question_results.json)
│       └── No question_results → crash during ingest/prepare phase
│           ├── Timeout → check Modal server logs: `modal app logs lens-llm`
│           ├── OOM → episode too large for adapter (compaction problem)
│           └── API error → check LENS_LLM_API_BASE, LENS_LLM_API_KEY
│
├── Agent produced answers but all empty/minimal (aq ≈ 0.05)
│   ├── Check: did agent make any tool calls? (tool_calls_made in results)
│   │   ├── 0 tool calls → TOOL CALLING IS BROKEN
│   │   │   ├── Check curl tool-call test (Level 1 step 3)
│   │   │   ├── If curl works but agent doesn't → agent harness not sending tools
│   │   │   └── If curl fails → chat template / vLLM parser issue
│   │   └── Many tool calls but empty answers → search returns no results
│   │       ├── Check: did ingest/prepare complete? (adapter state populated?)
│   │       ├── Check: are search results relevant? (look at retrieved_ref_ids)
│   │       └── Check: is the agent's final synthesis failing? (LLM issue)
│   └── Check: did agent hit budget limit? (budget_violations in results)
│       └── Yes → agent spinning on bad tool calls, burning budget
│
├── Agent produced answers but score is suspiciously low (aq < 0.15)
│   ├── Compare to null adapter on same scope
│   │   ├── Null is also low → scope may be hard, or LLM quality issue
│   │   └── Null is higher → adapter is hurting retrieval
│   └── Read actual answers — are they off-topic? Truncated? Hallucinated?
│
├── Run hangs (no progress for > 10 min)
│   ├── Check: which phase? (ingest, prepare, or question-answering)
│   │   ├── Ingest → adapter's ingest() is slow (e.g., Cognee, Graphiti)
│   │   ├── Prepare → adapter's prepare() is slow (e.g., triadv1 notebook growth)
│   │   └── Q&A → LLM not responding (Modal cold start or overload)
│   ├── Check Modal: `modal app logs lens-llm` — is vLLM healthy?
│   └── DO NOT just kill and retry. Identify which call is hanging.
│
└── Run completed but scoring fails
    ├── Judge model not responding → same Modal checks as above
    ├── scorecard.json missing → scoring crashed, check stderr
    └── composite=0 but aq>0 → evidence_grounding gate tripped (expected for some adapters)
```

### The Golden Rule

**If you don't understand why something failed, you don't have permission to retry it.**

Read the logs. Check the output. Trace the actual API calls. Understand the failure. Fix it. THEN run again.

---

## Critical Rules for the Rerun Session

1. **Fix the template FIRST. Verify ONE run works. Then scale.**
2. **If a run scores < 0.10 answer_quality, STOP. Don't run more.**
3. **One adapter at a time per Letta server. No concurrency.**
4. **Score immediately after each run. Don't batch.**
5. **Set min_containers=0 when done. Don't forget.**
6. **Follow the validation protocol at every level. No shortcuts.**
7. **If you don't understand a failure, don't retry. Diagnose first.**

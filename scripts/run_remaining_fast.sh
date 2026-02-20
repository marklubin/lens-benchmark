#!/usr/bin/env bash
# Phase 1: Run remaining benchmarks (fast, no scoring)
# Phase 2: Batch-score all unscored runs in parallel
set -euo pipefail
cd /home/mark/lens-benchmark

TK=$(grep '^TOGETHER_API_KEY=' .env | cut -d= -f2 | tr -d '"' | tr -d "'")
JUDGE="Qwen/Qwen3-235B-A22B-Instruct-2507-tput"

run_only() {
  local label="$1" config="$2"
  shift 2
  echo "[$(date +%H:%M:%S)] RUN: $label"
  eval "LENS_LLM_API_KEY=$TK LENS_LLM_API_BASE=https://api.together.xyz/v1 LENS_LLM_MODEL=$JUDGE OPENAI_API_KEY=$TK LENS_EMBED_BASE_URL=https://api.together.xyz/v1 LENS_EMBED_MODEL=Alibaba-NLP/gte-modernbert-base $* uv run lens run --config $config --parallel-questions 4 --cache-dir .cache/adapter -v 2>&1" | grep -E "Artifacts saved|Run .* complete"
}

echo "=== Phase 1: Run remaining benchmarks ==="

# Cognee (cached)
CENV="COGNEE_LLM_API_KEY=$TK COGNEE_LLM_API_BASE=https://api.together.xyz/v1 COGNEE_LLM_MODEL=meta-llama/Llama-3.3-70B-Instruct-Turbo COGNEE_EMBED_API_KEY=$TK COGNEE_EMBED_API_BASE=https://api.together.xyz/v1 COGNEE_EMBED_MODEL=Alibaba-NLP/gte-modernbert-base COGNEE_EMBED_DIM=768 ENABLE_BACKEND_ACCESS_CONTROL=false"
run_only "cognee 4K" "configs/cognee_scope01_4k.json" "$CENV"
run_only "cognee 2K" "configs/cognee_scope01_2k.json" "$CENV"

# Letta (no cache, full ingest — still only ~5 min each)
LENV="LETTA_BASE_URL=http://localhost:8283"
run_only "letta 4K" "configs/letta_scope01_4k.json" "$LENV"
run_only "letta 2K" "configs/letta_scope01_2k.json" "$LENV"

echo ""
echo "=== Phase 2: Batch-score all unscored runs (parallel) ==="

# Find all unscored runs
UNSCORED=()
for dir in output/*/; do
  [ -f "$dir/run_manifest.json" ] || continue
  [ -f "$dir/scores/scorecard.json" ] && continue
  UNSCORED+=("$(basename "$dir")")
done

echo "Unscored runs: ${#UNSCORED[@]}"

score_one() {
  local run_id="$1"
  OPENAI_API_KEY="$TK" OPENAI_BASE_URL="https://api.together.xyz/v1" \
    uv run lens score --run "output/$run_id" --judge-model "$JUDGE" -v 2>&1 | tail -3
  echo "SCORED: $run_id"
}

# Score in parallel batches of 4
BATCH=4
for ((i=0; i<${#UNSCORED[@]}; i+=BATCH)); do
  batch=("${UNSCORED[@]:i:BATCH}")
  echo "[$(date +%H:%M:%S)] Scoring batch: ${batch[*]}"
  pids=()
  for rid in "${batch[@]}"; do
    score_one "$rid" &
    pids+=($!)
  done
  for pid in "${pids[@]}"; do
    wait "$pid" || echo "WARN: $pid failed"
  done
done

echo ""
echo "=== All done at $(date) ==="

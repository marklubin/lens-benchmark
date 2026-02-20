#!/usr/bin/env bash
# Run constrained benchmark matrix: 5 adapters × 2 caps = 10 runs + compaction standard
# Plus score all runs with v3.2 (naive baseline advantage)
set -euo pipefail
cd /home/mark/lens-benchmark

TK=$(grep '^TOGETHER_API_KEY=' .env | cut -d= -f2 | tr -d '"' | tr -d "'")
JUDGE_MODEL="Qwen/Qwen3-235B-A22B-Instruct-2507-tput"

# Common env for all runs
BASE_ENV="LENS_LLM_API_KEY=$TK LENS_LLM_API_BASE=https://api.together.xyz/v1 LENS_LLM_MODEL=Qwen/Qwen3-235B-A22B-Instruct-2507-tput"
EMBED_ENV="OPENAI_API_KEY=$TK LENS_EMBED_BASE_URL=https://api.together.xyz/v1 LENS_EMBED_MODEL=Alibaba-NLP/gte-modernbert-base"

# Graphiti env
GRAPHITI_ENV="GRAPHITI_LLM_API_KEY=$TK GRAPHITI_LLM_BASE_URL=https://api.together.xyz/v1 GRAPHITI_LLM_MODEL=meta-llama/Llama-3.3-70B-Instruct-Turbo GRAPHITI_EMBED_API_KEY=$TK GRAPHITI_EMBED_BASE_URL=https://api.together.xyz/v1 GRAPHITI_EMBED_MODEL=Alibaba-NLP/gte-modernbert-base GRAPHITI_EMBED_DIM=768"

# Cognee env
COGNEE_ENV="COGNEE_LLM_API_KEY=$TK COGNEE_LLM_API_BASE=https://api.together.xyz/v1 COGNEE_LLM_MODEL=meta-llama/Llama-3.3-70B-Instruct-Turbo COGNEE_EMBED_API_KEY=$TK COGNEE_EMBED_API_BASE=https://api.together.xyz/v1 COGNEE_EMBED_MODEL=Alibaba-NLP/gte-modernbert-base COGNEE_EMBED_DIM=768 ENABLE_BACKEND_ACCESS_CONTROL=false"

# Letta env
LETTA_ENV="LETTA_BASE_URL=http://localhost:8283"

run_and_score() {
  local label="$1"
  local config="$2"
  local extra_env="$3"

  echo ""
  echo "=========================================="
  echo "[$(date +%H:%M:%S)] Running: $label"
  echo "=========================================="

  local cmd="$BASE_ENV $EMBED_ENV $extra_env uv run lens run --config $config --parallel-questions 4 --cache-dir .cache/adapter -v"

  # Run benchmark
  local run_id
  run_id=$(eval "$cmd" 2>&1 | tee /dev/stderr | grep "Artifacts saved to" | grep -oP 'output/\K[a-f0-9]+')

  if [ -z "$run_id" ]; then
    echo "ERROR: Failed to extract run_id for $label"
    return 1
  fi

  echo "[$(date +%H:%M:%S)] Run complete: $run_id"

  # Copy naive baseline cache only for standard-budget runs (constrained runs
  # need separate baselines with truncated episode context)
  local max_tokens
  max_tokens=$(python3 -c "import json; c=json.load(open('$config')); print(c.get('agent_budget',{}).get('max_cumulative_result_tokens', 0))" 2>/dev/null || echo "0")
  if [ "$max_tokens" = "0" ]; then
    local cache_src="output/be0003e5447b/scores/naive_baseline_cache.json"
    if [ -f "$cache_src" ]; then
      mkdir -p "output/$run_id/scores"
      cp "$cache_src" "output/$run_id/scores/naive_baseline_cache.json"
    fi
  fi

  # Score
  echo "[$(date +%H:%M:%S)] Scoring: $run_id"
  OPENAI_API_KEY="$TK" OPENAI_BASE_URL="https://api.together.xyz/v1" \
    uv run lens score --run "output/$run_id" --judge-model "$JUDGE_MODEL" -v 2>&1 | tail -5

  echo "[$(date +%H:%M:%S)] Done: $label (run_id=$run_id)"
}

echo "=== Constrained Benchmark Matrix ==="
echo "Starting at $(date)"
echo ""

# Group 1: No external services needed (can run in any order)
# chunked-hybrid: fast, no caching, ~3 min each
run_and_score "chunked-hybrid 4K" "configs/chunked_hybrid_scope01_4k.json" ""
run_and_score "chunked-hybrid 2K" "configs/chunked_hybrid_scope01_2k.json" ""

# compaction: standard first (for cache), then constrained
run_and_score "compaction standard" "configs/compaction_scope01.json" ""
run_and_score "compaction 4K" "configs/compaction_scope01_4k.json" ""
run_and_score "compaction 2K" "configs/compaction_scope01_2k.json" ""

# Group 2: External services
# graphiti: cached (skip ingest/prepare)
run_and_score "graphiti 4K" "configs/graphiti_scope01_4k.json" "$GRAPHITI_ENV"
run_and_score "graphiti 2K" "configs/graphiti_scope01_2k.json" "$GRAPHITI_ENV"

# cognee: cached
run_and_score "cognee 4K" "configs/cognee_scope01_4k.json" "$COGNEE_ENV"
run_and_score "cognee 2K" "configs/cognee_scope01_2k.json" "$COGNEE_ENV"

# letta: no caching, needs embed proxy
run_and_score "letta 4K" "configs/letta_scope01_4k.json" "$LETTA_ENV"
run_and_score "letta 2K" "configs/letta_scope01_2k.json" "$LETTA_ENV"

echo ""
echo "=========================================="
echo "All constrained runs complete at $(date)"
echo "=========================================="

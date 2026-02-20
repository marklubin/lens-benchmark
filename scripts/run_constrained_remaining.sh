#!/usr/bin/env bash
# Run remaining constrained benchmarks (chunked-hybrid already done)
# Includes compaction standard + 4K + 2K, graphiti/cognee/letta 4K + 2K
set -euo pipefail
cd /home/mark/lens-benchmark

TK=$(grep '^TOGETHER_API_KEY=' .env | cut -d= -f2 | tr -d '"' | tr -d "'")
JUDGE_MODEL="Qwen/Qwen3-235B-A22B-Instruct-2507-tput"
AGENT_MODEL="Qwen/Qwen3-235B-A22B-Instruct-2507-tput"

run_one() {
  local label="$1"
  local config="$2"
  shift 2
  local extra_env="$*"

  echo ""
  echo "=========================================="
  echo "[$(date +%H:%M:%S)] Running: $label"
  echo "=========================================="

  local cmd="LENS_LLM_API_KEY=$TK LENS_LLM_API_BASE=https://api.together.xyz/v1 LENS_LLM_MODEL=$AGENT_MODEL OPENAI_API_KEY=$TK LENS_EMBED_BASE_URL=https://api.together.xyz/v1 LENS_EMBED_MODEL=Alibaba-NLP/gte-modernbert-base $extra_env uv run lens run --config $config --parallel-questions 4 --cache-dir .cache/adapter -v 2>&1"

  local output
  output=$(eval "$cmd")
  echo "$output" | grep -E "lens |Run complete" | tail -10

  local run_id
  run_id=$(echo "$output" | grep "Artifacts saved to" | grep -oP 'output/\K[a-f0-9]+' || true)

  if [ -z "$run_id" ]; then
    echo "ERROR: No run_id for $label"
    return 1
  fi

  echo "[$(date +%H:%M:%S)] Run $run_id complete. Scoring..."

  # Score
  OPENAI_API_KEY="$TK" OPENAI_BASE_URL="https://api.together.xyz/v1" \
    uv run lens score --run "output/$run_id" --judge-model "$JUDGE_MODEL" -v 2>&1 | tail -5

  # Extract result
  local sc="output/$run_id/scores/scorecard.json"
  if [ -f "$sc" ]; then
    python3 -c "
import json
d = json.load(open('$sc'))
m = {x['name']: x['value'] for x in d['metrics']}
print(f'  Composite={d[\"composite_score\"]:.4f} NBA={m.get(\"naive_baseline_advantage\",0):.4f} AQ={m.get(\"answer_quality\",0):.4f} EC={m.get(\"evidence_coverage\",0):.4f}')
"
  fi
  echo "[$(date +%H:%M:%S)] Done: $label ($run_id)"
}

echo "Starting constrained runs at $(date)"

# Compaction (standard + 4K + 2K) — no external services
run_one "compaction standard" "configs/compaction_scope01.json"
run_one "compaction 4K" "configs/compaction_scope01_4k.json"
run_one "compaction 2K" "configs/compaction_scope01_2k.json"

# Graphiti (4K + 2K) — FalkorDB running, has cache
GENV="GRAPHITI_LLM_API_KEY=$TK GRAPHITI_LLM_BASE_URL=https://api.together.xyz/v1 GRAPHITI_LLM_MODEL=meta-llama/Llama-3.3-70B-Instruct-Turbo GRAPHITI_EMBED_API_KEY=$TK GRAPHITI_EMBED_BASE_URL=https://api.together.xyz/v1 GRAPHITI_EMBED_MODEL=Alibaba-NLP/gte-modernbert-base GRAPHITI_EMBED_DIM=768"
run_one "graphiti 4K" "configs/graphiti_scope01_4k.json" "$GENV"
run_one "graphiti 2K" "configs/graphiti_scope01_2k.json" "$GENV"

# Cognee (4K + 2K) — no container, has cache
CENV="COGNEE_LLM_API_KEY=$TK COGNEE_LLM_API_BASE=https://api.together.xyz/v1 COGNEE_LLM_MODEL=meta-llama/Llama-3.3-70B-Instruct-Turbo COGNEE_EMBED_API_KEY=$TK COGNEE_EMBED_API_BASE=https://api.together.xyz/v1 COGNEE_EMBED_MODEL=Alibaba-NLP/gte-modernbert-base COGNEE_EMBED_DIM=768 ENABLE_BACKEND_ACCESS_CONTROL=false"
run_one "cognee 4K" "configs/cognee_scope01_4k.json" "$CENV"
run_one "cognee 2K" "configs/cognee_scope01_2k.json" "$CENV"

# Letta (4K + 2K) — server running, no cache (full ingest needed)
LENV="LETTA_BASE_URL=http://localhost:8283"
run_one "letta 4K" "configs/letta_scope01_4k.json" "$LENV"
run_one "letta 2K" "configs/letta_scope01_2k.json" "$LENV"

echo ""
echo "=========================================="
echo "All remaining constrained runs complete at $(date)"
echo "=========================================="

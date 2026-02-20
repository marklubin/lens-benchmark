#!/usr/bin/env bash
# Re-score all scope 01 runs with v3.2 (naive baseline advantage).
# Step 1: Score one run to generate naive baseline cache (24 LLM calls)
# Step 2: Copy cache to all other runs
# Step 3: Score remaining runs in parallel batches

set -euo pipefail
cd /home/mark/lens-benchmark

TK=$(grep '^TOGETHER_API_KEY=' .env | cut -d= -f2 | tr -d '"' | tr -d "'")
JUDGE_MODEL="Qwen/Qwen3-235B-A22B-Instruct-2507-tput"
OUTDIR="output"

# All 14 canonical scope 01 run IDs
RUNS=(
  be0003e5447b  # letta
  8ef0786f0eb5  # letta-sleepy V3
  8c415eba299e  # letta-sleepy V0
  1cbe02135799  # letta-sleepy V1
  6e6e53e7581d  # letta-sleepy V2
  8581429063e7  # chunked-hybrid+batch
  fef20b05d46b  # embedding-openai
  830d711e5c17  # mem0-raw
  8b9e83ae9dec  # chunked-hybrid L=7
  77545ef2b9b8  # cognee
  2bc821424282  # graphiti
  040bb488abbd  # hindsight
  11d7bf53e4f0  # sqlite-fts
  a119b4906684  # mem0-extract
)

score_run() {
  local run_id="$1"
  local label="$2"
  echo "[$(date +%H:%M:%S)] Scoring $run_id ($label)..."
  OPENAI_API_KEY="$TK" OPENAI_BASE_URL="https://api.together.xyz/v1" \
    uv run lens score --run "$OUTDIR/$run_id" --judge-model "$JUDGE_MODEL" -v \
    2>&1 | tail -5
  echo "[$(date +%H:%M:%S)] Done: $run_id ($label)"
  echo "---"
}

# Step 1: Score first run to generate naive baseline cache
echo "=== Step 1: Generate naive baseline cache (letta) ==="
score_run "be0003e5447b" "letta"

CACHE_SRC="$OUTDIR/be0003e5447b/scores/naive_baseline_cache.json"
if [ ! -f "$CACHE_SRC" ]; then
  echo "ERROR: Naive baseline cache not generated at $CACHE_SRC"
  exit 1
fi
echo "Cache generated: $(wc -c < "$CACHE_SRC") bytes"

# Step 2: Copy cache to all other runs
echo ""
echo "=== Step 2: Distributing naive baseline cache ==="
for run_id in "${RUNS[@]}"; do
  if [ "$run_id" = "be0003e5447b" ]; then continue; fi
  mkdir -p "$OUTDIR/$run_id/scores"
  cp "$CACHE_SRC" "$OUTDIR/$run_id/scores/naive_baseline_cache.json"
done
echo "Cache distributed to ${#RUNS[@]} runs"

# Step 3: Score remaining runs in parallel (batches of 3)
echo ""
echo "=== Step 3: Scoring remaining runs (3 parallel) ==="
BATCH_SIZE=3
remaining=("${RUNS[@]:1}")  # Skip first (already scored)

for ((i=0; i<${#remaining[@]}; i+=BATCH_SIZE)); do
  batch=("${remaining[@]:i:BATCH_SIZE}")
  echo ""
  echo "--- Batch starting at index $i (${#batch[@]} runs) ---"
  pids=()
  for run_id in "${batch[@]}"; do
    adapter=$(python3 -c "import json; print(json.load(open('$OUTDIR/$run_id/run_manifest.json'))['adapter'])")
    score_run "$run_id" "$adapter" &
    pids+=($!)
  done
  # Wait for batch to complete
  for pid in "${pids[@]}"; do
    wait "$pid" || echo "WARNING: Process $pid failed"
  done
done

echo ""
echo "=== All scoring complete ==="
echo ""

# Print summary
echo "| Run ID | Adapter | Composite | NBA |"
echo "|--------|---------|-----------|-----|"
for run_id in "${RUNS[@]}"; do
  sc="$OUTDIR/$run_id/scores/scorecard.json"
  if [ -f "$sc" ]; then
    python3 -c "
import json
d = json.load(open('$sc'))
adapter = json.load(open('$OUTDIR/$run_id/run_manifest.json'))['adapter']
composite = d['composite_score']
nba_m = [m for m in d['metrics'] if m['name'] == 'naive_baseline_advantage']
nba = nba_m[0]['value'] if nba_m and not nba_m[0].get('details',{}).get('not_configured') else 'N/A'
nba_str = f'{nba:.4f}' if isinstance(nba, float) else nba
print(f'| {\"$run_id\"} | {adapter} | {composite:.4f} | {nba_str} |')
" 2>/dev/null || echo "| $run_id | ? | error | error |"
  fi
done

echo ""
echo "Done! $(date)"

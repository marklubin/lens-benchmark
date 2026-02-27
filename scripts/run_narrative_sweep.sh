#!/usr/bin/env bash
# Mass parallel benchmark sweep for narrative scopes 07, 08, 09
# 7 adapters × 3 scopes × 3 reps = 63 runs total
# Rep 1 already done for 5 adapters (sqlite-chunked-hybrid, compaction, mem0-raw, null, cognee)
# This run: letta rep1 + letta-sleepy rep1 + reps 2-3 for all 7
set -euo pipefail

export LENS_LLM_API_KEY=dummy
export LENS_LLM_API_BASE='https://synix--lens-llm-llm-serve.modal.run/v1'
export LENS_LLM_MODEL='casperhansen/Meta-Llama-3.3-70B-Instruct-AWQ-INT4'
export LENS_EMBED_BASE_URL='http://localhost:7878/v1'
export LENS_EMBED_API_KEY=dummy
export LENS_EMBED_MODEL='Alibaba-NLP/gte-modernbert-base'
export ENABLE_BACKEND_ACCESS_CONTROL=false
export LETTA_BASE_URL='http://localhost:8283'

LOGDIR="/tmp/lens_sweep_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOGDIR"
echo "Logs: $LOGDIR"

JUDGE_MODEL='casperhansen/Meta-Llama-3.3-70B-Instruct-AWQ-INT4'
MAX_PARALLEL=${1:-3}

run_one() {
  local config="$1"
  local rep="$2"
  local label
  label="$(basename "$config" .json)_r${rep}"
  local logfile="$LOGDIR/${label}.log"

  echo "[$(date +%H:%M:%S)] START $label"
  if uv run lens run --config "$config" -v > "$logfile" 2>&1; then
    local run_id
    run_id=$(grep -oP 'output/\K[a-f0-9]+' "$logfile" | tail -1)
    if [ -n "$run_id" ]; then
      echo "[$(date +%H:%M:%S)] SCORE $label -> $run_id"
      uv run lens score --run "output/$run_id" \
        --judge-model "$JUDGE_MODEL" \
        --no-baseline --no-gate >> "$logfile" 2>&1
      local score
      score=$(grep -oP 'Composite score: \K[0-9.]+' "$logfile" | tail -1)
      echo "[$(date +%H:%M:%S)] DONE  $label = $score (run=$run_id)"
      echo "$label,$run_id,$score" >> "$LOGDIR/results.csv"
    else
      echo "[$(date +%H:%M:%S)] FAIL  $label (no run_id)"
      echo "$label,FAIL_NO_ID,0" >> "$LOGDIR/results.csv"
    fi
  else
    local err
    err=$(tail -1 "$logfile" | head -c 120)
    echo "[$(date +%H:%M:%S)] FAIL  $label: $err"
    echo "$label,CRASH,0" >> "$LOGDIR/results.csv"
  fi
}

echo "config,run_id,composite" > "$LOGDIR/results.csv"

# ========================================================
# Phase 1: Letta + Letta-sleepy rep 1 (sequential, shared container)
# ========================================================
echo "=== Phase 1: letta variants rep 1 (sequential) ==="
for scope in 07 08 09; do
  for adapter in letta letta_sleepy; do
    config="configs/${adapter}_scope${scope}d_8k.json"
    [ -f "$config" ] && run_one "$config" 1
  done
done

# ========================================================
# Phase 2: Reps 2-3 for parallel adapters
# ========================================================
echo "=== Phase 2: reps 2-3 parallel adapters (max $MAX_PARALLEL concurrent) ==="
for rep in 2 3; do
  for scope in 07 08 09; do
    for adapter in sqlite_chunked_hybrid compaction mem0_raw null; do
      config="configs/${adapter}_scope${scope}d_8k.json"
      if [ -f "$config" ]; then
        run_one "$config" "$rep" &
        while [ "$(jobs -r | wc -l)" -ge "$MAX_PARALLEL" ]; do
          sleep 10
        done
      fi
    done
  done
done
wait
echo "=== Phase 2 complete ==="

# ========================================================
# Phase 3: Reps 2-3 for cognee (sequential, kuzu lock)
# ========================================================
echo "=== Phase 3: cognee reps 2-3 (sequential) ==="
for rep in 2 3; do
  for scope in 07 08 09; do
    config="configs/cognee_scope${scope}d_8k.json"
    if [ -f "$config" ]; then
      rm -f /home/mark/lens-benchmark/.venv/lib/python3.12/site-packages/cognee/.cognee_system/databases/cognee_graph_kuzu/lock 2>/dev/null
      run_one "$config" "$rep"
    fi
  done
done
echo "=== Phase 3 complete ==="

# ========================================================
# Phase 4: Letta + Letta-sleepy reps 2-3 (sequential)
# ========================================================
echo "=== Phase 4: letta variants reps 2-3 (sequential) ==="
for rep in 2 3; do
  for scope in 07 08 09; do
    for adapter in letta letta_sleepy; do
      config="configs/${adapter}_scope${scope}d_8k.json"
      [ -f "$config" ] && run_one "$config" "$rep"
    done
  done
done
echo "=== Phase 4 complete ==="

echo ""
echo "========================================="
echo "ALL DONE — Results:"
echo "========================================="
column -t -s',' "$LOGDIR/results.csv"
echo ""
echo "CSV: $LOGDIR/results.csv"

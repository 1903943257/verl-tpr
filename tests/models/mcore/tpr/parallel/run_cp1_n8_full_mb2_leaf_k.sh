#!/usr/bin/env bash
# N8 true-captured short PPO, controlled Full physical MB2 / TPR K1/2/4.
# Independent second phase reports stage NPU events (Root Push/Leaf/Root Pop).
set -euo pipefail
cd "$(dirname "$0")/../../../../.."

: "${TPR_CP1_N8_ROOT:?Set a NEW experiment root (no overwrites)}"
ROOT="$TPR_CP1_N8_ROOT"
if [[ -e "$ROOT" ]]; then
  echo "Refusing existing directory: $ROOT" >&2
  exit 2
fi
mkdir -p "$ROOT"
_CURRENT_LOG=""
finish() {
  local rc=$?
  trap - EXIT
  set +e
  echo
  echo "========= N8 FINAL RESULTS OR FAILURE exit=$rc ========="
  python -m tests.models.mcore.tpr.parallel.summarize_cp1_n8_fair \
    "$ROOT" --memory-margin-gib "${TPR_CP1_N8_MEMORY_MARGIN_GIB:-0.5}" || true
  if ((rc != 0)) && [[ -n "$_CURRENT_LOG" && -f "$_CURRENT_LOG" ]]; then
    echo "--- Failed stage: $_CURRENT_LOG ---"
    grep -E 'NPU out of memory|Traceback|^E[[:space:]]|FAILED|ERROR|TPR_CP1_N8_' \
      "$_CURRENT_LOG" | tail -n 65 || true
    echo "--- Current log tail ---"
    tail -n 60 "$_CURRENT_LOG" || true
  fi
  echo "========= N8 BENCH END exit=$rc ========="
  exit "$rc"
}
trap finish EXIT

SOURCE="${TPR_CP1_N8_SOURCE:-/workspace/tpr_actor_capture_4b_n8_run1/actor_update_rank0_batch0.pt}"
export TPR_QWEN_4B_PATH="${TPR_QWEN_4B_PATH:-/workspace/hf_models/Qwen3-4B-128K}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-8}"
[[ "$ASCEND_RT_VISIBLE_DEVICES" == 8 ]] || {
  echo "This controlled CP1 run reserves physical NPU 8" >&2; exit 2;
}
[[ -f "$SOURCE" ]] || { echo "Missing original Actor source: $SOURCE" >&2; exit 2; }
[[ -d "$TPR_QWEN_4B_PATH" ]] || { echo "Missing real Qwen3-4B model" >&2; exit 2; }
export HCCL_DETERMINISTIC=true
export TPR_OFFLINE_GRAD_SYNC_DTYPE=bf16
export PYTHONUNBUFFERED=1
export TPR_RUN_CP1_N8_FAIR=1
export TPR_TREE_POLICY=qwen_turn
export TPR_TURN_MIN_SEGMENT_TOKENS=128
export TPR_PPO_LOGPROB_CHUNK_TOKENS=256
export TPR_ROOT_NATIVE_RING=0
export TPR_RING_COALESCE_PREFIX_FULL=0
export TPR_RING_COALESCE_PREFIX_QUERY=0
export TPR_RING_COALESCE_PREFIX_PADDED=0
export TPR_LEAF_BATCH_TRACE=1
export TPR_LEAF_BATCH_MAX_TOKENS="${TPR_CP1_N8_LEAF_TOKEN_BUDGET:-8192}"
unset TPR_CP1_TRACE TPR_CP1_TRACE_DIR TPR_MEM_TRACE TPR_FIRST_LEAF_DIAG TPR_FIRST_LEAF_DIAG_SYNC || true
export TPR_CP1_N8_CAPTURE="$ROOT/n8_real_actor_short.pt"

echo "===== Derive N=8 unique original trajectories (no synthetic replication) ====="
_CURRENT_LOG="$ROOT/capture.log"
rows_args=()
if [[ -n "${TPR_CP1_N8_ROWS:-}" ]]; then
  rows_args+=(--rows "$TPR_CP1_N8_ROWS")
fi
python -m tests.models.mcore.tpr.parallel.prepare_tpr_cp1_n8_capture \
  "$SOURCE" "$TPR_CP1_N8_CAPTURE" \
  --cap "${TPR_CP1_N8_CAP:-2048}" \
  --prompt-window "${TPR_CP1_N8_PROMPT_WINDOW:-1024}" \
  "${rows_args[@]}" 2>&1 | tee "$_CURRENT_LOG"

if command -v npu-smi >/dev/null 2>&1; then
  echo "===== Check physical NPU occupancy before model loading ====="
  npu-smi info || true
fi

export TPR_CP1_N8_WARMUPS="${TPR_CP1_N8_WARMUPS:-1}"
export TPR_CP1_N8_REPEATS="${TPR_CP1_N8_REPEATS:-3}"
for arm in full_mb2 tpr_k1 tpr_k2 tpr_k4; do
  mkdir -p "$ROOT/$arm"
  case "$arm" in
    full_mb2) export TPR_CP1_N8_MODE=full_mb2 TPR_LEAF_BATCH_SIZE=1;;
    tpr_k1) export TPR_CP1_N8_MODE=tpr TPR_LEAF_BATCH_SIZE=1;;
    tpr_k2) export TPR_CP1_N8_MODE=tpr TPR_LEAF_BATCH_SIZE=2;;
    tpr_k4) export TPR_CP1_N8_MODE=tpr TPR_LEAF_BATCH_SIZE=4;;
  esac
  export TPR_CP1_N8_OUTPUT="$ROOT/$arm/result.json"
  export TPR_EXEC_PROFILE=0 TPR_CP1_N8_STAGE_PROFILE=0
  _CURRENT_LOG="$ROOT/$arm/run.log"
  echo "===== FAIR N8 F/B START arm=$arm K=$TPR_LEAF_BATCH_SIZE ====="
  torchrun --master_addr=127.0.0.1 --master_port=29818 \
    --nproc_per_node=1 -m pytest -q -s -x \
    tests/models/mcore/tpr/parallel/test_real_actor_cp1_n8_fair_npu.py \
    2>&1 | tee "$_CURRENT_LOG"
done

# Stage-event timings are run in dedicated fresh processes. Their cost must
# NOT be counted in the Full/TPR fair performance medians above.
if [[ "${TPR_CP1_N8_RUN_STAGES:-1}" == "1" ]]; then
  export TPR_CP1_N8_WARMUPS=0 TPR_CP1_N8_REPEATS=1
  export TPR_CP1_N8_MODE=tpr
  export TPR_EXEC_PROFILE=1 TPR_CP1_N8_STAGE_PROFILE=1
  for k in 2 4; do
    mkdir -p "$ROOT/stage_tpr_k$k"
    export TPR_LEAF_BATCH_SIZE="$k"
    export TPR_CP1_N8_OUTPUT="$ROOT/stage_tpr_k$k/result.json"
    _CURRENT_LOG="$ROOT/stage_tpr_k$k/run.log"
    echo "===== STAGE NPU EVENTS K=$k (separate from fair F/B timing) ====="
    torchrun --master_addr=127.0.0.1 --master_port=29818 \
      --nproc_per_node=1 -m pytest -q -s -x \
      tests/models/mcore/tpr/parallel/test_real_actor_cp1_n8_fair_npu.py \
      2>&1 | tee "$_CURRENT_LOG"
  done
fi

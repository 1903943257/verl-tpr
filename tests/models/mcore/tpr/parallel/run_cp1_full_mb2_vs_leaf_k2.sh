#!/usr/bin/env bash
# Four-arm two-row controlled Qwen3-4B PPO comparison on CP=TP=DP=1.
# "Full MB2" here is physical [2,S] square-CANN batch, NOT native VERL
# prepare_micro_batches scheduling. It is an isolated model F/B benchmark.
set -euo pipefail
cd "$(dirname "$0")/../../../../.."

: "${TPR_CP1_CAPTURE:?Set TPR_CP1_CAPTURE to the derived 2-row actor_pair capture}"
: "${TPR_CP1_FULL_K2_ROOT:?Set a NEW TPR_CP1_FULL_K2_ROOT for this 4-way run}"
ROOT="$TPR_CP1_FULL_K2_ROOT"
if [[ -e "$ROOT" ]]; then
  echo "Refusing existing directory $ROOT; choose a fresh experiment root" >&2
  exit 2
fi
mkdir -p "$ROOT"
_CURRENT_LOG=""
finish() {
  local rc=$?
  trap - EXIT
  set +e
  echo
  echo "======== FINAL ACTOR FULL-MB2 / TPR-K2 SUMMARY (exit=$rc) ========"
  python -m tests.models.mcore.tpr.parallel.summarize_cp1_full_mb2_leaf_k2 \
    "$ROOT" --memory-margin-gib "${TPR_CP1_MEMORY_MARGIN_GIB:-0.5}" || true
  if ((rc != 0)) && [[ -f "$_CURRENT_LOG" ]]; then
    echo "--- CURRENT failed arm: $_CURRENT_LOG ---"
    grep -E 'NPU out of memory|OutOfMemoryError|Traceback|FAILED|ERROR|^E[[:space:]]' \
      "$_CURRENT_LOG" | tail -n 30 || true
    tail -n 50 "$_CURRENT_LOG" || true
  fi
  echo "======== FINAL ACTOR FULL-MB2 / TPR-K2 END exit=$rc ========"
  exit "$rc"
}
trap finish EXIT

export TPR_QWEN_4B_PATH="${TPR_QWEN_4B_PATH:-/workspace/hf_models/Qwen3-4B-128K}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-8}"
if [[ "$ASCEND_RT_VISIBLE_DEVICES" != "8" ]]; then
  echo "Existing CP1 replay reserves physical NPU 8; got $ASCEND_RT_VISIBLE_DEVICES" >&2
  exit 2
fi
[[ -f "$TPR_CP1_CAPTURE" ]] || { echo "Missing capture $TPR_CP1_CAPTURE" >&2; exit 2; }
[[ -d "$TPR_QWEN_4B_PATH" ]] || { echo "Missing Qwen3-4B weights: $TPR_QWEN_4B_PATH" >&2; exit 2; }
export TPR_RUN_CP1_SHORT_ACTOR=1
export TPR_TREE_POLICY=qwen_turn
export TPR_TURN_MIN_SEGMENT_TOKENS=128
export TPR_PPO_LOGPROB_CHUNK_TOKENS=256
export TPR_ROOT_NATIVE_RING=0
export TPR_RING_COALESCE_PREFIX_FULL=0
export TPR_RING_COALESCE_PREFIX_QUERY=0
export TPR_RING_COALESCE_PREFIX_PADDED=0
export TPR_EXEC_PROFILE=0
export TPR_LEAF_BATCH_TRACE=1
export TPR_LEAF_BATCH_MAX_TOKENS="${TPR_LEAF_BATCH_MAX_TOKENS:-8192}"
export TPR_OFFLINE_GRAD_SYNC_DTYPE=bf16
export HCCL_DETERMINISTIC=true
export PYTHONUNBUFFERED=1
export TPR_CP1_BENCH_WARMUPS="${TPR_CP1_BENCH_WARMUPS:-1}"
export TPR_CP1_BENCH_REPEATS="${TPR_CP1_BENCH_REPEATS:-3}"
# Never turn on K1/K2 layer trace / huge GEMM replay for timing work.
unset TPR_CP1_TRACE TPR_CP1_TRACE_DIR TPR_MEM_TRACE TPR_FIRST_LEAF_DIAG TPR_FIRST_LEAF_DIAG_SYNC || true

if command -v npu-smi >/dev/null 2>&1; then
  echo "====== PHYSICAL NPU occupancy before benchmark ======"
  npu-smi info || true
fi

for arm in full_mb1 full_mb2 tpr_k1 tpr_k2; do
  case "$arm" in
    full_mb1)
      export TPR_CP1_MODE=full
      export TPR_LEAF_BATCH_SIZE=1
      ;;
    full_mb2)
      export TPR_CP1_MODE=full_batch2
      export TPR_LEAF_BATCH_SIZE=1
      ;;
    tpr_k1)
      export TPR_CP1_MODE=shared
      export TPR_LEAF_BATCH_SIZE=1
      ;;
    tpr_k2)
      export TPR_CP1_MODE=shared
      export TPR_LEAF_BATCH_SIZE=2
      ;;
  esac
  mkdir -p "$ROOT/$arm"
  export TPR_CP1_OUTPUT="$ROOT/$arm/result.json"
  _CURRENT_LOG="$ROOT/$arm/run.log"
  echo "TPR_CP1_4WAY_START arm=$arm full_mode=$TPR_CP1_MODE leaf_k=$TPR_LEAF_BATCH_SIZE"
  torchrun --master_addr=127.0.0.1 --master_port=29797 \
    --nproc_per_node=1 -m pytest -q -s -x \
    tests/models/mcore/tpr/parallel/test_real_actor_cp1_short_full_tpr_npu.py \
    2>&1 | tee "$_CURRENT_LOG"
done

#!/usr/bin/env bash
# First-round delayed dKV precision audit: Full / TPR-old / TPR-FP32.
# Exactly ONE NPU and TP=CP=DP=1; no optimizer step or second-round tests.
set -Eeuo pipefail
cd "$(dirname "$0")/../../../../.."

: "${TPR_CP1_CAPTURE:?Set TPR_CP1_CAPTURE to the existing validated, cropped two-row Actor capture}"
: "${TPR_CP1_DKV_OUTPUT_ROOT:?Set a NEW TPR_CP1_DKV_OUTPUT_ROOT directory}"
: "${ASCEND_RT_VISIBLE_DEVICES:?Set one available physical NPU ID}"
[[ "$ASCEND_RT_VISIBLE_DEVICES" != *,* ]] || {
  echo "ERROR: this experiment needs exactly one physical NPU" >&2
  exit 2
}
ROOT="$TPR_CP1_DKV_OUTPUT_ROOT"
if [[ -e "$ROOT" ]]; then
  echo "ERROR: output already exists: $ROOT" >&2
  exit 2
fi
[[ -f "$TPR_CP1_CAPTURE" ]] || { echo "ERROR: missing capture: $TPR_CP1_CAPTURE" >&2; exit 2; }
export TPR_QWEN_4B_PATH="${TPR_QWEN_4B_PATH:-/workspace/hf_models/Qwen3-4B-128K}"
[[ -d "$TPR_QWEN_4B_PATH" ]] || { echo "ERROR: missing model: $TPR_QWEN_4B_PATH" >&2; exit 2; }

TEST="tests/models/mcore/tpr/parallel/test_real_actor_cp1_short_full_tpr_npu.py"
COMPARE="tests.models.mcore.tpr.parallel.compare_cp1_delayed_dkv_precision"
[[ -f "$TEST" && -f "tests/models/mcore/tpr/parallel/compare_cp1_delayed_dkv_precision.py" ]] || {
  echo "ERROR: branch is missing first-round precision tests" >&2
  exit 2
}
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export HCCL_DETERMINISTIC=true
export TPR_RUN_CP1_SHORT_ACTOR=1
export TPR_CP1_SAVE_FULL_GRADS=1
export TPR_CP1_BENCH_WARMUPS=0
export TPR_CP1_BENCH_REPEATS=1
export TPR_OFFLINE_GRAD_SYNC_DTYPE=bf16
export TPR_PPO_LOGPROB_CHUNK_TOKENS=256
export TPR_TREE_POLICY=qwen_turn
export TPR_LEAF_BATCH_SIZE=1
export TPR_EXEC_PROFILE=0
unset TPR_CP1_TRACE TPR_CP1_TRACE_DIR TPR_MEM_TRACE TPR_FIRST_LEAF_DIAG TPR_FIRST_LEAF_DIAG_SYNC || true

mkdir -p "$ROOT"
echo "TPR_DKV_FP32_ROUND1 capture=$TPR_CP1_CAPTURE model=$TPR_QWEN_4B_PATH"
echo "TPR_DKV_FP32_ROUND1 topology=TP1CP1DP1 physical_npu=$ASCEND_RT_VISIBLE_DEVICES output=$ROOT"
echo "TPR_DKV_FP32_ROUND1 full_reference=controlled_square_CANN_not_native_VERL"
echo "TPR_DKV_FP32_ROUND1 full_grad_dump=all_BF16_parameters"
if command -v git >/dev/null 2>&1; then
  echo "TPR_DKV_FP32_ROUND1 source_head=$(git rev-parse HEAD 2>/dev/null || echo unavailable)"
fi

for arm in full tpr_old tpr_fp32; do
  mkdir -p "$ROOT/$arm"
  case "$arm" in
    full) export TPR_CP1_MODE=full TPR_EXPERIMENT_DELAYED_DKV_FP32=0 ;;
    tpr_old) export TPR_CP1_MODE=shared TPR_EXPERIMENT_DELAYED_DKV_FP32=0 ;;
    tpr_fp32) export TPR_CP1_MODE=shared TPR_EXPERIMENT_DELAYED_DKV_FP32=1 ;;
  esac
  export TPR_CP1_OUTPUT="$ROOT/$arm/result.json"
  echo "===== ROUND1 $arm mode=$TPR_CP1_MODE delayed_dkv_fp32=$TPR_EXPERIMENT_DELAYED_DKV_FP32 ====="
  torchrun --master_addr=127.0.0.1 --master_port="${TPR_TEST_MASTER_PORT:-29867}" \
    --nproc_per_node=1 -m tests.models.mcore.tpr._mindspeed_first_pytest \
    -q -s -x "$TEST" \
    2>&1 | tee "$ROOT/$arm/run.log"
done

echo "===== THREE-ARM FULL-GRADIENT REPORT ====="
python -m "$COMPARE" "$ROOT" 2>&1 | tee "$ROOT/compare.log"
echo "TPR_DKV_FP32_ROUND1_DONE output=$ROOT/comparison_delayed_dkv_fp32.json"

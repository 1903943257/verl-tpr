#!/usr/bin/env bash
# Round 1: eight real PPO trajectories, Full / TPR-old / TPR-FP32.
# Exactly ONE NPU, TP=CP=DP=1. No optimizer update, no Ring, no recompute.
set -Eeuo pipefail
cd "$(dirname "$0")/../../../../.."

: "${TPR_CP1_DKV_OUTPUT_ROOT:?Set a NEW experiment output directory}"
: "${ASCEND_RT_VISIBLE_DEVICES:?Select exactly one available physical NPU ID}"
[[ "$ASCEND_RT_VISIBLE_DEVICES" != *,* ]] || {
  echo "ERROR: this experiment requires exactly one physical NPU" >&2
  exit 2
}
ROOT="$TPR_CP1_DKV_OUTPUT_ROOT"
if [[ -e "$ROOT" ]]; then
  echo "ERROR: output already exists: $ROOT" >&2
  exit 2
fi
export TPR_QWEN_4B_PATH="${TPR_QWEN_4B_PATH:-/workspace/hf_models/Qwen3-4B-128K}"
[[ -d "$TPR_QWEN_4B_PATH" ]] || { echo "ERROR: missing Qwen3-4B checkpoint: $TPR_QWEN_4B_PATH" >&2; exit 2; }

TEST="tests/models/mcore/tpr/parallel/test_real_actor_cp1_n8_fair_npu.py"
COMPARE="tests.models.mcore.tpr.parallel.compare_cp1_delayed_dkv_precision"
[[ -f "$TEST" && -f "tests/models/mcore/tpr/parallel/compare_cp1_delayed_dkv_precision.py" ]] || {
  echo "ERROR: branch is missing the N8 delayed dKV experiment files" >&2
  exit 2
}
mkdir -p "$ROOT"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export HCCL_DETERMINISTIC=true
export TPR_RUN_CP1_N8_FAIR=1
export TPR_CP1_N8_SAVE_FULL_GRADS=1
export TPR_CP1_N8_WARMUPS=0
export TPR_CP1_N8_REPEATS=1
export TPR_CP1_N8_STAGE_PROFILE=0
export TPR_EXEC_PROFILE=0
export TPR_OFFLINE_GRAD_SYNC_DTYPE=bf16
export TPR_TREE_POLICY=qwen_turn
export TPR_PPO_LOGPROB_CHUNK_TOKENS=256
export TPR_LEAF_BATCH_SIZE=1
export TPR_ROOT_NATIVE_RING=0
unset TPR_CP1_TRACE TPR_CP1_TRACE_DIR TPR_MEM_TRACE TPR_FIRST_LEAF_DIAG TPR_FIRST_LEAF_DIAG_SYNC || true

if [[ -n "${TPR_CP1_N8_CAPTURE:-}" ]]; then
  [[ -f "$TPR_CP1_N8_CAPTURE" ]] || { echo "ERROR: missing preprocessed N8 capture: $TPR_CP1_N8_CAPTURE" >&2; exit 2; }
else
  SOURCE="${TPR_CP1_N8_SOURCE:-/workspace/tpr_actor_capture_4b_n8_run1/actor_update_rank0_batch0.pt}"
  [[ -f "$SOURCE" ]] || { echo "ERROR: missing N8 Actor source: $SOURCE (set TPR_CP1_N8_SOURCE)" >&2; exit 2; }
  mkdir -p "$ROOT/data"
  export TPR_CP1_N8_CAPTURE="$ROOT/data/n8_short.pt"
  echo "===== PREPARE N8 REAL PPO CAPTURE ====="
  python -m tests.models.mcore.tpr.parallel.prepare_tpr_cp1_n8_capture \
    "$SOURCE" "$TPR_CP1_N8_CAPTURE" \
    --cap "${TPR_CP1_N8_CAP:-2048}" \
    --prompt-window "${TPR_CP1_N8_PROMPT_WINDOW:-1024}" \
    2>&1 | tee "$ROOT/prepare.log"
fi

echo "TPR_DKV_FP32_ROUND1 capture=$TPR_CP1_N8_CAPTURE model=$TPR_QWEN_4B_PATH"
echo "TPR_DKV_FP32_ROUND1 topology=TP1CP1DP1 logical_trajectories=8 physical_npu=$ASCEND_RT_VISIBLE_DEVICES"
echo "TPR_DKV_FP32_ROUND1 full_reference=controlled_B1_square_CANN_not_native_VERL"
echo "TPR_DKV_FP32_ROUND1 complete_BF16_gradient_dumps=enabled (disk requirement ~24 GB)"
if command -v git >/dev/null 2>&1; then
  echo "TPR_DKV_FP32_ROUND1 source_head=$(git rev-parse HEAD 2>/dev/null || echo unavailable)"
fi

for arm in full tpr_old tpr_fp32; do
  mkdir -p "$ROOT/$arm"
  case "$arm" in
    full) export TPR_CP1_N8_MODE=full_mb1 TPR_EXPERIMENT_DELAYED_DKV_FP32=0 ;;
    tpr_old) export TPR_CP1_N8_MODE=tpr TPR_EXPERIMENT_DELAYED_DKV_FP32=0 ;;
    tpr_fp32) export TPR_CP1_N8_MODE=tpr TPR_EXPERIMENT_DELAYED_DKV_FP32=1 ;;
  esac
  export TPR_CP1_N8_OUTPUT="$ROOT/$arm/result.json"
  echo "===== ROUND1 arm=$arm mode=$TPR_CP1_N8_MODE delayed_dkv_fp32=$TPR_EXPERIMENT_DELAYED_DKV_FP32 ====="
  torchrun --master_addr=127.0.0.1 --master_port="${TPR_TEST_MASTER_PORT:-29867}" \
    --nproc_per_node=1 -m tests.models.mcore.tpr._mindspeed_first_pytest \
    -q -s -x "$TEST" 2>&1 | tee "$ROOT/$arm/run.log"
done

echo "===== ROUND1 ALL-PARAMETER GRADIENT COMPARISON ====="
python -m "$COMPARE" "$ROOT" 2>&1 | tee "$ROOT/compare.log"
echo "TPR_DKV_FP32_ROUND1_DONE output=$ROOT/comparison_delayed_dkv_fp32.json"

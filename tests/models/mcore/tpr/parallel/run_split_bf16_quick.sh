#!/usr/bin/env bash
# BF16 P32/S32 Full/connected/Replay, aggregate metrics only.
set -Eeuo pipefail
cd "$(dirname "$0")/../../../../.."
: "${TPR_SPLIT_BF16_OUTPUT:?Set a NEW output directory}"
: "${TPR_QWEN_4B_PATH:?Set the same real Qwen checkpoint as the FP32 run}"
: "${ASCEND_RT_VISIBLE_DEVICES:?Select one available physical NPU}"
[[ "$ASCEND_RT_VISIBLE_DEVICES" != *,* ]] || { echo 'Expected one NPU' >&2; exit 2; }
[[ -d "$TPR_QWEN_4B_PATH" ]] || { echo 'Checkpoint directory not found' >&2; exit 2; }
[[ ! -e "$TPR_SPLIT_BF16_OUTPUT" && ! -e "${TPR_SPLIT_BF16_OUTPUT}.log" ]] || {
  echo 'Refusing to overwrite experiment output' >&2; exit 2;
}
mkdir -p "$(dirname "$TPR_SPLIT_BF16_OUTPUT")"
export TPR_SPLIT_BF16_OUTPUT="$(cd "$(dirname "$TPR_SPLIT_BF16_OUTPUT")" && pwd -P)/$(basename "$TPR_SPLIT_BF16_OUTPUT")"
export TPR_RUN_SPLIT_BF16_QUICK=1
export TPR_SPLIT_BF16_PREFIX="${TPR_SPLIT_BF16_PREFIX:-${TPR_SPLIT_FP32_PREFIX:-32}}"
export TPR_SPLIT_BF16_SUFFIX="${TPR_SPLIT_BF16_SUFFIX:-${TPR_SPLIT_FP32_SUFFIX:-32}}"
export TPR_SPLIT_BF16_PROFILE="${TPR_SPLIT_BF16_PROFILE:-${TPR_SPLIT_FP32_PROFILE:-4b}}"
export TPR_EXPERIMENT_DELAYED_DKV_FP32="${TPR_EXPERIMENT_DELAYED_DKV_FP32:-1}"
[[ "$TPR_EXPERIMENT_DELAYED_DKV_FP32" =~ ^[01]$ ]] || { echo 'dKV gate must be 0 or 1' >&2; exit 2; }
export TPR_ROOT_NATIVE_RING=0 TPR_LEAF_BATCH_SIZE=1 TPR_EXEC_PROFILE=0
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 HCCL_DETERMINISTIC=true
unset TPR_DKV_REPLAY_DIAG_PATH TPR_MEM_TRACE TPR_FIRST_LEAF_DIAG TPR_FIRST_LEAF_DIAG_SYNC || true
export MASTER_ADDR=127.0.0.1
export MASTER_PORT="${TPR_SPLIT_BF16_MASTER_PORT:-29683}"
if ! [[ "$MASTER_PORT" =~ ^[0-9]+$ ]] || (( MASTER_PORT < 1 || MASTER_PORT > 65535 )); then
  echo 'Invalid TPR_SPLIT_BF16_MASTER_PORT' >&2; exit 2;
fi
echo "BF16 CANN, P=$TPR_SPLIT_BF16_PREFIX S=$TPR_SPLIT_BF16_SUFFIX, delayed_dKV_FP32=$TPR_EXPERIMENT_DELAYED_DKV_FP32"
echo 'Summary only; no tensor dumps. Two CPU BF16 gradient snapshots need about 16 GB RAM for Qwen3-4B.'
"${PYTHON_BIN:-python}" -m torch.distributed.run \
  --nnodes=1 --node_rank=0 --nproc_per_node=1 \
  --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" \
  -m tests.models.mcore.tpr._mindspeed_first_pytest -q -s -x \
  tests/models/mcore/tpr/parallel/test_real_actor_split_bf16_quick_npu.py \
  2>&1 | tee "${TPR_SPLIT_BF16_OUTPUT}.log"
[[ -s "${TPR_SPLIT_BF16_OUTPUT}/report.json" ]] || {
  echo 'BF16 experiment did not produce report.json (check for a skipped test)' >&2; exit 1;
}

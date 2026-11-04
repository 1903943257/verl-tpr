#!/usr/bin/env bash
# Single synthetic trajectory; real FP32 Qwen; no CANN fusion attention or PPO.
set -Eeuo pipefail
cd "$(dirname "$0")/../../../../.."
: "${TPR_SPLIT_FP32_OUTPUT:?Set a NEW output directory}"
: "${TPR_QWEN_4B_PATH:?Set the real Qwen3 checkpoint directory}"
: "${ASCEND_RT_VISIBLE_DEVICES:?Select one available physical NPU}"
[[ "$ASCEND_RT_VISIBLE_DEVICES" != *,* ]] || { echo 'Expected exactly one physical NPU' >&2; exit 2; }
[[ -d "$TPR_QWEN_4B_PATH" ]] || { echo 'Checkpoint directory not found' >&2; exit 2; }
[[ ! -e "$TPR_SPLIT_FP32_OUTPUT" && ! -e "${TPR_SPLIT_FP32_OUTPUT}.log" ]] || {
  echo 'Refusing to overwrite experiment output' >&2; exit 2;
}
mkdir -p "$(dirname "$TPR_SPLIT_FP32_OUTPUT")"
export TPR_SPLIT_FP32_OUTPUT="$(cd "$(dirname "$TPR_SPLIT_FP32_OUTPUT")" && pwd -P)/$(basename "$TPR_SPLIT_FP32_OUTPUT")"
export TPR_RUN_SPLIT_FP32_ATTRIBUTION=1
export TPR_SPLIT_FP32_PREFIX="${TPR_SPLIT_FP32_PREFIX:-32}"
export TPR_SPLIT_FP32_SUFFIX="${TPR_SPLIT_FP32_SUFFIX:-32}"
export TPR_SPLIT_FP32_PROFILE="${TPR_SPLIT_FP32_PROFILE:-4b}"
export TPR_ROOT_NATIVE_RING=0 TPR_LEAF_BATCH_SIZE=1 TPR_EXEC_PROFILE=0
export TPR_EXPERIMENT_DELAYED_DKV_FP32=1
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 HCCL_DETERMINISTIC=true
unset TPR_DKV_REPLAY_DIAG_PATH TPR_MEM_TRACE TPR_FIRST_LEAF_DIAG TPR_FIRST_LEAF_DIAG_SYNC || true
# Use an explicit IPv4 loopback rendezvous instead of --standalone. This avoids
# hostname/IPv6 resolution in containers and makes the single-rank endpoint clear.
export MASTER_ADDR=127.0.0.1
export MASTER_PORT="${TPR_SPLIT_FP32_MASTER_PORT:-29681}"
if ! [[ "$MASTER_PORT" =~ ^[0-9]+$ ]] || (( MASTER_PORT < 1 || MASTER_PORT > 65535 )); then
  echo "ERROR: invalid TPR_SPLIT_FP32_MASTER_PORT=$MASTER_PORT" >&2
  exit 2
fi

echo 'FP32 parameters + gradients + math attention. Qwen3-4B gradient dumps require about 48 GB disk.'
echo "Synthetic sequence: P=$TPR_SPLIT_FP32_PREFIX S=$TPR_SPLIT_FP32_SUFFIX"
echo "Distributed rendezvous: MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT"
"${PYTHON_BIN:-python}" -m torch.distributed.run \
  --nnodes=1 --node_rank=0 --nproc_per_node=1 \
  --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" \
  -m tests.models.mcore.tpr._mindspeed_first_pytest -q -s -x \
  tests/models/mcore/tpr/parallel/test_real_actor_split_fp32_attribution_npu.py \
  2>&1 | tee "${TPR_SPLIT_FP32_OUTPUT}.log"

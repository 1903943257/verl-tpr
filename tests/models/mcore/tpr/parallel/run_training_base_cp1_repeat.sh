#!/usr/bin/env bash
# One independent Base CP1 repeat, reading the completed historical run only.
set -Eeuo pipefail
cd "$(dirname "$0")/../../../../.."

: "${TPR_TRAIN_MODEL_PATH:?Use the same local checkpoint as the original experiment}"
: "${TPR_TRAIN10_REPEAT_SOURCE:?Set the original completed ten-step experiment directory}"
: "${TPR_TRAIN10_OUTPUT:?Set a NEW repeat output directory outside the original experiment}"
[[ -d "$TPR_TRAIN_MODEL_PATH" ]] || { echo 'ERROR: checkpoint missing' >&2; exit 2; }
[[ ! -e "$TPR_TRAIN10_OUTPUT" ]] || { echo 'ERROR: repeat output already exists' >&2; exit 2; }
repeat_devices="${TPR_TRAIN_DEVICES:-}"
export TPR_TRAIN10_REPEAT_DEVICE="${TPR_TRAIN10_REPEAT_DEVICE:-${repeat_devices%%,*}}"
: "${TPR_TRAIN10_REPEAT_DEVICE:?Set the same physical NPU used by the original Base CP1}"
[[ "$TPR_TRAIN10_REPEAT_DEVICE" =~ ^[0-9]+$ ]] || { echo 'ERROR: repeat requires one physical NPU ID' >&2; exit 2; }
export TPR_TRAIN10_REPEAT_SOURCE
export TPR_TRAIN_MODE=base_cp1_repeat
export TPR_TREE_POLICY=token_lcp TPR_TREE_BUILDER=lcp_sort TPR_MIN_SHARED_PREFIX_TOKENS=1024
export TPR_RING_COALESCE_PREFIX_FULL=0 TPR_RING_COALESCE_PREFIX_QUERY=0 TPR_RING_COALESCE_PREFIX_PADDED=0
export TPR_PPO_LOGPROB_CHUNK_TOKENS=0 HCCL_DETERMINISTIC=true
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
unset TPR_DKV_REPLAY_DIAG_PATH TPR_MEM_TRACE TPR_EXEC_PROFILE TPR_FIRST_LEAF_DIAG TPR_FIRST_LEAF_DIAG_SYNC || true
"${PYTHON_BIN:-python}" -m tests.models.mcore.tpr.parallel.training_multistep_repeat
mkdir -p "$TPR_TRAIN10_OUTPUT/base_cp1_repeat"
export TPR_TRAIN10_OUTPUT="$(cd "$TPR_TRAIN10_OUTPUT" && pwd -P)"
export TPR_TRAIN_MODE=base_cp1_repeat ASCEND_RT_VISIBLE_DEVICES="$TPR_TRAIN10_REPEAT_DEVICE"
echo 'TRAIN10_REPEAT_SETUP Only Base CP1; ten fresh native updates; original fixture and controls reused.'
"${PYTHON_BIN:-python}" -m torch.distributed.run --nnodes=1 --node_rank=0 \
    --nproc_per_node=1 --master_addr=127.0.0.1 --master_port="${TPR_TRAIN_PORT:-29701}" \
    -m tests.models.mcore.tpr.parallel.training_multistep_npu \
    2>&1 | tee "$TPR_TRAIN10_OUTPUT/base_cp1_repeat/run.log"

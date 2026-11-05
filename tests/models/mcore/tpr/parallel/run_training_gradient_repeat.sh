#!/usr/bin/env bash
# Two independent Base CP1 first backwards on the same physical NPU.
set -Eeuo pipefail
cd "$(dirname "$0")/../../../../.."

: "${TPR_TRAIN_MODEL_PATH:?Use the original local checkpoint}"
: "${TPR_TRAIN10_SOURCE:?Set the original completed ten-step experiment directory}"
: "${TPR_GRAD_REPEAT_OUTPUT:?Set a NEW output directory outside the historical experiment}"
[[ -d "$TPR_TRAIN_MODEL_PATH" ]] || { echo 'ERROR: checkpoint missing' >&2; exit 2; }
repeat_devices="${TPR_TRAIN_DEVICES:-}"
export TPR_GRAD_REPEAT_DEVICE="${TPR_GRAD_REPEAT_DEVICE:-${repeat_devices%%,*}}"
[[ "$TPR_GRAD_REPEAT_DEVICE" =~ ^[0-9]+$ ]] || { echo 'ERROR: set one physical NPU ID' >&2; exit 2; }
export TPR_TRAIN10_SOURCE TPR_GRAD_REPEAT_OUTPUT
export ASCEND_RT_VISIBLE_DEVICES="$TPR_GRAD_REPEAT_DEVICE"
export TPR_TREE_POLICY=token_lcp TPR_TREE_BUILDER=lcp_sort TPR_MIN_SHARED_PREFIX_TOKENS=1024
export TPR_RING_COALESCE_PREFIX_FULL=0 TPR_RING_COALESCE_PREFIX_QUERY=0 TPR_RING_COALESCE_PREFIX_PADDED=0
export TPR_PPO_LOGPROB_CHUNK_TOKENS=0 HCCL_DETERMINISTIC=true
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
unset TPR_DKV_REPLAY_DIAG_PATH TPR_MEM_TRACE TPR_EXEC_PROFILE TPR_FIRST_LEAF_DIAG TPR_FIRST_LEAF_DIAG_SYNC || true
unset TPR_GRAD_REPEAT_RUN || true
"${PYTHON_BIN:-python}" -m tests.models.mcore.tpr.parallel.training_gradient_repeat_io
mkdir -p "$TPR_GRAD_REPEAT_OUTPUT/first" "$TPR_GRAD_REPEAT_OUTPUT/second"
export TPR_GRAD_REPEAT_OUTPUT="$(cd "$TPR_GRAD_REPEAT_OUTPUT" && pwd -P)"
for repeat_run in first second; do
    export TPR_GRAD_REPEAT_RUN="$repeat_run"
    "${PYTHON_BIN:-python}" -m torch.distributed.run --nnodes=1 --node_rank=0 \
        --nproc_per_node=1 --master_addr=127.0.0.1 --master_port="${TPR_TRAIN_PORT:-29705}" \
        -m tests.models.mcore.tpr.parallel.training_gradient_repeat_npu \
        2>&1 | tee "$TPR_GRAD_REPEAT_OUTPUT/$repeat_run/run.log"
done
export TPR_GRAD_REPEAT_RUN=report
"${PYTHON_BIN:-python}" -m tests.models.mcore.tpr.parallel.training_gradient_repeat_io

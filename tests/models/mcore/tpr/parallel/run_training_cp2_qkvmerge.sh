#!/usr/bin/env bash
# Add only TPR CP2 joint Query/KV merge against the saved unmerged control.
set -Eeuo pipefail
cd "$(dirname "$0")/../../../../.."
: "${TPR_TRAIN_MODEL_PATH:?Use the original local checkpoint}"
: "${TPR_TRAIN10_REPEAT_SOURCE:?Set the completed original four-route experiment}"
: "${TPR_TRAIN10_OUTPUT:?Set a NEW supplemental output directory}"
: "${TPR_TRAIN_DEVICES:?Set the original two physical NPU IDs}"
[[ -d "$TPR_TRAIN_MODEL_PATH" ]] || { echo 'ERROR: checkpoint missing' >&2; exit 2; }
[[ ! -e "$TPR_TRAIN10_OUTPUT" ]] || { echo 'ERROR: output already exists' >&2; exit 2; }
IFS=, read -r DEVICE0 DEVICE1 EXTRA <<< "$TPR_TRAIN_DEVICES"
[[ -n "$DEVICE0" && -n "$DEVICE1" && -z "${EXTRA:-}" && "$DEVICE0" != "$DEVICE1" ]] || {
    echo 'ERROR: set exactly two distinct NPU IDs' >&2; exit 2;
}
export TPR_TRAIN_MODE=tpr_cp2_qkvmerge
export TPR_TREE_POLICY=token_lcp TPR_TREE_BUILDER=lcp_sort TPR_MIN_SHARED_PREFIX_TOKENS=1024
export TPR_RING_COALESCE_PREFIX_FULL=1 TPR_RING_COALESCE_PREFIX_QUERY=1 TPR_RING_COALESCE_PREFIX_PADDED=1
export TPR_PPO_LOGPROB_CHUNK_TOKENS=0 HCCL_DETERMINISTIC=true
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
unset TPR_DKV_REPLAY_DIAG_PATH TPR_MEM_TRACE TPR_EXEC_PROFILE TPR_FIRST_LEAF_DIAG TPR_FIRST_LEAF_DIAG_SYNC || true
"${PYTHON_BIN:-python}" -m tests.models.mcore.tpr.parallel.training_multistep_repeat
mkdir -p "$TPR_TRAIN10_OUTPUT/tpr_cp2_qkvmerge"
export TPR_TRAIN10_OUTPUT="$(cd "$TPR_TRAIN10_OUTPUT" && pwd -P)"
export ASCEND_RT_VISIBLE_DEVICES="$DEVICE0,$DEVICE1"
echo 'TRAIN10_QKVMERGE_SETUP Same historical controls; only Prefix Query/KV merge enabled.'
"${PYTHON_BIN:-python}" -m torch.distributed.run --nnodes=1 --node_rank=0 \
    --nproc_per_node=2 --master_addr=127.0.0.1 --master_port="${TPR_TRAIN_PORT:-29701}" \
    -m tests.models.mcore.tpr.parallel.training_multistep_npu \
    2>&1 | tee "$TPR_TRAIN10_OUTPUT/tpr_cp2_qkvmerge/run.log"

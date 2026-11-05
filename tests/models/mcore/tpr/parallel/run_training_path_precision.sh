#!/usr/bin/env bash
# Real training-engine P1024/S1024 controls, one fresh process per mode.
set -Eeuo pipefail
cd "$(dirname "$0")/../../../../.."

: "${TPR_TRAIN_MODEL_PATH:?Set a local real Qwen3 checkpoint path (1.7B recommended)}"
: "${TPR_TRAIN_OUTPUT:?Set a new experiment output directory}"
: "${TPR_TRAIN_DEVICES:?Set two physical NPU IDs, e.g. 0,1}"
[[ -d "$TPR_TRAIN_MODEL_PATH" && ! -e "$TPR_TRAIN_OUTPUT" ]] || {
    echo 'ERROR: checkpoint missing or output already exists' >&2; exit 2;
}
IFS=, read -r DEVICE0 DEVICE1 EXTRA <<< "$TPR_TRAIN_DEVICES"
[[ -n "$DEVICE0" && -n "$DEVICE1" && -z "${EXTRA:-}" && "$DEVICE0" != "$DEVICE1" ]] || {
    echo 'ERROR: set exactly two distinct NPU IDs' >&2; exit 2;
}
export TPR_TRAIN_GRAD_ACCUM_FUSION="${TPR_TRAIN_GRAD_ACCUM_FUSION:-1}"
export TPR_TRAIN_VANILLA_MBRIDGE="${TPR_TRAIN_VANILLA_MBRIDGE:-1}"
export TPR_TRAIN_KEEP_GRADS="${TPR_TRAIN_KEEP_GRADS:-0}"
export TPR_TRAIN_SEED="${TPR_TRAIN_SEED:-261105}"
for flag in TPR_TRAIN_GRAD_ACCUM_FUSION TPR_TRAIN_VANILLA_MBRIDGE TPR_TRAIN_KEEP_GRADS; do
    [[ "${!flag}" =~ ^[01]$ ]] || { echo "ERROR: $flag must be 0 or 1" >&2; exit 2; }
done

# Explicit production choices, identical across all modes. No math attention
# adapters, monkeypatches, custom finalize or post-hoc gradient casting.
export TPR_TREE_POLICY=token_lcp TPR_TREE_BUILDER=lcp_sort TPR_MIN_SHARED_PREFIX_TOKENS=1024
export TPR_LEAF_BATCH_SIZE=1 TPR_ROOT_NATIVE_RING=0
export TPR_EXPERIMENT_DELAYED_DKV_FP32="${TPR_EXPERIMENT_DELAYED_DKV_FP32:-0}"
[[ "$TPR_EXPERIMENT_DELAYED_DKV_FP32" =~ ^[01]$ ]] || {
    echo 'ERROR: TPR_EXPERIMENT_DELAYED_DKV_FP32 must be 0 or 1' >&2; exit 2;
}
export TPR_RING_COALESCE_PREFIX_FULL=0 TPR_RING_COALESCE_PREFIX_QUERY=0 TPR_RING_COALESCE_PREFIX_PADDED=0
export TPR_SELECTED_LOGITS_FP32=0 TPR_PPO_LOGPROB_CHUNK_TOKENS=0
export HCCL_DETERMINISTIC=true PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
unset TPR_DKV_REPLAY_DIAG_PATH TPR_MEM_TRACE TPR_EXEC_PROFILE TPR_FIRST_LEAF_DIAG TPR_FIRST_LEAF_DIAG_SYNC || true
mkdir -p "$TPR_TRAIN_OUTPUT"
export TPR_TRAIN_OUTPUT="$(cd "$TPR_TRAIN_OUTPUT" && pwd -P)"

echo "TRAIN_PRECISION setup=P1024+S1024 rows=2 shared_prefix=1024 output=$TPR_TRAIN_OUTPUT"
echo 'All modes use the actual MindspeedEngineWithLMHead, DDP, PPO and native gradient finalize.'
echo 'Native DistributedOptimizer owns gradient shards; comparisons read only those finalized slices.'
echo 'Two FP32 temporary references: about 14 GB disk for 1.7B, 33 GB for 4B; removed after success.'
echo 'Stops before optimizer clipping/update. No dense FP32 attention; no parameter-gradient console dumps.'

for mode in base_cp1 base_cp2 tpr_cp1 tpr_cp2; do
    export TPR_TRAIN_MODE="$mode"
    if [[ "$mode" == *_cp2 ]]; then
        export ASCEND_RT_VISIBLE_DEVICES="$DEVICE0,$DEVICE1"
        ranks=2
    else
        export ASCEND_RT_VISIBLE_DEVICES="$DEVICE0"
        ranks=1
    fi
    mkdir "$TPR_TRAIN_OUTPUT/$mode"
    "${PYTHON_BIN:-python}" -m torch.distributed.run --nnodes=1 --node_rank=0 \
        --nproc_per_node="$ranks" --master_addr=127.0.0.1 \
        --master_port="${TPR_TRAIN_PORT:-29701}" \
        -m tests.models.mcore.tpr.parallel.training_path_precision_npu \
        2>&1 | tee "$TPR_TRAIN_OUTPUT/$mode/run.log"
    [[ -s "$TPR_TRAIN_OUTPUT/$mode/result.json" ]] || {
        echo "ERROR: missing $mode result; references retained" >&2; exit 1;
    }
done
"${PYTHON_BIN:-python}" -m tests.models.mcore.tpr.parallel.training_precision_report

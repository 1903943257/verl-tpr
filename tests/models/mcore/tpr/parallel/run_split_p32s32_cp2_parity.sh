#!/usr/bin/env bash
# Controlled 64-token BF16: CP1 Full, MindSpeed CP2 Full, TPR CP2 Full/split.
# Full CP1/CP2 and TPR CP2 use identical checkpoint/tokens/fixed loss.
# Three temporary references (~24 GB for 4B); split is compared online.
set -Eeuo pipefail
cd "$(dirname "$0")/../../../../.."

: "${TPR_P32S32_OUTPUT:?Set a fresh output directory}"
: "${TPR_QWEN_4B_PATH:?Set the Qwen3-4B-128K checkpoint directory}"
: "${TPR_P32S32_DEVICES:?Set two physical NPU IDs as e.g. 0,1}"
[[ ! -e "$TPR_P32S32_OUTPUT" ]] || {
    echo "ERROR: refusing to overwrite existing output directory: $TPR_P32S32_OUTPUT" >&2
    exit 2
}
[[ -d "$TPR_QWEN_4B_PATH" ]] || {
    echo "ERROR: missing model checkpoint: $TPR_QWEN_4B_PATH" >&2
    exit 2
}
IFS=, read -r DEVICE0 DEVICE1 EXTRA <<< "$TPR_P32S32_DEVICES"
if [[ -z "$DEVICE0" || -z "$DEVICE1" || -n "${EXTRA:-}" || "$DEVICE0" == "$DEVICE1" ]]; then
    echo "ERROR: set exactly two distinct visible NPUs (example: 0,1)" >&2
    exit 2
fi
export TPR_RUN_P32S32_CP2_PARITY=1
export TPR_ROOT_NATIVE_RING=0
export TPR_LEAF_BATCH_SIZE=1
export TPR_EXPERIMENT_DELAYED_DKV_FP32="${TPR_EXPERIMENT_DELAYED_DKV_FP32:-1}"
export TPR_P32S32_PROFILE="${TPR_P32S32_PROFILE:-4b}"
export TPR_P32S32_GRAD_SYNC_DTYPE="${TPR_P32S32_GRAD_SYNC_DTYPE:-bf16}"
export TPR_P32S32_KEEP_GRADS="${TPR_P32S32_KEEP_GRADS:-0}"
export TPR_P32S32_NATIVE_RING="${TPR_P32S32_NATIVE_RING:-1}"
export TPR_OFFLINE_NATIVE_RECOMPUTE_LAYERS=0
[[ "$TPR_EXPERIMENT_DELAYED_DKV_FP32" =~ ^[01]$ && "$TPR_P32S32_KEEP_GRADS" =~ ^[01]$ && "$TPR_P32S32_NATIVE_RING" =~ ^[01]$ ]] || {
    echo 'dKV and keep-grads gates must be 0 or 1' >&2; exit 2;
}
[[ "$TPR_P32S32_GRAD_SYNC_DTYPE" == bf16 || "$TPR_P32S32_GRAD_SYNC_DTYPE" == fp32 ]] || {
    echo 'Gradient reduction dtype must be bf16 or fp32' >&2; exit 2;
}
export TPR_RING_COALESCE_PREFIX_FULL=0
export TPR_RING_COALESCE_PREFIX_QUERY=0
export TPR_RING_COALESCE_PREFIX_PADDED=0
export HCCL_DETERMINISTIC=true PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
unset TPR_DKV_REPLAY_DIAG_PATH TPR_MEM_TRACE TPR_FIRST_LEAF_DIAG TPR_FIRST_LEAF_DIAG_SYNC || true
mkdir -p "$TPR_P32S32_OUTPUT"
export TPR_P32S32_OUTPUT="$(cd "$TPR_P32S32_OUTPUT" && pwd -P)"

TEST=tests/models/mcore/tpr/parallel/test_split_p32s32_cp2_parity_npu.py
echo "P32S32_CP_TEST output=$TPR_P32S32_OUTPUT"
echo "P32S32_CP_TEST attention=CP1_square_CANN_vs_CP2_TPR_Ring dtype=BF16"
echo "P32S32_CP_TEST grad_reduce=$TPR_P32S32_GRAD_SYNC_DTYPE delayed_dKV_FP32=$TPR_EXPERIMENT_DELAYED_DKV_FP32"
echo "Native MindSpeed CP2 Full enabled=$TPR_P32S32_NATIVE_RING; separate from the TPR whole-segment control."
echo 'With the native control, temporary gradients require about 24 GB disk for 4B; cleaned after success.'
echo "P32S32_CP_TEST fixed_loss + diagnostic next-token logprobs; no PPO, no optimizer"

export ASCEND_RT_VISIBLE_DEVICES="$DEVICE0"
export TPR_P32S32_MODE=full_cp1
echo "===== P32S32_CP_STAGE 1: Full CP1 64 tokens on NPU $DEVICE0 ====="
"${PYTHON_BIN:-python}" -m torch.distributed.run --nnodes=1 --node_rank=0 \
    --nproc_per_node=1 --master_addr=127.0.0.1 \
    --master_port="${TPR_P32S32_PORT_CP1:-29691}" \
    -m tests.models.mcore.tpr._mindspeed_first_pytest \
    -q -s -x -rs "$TEST" 2>&1 | tee "$TPR_P32S32_OUTPUT/full_cp1.log"
[[ -s "$TPR_P32S32_OUTPUT/full_cp1/result.json" ]] || {
    echo "ERROR: Full CP1 did not produce result.json" >&2
    exit 1
}

export ASCEND_RT_VISIBLE_DEVICES="$DEVICE0,$DEVICE1"
if [[ "$TPR_P32S32_NATIVE_RING" == 1 ]]; then
    export TPR_P32S32_MODE=native_cp2
    echo '===== P32S32_CP_STAGE 2: Native MindSpeed Ring CP2 Full ====='
    "${PYTHON_BIN:-python}" -m torch.distributed.run --nnodes=1 --node_rank=0 \
        --nproc_per_node=2 --master_addr=127.0.0.1 \
        --master_port="${TPR_P32S32_PORT_NATIVE:-29694}" \
        -m tests.models.mcore.tpr._mindspeed_first_pytest \
        -q -s -x -rs "$TEST" 2>&1 | tee "$TPR_P32S32_OUTPUT/native_cp2.log"
    [[ -s "$TPR_P32S32_OUTPUT/native_cp2/result.json" ]] || {
        echo 'Native MindSpeed control did not produce result.json' >&2; exit 1;
    }
fi
export TPR_P32S32_MODE=cp2_both
echo "===== P32S32_CP_STAGE 3: TPR Ring Full CP2 vs TPR CP2 P32+S32 ====="
"${PYTHON_BIN:-python}" -m torch.distributed.run --nnodes=1 --node_rank=0 \
    --nproc_per_node=2 --master_addr=127.0.0.1 \
    --master_port="${TPR_P32S32_PORT_CP2:-29692}" \
    -m tests.models.mcore.tpr._mindspeed_first_pytest \
    -q -s -x -rs "$TEST" 2>&1 | tee "$TPR_P32S32_OUTPUT/cp2.log"

[[ -s "$TPR_P32S32_OUTPUT/comparison.json" ]] || {
    echo "ERROR: CP2 did not produce comparison.json" >&2
    exit 1
}
echo "P32S32_CP_DONE report=$TPR_P32S32_OUTPUT/comparison.json"

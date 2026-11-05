#!/usr/bin/env bash
# Ten real production optimizer updates, four sequential topology processes.
set -Eeuo pipefail
cd "$(dirname "$0")/../../../../.."

: "${TPR_TRAIN_MODEL_PATH:?Set a local Qwen3 checkpoint, including tokenizer}"
: "${TPR_TRAIN10_OUTPUT:?Set a NEW ten-step output directory}"
: "${TPR_TRAIN_DEVICES:?Set two distinct physical NPU IDs, e.g. 0,1}"
[[ -d "$TPR_TRAIN_MODEL_PATH" ]] || { echo 'ERROR: checkpoint missing' >&2; exit 2; }
export TPR_TRAIN10_RESUME="${TPR_TRAIN10_RESUME:-0}"
[[ "$TPR_TRAIN10_RESUME" =~ ^[01]$ ]] || { echo 'ERROR: TPR_TRAIN10_RESUME must be 0 or 1' >&2; exit 2; }
[[ ! -e "$TPR_TRAIN10_OUTPUT" || ( "$TPR_TRAIN10_RESUME" == 1 && -d "$TPR_TRAIN10_OUTPUT" ) ]] || {
    echo 'ERROR: output already exists; use a new directory or TPR_TRAIN10_RESUME=1' >&2; exit 2;
}
IFS=, read -r DEVICE0 DEVICE1 EXTRA <<< "$TPR_TRAIN_DEVICES"
[[ -n "$DEVICE0" && -n "$DEVICE1" && -z "${EXTRA:-}" && "$DEVICE0" != "$DEVICE1" ]] || {
    echo 'ERROR: set exactly two distinct NPU IDs' >&2; exit 2;
}
export TPR_TRAIN_SEED="${TPR_TRAIN_SEED:-261105}"
export TPR_TRAIN10_LR="${TPR_TRAIN10_LR:-1e-6}"
export TPR_TRAIN10_KEEP_UPDATES="${TPR_TRAIN10_KEEP_UPDATES:-0}"
export TPR_TRAIN_GRAD_ACCUM_FUSION="${TPR_TRAIN_GRAD_ACCUM_FUSION:-1}"
export TPR_TRAIN_VANILLA_MBRIDGE="${TPR_TRAIN_VANILLA_MBRIDGE:-1}"
export TPR_EXPERIMENT_DELAYED_DKV_FP32="${TPR_EXPERIMENT_DELAYED_DKV_FP32:-0}"
for flag in TPR_TRAIN10_KEEP_UPDATES TPR_TRAIN_GRAD_ACCUM_FUSION TPR_TRAIN_VANILLA_MBRIDGE TPR_EXPERIMENT_DELAYED_DKV_FP32; do
    [[ "${!flag}" =~ ^[01]$ ]] || { echo "ERROR: $flag must be 0 or 1" >&2; exit 2; }
done
export TPR_TREE_POLICY=token_lcp TPR_TREE_BUILDER=lcp_sort TPR_MIN_SHARED_PREFIX_TOKENS=1024
export TPR_LEAF_BATCH_SIZE=1 TPR_ROOT_NATIVE_RING=0
export TPR_RING_COALESCE_PREFIX_FULL=0 TPR_RING_COALESCE_PREFIX_QUERY=0 TPR_RING_COALESCE_PREFIX_PADDED=0
export TPR_SELECTED_LOGITS_FP32=0 TPR_PPO_LOGPROB_CHUNK_TOKENS=0
export HCCL_DETERMINISTIC=true PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
unset TPR_DKV_REPLAY_DIAG_PATH TPR_MEM_TRACE TPR_EXEC_PROFILE TPR_FIRST_LEAF_DIAG TPR_FIRST_LEAF_DIAG_SYNC || true
mkdir -p "$TPR_TRAIN10_OUTPUT"
export TPR_TRAIN10_OUTPUT="$(cd "$TPR_TRAIN10_OUTPUT" && pwd -P)"
if [[ "$TPR_TRAIN10_RESUME" == 1 ]]; then
    "${PYTHON_BIN:-python}" -m tests.models.mcore.tpr.parallel.training_multistep_resume
fi

echo "TRAIN10_SETUP P1024+S1024 rows=2 updates=10 lr=$TPR_TRAIN10_LR output=$TPR_TRAIN10_OUTPUT"
echo 'Native BF16 model + FP32 main_grad/Adam masters/moments; native clipping/optimizer/scheduler.'
echo 'Full update comparison only at step 10. Samples at 1/5/10. No full gradient dumps.'
echo 'Temporary disk: ~14 GB for 1.7B / ~33 GB for 4B. Native Adam moments increase NPU memory.'

for mode in base_cp1 base_cp2 tpr_cp1 tpr_cp2; do
    if [[ "$TPR_TRAIN10_RESUME" == 1 && -s "$TPR_TRAIN10_OUTPUT/$mode/result.json" ]]; then
        echo "TRAIN10_SKIP completed=$mode"
        continue
    fi
    export TPR_TRAIN_MODE="$mode"
    if [[ "$mode" == *_cp2 ]]; then
        export ASCEND_RT_VISIBLE_DEVICES="$DEVICE0,$DEVICE1"
        ranks=2
    else
        export ASCEND_RT_VISIBLE_DEVICES="$DEVICE0"
        ranks=1
    fi
    mkdir "$TPR_TRAIN10_OUTPUT/$mode"
    "${PYTHON_BIN:-python}" -m torch.distributed.run --nnodes=1 --node_rank=0 \
        --nproc_per_node="$ranks" --master_addr=127.0.0.1 \
        --master_port="${TPR_TRAIN_PORT:-29701}" \
        -m tests.models.mcore.tpr.parallel.training_multistep_npu \
        2>&1 | tee "$TPR_TRAIN10_OUTPUT/$mode/run.log"
    [[ -s "$TPR_TRAIN10_OUTPUT/$mode/result.json" ]] || { echo "ERROR: incomplete $mode" >&2; exit 1; }
done
if [[ ! -s "$TPR_TRAIN10_OUTPUT/comparison.json" ]]; then
    "${PYTHON_BIN:-python}" -m tests.models.mcore.tpr.parallel.training_multistep_report \
        2>&1 | tee "$TPR_TRAIN10_OUTPUT/summary.log"
fi

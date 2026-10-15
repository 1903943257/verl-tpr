#!/usr/bin/env bash
# Replay true Actor PPO through old and new TP2CP4 TPR separately.
# bash tests/models/mcore/tpr/parallel/run_old_new_tpr_tp2cp4_ppo.sh \
#   /workspace/verl-tpr-old /workspace/verl-tpr-review \
#   /workspace/tpr_actor_capture_4b_n8_run1/actor_update_rank0_batch0.pt \
#   /workspace/tpr_old_new_tp2cp4_ppo_v1
set -euo pipefail
if [[ "$#" -ne 4 ]]; then
    echo "Usage: $0 OLD_REPO NEW_REPO ACTOR_CAPTURE OUTPUT_ROOT" >&2
    exit 2
fi
OLD="$(realpath "$1")"
NEW="$(realpath "$2")"
CAPTURE="$(realpath "$3")"
OUT="$4"
[[ -f "$CAPTURE" ]] || { echo "Missing Actor capture: $CAPTURE" >&2; exit 2; }
[[ "$OLD" != "$NEW" ]] || { echo "Old and new repositories must be separate" >&2; exit 2; }
# The two checkouts are pinned and verified with Git on the host.
# Docker has no access to the host worktree's Git metadata.
[[ -f "$NEW/tests/models/mcore/tpr/parallel/compare_old_new_tpr_tp2cp4_ppo.py" ]] ||
    { echo "New repo missing cross-version comparator" >&2; exit 2; }
[[ ! -e "$OUT" ]] || { echo "Output directory already exists: $OUT" >&2; exit 2; }
mkdir -p "$OUT/old" "$OUT/new"
OUT="$(realpath "$OUT")"
export TPR_REAL_ACTOR_MINIBATCH="$CAPTURE"
export TPR_QWEN_4B_PATH="${TPR_QWEN_4B_PATH:-/workspace/hf_models/Qwen3-4B-128K}"
export TPR_OFFLINE_ROWS="${TPR_OFFLINE_ROWS:-4,6}"
export TPR_TREE_POLICY=qwen_turn
export TPR_TURN_MIN_SEGMENT_TOKENS=128
export TPR_PPO_LOGPROB_CHUNK_TOKENS=512
export TPR_OFFLINE_SGD_LR=1e-5
export TPR_OFFLINE_SAVE_FULL_GRADS="${TPR_OFFLINE_SAVE_FULL_GRADS:-1}"
export TPR_OFFLINE_RUN_MODES=tpr
export TPR_OFFLINE_FULL_RING_BACKEND=custom
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-8,9,10,11,12,13,14,15}"
export HCCL_DETERMINISTIC=true
export TPR_EXEC_PROFILE=0
export PYTHONUNBUFFERED=1
[[ -d "$TPR_QWEN_4B_PATH" ]] || { echo "Missing Qwen3-4B model" >&2; exit 2; }
[[ "$ASCEND_RT_VISIBLE_DEVICES" == "8,9,10,11,12,13,14,15" ]] ||
    { echo "Required physical NPUs 8..15" >&2; exit 2; }
for version in old new; do
    if [[ "$version" == old ]]; then
        source_dir="$OLD"
    else
        source_dir="$NEW"
    fi
    export TPR_OFFLINE_OUTPUT_DIR="$OUT/$version"
    echo "TPR_REPLAY version=$version checkout=$source_dir"
    (
        cd "$source_dir"
        export PYTHONPATH="$source_dir:${PYTHONPATH:-}"
        if [[ "$version" == old ]]; then
            python -c 'import inspect; from verl.models.mcore.tpr.megatron_adapter import run_tpr_forward_backward_batch as f; print("ACTUAL_TPR_ADAPTER", inspect.getfile(f), flush=True)'
        else
            python -c 'import inspect; from verl.workers.engine.megatron.tpr.adapter import run_tpr_forward_backward_batch as f; print("ACTUAL_TPR_ADAPTER", inspect.getfile(f), flush=True)'
        fi
        bash tests/models/mcore/tpr/parallel/run_real_actor_tp2cp4_offline_ppo.sh
    ) 2>&1 | tee "$OUT/$version.log"
done
if [[ "$TPR_OFFLINE_SAVE_FULL_GRADS" == "1" ]]; then
    python "$NEW/tests/models/mcore/tpr/parallel/compare_old_new_tpr_tp2cp4_ppo.py" \
        "$OUT/old" "$OUT/new" --require-full-grads | tee "$OUT/comparison.log"
else
    python "$NEW/tests/models/mcore/tpr/parallel/compare_old_new_tpr_tp2cp4_ppo.py" \
        "$OUT/old" "$OUT/new" | tee "$OUT/comparison.log"
fi
echo "TPR_OLD_NEW_REPLAY_DONE result=$OUT/comparison.log"

#!/usr/bin/env bash
# Isolated 8-NPU TP2 x RingCP4 true-PPO recorded-trajectory one-step profile.
# Must run from the synced Docker VERL root, no rollout/Ray is launched.
set -euo pipefail
cd "$(dirname "$0")/../../../../.."
export TPR_RUN_REAL_ACTOR_TP2CP4=1
export TPR_REAL_ACTOR_MINIBATCH="${TPR_REAL_ACTOR_MINIBATCH:-/workspace/tpr_actor_capture_4b_n8_run1/actor_update_rank0_batch0.pt}"
export TPR_QWEN_4B_PATH="${TPR_QWEN_4B_PATH:-/workspace/hf_models/Qwen3-4B-128K}"
# Production TPR no longer accepts the historical exact-tree policy.
# Force the only supported policy for this experiment even if a previous
# shell exported TPR_TREE_POLICY=exact.
export TPR_TREE_POLICY=qwen_turn
export TPR_RECOMPUTE_LOG_DIR="${TPR_RECOMPUTE_LOG_DIR:-/workspace/tpr_log/recompute}"
mkdir -p "$TPR_RECOMPUTE_LOG_DIR"
export TPR_OFFLINE_RECOMPUTE="${TPR_OFFLINE_RECOMPUTE:-0}"
export TPR_OFFLINE_SP="${TPR_OFFLINE_SP:-0}"
case "$TPR_OFFLINE_SP" in
    0|1) ;;
    *) echo "TPR_OFFLINE_SP must be 0 or 1" >&2; exit 2 ;;
esac
case "$TPR_OFFLINE_RECOMPUTE" in
    0) recompute_label=off ;;
    1) recompute_label=on ;;
    *) echo "TPR_OFFLINE_RECOMPUTE must be 0 or 1" >&2; exit 2 ;;
esac
export TPR_OFFLINE_ROWS="${TPR_OFFLINE_ROWS:-4,6}"
# OFF and ON must not overwrite one another's saved gradients / witnesses.
export TPR_OFFLINE_OUTPUT_DIR="${TPR_OFFLINE_OUTPUT_DIR:-/workspace/tpr_sp${TPR_OFFLINE_SP}_recompute_${recompute_label}_v1}"
: "${ASCEND_RT_VISIBLE_DEVICES:=0,1,2,3,4,5,6,7}"
export ASCEND_RT_VISIBLE_DEVICES
case "$ASCEND_RT_VISIBLE_DEVICES" in
    0,1,2,3,4,5,6,7|8,9,10,11,12,13,14,15) ;;
    *) echo "ERROR: CP4 replay expects eight physical NPUs (0-7 or 8-15), got $ASCEND_RT_VISIBLE_DEVICES" >&2; exit 2 ;;
esac
[[ -f "$TPR_REAL_ACTOR_MINIBATCH" ]] || { echo "Missing actor capture" >&2; exit 2; }
[[ -d "$TPR_QWEN_4B_PATH" ]] || { echo "Missing Qwen3-4B checkpoint" >&2; exit 2; }
mkdir -p "$TPR_OFFLINE_OUTPUT_DIR"
case "${TPR_OFFLINE_RUN_MODES:-both}" in
   both) modes=(full tpr) ;;
   full) modes=(full) ;;
   tpr) modes=(tpr) ;;
   split_no_share) modes=(split_no_share) ;;
   compare) modes=() ;;
   *) echo "Unknown TPR_OFFLINE_RUN_MODES" >&2; exit 2;;
esac
if [[ "${TPR_OFFLINE_RUN_MODES:-both}" == "split_no_share" ]]; then
    [[ "${TPR_OFFLINE_FULL_RING_BACKEND:-custom}" == "custom" ]] || {
        echo "CP4 split-no-share control requires custom Ring on all three arms" >&2
        exit 2
    }
    [[ "${TPR_OFFLINE_SGD_LR:-1e-5}" == "0" ||
       "${TPR_OFFLINE_SGD_LR:-1e-5}" == "0.0" ]] || {
        echo "CP4 split-no-share controlled parity requires TPR_OFFLINE_SGD_LR=0" >&2
        exit 2
    }
    python -m pytest -q \
      tests/models/mcore/tpr/parallel/test_cp4_multistep_replica_audit_cpu.py
fi
errors=()
for mode in "${modes[@]}"; do
    export TPR_OFFLINE_MODE="$mode"
    echo "TPR_OFFLINE_START mode=$mode sp=$TPR_OFFLINE_SP recompute=$TPR_OFFLINE_RECOMPUTE rows=$TPR_OFFLINE_ROWS device=$ASCEND_RT_VISIBLE_DEVICES results=$TPR_OFFLINE_OUTPUT_DIR"
    # An OOM in Full must not prevent the independent TPR memory experiment.
    if torchrun --nproc_per_node=8 --master_addr=127.0.0.1 --master_port=29797 \
       tests/models/mcore/tpr/parallel/run_cp4_pytest_argv_safe.py \
       2>&1 | tee "$TPR_RECOMPUTE_LOG_DIR/offline_tp2cp4_${mode}_sp${TPR_OFFLINE_SP}_recompute_${recompute_label}.log"; then
       echo "TPR_OFFLINE mode=$mode completed"
    else
       errors+=("$mode")
       echo "TPR_OFFLINE mode=$mode FAILED; retaining the log and trying remaining modes" >&2
    fi
done
if [[ "${#errors[@]}" == 0 ]] && \
   [[ "${TPR_OFFLINE_RUN_MODES:-both}" == "both" || "${TPR_OFFLINE_RUN_MODES:-both}" == "compare" ]]; then
    # The historical Full-vs-TPR provisional correctness thresholds are
    # not a Recompute parity gate. ON compares two different execution
    # layouts, so report all metrics without treating an unrelated 5%
    # gradient threshold as an infrastructure failure.
    compare_args=()
    if [[ "$TPR_OFFLINE_RECOMPUTE" == "1" ]]; then
        compare_args+=(--require-recompute --diagnostic-only)
    fi
    python tests/models/mcore/tpr/parallel/compare_real_actor_tp2cp4_offline_ppo.py \
      "$TPR_OFFLINE_OUTPUT_DIR" "${compare_args[@]}" \
      2>&1 | tee "$TPR_RECOMPUTE_LOG_DIR/offline_tp2cp4_comparison_sp${TPR_OFFLINE_SP}_recompute_${recompute_label}.log"
fi
# CP4 three-arm numerical attribution can reuse an existing full/tpr run.
# The split_no_share mode runs ONLY the missing third arm (8 NPUs once).
if [[ "${TPR_OFFLINE_RUN_MODES:-both}" == "split_no_share" ]] && \
   [[ "${#errors[@]}" == 0 ]]; then
    three_arms_ready=1
    for mode in full split_no_share tpr; do
        for rank in {0..7}; do
            [[ -f "$TPR_OFFLINE_OUTPUT_DIR/${mode}_rank${rank}.pt" ]] || three_arms_ready=0
        done
    done
    if [[ "$three_arms_ready" == 1 ]]; then
        python -m tests.models.mcore.tpr.parallel.compare_cp4_split_no_share_cpu \
          "$TPR_OFFLINE_OUTPUT_DIR" | \
          tee "$TPR_OFFLINE_OUTPUT_DIR/cp4_three_arm_comparison.log"
    else
        echo "TPR_CP4_SPLIT saved; no three-arm comparison: full/tpr snapshots missing"
    fi
fi
if (("${#errors[@]}" > 0)); then
    echo "TPR_OFFLINE failed_modes=${errors[*]} (logs preserved)" >&2
    exit 1
fi

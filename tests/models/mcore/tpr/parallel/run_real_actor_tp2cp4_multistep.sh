#!/usr/bin/env bash
# Real Actor PPO optimizer replay: fixed 1-50-step capture or chronological
# 8-Session step manifest. Native Megatron microbatch count equals input rows.
# Complete prior mode snapshots may be reused after the other mode fails.
# Resource note: CPU FP32-master AdamW allocates 3x TP-shard parameters
# on exactly two CP-leader processes (not 8 replicated optimizer states).
set -euo pipefail
cd "$(dirname "$0")/../../../../.."

export TPR_RUN_REAL_ACTOR_TP2CP4_MULTISTEP=1
export TPR_REAL_ACTOR_MINIBATCH="${TPR_REAL_ACTOR_MINIBATCH:-/workspace/tpr_actor_capture_4b_n8_run1/actor_update_siblings_row4_row6.pt}"
export TPR_QWEN_4B_PATH="${TPR_QWEN_4B_PATH:-/workspace/hf_models/Qwen3-4B-128K}"
export TPR_OFFLINE_ROWS="${TPR_OFFLINE_ROWS:-0,1}"
export TPR_OFFLINE_STEPS="${TPR_OFFLINE_STEPS:-50}"
export TPR_OFFLINE_OUTPUT_DIR="${TPR_OFFLINE_OUTPUT_DIR:-/workspace/tpr_offline_qwen4b_tp2cp4_fixed_50steps}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-8,9,10,11,12,13,14,15}"
export TPR_OFFLINE_ADAMW_LR="${TPR_OFFLINE_ADAMW_LR:-2e-5}"
export TPR_OFFLINE_ADAMW_CLIP_NORM="${TPR_OFFLINE_ADAMW_CLIP_NORM:-1.0}"
if [[ "$ASCEND_RT_VISIBLE_DEVICES" != "8,9,10,11,12,13,14,15" ]]; then
    echo "Requires visible physical NPU 8-15" >&2; exit 2
fi
[[ -f "$TPR_REAL_ACTOR_MINIBATCH" ]] || { echo "Missing recorded actor capture" >&2; exit 2; }
[[ -d "$TPR_QWEN_4B_PATH" ]] || { echo "Missing real 4B checkpoint" >&2; exit 2; }

mkdir -p "$TPR_OFFLINE_OUTPUT_DIR"
python -m pytest -q \
  tests/models/mcore/tpr/parallel/test_native_full_dynamic_microbatches_cpu.py \
  | tee "$TPR_OFFLINE_OUTPUT_DIR/native_batch_preflight.log"

# Re-run a failed mode without clobbering the opposite mode's successful
# 8-rank snapshots. Snapshot sets must be all-or-nothing; a partially
# completed optimizer trajectory CANNOT safely resume from step 1.
declare -a expected_steps=()
if [[ -n "${TPR_OFFLINE_CAPTURE_MANIFEST:-}" ]]; then
  for ((s=1;s<=TPR_OFFLINE_STEPS;s++)); do expected_steps+=("$s"); done
else
  for s in 1 2 5 10 20 30 40 50; do
    (( s <= TPR_OFFLINE_STEPS )) && expected_steps+=("$s")
  done
  if [[ " ${expected_steps[*]} " != *" ${TPR_OFFLINE_STEPS} "* ]]; then
    expected_steps+=("$TPR_OFFLINE_STEPS")
  fi
fi
case "${TPR_OFFLINE_RUN_MODES:-both}" in
    full) modes=(full);;
    tpr) modes=(tpr);;
    both) modes=(full tpr);;
    compare) modes=();;
    *) echo "TPR_OFFLINE_RUN_MODES must be full|tpr|both|compare" >&2; exit 2;;
esac
errors=()
for mode in "${modes[@]}"; do
    export TPR_OFFLINE_MODE="$mode"
    expected=0
    existing=0
    for step in "${expected_steps[@]}"; do
        for rank in {0..7}; do
            expected=$((expected+1))
            [[ -f "$TPR_OFFLINE_OUTPUT_DIR/${mode}_step${step}_rank${rank}.pt" ]] &&
                existing=$((existing+1))
        done
    done
    if (( existing == expected && expected > 0 )); then
        echo "TPR_FIXED_PPO_REUSE completed_mode=$mode snapshots=$existing"
        continue
    elif (( existing > 0 )); then
        echo "TPR_FIXED_PPO_PARTIAL mode=$mode snapshots=$existing/$expected; refusing unsafe restart. Use a new output directory for this mode." >&2
        errors+=("$mode")
        continue
    fi
    echo "TPR_FIXED_PPO start=$mode steps=$TPR_OFFLINE_STEPS rows=$TPR_OFFLINE_ROWS"
    if torchrun --nproc_per_node=8 --master_addr=127.0.0.1 --master_port=29786 \
        -m pytest -vv -s -x \
        tests/models/mcore/tpr/parallel/test_real_actor_tp2cp4_multistep_npu.py \
        2>&1 | tee "$TPR_OFFLINE_OUTPUT_DIR/${mode}_50step.log"; then
        echo "TPR_FIXED_PPO mode=$mode completed"
    else
        errors+=("$mode")
        echo "TPR_FIXED_PPO mode=$mode failed; keeping snapshots/logs; proceeding independently" >&2
    fi
done
if [[ "${#errors[@]}" == 0 ]] && \
   [[ "${TPR_OFFLINE_RUN_MODES:-both}" == "both" || "${TPR_OFFLINE_RUN_MODES:-both}" == "compare" ]]; then
    python tests/models/mcore/tpr/parallel/compare_real_actor_tp2cp4_multistep.py \
        "$TPR_OFFLINE_OUTPUT_DIR" --steps "$TPR_OFFLINE_STEPS" \
        | tee "$TPR_OFFLINE_OUTPUT_DIR/multistep_comparison.log"
fi
if (("${#errors[@]}" > 0)); then
    echo "TPR_FIXED_PPO failed_modes=${errors[*]} (logs retained)" >&2
    exit 1
fi

#!/usr/bin/env bash
# Route B: 10 real Actor PPO optimizer updates by default, Full Sequence vs Split-No-Share.
# The two arms run sequentially, one Qwen3-4B model at a time, from the same
# checkpoint and the same fixed cropped Actor capture.
set -euo pipefail
cd "$(dirname "$0")/../../../../.."

BASE="${TPR_CP1_BASE:-/workspace/tpr_experiments/cp1_sequence_split_4k_precision_20261025_v1}"
CAPTURE="${TPR_CP1_CAPTURE:-$BASE/actor_pair_cap4096_p3072.pt}"
CHECKPOINT="${TPR_QWEN_4B_PATH:-/workspace/hf_models/Qwen3-4B-128K}"
ROOT="${TPR_CP1_MULTISTEP_ROOT:-/workspace/tpr_experiments/cp1_sequence_split_route_b_10step_20261025_v1}"
STEPS="${TPR_CP1_MULTISTEP_STEPS:-10}"

[[ -f "$CAPTURE" ]] || { echo "Missing fixed Actor capture: $CAPTURE" >&2; exit 2; }
[[ -d "$CHECKPOINT" ]] || { echo "Missing Qwen3-4B checkpoint: $CHECKPOINT" >&2; exit 2; }
[[ "$STEPS" =~ ^[0-9]+$ ]] && [[ "$STEPS" -ge 2 ]] || {
  echo "TPR_CP1_MULTISTEP_STEPS must be an integer >= 2" >&2; exit 2;
}
export TPR_CP1_CAPTURE="$CAPTURE"
export TPR_QWEN_4B_PATH="$CHECKPOINT"
export TPR_CP1_MULTISTEP_STEPS="$STEPS"
export TPR_RUN_CP1_MULTISTEP=1
export TPR_RUN_CP1_SHORT_ACTOR=1
export TPR_TREE_POLICY=qwen_turn
export TPR_TURN_MIN_SEGMENT_TOKENS=128
export TPR_PPO_LOGPROB_CHUNK_TOKENS=256
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-8}"
export HCCL_DETERMINISTIC=true
export PYTHONUNBUFFERED=1
unset TPR_CP1_SAVE_FULL_GRADS TPR_CP1_SPLIT_FP32_LINEAR_DW TPR_CP1_TRACE
unset TPR_CP1_PROGRESS_LAYERS

mkdir -p "$ROOT"

needs_run=0
for mode in full split_no_share; do
  dir="$ROOT/$mode"
  trajectory="$dir/trajectory.jsonl"
  summary="$dir/trajectory.summary.json"
  final_state="$dir/trajectory.final_state.pt"
  if [[ -f "$trajectory" && -f "$summary" && -f "$final_state" ]]; then
    echo "ROUTE_B_REUSE_COMPLETE_ARM mode=$mode trajectory=$trajectory final_state=$final_state"
  elif [[ -e "$trajectory" || -e "$summary" || -e "$final_state" ]]; then
    echo "Incomplete Route B artifacts for $mode under $dir." >&2
    echo "Use a fresh TPR_CP1_MULTISTEP_ROOT to avoid mixing partial runs." >&2
    exit 2
  else
    needs_run=1
  fi
done

if [[ "$needs_run" == "1" ]]; then
  python tests/models/mcore/tpr/parallel/_mindspeed_first_pytest.py \
    --bootstrap-only 2>&1 | tee "$ROOT/bootstrap.log"
fi

run_arm() {
  local mode="$1"
  local port="$2"
  local dir="$ROOT/$mode"
  local trajectory="$dir/trajectory.jsonl"
  local summary="$dir/trajectory.summary.json"
  local final_state="$dir/trajectory.final_state.pt"

  if [[ -f "$trajectory" && -f "$summary" && -f "$final_state" ]]; then
    echo "ROUTE_B_SKIP_ARM mode=$mode existing_result=True"
    return 0
  fi
  mkdir -p "$dir"
  echo "ROUTE_B_START mode=$mode steps=$STEPS capture=$CAPTURE"
  TPR_CP1_MULTISTEP_MODE="$mode" \
  TPR_CP1_MULTISTEP_OUTPUT="$trajectory" \
    torchrun --nproc_per_node=1 \
      --master_addr=127.0.0.1 \
      --master_port="$port" \
      tests/models/mcore/tpr/parallel/_mindspeed_first_pytest.py \
      -xvs tests/models/mcore/tpr/parallel/test_real_actor_cp1_sequence_split_multistep_npu.py \
      2>&1 | tee "$ROOT/$mode.log"
  echo "ROUTE_B_ARM_DONE mode=$mode"
}

run_arm full 29818
run_arm split_no_share 29819

python tests/models/mcore/tpr/parallel/compare_real_actor_cp1_sequence_split_multistep.py \
  "$ROOT/full/trajectory.jsonl" \
  "$ROOT/split_no_share/trajectory.jsonl" \
  "$ROOT/comparison.json" 2>&1 | tee "$ROOT/comparison.log"

echo
echo "========== ROUTE B FINAL STATUS =========="
echo "steps=$STEPS"
echo "full_log=$ROOT/full.log"
echo "split_log=$ROOT/split_no_share.log"
echo "report=$ROOT/comparison.json"
grep -E 'ROUTE_B_STEP|ROUTE_B SUMMARY|ROUTE_B_ANALYSIS_COMPLETE|ROUTE_B_RESULT|ROUTE_B_ARM_DONE' \
  "$ROOT/comparison.log" "$ROOT/full.log" "$ROOT/split_no_share.log" 2>/dev/null || true
echo "ROUTE_B_FINISHED"

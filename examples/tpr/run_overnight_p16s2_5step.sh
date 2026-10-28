#!/usr/bin/env bash
# P16K/S2K, 5-step online Full then TPR; preserve FULL actor minibatch
# captures and original per-session trajectories for tomorrow's offline replay.
#
# Launch ONCE in an otherwise idle training container:
#   nohup bash examples/tpr/run_overnight_p16s2_5step.sh \
#     > /workspace/tpr_experiments/e2e/p16s2_overnight_controller.log 2>&1 < /dev/null &
#
# No Python/site-package installs, no git mutations, no new Docker containers.
# The canonical launcher will stop/reinitialize the local Ray runtime per run.
set -uo pipefail

CHECKOUT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)"
E2E="$CHECKOUT/tests/models/mcore/tpr/integration/run_uniagent_rebench_ppo_e2e.sh"
if [[ ! -f "$E2E" ]]; then echo "FATAL missing E2E $E2E"; exit 2; fi

N="${OVERNIGHT_N:-4}"
STEPS="${OVERNIGHT_STEPS:-5}"
if [[ ! "$N" =~ ^[0-9]+$ || ! "$STEPS" =~ ^[0-9]+$ ]] || ((N < 2 || STEPS < 1)); then
  echo "FATAL OVERNIGHT_N>=2, OVERNIGHT_STEPS>=1 required"; exit 2
fi
BASE="${OVERNIGHT_ROOT:-/workspace/tpr_experiments/e2e/p16s2_n${N}_5step_$(date +%Y%m%d_%H%M%S)}"
if [[ -e "$BASE" ]]; then echo "FATAL output directory exists: $BASE"; exit 2; fi
mkdir -p "$BASE"
cd "$CHECKOUT" || exit 2

export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
export RAY_DEDUP_LOGS=0
export HYDRA_FULL_ERROR=1
export ROOT="${UNIAGENT_ROOT:-/workspace/uni-agent}"
export TPR_VERL_ROOT="$CHECKOUT"
export MODEL_PATH="${MODEL_PATH:-/workspace/hf_models/Qwen3-4B-128K}"
export MODEL_NAME=Qwen3-4B
export TRAIN_FILE="${TRAIN_FILE:-/workspace/tpr_experiments/shortfix/pelita875_hint_h2_arm64_train.parquet}"
export TEST_FILE="$TRAIN_FILE"
export TASK_CONFIG="${TASK_CONFIG:-/workspace/tpr_experiments/shortfix/task_config_claude_code_pelita_docker.yaml}"
export N
export NNODES=1
export N_GPUS_PER_NODE=8
export TRAIN_TP=2
export TRAIN_PP=1
export TRAIN_CP=4
export TRAIN_SP=True
export ROLLOUT_TP=2
export TRAIN_PROMPT_BSZ=1
export GEN_BATCH_SIZE=1
export PPO_MINI_BATCH_SIZE=1
export TOTAL_TRAINING_STEPS="$STEPS"
export TOTAL_EPOCHS="$STEPS"
export MAX_PROMPT_LENGTH=16384
export MAX_RESPONSE_LENGTH=2048
export ACTOR_RECOMPUTE=True
export ROLLOUT_GPU_MEMORY_UTILIZATION=0.45
export CONCURRENCY=4
export GATEWAY_COUNT=1
export NUM_AGENT_WORKERS=1
export SESSION_HARD_CAP=True
export TRAJECTORY_SELECTION=all
export TPR_TREE_POLICY=qwen_turn
export TPR_E2E_EXTERNAL_LOG=1

{
  echo "START $(date -Is)"
  echo "RESULT_ROOT=$BASE"
  echo "FULL THEN TPR: N=$N steps=$STEPS P=16384 S=2048 TP=2 CP=4 DP=1"
  echo "Full actor minibatch snapshots enabled; TPR optimized Ring merge enabled."
  git rev-parse HEAD 2>/dev/null || true
} > "$BASE/experiment.txt"

failures=0
for mode in full tpr; do
  MODE_ROOT="$BASE/$mode"
  mkdir -p "$MODE_ROOT"
  export RUN_ROOT="$MODE_ROOT"
  export RUN_NAME="rebench-p16k-s2k-${mode}-tp2cp4-n${N}-step${STEPS}-$(basename "$BASE")"
  export TPR_CAPTURE_ACTOR_MINIBATCH_DIR="$MODE_ROOT/actor_captures"

  if [[ "$mode" == full ]]; then
    export TPR_ENABLED=False
    export CAPTURE_ACTOR=True
    export TPR_RING_COALESCE_PREFIX_FULL=0
    export TPR_RING_COALESCE_PREFIX_QUERY=0
    export TPR_RING_COALESCE_PREFIX_PADDED=0
  else
    export TPR_ENABLED=True
    export CAPTURE_ACTOR=False
    export TPR_RING_COALESCE_PREFIX_FULL=1
    export TPR_RING_COALESCE_PREFIX_QUERY=1
    export TPR_RING_COALESCE_PREFIX_PADDED=1
  fi

  echo "[$(date -Is)] PRECHECK mode=$mode" | tee -a "$BASE/experiment.txt"
  if ! PRECHECK_ONLY=True bash "$E2E" > "$MODE_ROOT/precheck.log" 2>&1; then
    echo 2 > "$MODE_ROOT/exit_code.txt"
    echo "[$(date -Is)] PRECHECK_FAILED mode=$mode; see $MODE_ROOT/precheck.log" | tee -a "$BASE/experiment.txt"
    failures=$((failures + 1))
    # If the common prerequisites are missing, trying the next mode is also invalid.
    break
  fi

  echo "[$(date -Is)] RUNNING mode=$mode captures=$TPR_CAPTURE_ACTOR_MINIBATCH_DIR" | tee -a "$BASE/experiment.txt"
  bash "$E2E" > "$MODE_ROOT/launcher.log" 2>&1
  rc=$?
  printf '%s\n' "$rc" > "$MODE_ROOT/exit_code.txt"
  echo "[$(date -Is)] FINISHED mode=$mode rc=$rc log=$MODE_ROOT/launcher.log" | tee -a "$BASE/experiment.txt"
  if (( rc != 0 )); then failures=$((failures + 1)); fi

  if [[ "$mode" == full ]]; then
    find "$TPR_CAPTURE_ACTOR_MINIBATCH_DIR" -type f -name 'actor_update_*.pt' -print \
      > "$MODE_ROOT/actor_capture_files.txt" 2>/dev/null || true
    capture_count=$(wc -l < "$MODE_ROOT/actor_capture_files.txt")
    echo "[$(date -Is)] FULL_CAPTURE_COUNT=$capture_count" | tee -a "$BASE/experiment.txt"
    if ((capture_count == 0)); then
      echo "[$(date -Is)] WARNING: Full produced no actor_update_*.pt snapshots" | tee -a "$BASE/experiment.txt"
      failures=$((failures + 1))
    fi
  fi
  grep -E 'training/global_step|timing_s/gen|timing_s/update_actor|actor/loss|STRICT_SESSION_CAP_PASS' \
    "$MODE_ROOT/launcher.log" | tail -25 > "$MODE_ROOT/key_metrics.txt" || true
done

echo "[$(date -Is)] END failures=$failures directory=$BASE" | tee -a "$BASE/experiment.txt"
if ((failures)); then exit 1; fi
exit 0

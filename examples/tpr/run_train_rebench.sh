#!/usr/bin/env bash
# SWE-reBench Qwen3-4B, Ascend 910B4 8x32GiB, one real UniAgent PPO update.
# Public recipe wrapper around the canonical E2E launcher, no duplicated Hydra args.
#
# Baseline measured on 2026-10-27:
#   MODE=full | TP2 CP4 DP1 | rollout TP2 n=2 | prompt=16384 response=8192
#   recompute + CPU offload, 1 PPO step, 1 Pelita-875 prompt, 2 agent sessions.
# Same-config TPR validation:
#   MODE=tpr bash examples/tpr/run_train_rebench.sh
# Non-invasive config precheck:
#   CHECK_ONLY=1 MODE=tpr bash examples/tpr/run_train_rebench.sh
#
# Before running, prepare TRAIN_FILE with local sandbox.image (ARM64) and
# preinstalled Reward dependencies; use a unique RUN_NAME for each experiment.
# Online Full and TPR each sample NEW agent trajectories: timings are not paired.
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
CHECKOUT="$(cd -- "$HERE/../.." && pwd -P)"
E2E="$CHECKOUT/tests/models/mcore/tpr/integration/run_uniagent_rebench_ppo_e2e.sh"
[[ -f "$E2E" ]] || { echo "ERROR: canonical launcher not found: $E2E" >&2; exit 2; }

MODE="${MODE:-tpr}"
case "$MODE" in
  full) export TPR_ENABLED=False ;;
  tpr)  export TPR_ENABLED=True ;;
  *) echo "ERROR: MODE must be full or tpr (got '$MODE')" >&2; exit 2 ;;
esac

# Same hardware/topology/optimization as the successfully completed Full step.
# NPU_IDS is a physical device mask, not TP/CP rank order on a multi-node run.
export ASCEND_RT_VISIBLE_DEVICES="${NPU_IDS:-0,1,2,3,4,5,6,7}"
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
export RAY_DEDUP_LOGS=0
export HYDRA_FULL_ERROR=1
export ROOT="${UNIAGENT_ROOT:-/workspace/uni-agent}"
export TPR_VERL_ROOT="$CHECKOUT"
export MODEL_PATH="${MODEL_PATH:-/workspace/hf_models/Qwen3-4B}"
export MODEL_NAME=Qwen3-4B
export TRAIN_FILE="${TRAIN_FILE:-/workspace/tpr_experiments/shortfix/pelita875_hint_h2_arm64_train.parquet}"
export TEST_FILE="$TRAIN_FILE"
export TASK_CONFIG="${TASK_CONFIG:-/workspace/tpr_experiments/shortfix/task_config_claude_code_pelita_docker.yaml}"

export N=2
export N_GPUS_PER_NODE=8
export TRAIN_TP=2
export TRAIN_CP=4
export ROLLOUT_TP=2
export TRAIN_PROMPT_BSZ=1
export GEN_BATCH_SIZE=1
export PPO_MINI_BATCH_SIZE=1
export TOTAL_TRAINING_STEPS=1
export TOTAL_EPOCHS=1
export MAX_PROMPT_LENGTH=16384
export MAX_RESPONSE_LENGTH=8192
export ACTOR_RECOMPUTE=True
export ROLLOUT_GPU_MEMORY_UTILIZATION=0.45
export CONCURRENCY=8
export GATEWAY_COUNT=1
export NUM_AGENT_WORKERS=1
export CAPTURE_ACTOR=False
export TPR_TREE_POLICY=qwen_turn

# The strict hard-cap patcher does not match the UniAgent checkout on the
# current ARM64 host. Preserve the validated Full configuration, but do not
# misrepresent this as a strict session-level cap. Gateway trajectory
# materialization at 24576 tokens remains active.
export SESSION_HARD_CAP=False

# No global PID reuse or log overwrites across runs.
# Ignore inherited RUN_NAME/RUN_ROOT from a previous Full/TPR run. Only
# REBENCH_RUN_NAME / REBENCH_RUN_ROOT are explicit overrides for this wrapper.
# Always create a fresh mode-specific output directory by default.
RUN_NAME="${REBENCH_RUN_NAME:-rebench-4b-16k8k-${MODE}-tp2cp4-n2-step1-$(date +%Y%m%d_%H%M%S)-$$}"
RUN_ROOT="${REBENCH_RUN_ROOT:-/workspace/tpr_experiments/e2e/$RUN_NAME}"
export RUN_NAME RUN_ROOT
if [[ -e "$RUN_ROOT/launcher.log" ]]; then
  echo "ERROR: refusing to overwrite existing log: $RUN_ROOT/launcher.log" >&2; exit 2
fi

[[ -f "$TRAIN_FILE" ]] || { echo "ERROR: missing TRAIN_FILE=$TRAIN_FILE" >&2; exit 2; }
[[ -f "$TASK_CONFIG" ]] || { echo "ERROR: missing TASK_CONFIG=$TASK_CONFIG" >&2; exit 2; }
[[ -f "$MODEL_PATH/config.json" ]] || { echo "ERROR: missing model config at $MODEL_PATH" >&2; exit 2; }
[[ -f "$ROOT/uni_agent/framework/entry.py" ]] || { echo "ERROR: UniAgent source missing: $ROOT" >&2; exit 2; }

# Check local sandbox name and avoid accidental online dependency installation.
# General datasets are supported by setting TRAIN_FILE; this wrapper expects
# the single tested Pelita instance to make the Full/TPR comparison interpretable.
TRAIN_FILE="$TRAIN_FILE" python - <<'PY'
import json, os
import pandas as pd
df = pd.read_parquet(os.environ['TRAIN_FILE'], columns=['extra_info'])
if len(df) != 1:
    raise SystemExit(f'ERROR: expected one prompt row (got {len(df)}); use the Pelita smoke dataset')
extra = df.iloc[0]['extra_info']
if isinstance(extra, str):
    extra = json.loads(extra)
task = extra['tools_kwargs']['task']
meta = task['metadata']
image = (task.get('sandbox') or {}).get('image')
if meta.get('instance_id') != 'ASPP__pelita-875':
    raise SystemExit(f'ERROR: expected ASPP__pelita-875, got {meta.get("instance_id")!r}')
if image != 'uniagent-swe-pelita-875:latest':
    raise SystemExit(f'ERROR: local ARM64 Sandbox must be uniagent-swe-pelita-875:latest, got {image!r}')
if str(meta.get('install') or '').strip() or meta.get('eval_commands'):
    raise SystemExit('ERROR: evaluation setup is nonempty; use noinstall dataset')
print('DATASET_OK: Pelita-875, ARM64 Docker sandbox, preinstalled reward dependencies')
PY

if [[ "$ASCEND_RT_VISIBLE_DEVICES" != "0,1,2,3,4,5,6,7" ]]; then
    echo "WARNING: physical NPU mask is $ASCEND_RT_VISIBLE_DEVICES; this 8-rank recipe requires 8 healthy visible devices."
fi
echo "================================================================"
echo "UniAgent SWE-reBench 4B Full/TPR 1-step training example"
echo "Mode=$MODE | NPU=$ASCEND_RT_VISIBLE_DEVICES | TP2 CP4 DP1"
echo "Model=$MODEL_PATH | Dataset=$TRAIN_FILE"
echo "Prompt=16384 Response=8192 | Rollout TP2 n=2 concurrency_cap=8 (at most 2 sessions from n=2)"
echo "Recompute=True | param/optimizer CPU offload=True | step=1"
echo "Session hard cap disabled (patcher not compatible); trajectory cap=24576"
echo "Output=$RUN_ROOT"
echo "================================================================"

case "${CHECK_ONLY:-0}" in
  1|true|True) export PRECHECK_ONLY=True ;;
  0|false|False) export PRECHECK_ONLY=False ;;
  *) echo "ERROR: CHECK_ONLY must be 0/1" >&2; exit 2 ;;
esac

# Canonical launcher stops LOCAL Ray before starting training; run on a
# dedicated container and do not run Full and TPR concurrently.
mkdir -p "$RUN_ROOT"
set +e
bash "$E2E" 2>&1 | tee "$RUN_ROOT/launcher.log"
rc="${PIPESTATUS[0]}"
set -e
printf '%s\n' "$rc" > "$RUN_ROOT/exit_code.txt"
if (( rc == 0 )); then
  echo "RUN_COMPLETED rc=0 MODE=$MODE LOG=$RUN_ROOT/launcher.log"
  echo "Check training/global_step=1, actor/loss, actor/grad_norm and timing_s/update_actor in the log."
else
  echo "RUN_FAILED rc=$rc MODE=$MODE LOG=$RUN_ROOT/launcher.log" >&2
fi
exit "$rc"

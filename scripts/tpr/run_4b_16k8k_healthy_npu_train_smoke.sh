#!/usr/bin/env bash
# Qwen3-4B / ARM64 Ascend 32GiB: one real Full PPO update, 16K prompt + 8K response.
# Defaults to the four healthy physical NPUs 0,2,4,6 (excludes Alarm 3/Critical 5).
# All-healthy host: NPU_IDS=0,1,2,3,4,5,6,7 bash this_script.sh
# Non-destructive precheck: CHECK_ONLY=1 bash this_script.sh
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
CHECKOUT="$(cd -- "$HERE/../.." && pwd -P)"
LAUNCHER="$CHECKOUT/tests/models/mcore/tpr/integration/run_uniagent_rebench_ppo_e2e.sh"
[[ -f "$LAUNCHER" ]] || { echo "ERROR: existing E2E launcher missing: $LAUNCHER" >&2; exit 2; }

# Physical device IDs (not local device ordinals).
NPU_IDS="${NPU_IDS:-0,2,4,6}"
IFS=, read -r -a SELECTED <<< "$NPU_IDS"
COUNT="${#SELECTED[@]}"
case "$COUNT" in
  4) TRAIN_CP=2 ;;
  8) TRAIN_CP=4 ;;
  *) echo "ERROR: require 4 or 8 healthy NPUs (got $NPU_IDS)" >&2; exit 2 ;;
esac
seen=","
for id in "${SELECTED[@]}"; do
  [[ "$id" =~ ^[0-7]$ ]] || { echo "ERROR: invalid NPU ID: $id" >&2; exit 2; }
  [[ "$seen" != *",$id,"* ]] || { echo "ERROR: duplicated NPU ID: $id" >&2; exit 2; }
  seen+="$id,"
done
[[ -z "${RAY_ADDRESS:-}" ]] || { echo "ERROR: RAY_ADDRESS is set; use a standalone node" >&2; exit 2; }

# Both variables are required on Ascend Ray/vLLM to keep the selected device mask.
export ASCEND_RT_VISIBLE_DEVICES="$NPU_IDS"
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
export RAY_DEDUP_LOGS=0 HYDRA_FULL_ERROR=1

# Use the real UniAgent -> TransferQueue -> Megatron Actor update E2E launcher.
export ROOT="${ROOT:-/workspace/uni-agent}"
export TPR_VERL_ROOT="${TPR_VERL_ROOT:-$CHECKOUT}"
export MODEL_PATH="${MODEL_PATH:-/workspace/hf_models/Qwen3-4B}"
export TRAIN_FILE="${TRAIN_FILE:-/workspace/tpr_experiments/shortfix/pelita875_hint_h2_noinstall.parquet}"
export TEST_FILE="$TRAIN_FILE"
export TASK_CONFIG="${TASK_CONFIG:-/workspace/tpr_experiments/shortfix/task_config_claude_code_pelita_docker.yaml}"
export TPR_ENABLED=False
export N_GPUS_PER_NODE="$COUNT" TRAIN_TP=2 TRAIN_CP ROLLOUT_TP=2
export MAX_PROMPT_LENGTH=16384 MAX_RESPONSE_LENGTH=8192
export TRAIN_PROMPT_BSZ=1 GEN_BATCH_SIZE=1 PPO_MINI_BATCH_SIZE=1
export TOTAL_TRAINING_STEPS=1 TOTAL_EPOCHS=1
export N="${N:-2}"
export ACTOR_RECOMPUTE=True
export ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.45}"
export CONCURRENCY="${CONCURRENCY:-2}" GATEWAY_COUNT=1 NUM_AGENT_WORKERS=1
export CAPTURE_ACTOR=False SESSION_HARD_CAP=True
export PRECHECK_ONLY="${CHECK_ONLY:-False}"

export RUN_NAME="${RUN_NAME:-rebench-4b-p16k-r8k-full-tp2cp${TRAIN_CP}-n${N}-step1-$(date +%Y%m%d_%H%M%S)-$$}"
export RUN_ROOT="${RUN_ROOT:-/workspace/tpr_experiments/e2e/$RUN_NAME}"

for f in "$MODEL_PATH/config.json" "$TRAIN_FILE" "$TASK_CONFIG" \
  "$ROOT/uni_agent/framework/entry.py" \
  "$TPR_VERL_ROOT/verl/workers/engine/megatron/tpr/adapter.py"; do
  [[ -f "$f" ]] || { echo "ERROR: missing required file: $f" >&2; exit 2; }
done

command -v docker >/dev/null || { echo "ERROR: main container needs Docker CLI/socket" >&2; exit 2; }
docker image inspect uniagent-swe-pelita-875:latest >/dev/null 2>&1 || {
  echo "ERROR: local ARM64 sandbox image missing" >&2; exit 2;
}
platform="$(docker image inspect uniagent-swe-pelita-875:latest --format '{{.Os}}/{{.Architecture}}')"
[[ "$platform" == linux/arm64 ]] || { echo "ERROR: invalid sandbox architecture $platform" >&2; exit 2; }

# A single known Pelita-875 row, preinstalled eval dependencies, no online pip install.
python - "$TRAIN_FILE" <<'PY'
import json, sys
import pandas as pd
df = pd.read_parquet(sys.argv[1], columns=['extra_info'])
if len(df) != 1:
    raise SystemExit(f'ERROR: expected 1 Pelita task, got {len(df)}')
extra = df.iloc[0]['extra_info']
if isinstance(extra, str):
    extra = json.loads(extra)
task = extra['tools_kwargs']['task']
meta = task['metadata']
image = (task.get('sandbox') or {}).get('image')
if meta.get('instance_id') != 'ASPP__pelita-875':
    raise SystemExit(f'ERROR: unexpected instance: {meta.get("instance_id")}')
if image != 'uniagent-swe-pelita-875:latest':
    raise SystemExit(f'ERROR: dataset requires {image!r}, expected uniagent-swe-pelita-875:latest')
if str(meta.get('install') or '').strip() or meta.get('eval_commands'):
    raise SystemExit('ERROR: install/eval_commands not empty; use noinstall parquet')
print('DATA_OK: Pelita-875, local ARM64 sandbox, offline Reward')
PY

if [[ -f /workspace/.venvs/uniagent-cc/bin/activate ]]; then
  source /workspace/.venvs/uniagent-cc/bin/activate
fi

python - "$COUNT" <<'PY'
import sys
import torch
import torch_npu
expected = int(sys.argv[1])
actual = torch.npu.device_count()
print(f'ASCEND_VISIBLE_NPUS expected={expected} actual={actual}')
if actual != expected:
    raise SystemExit('ERROR: device visibility mismatch. Check Docker NPU mounts and ASCEND_RT_VISIBLE_DEVICES.')
# Do not kill a card-occupying script automatically; refuse to start if cards are busy.
for i in range(actual):
    try:
        free, total = torch.npu.mem_get_info(i)
    except Exception as exc:
        print(f'WARN: logical NPU {i} memory query failed: {exc}')
        continue
    print(f'NPU logical={i}: free={free/2**30:.1f} GiB, total={total/2**30:.1f} GiB')
    if free < 16 * 2**30:
        raise SystemExit(f'ERROR: logical NPU {i} has <16 GiB free; release its occupancy reservation first.')
PY

echo "=========================================================="
echo "One-step Qwen3-4B Full train: 16384+8192 tokens"
echo "Physical NPUs: $NPU_IDS | TP2 CP$TRAIN_CP DP1 | rollout TP2 n=$N"
echo "BF16, param/optimizer offload, full activation recompute"
echo "Model: $MODEL_PATH"
echo "Data : $TRAIN_FILE"
echo "Precheck: $PRECHECK_ONLY"
echo "Output: $RUN_ROOT"
echo "=========================================================="
echo "NOTE: real mode calls 'ray stop --force'; run in a dedicated training container."
mkdir -p "$RUN_ROOT"
set +e
bash "$LAUNCHER" 2>&1 | tee "$RUN_ROOT/launcher.log"
rc="${PIPESTATUS[0]}"
set -e
printf '%s\n' "$rc" > "$RUN_ROOT/exit_code.txt"
if (( rc != 0 )); then
  echo "FAIL (rc=$rc): $RUN_ROOT/launcher.log" >&2
else
  echo "Launcher exited 0. Verify actual Actor optimizer/global_step in:"
  echo "  $RUN_ROOT/launcher.log"
fi
exit "$rc"

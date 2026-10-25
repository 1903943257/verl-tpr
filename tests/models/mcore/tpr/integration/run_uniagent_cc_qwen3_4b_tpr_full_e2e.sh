#!/usr/bin/env bash
# Run two genuine online Uni-Agent Actor PPO experiments from the same HF
# checkpoint: native Full, then TPR. Each mode performs real rollout + updates.
# Default is 10 steps per mode; set TOTAL_TRAINING_STEPS=20 for a longer run.
set -euo pipefail

ROOT=/workspace/uni-agent-cc
ACTIVE_VERL_ROOT="${ROOT}/verl"
SCRIPT_VERL_ROOT="${VERL_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)}"
ONLINE_SCRIPT="${SCRIPT_VERL_ROOT}/tests/models/mcore/tpr/integration/run_uniagent_cc_qwen17_swe_tpr_ppo_cp2.sh"
MODEL_PATH="${MODEL_PATH:-/workspace/hf_models/Qwen3-4B-128K}"
TRAIN_FILE=/workspace/data/uniagent-cc/django_11163_smoke_py311.parquet
TASK_CONFIG="${ROOT}/examples/claude_code_swe_task/task_config_claude_code_django11163.yaml"
STEPS="${TOTAL_TRAINING_STEPS:-10}"

case "${STEPS}" in
  10|20) ;;
  *) echo "ERROR: TOTAL_TRAINING_STEPS must be 10 or 20, got ${STEPS}" >&2; exit 2 ;;
esac

RUN_ROOT="${RUN_ROOT:-/workspace/tpr_full_e2e_${STEPS}step_$(date +%Y%m%d_%H%M%S)}"
if [[ -e "${RUN_ROOT}" ]]; then
  echo "ERROR: refusing to overwrite existing run directory: ${RUN_ROOT}" >&2
  exit 2
fi
mkdir -p "${RUN_ROOT}"/{logs,full,tpr}
printf '%s\n' "${RUN_ROOT}" > /workspace/tpr_full_e2e_latest.txt

echo "TPR_FULL_E2E_RUN_ROOT=${RUN_ROOT}"
echo "STEPS_PER_MODE=${STEPS}"
echo "MODEL_PATH=${MODEL_PATH}"
echo "ACTIVE_VERL_ROOT=${ACTIVE_VERL_ROOT}"
echo "SCRIPT_VERL_ROOT=${SCRIPT_VERL_ROOT}"

# The launcher uses ROOT/verl on PYTHONPATH. Refuse to benchmark a script from
# one checkout against a different runtime checkout.
for required in \
  "${ONLINE_SCRIPT}" \
  "${MODEL_PATH}/config.json" \
  "${TRAIN_FILE}" \
  "${TASK_CONFIG}" \
  "${ACTIVE_VERL_ROOT}/verl/workers/config/engine.py" \
  "${ACTIVE_VERL_ROOT}/verl/workers/engine/megatron/transformer_impl.py" \
  "${ACTIVE_VERL_ROOT}/verl/workers/engine/megatron/tpr/adapter.py" \
  "${ROOT}/verl/verl/workers/engine_workers.py"; do
  if [[ ! -e "${required}" ]]; then
    echo "ERROR: required input missing: ${required}" >&2
    exit 2
  fi
done

if [[ "${SCRIPT_VERL_ROOT}" != "${ACTIVE_VERL_ROOT}" ]]; then
  for rel in \
    tests/models/mcore/tpr/integration/run_uniagent_cc_qwen17_swe_tpr_ppo_cp2.sh \
    verl/workers/config/engine.py \
    verl/workers/engine/megatron/transformer_impl.py \
    verl/workers/engine/megatron/tpr/adapter.py; do
    if ! cmp -s "${SCRIPT_VERL_ROOT}/${rel}" "${ACTIVE_VERL_ROOT}/${rel}"; then
      echo "ERROR: script checkout and runtime checkout differ at ${rel}." >&2
      echo "Sync this branch into ${ACTIVE_VERL_ROOT} before running; refusing a mixed-code experiment." >&2
      exit 2
    fi
  done
fi

# Validate the real runtime route before spending hours on rollouts.
grep -Fq 'tpr_enabled: bool' "${ACTIVE_VERL_ROOT}/verl/workers/config/engine.py" || {
  echo "ERROR: active McoreEngineConfig lacks tpr_enabled." >&2; exit 2;
}
grep -Fq 'run_tpr_forward_backward_batch' "${ACTIVE_VERL_ROOT}/verl/workers/engine/megatron/transformer_impl.py" || {
  echo "ERROR: active Megatron Engine lacks the TPR PPO route." >&2; exit 2;
}
grep -Fq 'TPR PPO supports TP1 CP1/2 DP1, TP2 CP1/2/4 DP1' "${ACTIVE_VERL_ROOT}/verl/workers/engine/megatron/tpr/adapter.py" || {
  echo "ERROR: active TPR adapter does not support the expected TP2/CP4/DP1 topology." >&2; exit 2;
}
if ! grep -Fq 'TPR P0 real actor-update capture' "${ROOT}/verl/verl/workers/engine_workers.py"; then
  echo "ERROR: actual TrainingWorker actor-update capture hook is missing." >&2
  echo "Apply/check the Bridge actor-capture patch before this run." >&2
  exit 2
fi
if [[ ! -f "${SCRIPT_VERL_ROOT}/patches/apply_uniagent_session_hardcap.py" \
   && ! -f /workspace/bridge/patches/apply_uniagent_session_hardcap.py ]]; then
  echo "ERROR: session hard-cap patcher is missing from both VERL and /workspace/bridge." >&2
  exit 2
fi

# Keep the two modes on the same 8-NPU mesh. TP2 x CP4 x DP1 is the supported
# online TPR PPO topology; using 16 NPUs would produce gated DP2.
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-8,9,10,11,12,13,14,15}"
export PYTHONUNBUFFERED=1
export N=8
export TOTAL_TRAINING_STEPS="${STEPS}"
export TOTAL_EPOCHS="${STEPS}"
export N_GPUS_PER_NODE=8
export TRAIN_PROMPT_BSZ=1
export GEN_BATCH_SIZE=1
export PPO_MINI_BATCH_SIZE=1
export SESSION_HARD_CAP=True
export NATIVE_RECOMPUTE=False
export MODEL_PATH
export TPR_TREE_POLICY=qwen_turn
export TPR_TURN_MIN_SEGMENT_TOKENS=128
# Disable detailed per-segment profiling for fair wall-clock timing.
export TPR_EXEC_PROFILE=0

run_mode() {
  local mode="$1"
  local tpr_enabled="$2"
  local start end rc
  export TPR_ENABLED="${tpr_enabled}"
  export RUN_NAME="$(basename "${RUN_ROOT}")-${mode}"
  export TQ_DUMP_DIR="${RUN_ROOT}/${mode}/tq_dump"
  export TPR_CAPTURE_ACTOR_MINIBATCH_DIR="${RUN_ROOT}/${mode}/actor_capture"
  mkdir -p "${TQ_DUMP_DIR}" "${TPR_CAPTURE_ACTOR_MINIBATCH_DIR}"

  local log="${RUN_ROOT}/logs/${mode}.log"
  echo
  echo "============================================================"
  echo "START mode=${mode} TPR_ENABLED=${TPR_ENABLED} steps=${STEPS}"
  echo "RUN_NAME=${RUN_NAME}"
  echo "LOG=${log}"
  echo "============================================================"
  start=$(date +%s)
  rc=0
  bash "${ONLINE_SCRIPT}" 2>&1 | tee "${log}" || rc=$?
  end=$(date +%s)
  printf '%s\n' "$((end - start))" > "${RUN_ROOT}/${mode}_wall_seconds.txt"
  printf '%s\n' "${rc}" > "${RUN_ROOT}/${mode}_exit_code.txt"
  echo "END mode=${mode} exit=${rc} wall_seconds=$((end - start))"
  ray stop --force > "${RUN_ROOT}/logs/${mode}_ray_cleanup.log" 2>&1 || true
  return "${rc}"
}

full_rc=0
run_mode full False || full_rc=$?
tpr_rc=0
run_mode tpr True || tpr_rc=$?

echo
echo "========== Full / TPR timing summary =========="
python - "${RUN_ROOT}" "${STEPS}" "${full_rc}" "${tpr_rc}" <<'PY'
import json
import re
import sys
from pathlib import Path
from statistics import mean

root = Path(sys.argv[1])
steps = int(sys.argv[2])
full_rc, tpr_rc = int(sys.argv[3]), int(sys.argv[4])

def wall(mode):
    p = root / f"{mode}_wall_seconds.txt"
    return int(p.read_text().strip()) if p.exists() else None

def actor_update_times(mode):
    p = root / "logs" / f"{mode}.log"
    if not p.exists():
        return []
    text = p.read_text(errors="replace")
    pattern = re.compile(
        r"""timing_s/update_actor['"]?\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)"""
    )
    return [float(m.group(1)) for m in pattern.finditer(text)]

full_wall, tpr_wall = wall("full"), wall("tpr")
full_actor, tpr_actor = actor_update_times("full"), actor_update_times("tpr")
e2e_speedup = full_wall / tpr_wall if full_wall and tpr_wall else None
full_actor_mean = mean(full_actor) if full_actor else None
tpr_actor_mean = mean(tpr_actor) if tpr_actor else None
training_speedup = (
    full_actor_mean / tpr_actor_mean
    if full_actor_mean is not None and tpr_actor_mean else None
)

summary = {
    "run_root": str(root),
    "steps_per_mode": steps,
    "full_exit_code": full_rc,
    "tpr_exit_code": tpr_rc,
    "full_e2e_wall_seconds": full_wall,
    "tpr_e2e_wall_seconds": tpr_wall,
    "e2e_speedup_full_over_tpr": e2e_speedup,
    "full_actor_update_seconds": full_actor,
    "tpr_actor_update_seconds": tpr_actor,
    "full_actor_update_mean_seconds": full_actor_mean,
    "tpr_actor_update_mean_seconds": tpr_actor_mean,
    "training_actor_update_speedup_full_over_tpr": training_speedup,
}
(root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
if e2e_speedup is not None:
    print(f"E2E_SPEEDUP_FULL_OVER_TPR={e2e_speedup:.4f}x")
else:
    print("E2E_SPEEDUP_FULL_OVER_TPR=unavailable")
if training_speedup is not None:
    print(f"TRAINING_UPDATE_SPEEDUP_FULL_OVER_TPR={training_speedup:.4f}x")
else:
    print("TRAINING_UPDATE_SPEEDUP_FULL_OVER_TPR=unavailable")
PY

echo "Summary: ${RUN_ROOT}/summary.json"
echo "Full log: ${RUN_ROOT}/logs/full.log"
echo "TPR log:  ${RUN_ROOT}/logs/tpr.log"
echo "Agent logs: /workspace/logs/$(basename "${RUN_ROOT}")-full/agent and ...-tpr/agent"

if [[ "${full_rc}" != "0" || "${tpr_rc}" != "0" ]]; then
  echo "TPR_FULL_E2E_FAILED full_exit=${full_rc} tpr_exit=${tpr_rc}; inspect per-mode logs." >&2
  exit 1
fi
echo "TPR_FULL_E2E_PASS full_exit=0 tpr_exit=0"

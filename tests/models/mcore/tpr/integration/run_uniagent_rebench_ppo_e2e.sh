#!/usr/bin/env bash
set -euo pipefail
# Resolve this script while invoked from the TPR checkout, before any cd to ROOT.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
TPR_CHECKOUT="$(cd -- "${SCRIPT_DIR}/../../../../.." && pwd -P)"
# SWE-reBench full E2E: UniAgent Claude Code -> vLLM -> TQ -> GRPO/PPO -> Actor update.
# Forked from run_uniagent_cc_qwen17_swe_tpr_ppo_cp2.sh.
# A/B runs use the same data and settings; TPR_ENABLED alone selects the actor path.
# Full and TPR are independent online rollouts (not identical-trajectory replays).

ROOT=${ROOT:-/workspace/uni-agent}
# TPR checkout is the VERL runtime; UniAgent's upstream VERL submodule is not used.
TPR_VERL_ROOT=${TPR_VERL_ROOT:-${TPR_CHECKOUT}}
MODEL_PATH=${MODEL_PATH:-/workspace/hf_models/Qwen3-4B-128K}
MODEL_NAME=${MODEL_NAME:-Qwen3-4B}
# Use the normalized ReBench Parquet, NOT the legacy reward/env-layout input.
TRAIN_FILE=${TRAIN_FILE:-/workspace/tpr_experiments/shortfix/shortfix_candidates_task.parquet}
TEST_FILE=${TEST_FILE:-${TRAIN_FILE}}
TASK_CONFIG=${TASK_CONFIG:-${TPR_CHECKOUT}/examples/tpr/task_config_swe_rebench_docker.yaml}
TPR_ENABLED=${TPR_ENABLED:-False}
export TPR_TREE_POLICY=${TPR_TREE_POLICY:-qwen_turn}

# ------------------------------------------------------------
# Same rollout behavior as the original UniAgent launcher
# ------------------------------------------------------------
N=${N:-8}
TEMPERATURE=0.6
TOP_P=0.95
TOP_K=20
TOOL_PARSER=hermes
# Default session budget is prompt 16K + response 8K = 24K tokens.
# Gateway trajectory materialization uses rollout.prompt_length + response_length.
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-16384}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-8192}
MAX_MODEL_LEN=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))
ROLLOUT_TP=${ROLLOUT_TP:-2}
ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.60}
GATEWAY_COUNT=${GATEWAY_COUNT:-1}
CONCURRENCY=${CONCURRENCY:-8}
# 一个 prompt group 即可。
NUM_AGENT_WORKERS=${NUM_AGENT_WORKERS:-1}
# Session-level hard stop prevents split materializations; keep ALL to avoid
# silently choosing the wrong member of a group if a regression occurs.
TRAJECTORY_SELECTION=${TRAJECTORY_SELECTION:-all}
SESSION_HARD_CAP=${SESSION_HARD_CAP:-True}
MASK_UNFINISHED_EPISODE=False
# ------------------------------------------------------------
# Training side: production Megatron Full/TPR with TP2 x CP2 by default
# ------------------------------------------------------------
NNODES=1
N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-8}
TRAIN_TP=${TRAIN_TP:-2}
TRAIN_PP=1
TRAIN_CP=${TRAIN_CP:-2}
# VERL's native MCore default is SP=True for TP>1. Keep it explicit in E2E.
TRAIN_SP=${TRAIN_SP:-True}
case "${TRAIN_SP}" in
  True|true|1) TRAIN_SP=True ;;
  False|false|0) TRAIN_SP=False ;;
  *) echo "TRAIN_SP must be True or False" >&2; exit 2 ;;
esac
# SP is passed explicitly to both Engine and TransformerConfig below.
# _validate_tpr_parallel_sizes remains the source of truth for supported DP
# layouts: current PPO support includes TP2 x CP2 x DP2, but not arbitrary DP>1.
TPR_CP_BACKEND=${TPR_CP_BACKEND:-ring}
TRAIN_DP=$((N_GPUS_PER_NODE / (TRAIN_TP * TRAIN_PP * TRAIN_CP)))
TRAIN_PROMPT_BSZ=${TRAIN_PROMPT_BSZ:-$TRAIN_DP}
GEN_BATCH_SIZE=${GEN_BATCH_SIZE:-$TRAIN_DP}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-$TRAIN_DP}
PPO_MICRO_BATCH_SIZE_PER_GPU=1
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=1
# Keep the established MBridge/MindSpeed integration from the reference script.
USE_MBRIDGE=True
VANILLA_MBRIDGE=True
PARAM_OFFLOAD=True
OPTIMIZER_OFFLOAD=True
# Capture an authentic actor update first; do not pretend zero-advantage means PPO parity.
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-1}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-${TOTAL_TRAINING_STEPS}}
# E1 Native only: opt-in full activation recompute to reduce online OOM.
# Offline Native/TPR Replay must disable recompute independently.
ACTOR_RECOMPUTE=${ACTOR_RECOMPUTE:-True}
RECOMPUTE_ARGS=()
case "${ACTOR_RECOMPUTE}" in
  True|true|1)
    RECOMPUTE_ARGS=(
      "++actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full"
      "++actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform"
      "++actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1"
    )
    ;;
  False|false|0) ;;
  *) echo "ACTOR_RECOMPUTE must be True or False" >&2; exit 2 ;;
esac
# 不做 validation/checkpoint。
SAVE_FREQ=-1
TEST_FREQ=-1
# 不预取额外 prompt。
NUM_WARMUP_BATCHES=0
# ------------------------------------------------------------
# Output
# ------------------------------------------------------------
RUN_NAME=${RUN_NAME:-rebench-${TPR_ENABLED,,}-n${N}-tp${TRAIN_TP}cp${TRAIN_CP}dp${TRAIN_DP}-step${TOTAL_TRAINING_STEPS}-$(date +%Y%m%d_%H%M%S)}
RUN_ROOT=${RUN_ROOT:-/workspace/tpr_experiments/e2e/${RUN_NAME}}
AGENT_LOG_DIR=${RUN_ROOT}/agent
TRAIN_LOG_DIR=${RUN_ROOT}/logs
CKPTS_DIR=${RUN_ROOT}/ckpts
CAPTURE_ACTOR=${CAPTURE_ACTOR:-False}
TPR_CAPTURE_ACTOR_MINIBATCH_DIR=${TPR_CAPTURE_ACTOR_MINIBATCH_DIR:-${RUN_ROOT}/actor_captures}
# ============================================================
# Environment
# ============================================================
[[ -f "${ROOT}/uni_agent/framework/entry.py" ]] || {
    echo "ERROR: UniAgent checkout missing: ${ROOT}" >&2; exit 2;
}
[[ -f "${TPR_VERL_ROOT}/verl/workers/engine/megatron/tpr/adapter.py" ]] || {
    echo "ERROR: TPR VERL runtime missing: ${TPR_VERL_ROOT}" >&2; exit 2;
}
# Python -m puts cwd first on sys.path; run from the TPR fork rather than
# ROOT, otherwise ROOT/verl (upstream submodule) can silently shadow TPR.
cd "${TPR_VERL_ROOT}"
if [ -f /workspace/.venvs/uniagent-cc/bin/activate ]; then
    source /workspace/.venvs/uniagent-cc/bin/activate
fi
# Optional exact instance selection from the already-normalized ReBench Parquet.
# For TP2/CP2/DP2 choose two IDs; for one task use DP1 (4x CP2 or 8x CP4).
if [[ -n "${REBENCH_INSTANCE_IDS:-}" ]]; then
    SOURCE_TRAIN_FILE="${TRAIN_FILE}"
    mkdir -p "${RUN_ROOT}"
    TRAIN_FILE="${RUN_ROOT}/selected_rebench_tasks.parquet"
    SOURCE_TRAIN_FILE="${SOURCE_TRAIN_FILE}" TRAIN_FILE="${TRAIN_FILE}" \
      REBENCH_INSTANCE_IDS="${REBENCH_INSTANCE_IDS}" python - <<'PY_SELECT'
import json, os
from pathlib import Path
import pandas as pd
src = Path(os.environ["SOURCE_TRAIN_FILE"])
dest = Path(os.environ["TRAIN_FILE"])
ids = [s.strip() for s in os.environ["REBENCH_INSTANCE_IDS"].split(",") if s.strip()]
if not ids or len(set(ids)) != len(ids):
    raise SystemExit("ERROR: REBENCH_INSTANCE_IDS must contain distinct nonempty IDs")
if dest.exists():
    raise SystemExit(f"ERROR: selected dataset already exists: {dest}; choose a new RUN_NAME")
df = pd.read_parquet(src)
def instance_id(extra):
    if isinstance(extra, str):
        extra = json.loads(extra)
    return (((extra or {}).get("tools_kwargs") or {}).get("task") or {}).get(
        "metadata", {}
    ).get("instance_id")
instances = df["extra_info"].map(instance_id)
selected = []
for instance in ids:
    hit = df.loc[instances == instance]
    if len(hit) != 1:
        raise SystemExit(f"ERROR: instance {instance!r} has {len(hit)} matches in {src} (expected 1)")
    selected.append(hit)
pd.concat(selected, ignore_index=True).to_parquet(dest, index=False)
print(f"REBENCH_SELECTED ids={ids} file={dest}")
PY_SELECT
    if [[ "${TEST_FILE}" == "${SOURCE_TRAIN_FILE}" ]]; then TEST_FILE="${TRAIN_FILE}"; fi
fi
# Uni-Agent upstream implements per-trajectory materialization, NOT a hard
# session token stop. Optional installer is vendored in this E2E checkout.
if [[ "${SESSION_HARD_CAP}" == "True" || "${SESSION_HARD_CAP}" == "true" || "${SESSION_HARD_CAP}" == "1" ]]; then
    SESSION_PATCHER="${UNIAGENT_SESSION_PATCHER:-${TPR_CHECKOUT}/patches/apply_uniagent_session_hardcap.py}"
    if [[ ! -f "${SESSION_PATCHER}" ]]; then
        echo "ERROR: session-cap patcher not found: ${SESSION_PATCHER}; set UNIAGENT_SESSION_PATCHER" >&2
        exit 2
    fi
# During PRECHECK_ONLY don't mutate UniAgent; only validate patch compatibility.
    if [[ "${PRECHECK_ONLY:-False}" == "True" || "${PRECHECK_ONLY:-False}" == "true" || "${PRECHECK_ONLY:-False}" == "1" ]]; then
        python "${SESSION_PATCHER}" --check "${ROOT}" || exit 2
    else
        python "${SESSION_PATCHER}" "${ROOT}" || exit 2
    fi
fi
# IMPORTANT:
# 不加入 /workspace/Megatron-Bridge/src。
# 本次明确走 vanilla mbridge。
VLLM_ASCEND_SRC=""
if [[ -d /workspace/vllm-ascend ]]; then
    VLLM_ASCEND_SRC=/workspace/vllm-ascend:
elif [[ -d /vllm-ascend ]]; then
    VLLM_ASCEND_SRC=/vllm-ascend:
fi
export PYTHONPATH="${TPR_VERL_ROOT}:${ROOT}:${VLLM_ASCEND_SRC}${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export VLLM_USE_V1=1
export TASK_QUEUE_ENABLE=1
export VLLM_ASCEND_ENABLE_TOPK_OPTIMIZE=1
export TQ_SIMPLE_STORAGE_SEND_RECV_TIMEOUT=2400
# Capture only for the prefix/suffix screening pass, not timed E2E benchmarking:
# torch.save on every actor minibatch affects wall time. Requires the patched worker.
unset TPR_REAL_ACTOR_CAPTURE || true
case "${CAPTURE_ACTOR}" in
  True|true|1)
    export TPR_CAPTURE_ACTOR_MINIBATCH_DIR
    mkdir -p "${TPR_CAPTURE_ACTOR_MINIBATCH_DIR}"
    ;;
  False|false|0) unset TPR_CAPTURE_ACTOR_MINIBATCH_DIR ;;
  *) echo "ERROR: CAPTURE_ACTOR must be True or False" >&2; exit 2 ;;
esac
mkdir -p "${AGENT_LOG_DIR}" "${TRAIN_LOG_DIR}" "${CKPTS_DIR}"
# ============================================================
# Preflight
# ============================================================
test -d "${MODEL_PATH}" || {
    echo "ERROR: model not found: ${MODEL_PATH}"
    exit 1
}
test -f "${TRAIN_FILE}" || {
    echo "ERROR: dataset not found: ${TRAIN_FILE}"
    exit 1
}
test -f "${TASK_CONFIG}" || {
    echo "ERROR: task config not found: ${TASK_CONFIG}" >&2
    exit 1
}
# Reject legacy reward/env Parquet before an expensive Ray/NPU launch.
TRAIN_FILE="${TRAIN_FILE}" TRAIN_PROMPT_BSZ="${TRAIN_PROMPT_BSZ}" python - <<'PY_DATA_CHECK'
import json, os
import pandas as pd

path = os.environ["TRAIN_FILE"]
df = pd.read_parquet(path, columns=["extra_info"])
bsz = int(os.environ["TRAIN_PROMPT_BSZ"])
if len(df) < bsz:
    raise SystemExit(f"ERROR: dataset {path} has {len(df)} rows, needs >= {bsz} for one train batch")
images = []
for idx, extra in enumerate(df["extra_info"]):
    if isinstance(extra, str):
        extra = json.loads(extra)
    if not isinstance(extra, dict):
        raise SystemExit(f"ERROR: extra_info[{idx}] is not a mapping")
    task = (extra.get("tools_kwargs") or {}).get("task")
    if not isinstance(task, dict) or task.get("name") != "swe_rebench":
        raise SystemExit(
            "ERROR: ReBench dataset needs tools_kwargs.task with name=swe_rebench. "
            "Run filter_swe_rebench_shortfix.py and use shortfix_candidates_task.parquet."
        )
    image = (task.get("sandbox") or {}).get("image")
    if not isinstance(image, str) or not image.strip() or image == "swe-local:3.12":
        raise SystemExit(f"ERROR: invalid/legacy ReBench sandbox image at row {idx}: {image!r}")
    if not (task.get("metadata") or {}).get("instance_id"):
        raise SystemExit(f"ERROR: missing task.metadata.instance_id at row {idx}")
    images.append(image)
print(f"REBENCH_DATASET rows={len(df)} train_batch_size={bsz} images={sorted(set(images))[:5]}")
PY_DATA_CHECK
# The official Qwen3-4B is native 32K / config 40960.
# Full 131072 tokens require a model config with YaRN enabled for BOTH
# rollout and actor, not just a larger max_model_len CLI override.
MODEL_PATH="${MODEL_PATH}" MAX_MODEL_LEN="${MAX_MODEL_LEN}" python - <<'PY_CTX_CHECK'
import json, os
from pathlib import Path
p = Path(os.environ["MODEL_PATH"]) / "config.json"
if not p.is_file():
    raise SystemExit(f"ERROR: no model config: {p}")
c = json.loads(p.read_text())
limit = int(os.environ["MAX_MODEL_LEN"])
rope = c.get("rope_scaling") or {}
if limit > 40960:
    if (rope.get("rope_type", rope.get("type")) != "yarn"
        or float(rope.get("factor", 0)) < 4.0):
        raise SystemExit(
            "ERROR: 131072-token Qwen3-4B needs YaRN in the model "
            "config.json (rope_type=yarn, factor>=4, "
            "original_max_position_embeddings=32768). "
            "Prepare a separate 128K config/checkpoint directory and set MODEL_PATH."
        )
print(f"LONG_CONTEXT_CHECK max_model_len={limit} rope_scaling={rope}")
PY_CTX_CHECK
if (( TRAIN_TP < 1 || TRAIN_CP < 1 || N_GPUS_PER_NODE < 1 || N_GPUS_PER_NODE % (TRAIN_TP * TRAIN_PP * TRAIN_CP) != 0 )); then
    echo "ERROR: N_GPUS_PER_NODE must be divisible by TP*PP*CP" >&2
    exit 2
fi
case "${TPR_ENABLED}" in
  True|true|1) TPR_ENABLED=True ;;
  False|false|0) TPR_ENABLED=False ;;
  *) echo "ERROR: TPR_ENABLED must be True or False" >&2; exit 2 ;;
esac
if (( TRAIN_PROMPT_BSZ % TRAIN_DP != 0 || GEN_BATCH_SIZE % TRAIN_DP != 0 || PPO_MINI_BATCH_SIZE % TRAIN_DP != 0 )); then
    echo "ERROR: train/gen/ppo mini batch sizes must be divisible by TRAIN_DP=${TRAIN_DP}." >&2
    exit 2
fi
if (( TRAIN_DP > 1 )); then
    echo "NOTE: DP${TRAIN_DP} will distribute actor mini-batches; verify per-DP sibling grouping and reward diversity." >&2
fi
if [[ "${CAPTURE_ACTOR}" == "True" || "${CAPTURE_ACTOR}" == "true" || "${CAPTURE_ACTOR}" == "1" ]]; then
    WORKER_FILE="${TPR_VERL_ROOT}/verl/workers/engine_workers.py"
    if [[ ! -f "${WORKER_FILE}" ]] || ! grep -Fq 'TPR P0 real actor-update capture' "${WORKER_FILE}"; then
        echo "ERROR: CAPTURE_ACTOR=True requires patched actor-update hook in ${WORKER_FILE}" >&2
        exit 2
    fi
    if find "${TPR_CAPTURE_ACTOR_MINIBATCH_DIR}" -maxdepth 1 -name 'actor_update_*.pt' | grep -q .; then
        echo "ERROR: Actor capture directory already has .pt snapshots; use a new RUN_NAME" >&2
        exit 2
    fi
fi
echo "============================================================"
echo "Uni-Agent TRAIN/TQ E2E"
echo "============================================================"
SOURCE_CHECKOUT="${TPR_CHECKOUT}"
echo "RUNNER_CHECKOUT   = ${SOURCE_CHECKOUT}"
echo "UNIAGENT_ROOT     = ${ROOT}"
echo "RUNTIME_VERL      = ${TPR_VERL_ROOT} (selected trainer Python package)"
echo "MODEL_PATH        = ${MODEL_PATH}"
echo "MODEL_NAME        = ${MODEL_NAME}"
echo "TPR_ENABLED       = ${TPR_ENABLED}"
echo "TRAIN_FILE        = ${TRAIN_FILE}"
echo "TASK_CONFIG       = ${TASK_CONFIG}"
echo
echo "SESSION_HARD_CAP  = ${SESSION_HARD_CAP}"
echo "N                 = ${N}"
echo "TEMPERATURE       = ${TEMPERATURE}"
echo "TOP_P             = ${TOP_P}"
echo "TOP_K             = ${TOP_K}"
echo "TOOL_PARSER       = ${TOOL_PARSER}"
echo
echo "MAX_PROMPT        = ${MAX_PROMPT_LENGTH}"
echo "MAX_RESPONSE      = ${MAX_RESPONSE_LENGTH}"
echo "MAX_MODEL_LEN     = ${MAX_MODEL_LEN}"
echo
echo "ROLLOUT_TP        = ${ROLLOUT_TP}"
echo "TRAIN_TP          = ${TRAIN_TP}"
echo "TRAIN_PP          = ${TRAIN_PP}"
echo "TRAIN_CP          = ${TRAIN_CP}"
echo "TRAIN_SP          = ${TRAIN_SP}"
echo "TPR_CP_BACKEND    = ${TPR_CP_BACKEND}"
echo
echo "NNODES            = ${NNODES}"
echo "N_GPUS_PER_NODE   = ${N_GPUS_PER_NODE}"
echo "TRAIN_DP          = ${TRAIN_DP}"
echo "TRAIN_PROMPT_BSZ  = ${TRAIN_PROMPT_BSZ}"
echo "GEN_BATCH_SIZE    = ${GEN_BATCH_SIZE}"
echo "PPO_MINI_BSZ      = ${PPO_MINI_BATCH_SIZE}"
echo "ACTOR_RECOMPUTE  = ${ACTOR_RECOMPUTE}"
echo "TOTAL_STEPS       = ${TOTAL_TRAINING_STEPS}"
echo "TOTAL_EPOCHS      = ${TOTAL_EPOCHS}"
echo
echo "USE_MBRIDGE       = ${USE_MBRIDGE}"
echo "VANILLA_MBRIDGE   = ${VANILLA_MBRIDGE}"
echo
echo "RUN_ROOT          = ${RUN_ROOT}"
echo "CAPTURE_ACTOR     = ${CAPTURE_ACTOR}"
echo "TPR_TREE_POLICY   = ${TPR_TREE_POLICY}"
echo "AGENT_LOG_DIR     = ${AGENT_LOG_DIR}"
echo "ACTOR_CAPTURE_DIR = ${TPR_CAPTURE_ACTOR_MINIBATCH_DIR:-disabled}"
echo "============================================================"
# Fail closed if Python is importing an unrelated VERL checkout.
TPR_VERL_ROOT="${TPR_VERL_ROOT}" ROOT="${ROOT}" python - <<'PY_TPR_IMPORT'
import importlib.util
import os
from pathlib import Path
for mod, expected in (("verl", "TPR_VERL_ROOT"), ("uni_agent", "ROOT")):
    spec = importlib.util.find_spec(mod)
    origin = getattr(spec, "origin", None)
    if origin is None:
        raise SystemExit(f"ERROR: {mod} is not importable as a regular package")
    actual = Path(origin).resolve()
    required = Path(os.environ[expected]).resolve()
    if not actual.is_relative_to(required):
        raise SystemExit(f"ERROR: {mod} resolved to {actual}, expected inside {required}")
    print(f"IMPORT_SOURCE_OK {mod}={actual}")
PY_TPR_IMPORT
python - <<'PY'
import importlib.util
import sys
import torch
print("python:", sys.executable)
print("torch:", torch.__version__)
for m in [
    "torch_npu",
    "verl",
    "uni_agent",
    "mbridge",
    "megatron.core",
    "megatron.bridge",
]:
    try:
        print(m, "=>", importlib.util.find_spec(m))
    except Exception as e:
        print(m, "=> ERROR:", repr(e))
PY
# PRECHECK_ONLY validates the new container's model, dataset and task configuration
# without killing its Ray runtime or allocating NPU resources.
case "${PRECHECK_ONLY:-False}" in
  True|true|1) echo "PRECHECK_ONLY PASS: no Ray/NPU jobs started"; exit 0 ;;
  False|false|0) ;;
  *) echo "ERROR: PRECHECK_ONLY must be True or False" >&2; exit 2 ;;
esac
# ============================================================
# Clean stale local Ray
# ============================================================
ray stop --force >/dev/null 2>&1 || true
# ============================================================
# Run normal VERL / Uni-Agent V1 training flow
# ============================================================
python -m verl.trainer.main_ppo \
    --config-name=ppo_megatron_trainer \
    trainer.use_v1=True \
    trainer.v1.trainer_mode=colocate_async \
    trainer.v1.colocate_async.num_warmup_batches="${NUM_WARMUP_BATCHES}" \
    \
    transfer_queue.enable=True \
    transfer_queue.metrics.enabled=True \
    \
    trainer.device=npu \
    trainer.nnodes="${NNODES}" \
    trainer.n_gpus_per_node="${N_GPUS_PER_NODE}" \
    \
    actor_rollout_ref.hybrid_engine=True \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=True \
    +actor_rollout_ref.model.override_config.model_config.max_position_embeddings="${MAX_MODEL_LEN}" \
    \
    "data.train_files=['${TRAIN_FILE}']" \
    "data.val_files=['${TEST_FILE}']" \
    data.prompt_key=prompt \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
    data.max_response_length="${MAX_RESPONSE_LENGTH}" \
    data.train_batch_size="${TRAIN_PROMPT_BSZ}" \
    data.gen_batch_size="${GEN_BATCH_SIZE}" \
    data.return_raw_chat=True \
    data.trust_remote_code=True \
    data.dataloader_num_workers=0 \
    \
    algorithm.adv_estimator=grpo \
    algorithm.filter_groups.enable=False \
    algorithm.use_kl_in_reward=False \
    \
    actor_rollout_ref.actor.optim.use_precision_aware_optimizer=True \
    actor_rollout_ref.actor.optim.main_grads_dtype=bf16 \
    +actor_rollout_ref.actor.optim.override_optimizer_config.use_precision_aware_optimizer=True \
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_cpu_offload=True \
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_offload_fraction=1.0 \
    +actor_rollout_ref.actor.optim.override_optimizer_config.overlap_cpu_optimizer_d2h_h2d=True \
    \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    actor_rollout_ref.actor.policy_loss.loss_mode=vanilla \
    actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE_PER_GPU}" \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${MAX_MODEL_LEN}" \
    actor_rollout_ref.actor.checkpoint.strict=False \
    \
    actor_rollout_ref.actor.megatron.use_mbridge="${USE_MBRIDGE}" \
    actor_rollout_ref.actor.megatron.vanilla_mbridge="${VANILLA_MBRIDGE}" \
    actor_rollout_ref.actor.megatron.use_dist_checkpointing=False \
    actor_rollout_ref.actor.megatron.param_offload="${PARAM_OFFLOAD}" \
    actor_rollout_ref.actor.megatron.optimizer_offload="${OPTIMIZER_OFFLOAD}" \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size="${TRAIN_TP}" \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size="${TRAIN_PP}" \
    actor_rollout_ref.actor.megatron.context_parallel_size="${TRAIN_CP}" \
    actor_rollout_ref.actor.megatron.sequence_parallel="${TRAIN_SP}" \
    ++actor_rollout_ref.actor.megatron.override_transformer_config.sequence_parallel="${TRAIN_SP}" \
    actor_rollout_ref.actor.megatron.use_remove_padding=True \
    "${RECOMPUTE_ARGS[@]}" \
    +actor_rollout_ref.actor.megatron.tpr_enabled="${TPR_ENABLED}" \
    +actor_rollout_ref.actor.megatron.tpr_cp_backend="${TPR_CP_BACKEND}" \
    \
    algorithm.rollout_correction.bypass_mode=True \
    algorithm.rollout_correction.loss_type=ppo_clip \
    \
    actor_rollout_ref.rollout.n="${N}" \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP}" \
    actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEMORY_UTILIZATION}" \
    actor_rollout_ref.rollout.prompt_length="${MAX_PROMPT_LENGTH}" \
    actor_rollout_ref.rollout.response_length="${MAX_RESPONSE_LENGTH}" \
    actor_rollout_ref.rollout.max_model_len="${MAX_MODEL_LEN}" \
    actor_rollout_ref.rollout.max_num_batched_tokens="${MAX_MODEL_LEN}" \
    actor_rollout_ref.rollout.max_num_seqs="${CONCURRENCY}" \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    +actor_rollout_ref.rollout.enable_sleep_mode=True \
    \
    actor_rollout_ref.rollout.temperature="${TEMPERATURE}" \
    actor_rollout_ref.rollout.top_p="${TOP_P}" \
    actor_rollout_ref.rollout.top_k="${TOP_K}" \
    \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}" \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="${MAX_MODEL_LEN}" \
    \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.max_parallel_calls=1 \
    ++actor_rollout_ref.rollout.multi_turn.format="${TOOL_PARSER}" \
    \
    actor_rollout_ref.rollout.agent.num_workers="${NUM_AGENT_WORKERS}" \
    ++actor_rollout_ref.rollout.agent.agent_loop_manager_class=uni_agent.framework.entry.AgentFrameworkRolloutAdapter \
    \
    ++actor_rollout_ref.rollout.custom.agent_framework.gateway_count="${GATEWAY_COUNT}" \
    ++actor_rollout_ref.rollout.custom.agent_framework.log_dir="${AGENT_LOG_DIR}" \
    ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.runner_fqn=uni_agent.framework.task_runner.run_task \
    ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.dispatch_mode=ray_task \
    ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.max_concurrent_sessions="${CONCURRENCY}" \
    ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.trajectory_selection="${TRAJECTORY_SELECTION}" \
    ++actor_rollout_ref.rollout.custom.agent_framework.hard_stop_on_trajectory_capacity="${SESSION_HARD_CAP}" \
    ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.runner_kwargs.task_config_path="${TASK_CONFIG}" \
    ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.runner_kwargs.model_name="${MODEL_NAME}" \
    ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.runner_kwargs.report_reward=True \
    ++actor_rollout_ref.rollout.custom.agent_framework.mask_unfinished_episode="${MASK_UNFINISHED_EPISODE}" \
    \
    reward.reward_manager.name=dapo \
    reward.custom_reward_function.path=pkg://uni_agent.framework.task_runner \
    reward.custom_reward_function.name=score_from_runner_result \
    \
    'trainer.logger=["console"]' \
    trainer.project_name=uniagent-rebench-e2e \
    trainer.experiment_name="${RUN_NAME}" \
    trainer.val_before_train=False \
    trainer.resume_mode=disable \
    trainer.save_freq="${SAVE_FREQ}" \
    trainer.test_freq="${TEST_FREQ}" \
    trainer.total_epochs="${TOTAL_EPOCHS}" \
    trainer.total_training_steps="${TOTAL_TRAINING_STEPS}" \
    trainer.default_local_dir="${CKPTS_DIR}" \
    \
    skip.rollout_tq.enable=False
echo
echo "============================================================"
echo "Trainer exited (online rollout + genuine Actor optimizer path)"
echo "Outputs: ${RUN_ROOT}"
if [[ "${CAPTURE_ACTOR}" == "True" || "${CAPTURE_ACTOR}" == "true" || "${CAPTURE_ACTOR}" == "1" ]]; then
    find "${TPR_CAPTURE_ACTOR_MINIBATCH_DIR}" -name 'actor_update_*.pt' -print || true
fi
echo "============================================================"
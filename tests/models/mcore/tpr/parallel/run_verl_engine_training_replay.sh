#!/usr/bin/env bash
# REAL training-side Full/TPR parity: Actor capture -> genuine VERL
# MindspeedEngine + DDP -> PPO Forward/Backward -> optimizer step.
# No rollout, Ray, vLLM, mock engine, CPU AdamW or in-place code changes.
#
# Usage: bash run_verl_engine_training_replay.sh CAPTURE_RUN_ROOT [STEPS=1] [both|full|tpr|compare] [NEW_OUTPUT_ROOT]
set -euo pipefail
cd "$(dirname "$0")/../../../../.."
VERL_ROOT="$PWD"

CAPTURE_ROOT="${1:?Usage: bash $0 CAPTURE_RUN_ROOT [STEPS=1] [MODE=both] [OUTPUT_ROOT]}"
STEPS="${2:-1}"
MODE="${3:-both}"
OUTPUT_ROOT="${4:-/workspace/tpr_experiments/verl_engine_training_replay_$(date +%Y%m%d_%H%M%S)}"
case "$MODE" in both|full|tpr|compare) ;; *) echo "MODE must be both|full|tpr|compare" >&2; exit 2;; esac
if ! [[ "$STEPS" =~ ^([1-9]|1[0-9]|20)$ ]]; then
  echo "STEPS must be 1..20" >&2; exit 2
fi
# Accept either a historical merged-manifest root or original rank0
# Actor captures. Rank1..rank7 are TP/CP replicas, NOT independent sessions.
# Build a NEW merged dataset under this run's output root; never touch input.
export TPR_REPLAY_STEPS="$STEPS"
if [[ -f "$CAPTURE_ROOT/capture_manifest.json" ]]; then
  MANIFEST="$(realpath "$CAPTURE_ROOT/capture_manifest.json")"
  echo "REAL_ENGINE_CAPTURE_INPUT kind=existing_manifest path=$MANIFEST"
elif compgen -G "$CAPTURE_ROOT/actor_update_step*_rank0_capture*_batch*.pt" > /dev/null; then
  mkdir -p "$OUTPUT_ROOT"
  MANIFEST="$OUTPUT_ROOT/capture_manifest.json"
  echo "REAL_ENGINE_CAPTURE_INPUT kind=raw_rank0 source=$CAPTURE_ROOT steps=$STEPS"
  python -m tests.models.mcore.tpr.parallel.prepare_real_actor_20step_manifest \
    "$CAPTURE_ROOT" "$MANIFEST" \
    --steps "$STEPS" --max-seq-len 49152 --expected-sessions 8
else
  echo "No manifest or rank0 Actor Capture files found under $CAPTURE_ROOT" >&2
  exit 2
fi
[[ -d /workspace/hf_models/Qwen3-4B-128K ]] || {
  echo "Missing Qwen3-4B original weights" >&2; exit 2;
}
if [[ -n "${TPR_MEGATRON_BRIDGE_SRC:-}" ]]; then
  echo "Unset TPR_MEGATRON_BRIDGE_SRC: real online E2E used vanilla_mbridge=True" >&2
  exit 2
fi
# Always import the NEW WORKTREE first. The old Full launcher prepended
# /workspace/uni-agent-cc/verl, which can silently bypass this PR's TPR audit.
CLEAN_PATH="$(python - <<'PY'
import os
print(":".join(
    part for part in os.environ.get("PYTHONPATH", "").split(":")
    if part not in ("", "/workspace/Megatron-Bridge/src", "/workspace/uni-agent-cc/verl")
))
PY
)"
export PYTHONPATH="$VERL_ROOT:/workspace/uni-agent-cc:/workspace/vllm-ascend:$CLEAN_PATH"
export PYTHONUNBUFFERED=1
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-8,9,10,11,12,13,14,15}"
[[ "$ASCEND_RT_VISIBLE_DEVICES" == "8,9,10,11,12,13,14,15" ]] || {
  echo "Requires validated physical NPUs 8..15; do not change topology silently" >&2; exit 2;
}
export TPR_OFFLINE_CAPTURE_MANIFEST="$MANIFEST"
export TPR_QWEN_4B_PATH="/workspace/hf_models/Qwen3-4B-128K"
export TPR_OFFLINE_ADAMW_LR="${TPR_OFFLINE_ADAMW_LR:-2e-5}"
export TPR_OFFLINE_ADAMW_CLIP_NORM="${TPR_OFFLINE_ADAMW_CLIP_NORM:-1.0}"
export TPR_TREE_POLICY=qwen_turn
export TPR_TURN_MIN_SEGMENT_TOKENS=128
# Chunking recomputes PPO selected-logit intermediates, not attention or
# Q/K/V dtype. May be disabled to compare memory/numerical performance.
export TPR_PPO_LOGPROB_CHUNK_TOKENS="${TPR_PPO_LOGPROB_CHUNK_TOKENS:-512}"
export TPR_OFFLINE_OUTPUT_DIR="$OUTPUT_ROOT/offline"
mkdir -p "$OUTPUT_ROOT/logs" "$TPR_OFFLINE_OUTPUT_DIR"

python - <<'PY'
import inspect, json, os
import mbridge
from verl.workers.config import McoreEngineConfig
from verl.workers.engine.mindspeed.transformer_impl import MindspeedEngineWithLMHead
import verl
fields = inspect.signature(McoreEngineConfig).parameters
needed = {"vanilla_mbridge", "tpr_enabled", "tpr_cp_backend"}
if not needed <= set(fields):
    raise SystemExit(f"REAL_ENGINE_PREFLIGHT_FAIL missing config fields: {needed - set(fields)}")
p = os.environ["TPR_OFFLINE_CAPTURE_MANIFEST"]
m = json.load(open(p))
required_steps = int(os.environ["TPR_REPLAY_STEPS"])
if (m.get("protocol") != "real_actor_session_first_8merged_v3"
        or m.get("online_steps") != len(m.get("items", []))
        or not required_steps <= len(m["items"]) <= 20
        or m.get("expected_sessions") != 8
        or any(len(item["rows"]) != 8 for item in m["items"][:required_steps])):
    raise SystemExit(
        "REAL_ENGINE_PREFLIGHT_FAIL: require enough genuine 8-Session "
        f"merged Actor steps; requested={required_steps} available={len(m.get('items', []))}"
    )
for item in m["items"][:required_steps]:
    if not os.path.isfile(item["capture"]):
        raise SystemExit("REAL_ENGINE_PREFLIGHT_FAIL: missing original capture " + item["capture"])
print("REAL_ENGINE_PREFLIGHT_PASS", "verl=", verl.__file__,
      "mbridge=", mbridge.__file__, "engine=", MindspeedEngineWithLMHead.__name__,
      f"original_updates_checked={required_steps}", flush=True)
PY

export TPR_RUN_VERL_ENGINE_FULL_REPLAY=0
export TPR_RUN_VERL_ENGINE_TPR_REPLAY=0
if [[ "$STEPS" == 20 ]]; then
  FULL_PREFIX=verl_full
  TPR_PREFIX=verl_tpr
elif [[ "$STEPS" == 1 ]]; then
  FULL_PREFIX=verl_full_smoke
  TPR_PREFIX=verl_tpr_smoke
else
  FULL_PREFIX="verl_full_test${STEPS}"
  TPR_PREFIX="verl_tpr_test${STEPS}"
fi
failed=()
if [[ "$MODE" == both || "$MODE" == full ]]; then
  export TPR_RUN_VERL_ENGINE_FULL_REPLAY=1
  export TPR_VERL_FULL_REPLAY_STEPS="$STEPS"
  export TPR_VERL_FULL_REPLAY_PREFIX="$FULL_PREFIX"
  echo "REAL_ENGINE_TRAIN_START mode=full steps=$STEPS output=$TPR_OFFLINE_OUTPUT_DIR"
  if torchrun --nproc_per_node=8 --master_addr=127.0.0.1 --master_port=29787 \
    -m pytest -vv -s -x tests/models/mcore/tpr/parallel/test_verl_engine_full_replay_npu.py \
    2>&1 | tee "$OUTPUT_ROOT/logs/${FULL_PREFIX}.log"; then
    echo "REAL_ENGINE_TRAIN_PASS mode=full"
  else
    failed+=(full)
  fi
  export TPR_RUN_VERL_ENGINE_FULL_REPLAY=0
fi
if [[ "$MODE" == both || "$MODE" == tpr ]]; then
  export TPR_RUN_VERL_ENGINE_TPR_REPLAY=1
  export TPR_VERL_TPR_REPLAY_STEPS="$STEPS"
  export TPR_VERL_TPR_REPLAY_PREFIX="$TPR_PREFIX"
  # Diagnostics default ON here, never in actual production Actor updates.
  export TPR_QKV_GRAD_AUDIT="${TPR_QKV_GRAD_AUDIT:-1}"
  echo "REAL_ENGINE_TRAIN_START mode=tpr steps=$STEPS qkv_audit=$TPR_QKV_GRAD_AUDIT"
  if torchrun --nproc_per_node=8 --master_addr=127.0.0.1 --master_port=29788 \
    -m pytest -vv -s -x tests/models/mcore/tpr/parallel/test_verl_engine_tpr_replay_npu.py \
    2>&1 | tee "$OUTPUT_ROOT/logs/${TPR_PREFIX}.log"; then
    echo "REAL_ENGINE_TRAIN_PASS mode=tpr"
  else
    failed+=(tpr)
  fi
  export TPR_RUN_VERL_ENGINE_TPR_REPLAY=0
fi
if [[ ("$MODE" == both && "${#failed[@]}" == 0) || "$MODE" == compare ]]; then
  python -m tests.models.mcore.tpr.parallel.compare_verl_engine_training_replay \
    "$TPR_OFFLINE_OUTPUT_DIR" --steps "$STEPS" \
    --full-prefix "$FULL_PREFIX" --tpr-prefix "$TPR_PREFIX" \
    2>&1 | tee "$OUTPUT_ROOT/logs/comparison.log"
fi
echo "REAL_ENGINE_TRAIN_ROOT=$OUTPUT_ROOT steps=$STEPS mode=$MODE failed=${failed[*]:-none}"
if (("${#failed[@]}" > 0)); then exit 1; fi

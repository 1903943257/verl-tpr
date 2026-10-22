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
# The original online prompt+response 49152 setting is NOT a verified hard
# cap on the captured full agent session. Run2 contains a genuine 50435-token
# trajectory. Do not crop PPO inputs/targets or silently replace a session.
# Limit is a data-validity guard, NOT a guarantee Full fits NPU HBM.
TPR_REPLAY_MAX_SEQ_LEN="${TPR_REPLAY_MAX_SEQ_LEN:-65536}"
if ! [[ "$TPR_REPLAY_MAX_SEQ_LEN" =~ ^[1-9][0-9]*$ ]] ||
  (( 10#$TPR_REPLAY_MAX_SEQ_LEN > 131072 )); then
  echo "TPR_REPLAY_MAX_SEQ_LEN must be an integer between 1 and 131072" >&2
  exit 2
fi
echo "REAL_ENGINE_CAPTURE_LENGTH_GUARD max_seq_len=$TPR_REPLAY_MAX_SEQ_LEN no_truncation=1"
# Prefer existing merged manifests. For a historical raw rank0 directory
# or already-derived eight-row capture .pt, use the validated controlled
# 10K / 30K source-data transforms, NOT an OOM-prone 71K full session.
# Never modify original captures or mix data from CP-replicated ranks.
export TPR_REPLAY_STEPS="$STEPS"
TPR_REPLAY_SHORT_CAP="${TPR_REPLAY_SHORT_CAP:-10240}"
TPR_REPLAY_REQUIRE_SHARED_PREFIX="${TPR_REPLAY_REQUIRE_SHARED_PREFIX:-1}"
case "$TPR_REPLAY_SHORT_CAP" in 0|10240|30720) ;; *)
  echo "TPR_REPLAY_SHORT_CAP must be 0, 10240 or 30720" >&2; exit 2;; esac
case "$TPR_REPLAY_REQUIRE_SHARED_PREFIX" in 0|1) ;; *)
  echo "TPR_REPLAY_REQUIRE_SHARED_PREFIX must be 0 or 1" >&2; exit 2;; esac
if [[ -d "$CAPTURE_ROOT" && -f "$CAPTURE_ROOT/capture_manifest.json" ]]; then
  MANIFEST="$(realpath "$CAPTURE_ROOT/capture_manifest.json")"
  echo "REAL_ENGINE_CAPTURE_INPUT kind=existing_manifest path=$MANIFEST"
elif [[ -f "$CAPTURE_ROOT" && "$CAPTURE_ROOT" == *.pt ]] ||
     { [[ -d "$CAPTURE_ROOT" ]] && compgen -G "$CAPTURE_ROOT/actor_update_step*_rank0_capture*_batch*.pt" > /dev/null; }; then
  mkdir -p "$OUTPUT_ROOT"
  MANIFEST="$OUTPUT_ROOT/capture_manifest.json"
  echo "REAL_ENGINE_CAPTURE_INPUT kind=short_or_raw_actor source=$CAPTURE_ROOT steps=$STEPS cap=$TPR_REPLAY_SHORT_CAP"
  export TPR_SHORT_CAPTURE_SOURCE="$CAPTURE_ROOT"
  export TPR_SHORT_CAPTURE_MANIFEST_ROOT="$OUTPUT_ROOT"
  export TPR_REPLAY_SHORT_CAP
  export TPR_REPLAY_REQUIRE_SHARED_PREFIX
  python - <<'PY'
import os
from pathlib import Path
from tests.models.mcore.tpr.parallel.prepare_verl_engine_short_manifest import prepare_short_manifest
prepare_short_manifest(
    Path(os.environ["TPR_SHORT_CAPTURE_SOURCE"]),
    Path(os.environ["TPR_SHORT_CAPTURE_MANIFEST_ROOT"]),
    steps=int(os.environ["TPR_REPLAY_STEPS"]),
    cap=int(os.environ["TPR_REPLAY_SHORT_CAP"]),
    require_shared_prefix=os.environ["TPR_REPLAY_REQUIRE_SHARED_PREFIX"] == "1",
)
PY
else
  echo "Expected an existing capture_manifest.json, a direct eight-row Actor .pt file, or rank0 capture directory: $CAPTURE_ROOT" >&2
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
    maximum = int(item["max_sequence_length"])
    online_limit = 49152
    if maximum > int(os.environ["TPR_REPLAY_MAX_SEQ_LEN"]):
        raise SystemExit(
            "REAL_ENGINE_PREFLIGHT_FAIL: source session longer than "
            "TPR_REPLAY_MAX_SEQ_LEN; refuse cropping"
        )
    print(
        "REAL_ENGINE_CAPTURE_SHAPE",
        f"step={item['source_global_step']}",
        f"selected_sessions={item['session_count']}",
        f"total_full_tokens={item['logical_full_tokens']}",
        f"longest_session={maximum}",
        f"exceeds_online_49152={maximum > online_limit}",
        "truncated=False",
        flush=True,
    )
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

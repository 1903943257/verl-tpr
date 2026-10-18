#!/usr/bin/env bash
# TP2×RingCP2×DP2, 8 Ascend NPU: real PPO OFF/ON recompute correctness replay.
# Two fresh HF-weight jobs; complete TP-shard BF16 gradients and one AdamW step.
# Usage: bash this_script.sh CAPTURE_MANIFEST NEW_OUTPUT_DIR [SOURCE_STEP=6]
# Inputs are immutable; output must be a NEW directory, never /workspace/*.log.
set -euo pipefail
cd "$(dirname "$0")/../../../../.."

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: $0 40K_OR_DERIVED_MANIFEST NEW_OUTPUT_DIR [SOURCE_STEP=6]" >&2
  exit 2
fi
MANIFEST="$(realpath "$1")"
OUT="$2"
STEP="${3:-6}"
[[ -f "$MANIFEST" ]] || { echo "Manifest not found: $MANIFEST" >&2; exit 2; }
[[ ! -e "$OUT" ]] || { echo "Output exists: $OUT (refuse overwrite)" >&2; exit 2; }
mkdir -p "$OUT/logs" "$OUT/off" "$OUT/on"
OUT="$(realpath "$OUT")"

export PYTHONUNBUFFERED=1
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-8,9,10,11,12,13,14,15}"
[[ "$ASCEND_RT_VISIBLE_DEVICES" == "8,9,10,11,12,13,14,15" ]] || {
  echo "This reproducible test requires physical NPUs 8..15" >&2; exit 2;
}
export TPR_QWEN_4B_PATH="${TPR_QWEN_4B_PATH:-/workspace/hf_models/Qwen3-4B-128K}"
[[ -d "$TPR_QWEN_4B_PATH" ]] || { echo "Missing checkpoint: $TPR_QWEN_4B_PATH" >&2; exit 2; }
export TPR_OFFLINE_CAPTURE_MANIFEST="$MANIFEST"
export TPR_BENCH_SOURCE_STEP="$STEP"
export TPR_OFFLINE_MODE=tpr
export TPR_RUN_TP2CP2DP2_ONE_STEP=1
export TPR_EXPERIMENTAL_OFFLINE_DP2=1
export TPR_TREE_POLICY=qwen_turn
export TPR_TURN_MIN_SEGMENT_TOKENS=128
export TPR_OFFLINE_NATIVE_RECOMPUTE_LAYERS=0
export TPR_BENCH_WARMUPS="${TPR_BENCH_WARMUPS:-1}"
export TPR_BENCH_REPEATS="${TPR_BENCH_REPEATS:-3}"
export TPR_BENCH_CAPTURE_LOGPROBS=1
# Controlled correctness gate: 910B HCCL BF16 SUM/4 has shown
# non-bitwise-identical post-finalize replicas even without recompute.
# Use FP32 per-parameter staging to isolate KV-aware recompute correctness.
# Explicit TPR_OFFLINE_GRAD_SYNC_DTYPE=bf16 remains available for diagnosis,
# but BF16 strict replica failures must NOT be reported as recompute failures.
export TPR_OFFLINE_GRAD_SYNC_DTYPE="${TPR_OFFLINE_GRAD_SYNC_DTYPE:-fp32}"
if [[ "$TPR_OFFLINE_GRAD_SYNC_DTYPE" != "bf16" && "$TPR_OFFLINE_GRAD_SYNC_DTYPE" != "fp32" ]]; then
  echo "TPR_OFFLINE_GRAD_SYNC_DTYPE must be fp32 or bf16" >&2
  exit 2
fi
echo "TPR_DP2_RECOMPUTE_GRAD_SYNC precision=$TPR_OFFLINE_GRAD_SYNC_DTYPE strict_replicas=1 (offline only)"
export TPR_PPO_LOGPROB_CHUNK_TOKENS="${TPR_PPO_LOGPROB_CHUNK_TOKENS:-512}"
export TPR_BENCH_GRAD_WITNESS=1
export TPR_BENCH_GRAD_REPLICA_POLICY=strict
export TPR_DP2_SAVE_FULL_GRADS=1
export TPR_DP2_OPTIMIZER_STEP=1
export TPR_OFFLINE_ADAMW_LR="${TPR_OFFLINE_ADAMW_LR:-2e-5}"
export TPR_OFFLINE_ADAMW_CLIP_NORM="${TPR_OFFLINE_ADAMW_CLIP_NORM:-1.0}"
export TPR_EXEC_PROFILE=0
export TPR_RECOMPUTE_REF_AUDIT=0
unset TPR_MEM_TRACE TPR_FIRST_LEAF_DIAG TPR_FIRST_LEAF_DIAG_SYNC || true

case "${TPR_DP2_VALIDATION_MODES:-both}" in
  both) MODES=(off on) ;;
  off) MODES=(off) ;;
  on) MODES=(on) ;;
  *) echo "TPR_DP2_VALIDATION_MODES must be both|off|on" >&2; exit 2 ;;
esac

python - "$MANIFEST" "$STEP" <<'PY'
import json, sys
from pathlib import Path
m = json.loads(Path(sys.argv[1]).read_text())
step = int(sys.argv[2])
assert m["protocol"] == "real_actor_session_first_8merged_v3"
assert m["steps"] == 20 and m["expected_sessions"] == 8
assert m["max_seq_len"] == 40960 and m["historical_ppo_labels_modified"] is False
assert 1 <= step <= 20
x = m["items"][step - 1]
assert x["step"] == x["source_global_step"] == step
assert x["rows"] == list(range(8)) and x["session_count"] == 8
assert x["max_sequence_length"] <= 40960 and Path(x["capture"]).is_file()
print(f"TPR_DP2_RECOMPUTE_INPUT step={step} ppo_tokens={x['ppo_tokens']} "
      f"full_tokens={x['logical_full_tokens']} source={x['capture']}", flush=True)
PY

declare -a FAILED=()
for mode in "${MODES[@]}"; do
  export TPR_DP2_RECOMPUTE=0
  [[ "$mode" == "on" ]] && export TPR_DP2_RECOMPUTE=1
  export TPR_BENCH_OUTPUT_DIR="$OUT/$mode"
  echo "TPR_DP2_RECOMPUTE_START step=$STEP mode=$mode tp=2 cp=2 dp=2"
  if torchrun --nproc_per_node=8 \
    --master_addr=127.0.0.1 --master_port="${TPR_BENCH_MASTER_PORT:-29797}" \
    -m pytest -q -s -x \
    tests/models/mcore/tpr/parallel/test_tpr_tp2cp2dp2_one_step_npu.py \
    2>&1 | tee "$OUT/logs/step${STEP}_${mode}.log"; then
    echo "TPR_DP2_RECOMPUTE_DONE step=$STEP mode=$mode"
  else
    FAILED+=("$mode")
    echo "TPR_DP2_RECOMPUTE_FAILED mode=$mode; logs and any artifacts retained" >&2
  fi
done

OFF="$OUT/off/step${STEP}_tpr_tp2cp2dp2.json"
ON="$OUT/on/step${STEP}_tpr_tp2cp2dp2_recompute_on.json"
if [[ "${TPR_DP2_VALIDATION_MODES:-both}" != "both" ]]; then
  # An intentional 40K ON-only capacity/optimizer run is not an OFF/ON
  # correctness comparison. Do not mislabel it as a failed pair.
  echo "TPR_DP2_RECOMPUTE_SINGLE_MODE_COMPLETE mode=${MODES[0]} comparison=not_requested"
elif [[ -f "$OFF" && -f "$ON" && ${#FAILED[@]} -eq 0 ]]; then
  python tests/models/mcore/tpr/parallel/compare_tpr_dp2_recompute.py \
    "$OFF" "$ON" "$OUT/comparison.json" \
    2>&1 | tee "$OUT/logs/comparison.log"
else
  echo "TPR_DP2_RECOMPUTE_PAIR_INCOMPLETE off=$OFF on=$ON; no parity claim" >&2
fi
if (( ${#FAILED[@]} )); then
  printf 'TPR_DP2_RECOMPUTE_FAILED_MODES=%s\n' "${FAILED[*]}" >&2
  exit 1
fi
echo "TPR_DP2_RECOMPUTE_ARTIFACTS=$OUT"

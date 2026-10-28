#!/usr/bin/env bash
# Recorded N=8 real-Actor PPO replay on TP8 x CP1 x DP1.
# Measures pure F/B, F/B+finalize, and one-step F/B+finalize+diagnostic SGD.
# This is offline replay; it does not run rollout or production AdamW.
set -euo pipefail
cd "$(dirname "$0")/../../../../.."

CAPTURE="${1:?Usage: $0 /path/to/actor_update_mini_batch.pt [output_root]}"
ROOT="${2:-/workspace/tpr_experiments/p16s2_n8_full_training_tp8_recompute_ablation_20261028}"
CAPTURE="$(realpath "$CAPTURE")"
[[ -f "$CAPTURE" ]] || { echo "Missing actor capture: $CAPTURE" >&2; exit 2; }
[[ ! -e "$ROOT" ]] || { echo "Refusing existing output root: $ROOT" >&2; exit 2; }
mkdir -p "$ROOT/logs"
ROOT="$(realpath "$ROOT")"

export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
[[ "$ASCEND_RT_VISIBLE_DEVICES" == "0,1,2,3,4,5,6,7" || "$ASCEND_RT_VISIBLE_DEVICES" == "8,9,10,11,12,13,14,15" ]] || {
  echo "Expected exactly eight visible NPUs, got $ASCEND_RT_VISIBLE_DEVICES" >&2
  exit 2
}
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
export TPR_RUN_REAL_ACTOR_TP8=1
export TPR_EXPERIMENTAL_OFFLINE_TP8_REAL_ACTOR=1
export TPR_OFFLINE_TP=8
export TPR_OFFLINE_CP=1
export TPR_OFFLINE_DP=1
export TPR_REAL_ACTOR_MINIBATCH="$CAPTURE"
export TPR_QWEN_4B_PATH="${TPR_QWEN_4B_PATH:-/workspace/hf_models/Qwen3-4B-128K}"
export TPR_TREE_POLICY=qwen_turn
export TPR_TURN_MIN_SEGMENT_TOKENS=128
export TPR_OFFLINE_SP=0
export TPR_OFFLINE_ROWS=all
export TPR_OFFLINE_SGD_LR="${TPR_OFFLINE_SGD_LR:-1e-5}"
export TPR_OFFLINE_FULL_RING_BACKEND=custom
export TPR_OFFLINE_GRAD_SYNC_DTYPE=bf16
export TPR_RECOMPUTE_LOG_DIR="$ROOT/logs"
export PYTHONUNBUFFERED=1
unset TPR_RUN_REAL_ACTOR_TP2CP4 TPR_EXPERIMENTAL_OFFLINE_DP2 TPR_RUN_10K_PARALLEL_PARITY TPR_10K_PARITY_ONLY || true

[[ -d "$TPR_QWEN_4B_PATH" ]] || { echo "Missing checkpoint: $TPR_QWEN_4B_PATH" >&2; exit 2; }

# Verify the source is a genuine Actor-update capture and print its row shapes.
python - "$CAPTURE" <<'PY'
import sys, torch
from pathlib import Path
p = Path(sys.argv[1])
raw = torch.load(p, map_location="cpu", weights_only=False)
if raw.get("capture_stage") != "actor_update_mini_batch":
    raise SystemExit(f"Expected actor_update_mini_batch capture, got {raw.get('capture_stage')!r}")
keys = raw.get("keys", ())
td = raw.get("tensordict")
if not isinstance(td, dict) and not hasattr(td, "__contains__"):
    raise SystemExit("Capture has no TensorDict-like tensordict")
if len(keys) != 8:
    raise SystemExit(f"Expected N=8 rows for TP8 single-tree replay, got {len(keys)}")
print(f"TPR_REAL_ACTOR_CAPTURE path={p} rows={len(keys)} keys={tuple(keys)}")
for name in ("input_ids", "prompts", "responses", "response_mask", "loss_mask"):
    if name in td:
        value = td[name]
        try:
            shapes = [tuple(x.shape) for x in value]
        except Exception:
            shapes = tuple(getattr(value, "shape", ()))
        print(f"TPR_CAPTURE_FIELD {name} shapes={shapes}")
PY

RUNNER="tests/models/mcore/tpr/parallel/run_real_actor_offline_ppo_argv_safe.py"
failures=0

run_case() {
  local recompute="$1" mode="$2"
  local out="$ROOT/recompute_${recompute}"
  local log="$ROOT/logs/recompute_${recompute}_${mode}.log"
  mkdir -p "$out"
  export TPR_OFFLINE_RECOMPUTE="$recompute"
  export TPR_OFFLINE_MODE="$mode"
  export TPR_OFFLINE_OUTPUT_DIR="$out"
  echo "TPR_REAL_ACTOR_TP8_CASE_START mode=$mode tp=8 cp=1 dp=1 recompute=$recompute rows=all capture=$CAPTURE output=$out"
  local rc=0
  if torchrun --nproc_per_node=8 --master_addr=127.0.0.1 \
      --master_port="${TPR_REAL_ACTOR_MASTER_PORT:-29797}" "$RUNNER" 2>&1 | tee "$log"; then
    rc=0
  else
    rc=$?
    failures=$((failures + 1))
    echo "TPR_REAL_ACTOR_TP8_CASE_FAILED mode=$mode recompute=$recompute rc=$rc log=$log" | tee -a "$log"
  fi
  printf '%s\n' "$rc" > "$out/$mode.exit_code"
}

for recompute in 0 1; do
  run_case "$recompute" full
  run_case "$recompute" tpr
  if [[ -f "$ROOT/recompute_${recompute}/full_rank0.pt" && -f "$ROOT/recompute_${recompute}/tpr_rank0.pt" ]]; then
    args=()
    if [[ "$recompute" == "1" ]]; then args+=(--require-recompute); fi
    python tests/models/mcore/tpr/parallel/compare_real_actor_tp2cp4_offline_ppo.py \
      "$ROOT/recompute_${recompute}" --diagnostic-only "${args[@]}" 2>&1 | \
      tee "$ROOT/logs/recompute_${recompute}_comparison.log" || failures=$((failures + 1))
  else
    echo "TPR_REAL_ACTOR_TP8_COMPARISON_SKIPPED recompute=$recompute missing full/tpr snapshots"
    failures=$((failures + 1))
  fi
done

echo "TPR_REAL_ACTOR_TP8_RESULTS=$ROOT"
grep -hE 'TPR_OFFLINE_COMPARISON|TPR_OFFLINE_PERF|TPR_OFFLINE_DIAGNOSTIC_ONLY' "$ROOT"/logs/recompute_*_comparison.log || true
if (( failures > 0 )); then
  echo "TPR_REAL_ACTOR_TP8_ABLATION_FINISHED_WITH_FAILURES=$failures"
  exit 1
fi
echo "TPR_REAL_ACTOR_TP8_ABLATION_FINISHED=PASS"

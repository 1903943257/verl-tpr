#!/usr/bin/env bash
# Recorded N=8 real-Actor PPO replay on TP8 x CP1 x DP1.
# Sequentially replays the five captured Actor minibatches on one evolving model.
# Measures pure F/B, gradient norm, and persistent CPU FP32-master AdamW update.
# Rollout is excluded; this is a controlled training-side replay.
set -euo pipefail
cd "$(dirname "$0")/../../../../.."

CAPTURE="${1:?Usage: $0 /path/to/actor_update_step000001_rank0_capture000000_batch0.pt [output_root]}"
ROOT="${2:-/workspace/tpr_experiments/p16s2_n8_full_training_tp8_adamw_recompute_ablation_$(date +%Y%m%d_%H%M%S)}"
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
CAPTURE_DIR="$(dirname "$CAPTURE")"
mapfile -t CAPTURES < <(find "$CAPTURE_DIR" -maxdepth 1 -type f \
  -name 'actor_update_step*_rank0_capture*_batch0.pt' -print | sort)
if [[ "${#CAPTURES[@]}" -ne 5 ]]; then
  echo "Expected exactly five rank0 Actor captures in $CAPTURE_DIR, found ${#CAPTURES[@]}" >&2
  printf '  %s\\n' "${CAPTURES[@]}" >&2
  exit 2
fi
export TPR_REAL_ACTOR_MINIBATCH="$CAPTURE"
export TPR_OFFLINE_CAPTURE_LIST="$(IFS=:; echo "${CAPTURES[*]}")"
export TPR_QWEN_4B_PATH="${TPR_QWEN_4B_PATH:-/workspace/hf_models/Qwen3-4B-128K}"
export TPR_TREE_POLICY=qwen_turn
export TPR_TURN_MIN_SEGMENT_TOKENS=128
export TPR_OFFLINE_SP=0
export TPR_OFFLINE_ROWS=all
export TPR_OFFLINE_ADAMW_LR="${TPR_OFFLINE_ADAMW_LR:-1e-6}"
export TPR_OFFLINE_ADAMW_CLIP_NORM="${TPR_OFFLINE_ADAMW_CLIP_NORM:-1.0}"
export TPR_OFFLINE_FULL_RING_BACKEND=custom
export TPR_OFFLINE_GRAD_SYNC_DTYPE=bf16
export TPR_RECOMPUTE_LOG_DIR="$ROOT/logs"
export PYTHONUNBUFFERED=1
unset TPR_RUN_REAL_ACTOR_TP2CP4 TPR_EXPERIMENTAL_OFFLINE_DP2 TPR_RUN_10K_PARALLEL_PARITY TPR_10K_PARITY_ONLY || true

[[ -d "$TPR_QWEN_4B_PATH" ]] || { echo "Missing checkpoint: $TPR_QWEN_4B_PATH" >&2; exit 2; }

# Verify the source is a genuine Actor-update capture and print its row shapes.
python - "${CAPTURES[@]}" <<'PY'
import sys, torch
from pathlib import Path
paths = [Path(x) for x in sys.argv[1:]]
for p in paths:
    raw = torch.load(p, map_location="cpu", weights_only=False)
    if raw.get("capture_stage") != "actor_update_mini_batch":
        raise SystemExit(f"Expected actor_update_mini_batch capture, got {raw.get('capture_stage')!r}: {p}")
    keys = raw.get("keys", ())
    td = raw.get("tensordict")
    if not isinstance(td, dict) and not hasattr(td, "__contains__"):
        raise SystemExit(f"Capture has no TensorDict-like tensordict: {p}")
    if len(keys) != 8:
        raise SystemExit(f"Expected N=8 rows for TP8 replay, got {len(keys)} in {p}")
    print(f"TPR_REAL_ACTOR_CAPTURE path={p} rows={len(keys)}")
    for name in ("input_ids", "prompts", "responses", "response_mask", "loss_mask"):
        if name in td:
            value = td[name]
            try:
                shapes = [tuple(x.shape) for x in value]
            except Exception:
                shapes = tuple(getattr(value, "shape", ()))
            print(f"TPR_CAPTURE_FIELD step_capture={p.name} {name} shapes={shapes}")
PY

RUNNER="tests/models/mcore/tpr/parallel/run_tp8_adamw_recompute_ablation_argv_safe.py"
TEST="tests/models/mcore/tpr/parallel/test_real_actor_tp8_adamw_recompute_ablation_npu.py"
SUMMARIZER="tests/models/mcore/tpr/parallel/summarize_real_actor_tp8_adamw_recompute_ablation.py"
failures=0

run_case() {
  local recompute="$1" mode="$2" port="${3:-${TPR_REAL_ACTOR_MASTER_PORT:-29797}}"
  local out="$ROOT/recompute_${recompute}"
  local log="$ROOT/logs/recompute_${recompute}_${mode}.log"
  mkdir -p "$out"
  export TPR_OFFLINE_RECOMPUTE="$recompute"
  export TPR_OFFLINE_MODE="$mode"
  export TPR_OFFLINE_OUTPUT_DIR="$out"
  echo "TPR_REAL_ACTOR_TP8_CASE_START mode=$mode tp=8 cp=1 dp=1 recompute=$recompute sequential_steps=${#CAPTURES[@]} capture_dir=$CAPTURE_DIR output=$out"
  local rc=0
  if torchrun --nproc_per_node=8 --master_addr=127.0.0.1 \
      --master_port="$port" "$RUNNER" 2>&1 | tee "$log"; then
    rc=0
  else
    rc=$?
    failures=$((failures + 1))
    echo "TPR_REAL_ACTOR_TP8_CASE_FAILED mode=$mode recompute=$recompute rc=$rc log=$log" | tee -a "$log"
  fi
  printf '%s\n' "$rc" > "$out/$mode.exit_code"
  printf '%s\n' "$rc" > "$ROOT/exit_${mode}_recompute_${recompute}.txt"
}

# Syntax-check before spending NPU time.
python -m py_compile "$TEST" "$RUNNER" "$SUMMARIZER" || exit 2
bash -n tests/models/mcore/tpr/parallel/run_real_actor_tp8_adamw_recompute_ablation.sh || exit 2

# Four independent configurations. Full/recompute-OFF is allowed to OOM.
run_case 0 full 29797
run_case 0 tpr 29798
run_case 1 full 29799
run_case 1 tpr 29800

python "$SUMMARIZER" "$ROOT" || {
  echo "Summarization failed; logs and partial snapshots are preserved at $ROOT" >&2
  exit 1
}
echo
echo "SUMMARY=$ROOT/SUMMARY.md"
echo "SUMMARY_JSON=$ROOT/summary.json"
cat "$ROOT/SUMMARY.md"

python - "$ROOT/summary.json" <<'PY'
import json, sys
from pathlib import Path
data = json.loads(Path(sys.argv[1]).read_text())
cfg = data["configs"]
expected = cfg["full_recompute_0"]["status"] == "OOM"
others_ok = all(cfg[k]["status"] == "PASS" for k in (
    "tpr_recompute_0", "full_recompute_1", "tpr_recompute_1"
))
print(f"EXPECTED_FULL_RECOMPUTE_OFF_OOM={expected}")
print(f"OTHER_THREE_CONFIGS_PASS={others_ok}")
if expected and others_ok:
    raise SystemExit(0)
if all(v["status"] == "PASS" for v in cfg.values()):
    raise SystemExit(0)
raise SystemExit(1)
PY
exit "$?"

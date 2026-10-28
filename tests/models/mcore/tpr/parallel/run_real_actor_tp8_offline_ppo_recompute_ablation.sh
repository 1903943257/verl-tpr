#!/usr/bin/env bash
# Recorded N=8 real-Actor PPO replay on TP8 x CP1 x DP1.
# Sequentially replays the five captured Actor minibatches on one evolving model.
# Measures pure F/B, F/B+finalize, and F/B+finalize+diagnostic SGD.
# This does not run rollout or production Megatron AdamW.
set -euo pipefail
cd "$(dirname "$0")/../../../../.."

CAPTURE="${1:?Usage: $0 /path/to/actor_update_step000001_rank0_capture000000_batch0.pt [output_root]}"
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
export TPR_OFFLINE_SGD_LR="${TPR_OFFLINE_SGD_LR:-1e-5}"
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
  echo "TPR_REAL_ACTOR_TP8_CASE_START mode=$mode tp=8 cp=1 dp=1 recompute=$recompute sequential_steps=${#CAPTURES[@]} capture_dir=$CAPTURE_DIR output=$out"
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
  args=()
  if [[ "$recompute" == "1" ]]; then args+=(--require-recompute); fi
  for step in 01 02 03 04 05; do
    step_root="$ROOT/recompute_${recompute}/step_${step}"
    if [[ -f "$step_root/full_rank0.pt" && -f "$step_root/tpr_rank0.pt" ]]; then
      python tests/models/mcore/tpr/parallel/compare_real_actor_tp2cp4_offline_ppo.py \
        "$step_root" --diagnostic-only "${args[@]}" 2>&1 | \
        tee "$ROOT/logs/recompute_${recompute}_step_${step}_comparison.log" || failures=$((failures + 1))
    else
      echo "TPR_REAL_ACTOR_TP8_COMPARISON_SKIPPED recompute=$recompute step=$step missing full/tpr snapshots"
      failures=$((failures + 1))
    fi
  done
done

echo "TPR_REAL_ACTOR_TP8_RESULTS=$ROOT"
grep -hE 'TPR_OFFLINE_COMPARISON|TPR_OFFLINE_PERF|TPR_OFFLINE_DIAGNOSTIC_ONLY' "$ROOT"/logs/recompute_*_step_*_comparison.log || true
python - "$ROOT" <<'PY'
import json, sys, torch
from pathlib import Path
root = Path(sys.argv[1])
for rec in (0, 1):
    sums = {}
    peak = {}
    for mode in ("full", "tpr"):
        per_step = []
        peaks = []
        for step in range(1, 6):
            folder = root / f"recompute_{rec}" / f"step_{step:02d}"
            files = sorted(folder.glob(f"{mode}_rank*.pt"))
            if len(files) != 8:
                raise SystemExit(f"Expected 8 rank snapshots, got {len(files)}: {folder}/{mode}")
            rows = [torch.load(p, map_location="cpu", weights_only=False) for p in files]
            fbfin = max(float(x["time_forward_backward_finalize_s"]) for x in rows)
            fin = max(float(x["time_finalize_s"]) for x in rows)
            sgd = max(float(x["time_sgd_s"]) for x in rows)
            pure_fb = max(0.0, fbfin - fin)
            per_step.append({"fwd_bwd": pure_fb, "fwd_bwd_finalize": fbfin,
                             "training_side_diag_sgd": fbfin + sgd})
            peaks.append(max(float(x["peak_allocated_bytes"]) for x in rows) / 2**30)
        sums[mode] = {key: sum(x[key] for x in per_step) for key in per_step[0]}
        peak[mode] = max(peaks)
    print(f"TPR_5STEP_SUMMARY recompute={rec} full={json.dumps(sums['full'],sort_keys=True)} "
          f"tpr={json.dumps(sums['tpr'],sort_keys=True)} "
          f"pure_fb_speedup={sums['full']['fwd_bwd']/sums['tpr']['fwd_bwd']:.4f}x "
          f"training_side_diag_sgd_speedup={sums['full']['training_side_diag_sgd']/sums['tpr']['training_side_diag_sgd']:.4f}x "
          f"peak_alloc_gib_full={peak['full']:.3f} peak_alloc_gib_tpr={peak['tpr']:.3f} "
          f"memory_saved_gib={peak['full']-peak['tpr']:.3f}")
PY
if (( failures > 0 )); then
  echo "TPR_REAL_ACTOR_TP8_ABLATION_FINISHED_WITH_FAILURES=$failures"
  exit 1
fi
echo "TPR_REAL_ACTOR_TP8_ABLATION_FINISHED=PASS"

#!/usr/bin/env bash
# Controlled TP8 x CP1 x DP1 replay: unshared/TPR, recompute OFF/ON.
# "unshared" is a Full-like control through the isolated TPR executor, not
# native Megatron Full scheduling.
set -euo pipefail
cd "$(dirname "$0")/../../../../.."

CAPTURE="${1:-/workspace/tpr_experiments/n8_short_shapes/n8_p8192_s2048.pt}"
ROOT="${2:-/workspace/tpr_experiments/tp8dp1_p8k_s2k_recompute_ablation_20261028}"
CAPTURE="$(realpath "$CAPTURE")"
[[ -f "$CAPTURE" ]] || { echo "Missing N8 Actor Capture: $CAPTURE" >&2; exit 2; }
[[ ! -e "$ROOT" ]] || { echo "Refusing existing output root: $ROOT" >&2; exit 2; }
mkdir -p "$ROOT"
ROOT="$(realpath "$ROOT")"

export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
IFS=',' read -r -a DEVICES <<< "$ASCEND_RT_VISIBLE_DEVICES"
[[ "${#DEVICES[@]}" == "8" ]] || { echo "Need exactly 8 visible NPUs" >&2; exit 2; }
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
export TPR_RUN_10K_PARALLEL_PARITY=1
export TPR_10K_PARITY_ONLY=1
export TPR_EXPERIMENTAL_OFFLINE_TP8=1
export TPR_10K_TP=8
export TPR_10K_CP=1
export TPR_10K_DP=1
export TPR_10K_CAPTURE="$CAPTURE"
export TPR_QWEN_4B_PATH="${TPR_QWEN_4B_PATH:-/workspace/hf_models/Qwen3-4B}"
export HCCL_DETERMINISTIC=true
export TPR_OFFLINE_GRAD_SYNC_DTYPE=bf16
export TPR_TREE_POLICY=qwen_turn
export TPR_TURN_MIN_SEGMENT_TOKENS=128
export TPR_PPO_LOGPROB_CHUNK_TOKENS=512
export TPR_LEAF_BATCH_SIZE=1
export TPR_EXEC_PROFILE=0
export TPR_10K_WARMUPS="${TPR_10K_WARMUPS:-1}"
export TPR_10K_REPEATS="${TPR_10K_REPEATS:-3}"
export TPR_10K_SUMMARY_ONLY=1
export PYTHONUNBUFFERED=1
unset TPR_EXPERIMENTAL_OFFLINE_DP2 TPR_MEM_TRACE TPR_FIRST_LEAF_DIAG TPR_FIRST_LEAF_DIAG_SYNC || true

[[ -d "$TPR_QWEN_4B_PATH" ]] || { echo "Missing checkpoint: $TPR_QWEN_4B_PATH" >&2; exit 2; }

RUNNER="$(mktemp /tmp/run_tpr_tp8_p8s2_XXXXXX.py)"
trap 'rm -f "$RUNNER"' EXIT
cat > "$RUNNER" <<'PY'
import sys
sys.argv[:] = [sys.argv[0]]
from tests.models.mcore.tpr.parallel.test_tpr_10k_parallel_parity_npu import (
    test_10k_parallel_factorial_one_topology,
)
test_10k_parallel_factorial_one_topology()
PY

failures=0
run_case() {
  local recompute="$1"
  local mode="$2"
  local label="recompute_${recompute}"
  local case_root="$ROOT/$label"
  local log="$case_root/$mode.log"
  mkdir -p "$case_root"
  export TPR_10K_RECOMPUTE="$recompute"
  export TPR_10K_MODE="$mode"
  export TPR_10K_OUTPUT="$case_root/$mode"
  echo "TPR_TP8_P8S2_CASE_START mode=$mode tp=8 cp=1 dp=1 recompute=$recompute capture=$CAPTURE output=$TPR_10K_OUTPUT"
  local rc=0
  if torchrun --nproc_per_node=8 \
      --master_addr=127.0.0.1 \
      --master_port="${TPR_BENCH_MASTER_PORT:-29874}" \
      "$RUNNER" 2>&1 | tee "$log"; then
    rc=0
  else
    rc=$?
    failures=$((failures + 1))
    echo "TPR_TP8_P8S2_CASE_FAILED mode=$mode recompute=$recompute rc=$rc log=$log" | tee -a "$log"
  fi
  printf '%s\n' "$rc" > "$case_root/$mode.exit_code"
  return 0
}

# Continue after a failed/OOM case so the other recompute setting still runs.
for recompute in 0 1; do
  run_case "$recompute" unshared
  run_case "$recompute" tpr
done

python - "$ROOT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = []
for recompute in (0, 1):
    label = f"recompute_{recompute}"
    for mode in ("unshared", "tpr"):
        case_root = root / label
        result_path = case_root / mode / "result.json"
        status_path = case_root / f"{mode}.exit_code"
        status = status_path.read_text().strip() if status_path.exists() else "missing"
        row = {
            "recompute": recompute,
            "mode": mode,
            "status": status,
            "result_path": str(result_path),
        }
        if result_path.is_file():
            data = json.loads(result_path.read_text())
            by_dp = data.get("execution_metrics_by_dp") or []
            logical = sum(x["logical_input_tokens"] for x in by_dp)
            unique = sum(x["unique_tree_tokens"] for x in by_dp)
            row.update({
                "median_s": data.get("median_fwd_bwd_seconds_max_rank"),
                "peak_gib": data.get("memory_peak_allocated_gib_worst"),
                "ppo_tokens": data.get("global_ppo_tokens"),
                "executed_logical_tokens": logical,
                "executed_unique_tree_tokens": unique,
                "executed_reuse_ratio": logical / unique if unique else None,
                "push_tokens": sum(x["push_forward_tokens"] for x in by_dp),
                "pop_tokens": sum(x["pop_forward_tokens"] for x in by_dp),
                "leaf_tokens": sum(x["leaf_forward_tokens"] + x["grouped_leaf_forward_tokens"] for x in by_dp),
                "physical_segments": sum(x["physical_segments"] for x in by_dp),
                "forward_calls": sum(x["model_forward_calls"] for x in by_dp),
                "input_sha256": data.get("input_sha256"),
                "recompute_config": data.get("recompute"),
            })
        rows.append(row)

print("TP8xCP1xDP1 | Qwen3-4B | P8K/S2K | no optimizer / no weight update")
print("RECOMPUTE\tMODE\tSTATUS\tMEDIAN_F_B_S\tPEAK_GIB\tEXEC_REUSE\tPUSH_TOKENS\tPOP_TOKENS\tLEAF_TOKENS\tSEGMENTS\tFORWARD_CALLS")
for row in rows:
    print("\t".join(str(row.get(key, "n/a")) if row.get(key) is not None else "n/a" for key in (
        "recompute", "mode", "status", "median_s", "peak_gib",
        "executed_reuse_ratio", "push_tokens", "pop_tokens", "leaf_tokens",
        "physical_segments", "forward_calls",
    )))

for recompute in (0, 1):
    pair = {r["mode"]: r for r in rows if r["recompute"] == recompute}
    a, b = pair["unshared"], pair["tpr"]
    if a.get("median_s") and b.get("median_s"):
        print(f"RECOMPUTE_{recompute}_UNSHARED_OVER_TPR_SPEEDUP={a['median_s'] / b['median_s']:.4f}x")
        if a.get("peak_gib") is not None and b.get("peak_gib") is not None:
            print(f"RECOMPUTE_{recompute}_TPR_MEMORY_SAVED_GIB={a['peak_gib'] - b['peak_gib']:.3f}")

for mode in ("unshared", "tpr"):
    off = next(r for r in rows if r["recompute"] == 0 and r["mode"] == mode)
    on = next(r for r in rows if r["recompute"] == 1 and r["mode"] == mode)
    if off.get("median_s") and on.get("median_s"):
        print(f"{mode.upper()}_RECOMPUTE_OFF_BASELINE_TO_ON_SPEEDUP={off['median_s'] / on['median_s']:.4f}x")
        if off.get("peak_gib") is not None and on.get("peak_gib") is not None:
            print(f"{mode.upper()}_RECOMPUTE_MEMORY_SAVED_GIB={off['peak_gib'] - on['peak_gib']:.3f}")

shas = {r.get("input_sha256") for r in rows if r.get("input_sha256")}
if len(shas) > 1:
    raise SystemExit(f"Input SHA mismatch across cases: {sorted(shas)}")
print(f"INPUT_SHA256={next(iter(shas)) if shas else 'no successful cases'}")
print("NOTE=unshared is the TPR executor without prefix sharing, not native Megatron Full. F/B includes gradient finalize, excludes optimizer and online rollout.")
if any(not r.get("median_s") for r in rows):
    print("WARNING=At least one case has no result; inspect its log and exit_code. Remaining cases were still attempted.")
PY

if (( failures > 0 )); then
  echo "TPR_TP8_P8S2_ABLATION_FINISHED_WITH_FAILURES=$failures"
  exit 1
fi
echo "TPR_TP8_P8S2_ABLATION_FINISHED=PASS"

#!/usr/bin/env bash
# Controlled Qwen3-4B N8 P:S sweep at 18K/27K/36K, TP2 CP2 or CP4 DP1.
# Reuses existing native Megatron Full vs production TPR Ring benchmark.
# Original PPO labels retained; reject any source unable to supply real tokens.
set -euo pipefail
cd "$(dirname "$0")/../../../../.."

if [[ "$#" != 2 ]]; then
  echo "Usage: bash $0 {1to1|8to1|18k_18k|32k_4k|16k_2k|9k_9k|6k_6k|12k_6k} SOURCE_ACTOR_CAPTURE.pt" >&2
  exit 2
fi
RATIO="$1"
SOURCE="$(realpath "$2")"
case "$RATIO" in
  1to1)    CP=4; P=13824; S=13824; SHAPE="27k_1to1" ;;
  8to1)    CP=4; P=24576; S=3072;  SHAPE="27k_8to1" ;;
  18k_18k) CP=4; P=18432; S=18432; SHAPE="36k_1to1" ;;
  32k_4k)  CP=4; P=32768; S=4096;  SHAPE="36k_8to1" ;;
  16k_2k)  CP=2; P=16384; S=2048;  SHAPE="18k_8to1" ;;
  9k_9k)   CP=2; P=9216;  S=9216;  SHAPE="18k_1to1" ;;
  6k_6k)   CP=2; P=6016;  S=6016;  SHAPE="12k_1to1" ;;
  12k_6k)  CP=2; P=12416; S=6016;  SHAPE="18k_2to1" ;;
  *) echo "Unsupported shape: $RATIO" >&2; exit 2 ;;
esac
ROOT="${TPR_CP_N8_ROOT:-${TPR_CP4_N8_ROOT:-/workspace/tpr_experiments/performance/cp${CP}_n8_${SHAPE}_v1}}"
DATA="$ROOT/data"
LOGS="$ROOT/logs"
RESULTS="$ROOT/results"
CAPTURE="$DATA/cp${CP}_n8_p${P}_s${S}.pt"
AUDIT="${CAPTURE%.pt}.audit.json"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1

[[ -f "$SOURCE" ]] || { echo "Missing original 8-row PPO Capture: $SOURCE" >&2; exit 2; }
mkdir -p "$DATA" "$LOGS" "$RESULTS"

# Always revalidate an existing derived file's shape and provenance before reuse.
# No NPU is initialized until the CPU capture-preparation gate succeeds.
if [[ ! -f "$CAPTURE" ]]; then
  python -m tests.models.mcore.tpr.parallel.prepare_tpr_cp1_n8_perf_capture \
    "$SOURCE" "$CAPTURE" --prefix "$P" --suffix "$S" \
    2>&1 | tee "$LOGS/prepare.log"
fi
python - "$AUDIT" "$SOURCE" "$P" "$S" <<'PY' 2>&1 | tee "$LOGS/audit.log"
import json, pathlib, sys
audit = json.loads(pathlib.Path(sys.argv[1]).read_text())
source = pathlib.Path(sys.argv[2]).resolve()
p, s = int(sys.argv[3]), int(sys.argv[4])
assert pathlib.Path(audit["source"]).resolve() == source
assert (audit["prefix"], audit["suffix"], audit["length"], audit["n"]) == (p, s, p+s, 8)
assert audit["forest_trees"] == 1 and audit["forest_segments"] >= 9
assert p-127 <= audit["actual_shared_root_tokens"] < p+s
assert audit["min_turn_segment_tokens"] == 128
assert audit["synthetic_shared_uid"] and audit["changed_conditioning"]
print("TPR_CP_N8_CAPTURE_AUDIT_PASS", audit["input_sha256"],
      "actual_shared_root_tokens=", audit["actual_shared_root_tokens"], flush=True)
PY

if [[ "${TPR_SHAPE_PREFLIGHT_ONLY:-0}" == "1" ]]; then
  echo "TPR_CP_N8_PREFLIGHT_ONLY_SUCCESS ratio=$RATIO capture=$CAPTURE"
  exit 0
fi

if [[ "$CP" == "2" ]]; then
  EXPECTED_DEVICES="8,9,10,11"
else
  EXPECTED_DEVICES="8,9,10,11,12,13,14,15"
fi
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-$EXPECTED_DEVICES}"
if [[ "$ASCEND_RT_VISIBLE_DEVICES" != "$EXPECTED_DEVICES" ]]; then
  echo "TP2 CP$CP requires Ascend devices $EXPECTED_DEVICES" >&2
  exit 2
fi
WORLD=$((2 * CP))
export TPR_QWEN_4B_PATH="${TPR_QWEN_4B_PATH:-/workspace/hf_models/Qwen3-4B-128K}"
[[ -d "$TPR_QWEN_4B_PATH" ]] || { echo "Missing Qwen3-4B checkpoint" >&2; exit 2; }
export HCCL_DETERMINISTIC=true
export TPR_OFFLINE_GRAD_SYNC_DTYPE=bf16
export TPR_TREE_POLICY=qwen_turn
export TPR_TURN_MIN_SEGMENT_TOKENS=128
export TPR_BENCH_CONTROLLED_N8_AUDIT="$AUDIT"
export TPR_BENCH_SOURCE_STEP=0
export TPR_BENCH_OUTPUT_DIR="$RESULTS"
export TPR_RUN_FIXED_SHAPE_40K_BENCH=1
export TPR_BENCH_CP_SIZE="$CP"
export TPR_BENCH_CAPTURE_LOGPROBS=0
export TPR_BENCH_RECOMPUTE=0
export TPR_BENCH_MEM_AUDIT=0
export TPR_BENCH_GRAD_WITNESS=0
export TPR_OFFLINE_NATIVE_RECOMPUTE_LAYERS=0
export TPR_EXEC_PROFILE=0
export TPR_BENCH_WARMUPS="${TPR_BENCH_WARMUPS:-1}"
export TPR_BENCH_REPEATS="${TPR_BENCH_REPEATS:-3}"
unset TPR_MEM_TRACE TPR_FIRST_LEAF_DIAG TPR_FIRST_LEAF_DIAG_SYNC || true
# Keep pytest short flags out of sys.argv: MindSpeed parses unknown -q/-s/-x
# as empty config field names and make_dataclass(TransformerConfig) then fails.
# PYTEST_ADDOPTS is consumed by pytest, not by MindSpeed argv parsing.
export PYTEST_ADDOPTS="-q -s -x"

failed=0
for mode in full tpr; do
  export TPR_OFFLINE_MODE="$mode"
  echo "TPR_CP_N8_START cp=$CP ratio=$RATIO mode=$mode p=$P s=$S"
  if ! torchrun --nproc_per_node="$WORLD" --master_port="${TPR_CP_N8_PORT:-${TPR_CP4_N8_PORT:-29832}}" \
      -m pytest \
      tests/models/mcore/tpr/parallel/test_tpr_fixed_shape_40k_benchmark_npu.py \
      2>&1 | tee "$LOGS/${mode}.log"; then
    echo "TPR_CP_N8_FAILED cp=$CP ratio=$RATIO mode=$mode. Inspect OOM/errors in $LOGS." >&2
    failed=1
  fi
done
if ((failed)); then
  echo "At least one mode failed; preserve individual results and logs, do not report speedup." >&2
  exit 1
fi

python - "$RESULTS" "$RATIO" "$P" "$S" "$CP" <<'PY' 2>&1 | tee "$LOGS/compare.log"
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
ratio, p, s, cp = sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
full = json.loads((root/"step0_full_fixed_shape.json").read_text())
tpr = json.loads((root/"step0_tpr_fixed_shape.json").read_text())
for key in ("input_sha256", "source_capture", "checkpoint", "warmups", "repeats",
            "effective_sequence_cap", "capture_logprobs", "recompute", "profiler_enabled",
            "gradient_sync_reduce_dtype", "hccl_deterministic", "topology"):
    assert full[key] == tpr[key], f"Full/TPR mismatch: {key}"
assert full["controlled_performance_shape"] == tpr["controlled_performance_shape"]
assert full["controlled_performance_shape"]["prefix"] == p
assert full["controlled_performance_shape"]["suffix"] == s
assert full["topology"] == {"tp":2,"cp":cp,"dp":1,"world":2*cp}
assert not full["updated_weights"] and not tpr["updated_weights"]
assert not full["recompute"] and not tpr["recompute"]
f,t=full["median_fwd_bwd_s"],tpr["median_fwd_bwd_s"]
assert f>0 and t>0
report = {
    "ratio": ratio, "prefix": p, "suffix": s, "length": p+s, "n": 8,
    "world": 2*cp, "topology": full["topology"], "recompute": False,
    "optimizer_included": False,
    "baseline": "NATIVE_MEGATRON_MINDSPEED_FULL",
    "tpr": "TPR_RING_QWEN_TURN",
    "input_sha256": full["input_sha256"],
    "actual_shared_root_tokens": full["controlled_performance_shape"]["actual_shared_root_tokens"],
    "unique_tree_tokens": full["controlled_performance_shape"]["unique_tree_tokens"],
    "full_median_fwd_bwd_s": f, "tpr_median_fwd_bwd_s": t,
    "speedup": f/t,
    "full_peak_allocated_gib": max(v["peak_allocated_gib"] for v in full["runs"] if not v["warmup"]),
    "tpr_peak_allocated_gib": max(v["peak_allocated_gib"] for v in tpr["runs"] if not v["warmup"]),
    "performance_only_changed_conditioning": True,
}
target = root/"comparison.json"
if target.exists(): raise FileExistsError(f"Refuse to overwrite: {target}")
target.write_text(json.dumps(report,indent=2))
print("TPR_CP_N8_COMPARE "+json.dumps(report,sort_keys=True),flush=True)
PY
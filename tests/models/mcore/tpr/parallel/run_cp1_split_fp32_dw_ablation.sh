#!/usr/bin/env bash
# Compare existing Full/Split BF16 baselines against all-layer FP32 Linear dW,
# then reverse to BF16 to verify the original split error returns.
set -euo pipefail
cd "$(dirname "$0")/../../../../.."

BASE="${TPR_CP1_BASE:-/workspace/tpr_experiments/cp1_sequence_split_4k_precision_20261025_v1}"
ROOT="${TPR_CP1_FP32_DW_ROOT:-/workspace/tpr_experiments/cp1_sequence_split_all_linear_dw_fp32_20261025_v1}"
CHECKPOINT="${TPR_QWEN_4B_PATH:-/workspace/hf_models/Qwen3-4B-128K}"
CAPTURE="$BASE/actor_pair_cap4096_p3072.pt"

[[ -f "$CAPTURE" ]] || { echo "Missing capture: $CAPTURE" >&2; exit 2; }
[[ -d "$CHECKPOINT" ]] || { echo "Missing checkpoint: $CHECKPOINT" >&2; exit 2; }
[[ -f "$BASE/full/result.json" ]] || { echo "Missing baseline: $BASE/full/result.json" >&2; exit 2; }
[[ -f "$BASE/split_no_share/result.json" ]] || { echo "Missing baseline: $BASE/split_no_share/result.json" >&2; exit 2; }
[[ ! -e "$ROOT" ]] || { echo "Output exists: $ROOT (set TPR_CP1_FP32_DW_ROOT to a fresh path)" >&2; exit 2; }

# Reuse the existing full-gradient baselines; fail early if the artifacts are incomplete.
python - "$BASE" <<'PY'
import json, sys
from pathlib import Path
base = Path(sys.argv[1])
for mode in ("full", "split_no_share"):
    record = json.loads((base / mode / "result.json").read_text())
    grad_dir = record.get("full_grad_dir")
    if not grad_dir or not (Path(grad_dir) / "manifest.pt").is_file():
        raise SystemExit(f"Missing complete gradient manifest for baseline mode={mode}: {grad_dir}")
    print(f"BASELINE_READY mode={mode} gradients={grad_dir}", flush=True)
PY

mkdir -p "$ROOT"
export TPR_CP1_CAPTURE="$CAPTURE"
export TPR_QWEN_4B_PATH="$CHECKPOINT"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-8}"
[[ "$ASCEND_RT_VISIBLE_DEVICES" == "8" ]] || {
  echo "Only single physical Ascend NPU 8 is allowed" >&2; exit 2;
}
export HCCL_DETERMINISTIC=true
export TPR_RUN_CP1_SHORT_ACTOR=1
export TPR_TREE_POLICY=qwen_turn
export TPR_TURN_MIN_SEGMENT_TOKENS=128
export TPR_PPO_LOGPROB_CHUNK_TOKENS=256
export TPR_CP1_SAVE_FULL_GRADS=1
export TPR_CP1_TRACE=0
export TPR_EXEC_PROFILE=0
export TPR_OFFLINE_GRAD_SYNC_DTYPE=bf16
export PYTHONUNBUFFERED=1
unset TPR_QWEN17_SPLIT_FP32_DW_BACKWARD TPR_CP1_SPLIT_FP32_LINEAR_DW
unset TPR_CP1_PROGRESS_LAYERS

python tests/models/mcore/tpr/parallel/_mindspeed_first_pytest.py \
  --bootstrap-only 2>&1 | tee "$ROOT/bootstrap.log"

run_split() {
  local label="$1"
  local use_fp32="$2"
  local port="$3"

  mkdir -p "$ROOT/$label"
  export TPR_CP1_MODE=split_no_share
  export TPR_CP1_OUTPUT="$ROOT/$label/result.json"
  if [[ "$use_fp32" == "1" ]]; then
    export TPR_CP1_SPLIT_FP32_LINEAR_DW=1
  else
    unset TPR_CP1_SPLIT_FP32_LINEAR_DW
  fi

  echo "CP1_FP32_DW_START mode=$label use_fp32=$use_fp32"
  torchrun --nproc_per_node=1 \
    --master_addr=127.0.0.1 \
    --master_port="$port" \
    tests/models/mcore/tpr/parallel/_mindspeed_first_pytest.py \
    -xvs tests/models/mcore/tpr/parallel/test_real_actor_cp1_short_full_tpr_npu.py \
    2>&1 | tee "$ROOT/$label.log"
  local rc=${PIPESTATUS[0]}
  echo "CP1_FP32_DW_RUN_STATUS mode=$label exit_code=$rc"
  return "$rc"
}

set +e
run_split split_fp32 1 29806
FP32_RC=$?
run_split split_bf16_reverse 0 29807
BF16_RC=$?
set -e

echo
echo "========== RUN STATUS =========="
echo "fp32_exit_code=$FP32_RC"
echo "bf16_reverse_exit_code=$BF16_RC"
echo "root=$ROOT"

echo
echo "========== KEY LOG =========="
grep -E \
  'TPR_CP1_GRADIENT_INTERVENTION|TPR_CP1_SHORT_RESULT|FP32 dW|AssertionError|Traceback|ERROR|FAILED|passed|failed' \
  "$ROOT/split_fp32.log" "$ROOT/split_bf16_reverse.log" 2>/dev/null || true

if [[ "$FP32_RC" -ne 0 || "$BF16_RC" -ne 0 ]]; then
  echo "ERROR: one or both intervention runs failed; see logs above" >&2
  exit 1
fi

# Compare complete gradients, not sampled witnesses, and prove the FP32
# intervention changes no forward values and BF16 reversal reproduces baseline.
python - "$BASE" "$ROOT" <<'PY'
import json
import sys
from pathlib import Path

from tests.models.mcore.tpr.parallel.compare_real_actor_cp1_short_full_tpr import (
    _complete_cp1_parameter_gradient_metrics, _metrics,
)

base, root = map(Path, sys.argv[1:])
records = {
    "full": json.loads((base / "full/result.json").read_text()),
    "split_bf16": json.loads((base / "split_no_share/result.json").read_text()),
    "split_fp32": json.loads((root / "split_fp32/result.json").read_text()),
    "split_bf16_reverse": json.loads((root / "split_bf16_reverse/result.json").read_text()),
}
full = records["full"]
baseline = records["split_bf16"]
candidate = records["split_fp32"]
reverse = records["split_bf16_reverse"]

common = (
    "source_capture", "input_sha256", "checkpoint",
    "initial_weights_witness_sha256", "ppo_tokens",
    "original_keys", "logical_input_tokens",
)
for label, record in records.items():
    if label != "full" and record["mode"] != "split_no_share":
        raise AssertionError(f"{label}: expected split_no_share result")
    if not record.get("full_grad_dir"):
        raise AssertionError(f"{label}: complete parameter gradients were not saved")
    if not (Path(record["full_grad_dir"]) / "manifest.pt").is_file():
        raise AssertionError(f"{label}: missing full-gradient manifest")
    for key in common:
        if record[key] != full[key]:
            raise AssertionError(f"{label}: mismatched {key}")

intervention = candidate.get("gradient_intervention", {})
if not intervention.get("enabled"):
    raise AssertionError("FP32 candidate did not record the intervention as enabled")
if intervention.get("layers") != 36 or intervention.get("linear_modules") != 144:
    raise AssertionError(f"Incomplete layer/module coverage: {intervention}")
if not intervention.get("all_modules_backwarded") or not intervention.get("flushed_before_native_finalizer"):
    raise AssertionError(f"FP32 dW intervention did not complete correctly: {intervention}")

def logprobs(record):
    values = {(int(row), int(offset)): float(value) for row, offset, value in record["logprobs"]}
    if len(values) != record["ppo_tokens"] or len(values) != len(record["logprobs"]):
        raise AssertionError("Missing or duplicate PPO logprob coordinates")
    return values

def forward_equal(a, b, label):
    if a["loss"] != b["loss"] or logprobs(a) != logprobs(b):
        raise AssertionError(f"{label}: dW-only intervention changed forward loss/logprobs")

forward_equal(baseline, candidate, "BF16 vs all-layer FP32 dW")
forward_equal(baseline, reverse, "BF16 baseline vs BF16 reverse")

def compare(label, a, b):
    grad = _complete_cp1_parameter_gradient_metrics(a, b)
    if grad is None:
        raise AssertionError(f"{label}: complete gradient comparison unavailable")
    la, lb = logprobs(a), logprobs(b)
    keys = sorted(la)
    return {
        "loss_abs_gap": abs(a["loss"] - b["loss"]),
        "logprob": _metrics([la[k] for k in keys], [lb[k] for k in keys]),
        "full_parameter_gradient": grad,
    }

comparisons = {
    "full_vs_split_bf16": compare("full_vs_split_bf16", full, baseline),
    "full_vs_split_all_linear_dw_fp32": compare("full_vs_split_all_linear_dw_fp32", full, candidate),
    "split_bf16_vs_split_all_linear_dw_fp32": compare("split_bf16_vs_split_all_linear_dw_fp32", baseline, candidate),
    "split_bf16_vs_bf16_reverse": compare("split_bf16_vs_bf16_reverse", baseline, reverse),
}
reverse_error = comparisons["split_bf16_vs_bf16_reverse"]["full_parameter_gradient"]["relative_l2"]
if reverse_error > 1e-6:
    raise AssertionError(f"BF16 reverse did not reproduce baseline gradients: rel_l2={reverse_error}")

baseline_error = comparisons["full_vs_split_bf16"]["full_parameter_gradient"]["relative_l2"]
fp32_error = comparisons["full_vs_split_all_linear_dw_fp32"]["full_parameter_gradient"]["relative_l2"]
report = {
    "scope": "Qwen3-4B BF16, CP1, sequence split without shared prefix",
    "intervention": intervention,
    "forward_loss_and_logprobs_unchanged": True,
    "bf16_reverse_reproduced_baseline": True,
    "full_vs_split_bf16_gradient_rel_l2": baseline_error,
    "full_vs_split_all_linear_dw_fp32_gradient_rel_l2": fp32_error,
    "gradient_error_reduction_factor": baseline_error / max(fp32_error, 1e-30),
    "fp32_reduced_full_gradient_error": fp32_error < baseline_error,
    "comparisons": comparisons,
}
output = root / "comparison_fp32_dw_ablation.json"
output.write_text(json.dumps(report, indent=2))

for name, item in comparisons.items():
    grad = item["full_parameter_gradient"]
    print(
        "CP1_FP32_DW_COMPARE", name,
        f"loss_gap={item['loss_abs_gap']:.9g}",
        f"logprob_rel_l2={item['logprob']['relative_l2']:.9g}",
        f"grad_rel_l2={grad['relative_l2']:.9g}",
        f"grad_cos={grad['cosine']:.9g}",
        f"norm_relative_gap={grad['norm_relative_gap']:.9g}",
        flush=True,
    )
print(
    "CP1_FP32_DW_EFFECT",
    f"baseline_grad_rel_l2={baseline_error:.9g}",
    f"all_layer_fp32_grad_rel_l2={fp32_error:.9g}",
    f"error_reduction_factor={baseline_error/max(fp32_error, 1e-30):.6g}",
    f"improved={fp32_error < baseline_error}",
    flush=True,
)
print("CP1_FP32_DW_TOP_ERRORS full_vs_split_all_linear_dw_fp32:")
for name, error in comparisons["full_vs_split_all_linear_dw_fp32"]["full_parameter_gradient"]["top_absolute_gradient_errors"][:8]:
    print(f"  {name}: {error:.9g}", flush=True)
print(f"CP1_FP32_DW_REPORT={output}", flush=True)
print("CP1_FP32_DW_ABLATION_VALIDATED forward_unchanged=True bf16_reverse_reproduced=True", flush=True)
PY

echo
echo "========== FINAL STATUS =========="
echo "fp32_exit_code=$FP32_RC"
echo "bf16_reverse_exit_code=$BF16_RC"
echo "report=$ROOT/comparison_fp32_dw_ablation.json"
echo "CP1_FP32_DW_ABLATION_PASS"

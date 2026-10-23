#!/usr/bin/env bash
# Single Ascend NPU, Qwen3-4B BF16, <=4096 tokens per trajectory, two siblings.
# Strict numerical scope: full sequence -> split without reuse -> split with reuse.
# Reuses the existing CP1 test/model/SegmentExecutor and original Actor labels.
# NO TP/CP/DP communication, DDP, optimizer, recompute, or performance acceptance.
set -euo pipefail
cd "$(dirname "$0")/../../../../.."

SOURCE="${TPR_CP1_SOURCE:-/workspace/tpr_actor_capture_4b_n8_run1/actor_update_rank0_batch0.pt}"
ROOT="${TPR_CP1_ROOT:-/workspace/tpr_experiments/cp1_sequence_split_4k_v1}"
CHECKPOINT="${TPR_QWEN_4B_PATH:-/workspace/hf_models/Qwen3-4B-128K}"

[[ -f "$SOURCE" ]] || { echo "Missing captured Actor pair: $SOURCE (set TPR_CP1_SOURCE)" >&2; exit 2; }
[[ -d "$CHECKPOINT" ]] || { echo "Missing Qwen3-4B checkpoint: $CHECKPOINT" >&2; exit 2; }
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
unset TPR_SELECTED_LOGITS_FP32 TPR_QWEN17_SPLIT_FP32_DW_BACKWARD
unset TPR_ROOT_NATIVE_RING TPR_RING_COALESCE_PREFIX_FULL
unset TPR_RING_COALESCE_PREFIX_QUERY TPR_RING_COALESCE_PREFIX_PADDED

mkdir -p "$ROOT"
export TPR_CP1_CAPTURE="$ROOT/actor_pair_cap4096_p3072.pt"
if [[ ! -e "$TPR_CP1_CAPTURE" ]]; then
  python -m tests.models.mcore.tpr.parallel.prepare_tpr_cp1_short_capture \
    "$SOURCE" "$TPR_CP1_CAPTURE" --rows 4,6 --cap 4096 --prompt-window 3072
fi

python - "$SOURCE" "$TPR_CP1_CAPTURE" <<'PY'
import json, sys
from pathlib import Path
source, capture = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
audit = json.loads(capture.with_suffix(".audit.json").read_text())
assert audit["source"] == str(source)
assert audit["cap"] == 4096 and audit["prompt_window"] == 3072
assert audit["original_rows"] == [4, 6]
assert len(audit["retained_lengths"]) == 2
assert all(1 < n <= 4096 for n in audit["retained_lengths"])
assert all(n > 0 for n in audit["supervised_tokens_per_row"])
assert audit["conditioning_changed"]
print("CP1_4K_CAPTURE_AUDIT", json.dumps(audit, sort_keys=True), flush=True)
PY

# Three arms are necessary to attribute an error to splitting vs sharing:
#   full:          two ordinary complete GPT calls (no attention context)
#   split_no_share: two separate Prefix -> Suffix chains on the same executor
#   shared:        one Prefix -> two sibling Suffix chains on the same executor
# Each independent process loads the identical checkpoint and cropped input.
for mode in full split_no_share shared; do
  export TPR_CP1_MODE="$mode"
  export TPR_CP1_OUTPUT="$ROOT/$mode/result.json"
  if [[ -f "$TPR_CP1_OUTPUT" ]]; then
    python - "$TPR_CP1_OUTPUT" <<'PY'
import json, sys
from pathlib import Path
item = json.loads(Path(sys.argv[1]).read_text())
directory = item.get("full_grad_dir")
assert directory and (Path(directory) / "manifest.pt").is_file(), (
    "Stale partial result without complete BF16 gradients; use a new TPR_CP1_ROOT"
)
PY
    echo "CP1_4K_REUSE mode=$mode"
    continue
  fi
  [[ ! -e "$ROOT/$mode" ]] || {
    echo "Partial output for $mode; use a new TPR_CP1_ROOT" >&2; exit 2;
  }
  echo "CP1_4K_START mode=$mode"
  torchrun --nproc_per_node=1 --master_addr=127.0.0.1 --master_port=29797 \
    -m pytest -xvs \
    tests/models/mcore/tpr/parallel/test_real_actor_cp1_short_full_tpr_npu.py \
    2>&1 | tee "$ROOT/$mode.log"
done

# Full parameter gradient compare is intentionally REQUIRED, not sampled.
# This report does not use performance, optimizer, or distributed correctness.
python - "$ROOT" <<'PY'
import json, math, os, sys
from pathlib import Path
from tests.models.mcore.tpr.parallel.compare_real_actor_cp1_short_full_tpr import (
    _complete_cp1_parameter_gradient_metrics, _metrics,
)
root = Path(sys.argv[1])
records = {
    mode: json.loads((root / mode / "result.json").read_text())
    for mode in ("full", "split_no_share", "shared")
}
full = records["full"]
for mode, record in records.items():
    assert record["mode"] == mode
    assert record["topology"] == {"tp": 1, "cp": 1, "dp": 1, "world": 1}
    assert not record["optimizer_step"]
    for key in (
        "source_capture", "input_sha256", "checkpoint",
        "initial_weights_witness_sha256", "ppo_tokens",
        "original_keys", "logical_input_tokens",
    ):
        assert record[key] == full[key], (mode, key)
assert (full["forest_trees"], full["forest_segments"]) == (2, 2)
assert (records["split_no_share"]["forest_trees"],
        records["split_no_share"]["forest_segments"]) == (2, 4)
assert (records["shared"]["forest_trees"],
        records["shared"]["forest_segments"]) == (1, 3)
assert records["split_no_share"]["unique_tree_tokens"] == full["logical_input_tokens"]
assert records["shared"]["unique_tree_tokens"] < full["logical_input_tokens"]

logprobs = {
    mode: {(int(row), int(offset)): float(value) for row, offset, value in r["logprobs"]}
    for mode, r in records.items()
}
keys = sorted(logprobs["full"])
assert len(keys) == full["ppo_tokens"]
assert all(set(values) == set(keys) for values in logprobs.values())
tolerances = {
    "logprob_relative_l2": float(os.getenv("TPR_CP1_LOGPROB_REL_TOL", "0.05")),
    "gradient_relative_l2": float(os.getenv("TPR_CP1_GRAD_REL_TOL", "0.05")),
    "loss_abs": float(os.getenv("TPR_CP1_LOSS_ABS_TOL", "0.02")),
}
assert all(math.isfinite(v) and v > 0 for v in tolerances.values())
pairs = {}
for label, left, right in (
    ("full_vs_split_no_share", "full", "split_no_share"),
    ("split_no_share_vs_shared", "split_no_share", "shared"),
    ("full_vs_shared", "full", "shared"),
):
    a, b = records[left], records[right]
    grad = _complete_cp1_parameter_gradient_metrics(a, b)
    assert grad is not None and grad["complete_parameter_elements"] > 0
    logp = _metrics([logprobs[left][k] for k in keys],
                    [logprobs[right][k] for k in keys])
    loss_gap = abs(a["loss"] - b["loss"])
    gates = {
        "loss": loss_gap <= tolerances["loss_abs"],
        "logprob": logp["relative_l2"] <= tolerances["logprob_relative_l2"],
        "all_parameter_gradient": grad["relative_l2"] <= tolerances["gradient_relative_l2"],
    }
    pairs[label] = {
        "loss_abs_gap": loss_gap,
        "logprob": logp,
        "full_parameter_gradient": grad,
        "gates": gates,
        "passed": all(gates.values()),
    }
report = {
    "scope": "single_Ascend_NPU_TP1_CP1_DP1_no_DDP_optimizer_recompute",
    "model": "Qwen3-4B BF16",
    "per_trajectory_cap": 4096,
    "prompt_window": 3072,
    "shared_prefix_physical_tokens": (
        full["logical_input_tokens"] - records["shared"]["unique_tree_tokens"]
    ),
    "ppo_tokens": full["ppo_tokens"],
    "input_sha256": full["input_sha256"],
    "tolerances": tolerances,
    "comparisons": pairs,
    "all_gates_passed": all(x["passed"] for x in pairs.values()),
    "note": (
        "Controlled cropped Actor PPO labels (conditioning changed). "
        "The full vs split_no_share contrast isolates sequence splitting; "
        "split_no_share vs shared isolates prefix reuse. "
        "Compares ALL BF16 parameter gradients, not DDP main_grad samples. "
        "Thresholds are provisional diagnostics, not bitwise parity."
    ),
}
output = root / "comparison_cp1_4k_sequence_split.json"
output.write_text(json.dumps(report, indent=2))
for label, item in pairs.items():
    print(
        "CP1_4K_PARITY", label,
        f"loss_gap={item['loss_abs_gap']:.8g}",
        f"logprob_rel_l2={item['logprob']['relative_l2']:.8g}",
        f"full_grad_rel_l2={item['full_parameter_gradient']['relative_l2']:.8g}",
        f"full_grad_cos={item['full_parameter_gradient']['cosine']:.8g}",
        f"passed={item['passed']}",
        flush=True,
    )
print(f"CP1_4K_REPORT={output} all_gates_passed={report['all_gates_passed']}", flush=True)
if not report["all_gates_passed"]:
    raise SystemExit(1)
PY

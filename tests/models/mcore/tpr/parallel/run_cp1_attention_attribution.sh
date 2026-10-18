#!/usr/bin/env bash
# Single command. Runs genuine CP1 Qwen3-4B PPO numerical attribution and
# prints an AUTOMATIC final conclusion. Detailed logs remain archived.
# No Fixed Tile, no modified Linear/backward, no optimizer update.
set -euo pipefail
cd "$(dirname "$0")/../../../../.."

export TPR_CP1_ROOT="${TPR_CP1_AUDIT_ROOT:-/workspace/tpr_experiments/cp1_attention_attribution_2k}"
export TPR_CP1_TRACE=1
export TPR_CP1_TRACE_DIR="$TPR_CP1_ROOT"
export TPR_CP1_CAP=2048
export TPR_CP1_PROMPT_WINDOW=1024
export TPR_ROOT_NATIVE_RING=0
export TPR_TREE_POLICY=qwen_turn
export TPR_RING_COALESCE_PREFIX_FULL=0
export TPR_RING_COALESCE_PREFIX_QUERY=0
export TPR_RING_COALESCE_PREFIX_PADDED=0
# Never import or enable the historical test-only Fixed Tile interventions.
unset TPR_QWEN17_PPO_TILE_GEMM TPR_QWEN17_GPT_TILE_GEMM \
  TPR_QWEN17_GPT_FP32_GEMM TPR_QWEN17_GPT_GROUPED_GEMM_GROUPS \
  TPR_QWEN17_SPLIT_FP32_DW_BACKWARD || true
export TPR_CP1_SOURCE="${TPR_CP1_SOURCE:-/workspace/tpr_actor_capture_4b_n8_run1/actor_update_rank0_batch0.pt}"
export TPR_QWEN_4B_PATH="${TPR_QWEN_4B_PATH:-/workspace/hf_models/Qwen3-4B-128K}"
: "${ASCEND_RT_VISIBLE_DEVICES:=8}"
export ASCEND_RT_VISIBLE_DEVICES

if [[ "$ASCEND_RT_VISIBLE_DEVICES" != "8" ]]; then
  echo "ERROR: isolate single physical Ascend 8 for CP1 attribution" >&2
  exit 2
fi
if [[ ! -f "$TPR_CP1_SOURCE" || ! -d "$TPR_QWEN_4B_PATH" ]]; then
  echo "ERROR: missing real Actor PPO capture or authentic Qwen3-4B checkpoint" >&2
  exit 2
fi
if [[ -e "$TPR_CP1_ROOT/ATTENTION_ATTRIBUTION_FINAL.json" ]]; then
  echo "Existing validated audit (immutable):"
  cat "$TPR_CP1_ROOT/ATTENTION_ATTRIBUTION_CONCLUSION.txt"
  exit 0
fi
mkdir -p "$TPR_CP1_ROOT"

# Catch syntax, stage-coordinate and diagnostic attribution errors before
# spending NPU time on four separate 4B checkpoint loads.
python -m py_compile \
  tests/models/mcore/tpr/parallel/cp1_attention_attribution.py \
  tests/models/mcore/tpr/parallel/analyze_cp1_attention_attribution.py \
  tests/models/mcore/tpr/parallel/test_real_actor_cp1_short_full_tpr_npu.py
if ! python -m pytest -q \
  tests/models/mcore/tpr/parallel/test_cp1_attention_attribution_cpu.py \
  > "$TPR_CP1_ROOT/cpu_preflight.log" 2>&1; then
  echo "FAILED CPU attribution preflight:" >&2
  tail -n 100 "$TPR_CP1_ROOT/cpu_preflight.log" >&2
  exit 1
fi

echo "CP1 4B attention attribution: isolated NPU8, 2K, authentic Actor PPO."
echo "Detailed NPU logs: $TPR_CP1_ROOT/full_run.log"
# Reuses existing loss/witness procedures; tracing is an opt-in test hook.
# Full -> split_no_share -> shared is required by the QKV replay.
if ! bash tests/models/mcore/tpr/parallel/run_real_actor_cp1_short_full_tpr.sh \
    > "$TPR_CP1_ROOT/full_run.log" 2>&1; then
  echo "FAILED during NPU attribution. Last 100 log lines:" >&2
  tail -n 100 "$TPR_CP1_ROOT/full_run.log" >&2
  exit 1
fi

for mode in full split_no_share shared; do
  if [[ ! -f "$TPR_CP1_ROOT/$mode/trace.pt" ]]; then
    echo "ERROR: missing $mode trace; the runner reused a non-instrumented snapshot" >&2
    exit 1
  fi
done
if [[ ! -f "$TPR_CP1_ROOT/comparison_split_no_share.json" ]]; then
  echo "ERROR: prior four-arm PPO comparator did not complete" >&2
  exit 1
fi

# Prints a human-readable verdict; no manual log inspection needed.
python -m tests.models.mcore.tpr.parallel.analyze_cp1_attention_attribution \
  "$TPR_CP1_ROOT"

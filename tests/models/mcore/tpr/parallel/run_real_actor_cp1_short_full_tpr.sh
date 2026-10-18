#!/usr/bin/env bash
# Qwen3-4B real-label Actor PPO: one NPU, 2K Full / one-node / shared TPR.
# Run inside the uniagent_16npu Docker container from the synced VERL worktree.
# For strict device isolation reserve physical NPU 8 and port 29797.
set -euo pipefail
cd "$(dirname "$0")/../../../../.."

SOURCE="${TPR_CP1_SOURCE:-/workspace/tpr_actor_capture_4b_n8_run1/actor_update_rank0_batch0.pt}"
ROOT="${TPR_CP1_ROOT:-/workspace/tpr_experiments/cp1_full_tpr_2k}"
CAP="${TPR_CP1_CAP:-2048}"
PROMPT_WINDOW="${TPR_CP1_PROMPT_WINDOW:-$((CAP / 2))}"
if ((CAP < 128 || CAP > 4096 || PROMPT_WINDOW < 1 || PROMPT_WINDOW >= CAP)); then
  echo "Invalid short capture cap/prompt_window: $CAP/$PROMPT_WINDOW" >&2
  exit 2
fi
export TPR_QWEN_4B_PATH="${TPR_QWEN_4B_PATH:-/workspace/hf_models/Qwen3-4B-128K}"
: "${ASCEND_RT_VISIBLE_DEVICES:=8}"
export ASCEND_RT_VISIBLE_DEVICES
if [[ "$ASCEND_RT_VISIBLE_DEVICES" != "8" ]]; then
  echo "Single-card parity requires ASCEND_RT_VISIBLE_DEVICES=8, got $ASCEND_RT_VISIBLE_DEVICES" >&2
  exit 2
fi
[[ -f "$SOURCE" ]] || { echo "Missing source Actor PPO capture: $SOURCE" >&2; exit 2; }
[[ -d "$TPR_QWEN_4B_PATH" ]] || { echo "Missing real Qwen3-4B checkpoint" >&2; exit 2; }

export HCCL_DETERMINISTIC=true
export TPR_OFFLINE_GRAD_SYNC_DTYPE=bf16
export TPR_RUN_CP1_SHORT_ACTOR=1
export TPR_TREE_POLICY=qwen_turn
export TPR_TURN_MIN_SEGMENT_TOKENS=128
export TPR_PPO_LOGPROB_CHUNK_TOKENS=256
export TPR_ROOT_NATIVE_RING=0
export TPR_RING_COALESCE_PREFIX_FULL=0
export TPR_RING_COALESCE_PREFIX_QUERY=0
export TPR_RING_COALESCE_PREFIX_PADDED=0
export TPR_EXEC_PROFILE=0
export PYTHONUNBUFFERED=1
unset TPR_MEM_TRACE TPR_FIRST_LEAF_DIAG TPR_FIRST_LEAF_DIAG_SYNC || true

mkdir -p "$ROOT"
export TPR_CP1_CAPTURE="$ROOT/actor_pair_cap${CAP}_p${PROMPT_WINDOW}.pt"
if [[ ! -e "$TPR_CP1_CAPTURE" ]]; then
  python -m tests.models.mcore.tpr.parallel.prepare_tpr_cp1_short_capture \
    "$SOURCE" "$TPR_CP1_CAPTURE" --rows 4,6 --cap "$CAP" --prompt-window "$PROMPT_WINDOW"
fi
[[ -f "$TPR_CP1_CAPTURE" ]] || { echo "Missing derived capture" >&2; exit 2; }
[[ -f "${TPR_CP1_CAPTURE%.pt}.audit.json" ]] || {
  echo "Missing short capture audit; refusing stale input" >&2; exit 2;
}
python - "$SOURCE" "$TPR_CP1_CAPTURE" "$CAP" "$PROMPT_WINDOW" <<'PY'
import json, sys
from pathlib import Path
source, capture = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
cap, prompt = int(sys.argv[3]), int(sys.argv[4])
audit = json.loads(capture.with_suffix(".audit.json").read_text())
assert audit["source"] == str(source), "Derived capture belongs to a different source"
assert audit["cap"] == cap and audit["prompt_window"] == prompt, "Old crop settings"
assert audit["original_rows"] == [4, 6], "Wrong sibling source rows"
assert all(x <= cap for x in audit["retained_lengths"])
assert all(n > 0 for n in audit["supervised_tokens_per_row"])
PY

# Each arm loads exactly the same initial Qwen3-4B weights from checkpoint.
# A separate torchrun exits after backward; no optimizer step is performed.
for mode in full one_node split_no_share shared; do
  export TPR_CP1_MODE="$mode"
  export TPR_CP1_OUTPUT="$ROOT/$mode/result.json"
  if [[ -f "$TPR_CP1_OUTPUT" ]]; then
    echo "TPR_CP1_SHORT_REUSE mode=$mode result=$TPR_CP1_OUTPUT"
    continue
  fi
  if [[ -e "$ROOT/$mode" ]]; then
    echo "Refusing nonempty partial case directory: $ROOT/$mode" >&2
    exit 2
  fi
  echo "TPR_CP1_SHORT_START mode=$mode source=$TPR_CP1_CAPTURE"
  torchrun --master_addr=127.0.0.1 --master_port=29797 \
    --nproc_per_node=1 \
    -m pytest -xvs \
    tests/models/mcore/tpr/parallel/test_real_actor_cp1_short_full_tpr_npu.py \
    2>&1 | tee "$ROOT/$mode.log"
done

if [[ ! -e "$ROOT/comparison_split_no_share.json" ]]; then
  python -m tests.models.mcore.tpr.parallel.compare_real_actor_cp1_short_full_tpr \
    "$ROOT" | tee "$ROOT/comparison.log"
else
  echo "TPR_CP1_SHORT_COMPARISON_REUSE=$ROOT/comparison_split_no_share.json"
fi

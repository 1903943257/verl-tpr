#!/usr/bin/env bash
# Supplement an EXISTING completed N8 comparison with ONLY physical Full MB1.
# Do not re-derive captures or re-run Full MB2 / TPR K1/K2/K4.
# Intended to execute INSIDE the synced leaf-microbatch Docker worktree.
set -euo pipefail
cd "$(dirname "$0")/../../../../.."

ROOT="${TPR_CP1_N8_ROOT:-/workspace/tpr_experiments/cp1_n8_full_mb2_vs_leaf_k_v1}"
export TPR_CP1_N8_CAPTURE="$ROOT/n8_real_actor_short.pt"
NEW_DIR="$ROOT/full_mb1"
CURRENT_LOG="$NEW_DIR/run.log"

if [[ ! -f "$TPR_CP1_N8_CAPTURE" ]]; then
  echo "Missing original N8 derived capture: $TPR_CP1_N8_CAPTURE" >&2
  echo "No synthetic copies: rerun original capture preparation separately if needed." >&2
  false
fi
for arm in full_mb2 tpr_k1 tpr_k2 tpr_k4; do
  if [[ ! -f "$ROOT/$arm/result.json" ]]; then
    echo "Missing original comparison arm $arm; refusing to mix incomplete runs" >&2
    false
  fi
done
if [[ -e "$NEW_DIR" ]]; then
  echo "Refusing to overwrite existing Full MB1 attempt: $NEW_DIR" >&2
  false
fi

# Validate the captured SHA and old four-arm numerical results before NPU load.
python -m tests.models.mcore.tpr.parallel.summarize_cp1_n8_fair \
  "$ROOT" --memory-margin-gib "${TPR_CP1_N8_MEMORY_MARGIN_GIB:-0.5}" \
  > "$ROOT/pre_full_mb1_summary.txt"

export TPR_QWEN_4B_PATH="${TPR_QWEN_4B_PATH:-/workspace/hf_models/Qwen3-4B-128K}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-8}"
if [[ "$ASCEND_RT_VISIBLE_DEVICES" != "8" || ! -d "$TPR_QWEN_4B_PATH" ]]; then
  echo "Full MB1 experiment requires physical NPU 8 and genuine Qwen3-4B model" >&2
  false
fi
export TPR_RUN_CP1_N8_FAIR=1
export TPR_CP1_N8_MODE=full_mb1
export TPR_LEAF_BATCH_SIZE=1
export TPR_CP1_N8_OUTPUT="$NEW_DIR/result.json"
export TPR_CP1_N8_WARMUPS="${TPR_CP1_N8_WARMUPS:-1}"
export TPR_CP1_N8_REPEATS="${TPR_CP1_N8_REPEATS:-3}"
export TPR_CP1_N8_STAGE_PROFILE=0
export TPR_EXEC_PROFILE=0
export TPR_TREE_POLICY=qwen_turn
export TPR_TURN_MIN_SEGMENT_TOKENS=128
export TPR_PPO_LOGPROB_CHUNK_TOKENS=256
export TPR_ROOT_NATIVE_RING=0
export TPR_RING_COALESCE_PREFIX_FULL=0
export TPR_RING_COALESCE_PREFIX_QUERY=0
export TPR_RING_COALESCE_PREFIX_PADDED=0
export HCCL_DETERMINISTIC=true
export TPR_OFFLINE_GRAD_SYNC_DTYPE=bf16
export PYTHONUNBUFFERED=1
unset TPR_CP1_TRACE TPR_CP1_TRACE_DIR TPR_MEM_TRACE TPR_FIRST_LEAF_DIAG TPR_FIRST_LEAF_DIAG_SYNC || true

mkdir "$NEW_DIR"

finish() {
  local rc=$?
  trap - EXIT
  set +e
  echo
  echo "===== N8 FULL MB1 SUPPLEMENT: FINAL SUMMARY (runner_rc=$rc) ====="
  python -m tests.models.mcore.tpr.parallel.summarize_cp1_n8_fair \
    "$ROOT" --memory-margin-gib "${TPR_CP1_N8_MEMORY_MARGIN_GIB:-0.5}" || true
  if ((rc != 0)); then
    echo "===== MB1 CURRENT FAILURE (no historical logs) ====="
    grep -E 'Traceback|ERROR|FAILED|NPU out of memory|^E[[:space:]]' "$CURRENT_LOG" | tail -n 40 || true
    tail -n 60 "$CURRENT_LOG" || true
  fi
  echo "===== MB1 RESULT FILE: $NEW_DIR/result.json ====="
  echo "===== N8 SUMMARY FILE: $ROOT/SUMMARY.txt ====="
  # The original pytest failure remains visible. Script does not alter other
  # arms' result JSON or force an interactive shell logout.
}
trap finish EXIT

echo "===== ONLY NEW ARM: Full MB=1, 8 physical [1,S] forwards ====="
torchrun --master_addr=127.0.0.1 --master_port=29818 \
  --nproc_per_node=1 -m pytest -q -s -x \
  tests/models/mcore/tpr/parallel/test_real_actor_cp1_n8_fair_npu.py \
  2>&1 | tee "$CURRENT_LOG"

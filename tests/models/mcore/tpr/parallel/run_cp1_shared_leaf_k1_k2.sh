#!/usr/bin/env bash
# Exactly two Qwen3-4B CP1 shared-tree runs (K=1 vs K=2).
# Requires an already-derived CP1 short actor capture from the earlier
# run_real_actor_cp1_short_full_tpr.sh script.
set -euo pipefail
cd "$(dirname "$0")/../../../../.."

: "${TPR_CP1_CAPTURE:?Set TPR_CP1_CAPTURE to the previously derived actor_pair_cap*_p*.pt}"
ROOT="${TPR_CP1_LEAF_ROOT:-/workspace/tpr_experiments/leaf_k1_k2_parity}"
export TPR_QWEN_4B_PATH="${TPR_QWEN_4B_PATH:-/workspace/hf_models/Qwen3-4B-128K}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-8}"
if [[ "$ASCEND_RT_VISIBLE_DEVICES" != "8" ]]; then
  echo "Existing CP1 parity gate reserves physical NPU 8; got $ASCEND_RT_VISIBLE_DEVICES" >&2
  exit 2
fi
[[ -f "$TPR_CP1_CAPTURE" ]] || { echo "Missing capture: $TPR_CP1_CAPTURE" >&2; exit 2; }
[[ -d "$TPR_QWEN_4B_PATH" ]] || { echo "Missing Qwen3-4B weights" >&2; exit 2; }
export TPR_RUN_CP1_SHORT_ACTOR=1
export TPR_CP1_MODE=shared
export TPR_TREE_POLICY=qwen_turn
export TPR_TURN_MIN_SEGMENT_TOKENS=128
export TPR_PPO_LOGPROB_CHUNK_TOKENS=256
export TPR_ROOT_NATIVE_RING=0
export TPR_RING_COALESCE_PREFIX_FULL=0
export TPR_RING_COALESCE_PREFIX_QUERY=0
export TPR_RING_COALESCE_PREFIX_PADDED=0
export TPR_EXEC_PROFILE=0
export TPR_LEAF_BATCH_TRACE=1
export TPR_LEAF_BATCH_MAX_TOKENS="${TPR_LEAF_BATCH_MAX_TOKENS:-8192}"
export TPR_OFFLINE_GRAD_SYNC_DTYPE=bf16
export HCCL_DETERMINISTIC=true
export PYTHONUNBUFFERED=1
unset TPR_MEM_TRACE TPR_FIRST_LEAF_DIAG TPR_FIRST_LEAF_DIAG_SYNC || true

for k in 1 2; do
  export TPR_LEAF_BATCH_SIZE="$k"
  export TPR_CP1_OUTPUT="$ROOT/k$k/result.json"
  # Never silently reuse an old comparison produced by a different branch.
  if [[ -e "$ROOT/k$k" ]]; then
    echo "Refusing existing result directory: $ROOT/k$k; select a fresh TPR_CP1_LEAF_ROOT" >&2
    exit 2
  fi
  mkdir -p "$ROOT/k$k"
  echo "TPR_CP1_LEAF_K_START K=$k capture=$TPR_CP1_CAPTURE output=$TPR_CP1_OUTPUT"
  torchrun --master_addr=127.0.0.1 --master_port=29797 \
    --nproc_per_node=1 -m pytest -xvs \
    tests/models/mcore/tpr/parallel/test_real_actor_cp1_short_full_tpr_npu.py \
    2>&1 | tee "$ROOT/k$k.log"
done

python -m tests.models.mcore.tpr.parallel.compare_real_actor_cp1_leaf_batch \
  "$ROOT/k1/result.json" "$ROOT/k2/result.json" \
  --output "$ROOT/k1_vs_k2.json" | tee "$ROOT/comparison.log"

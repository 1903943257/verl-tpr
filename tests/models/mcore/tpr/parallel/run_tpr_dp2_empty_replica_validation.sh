#!/usr/bin/env bash
# One recorded Actor step with DP0 completely unsupervised and DP1 supervised.
# Uses the EXISTING offline gradient finalizer, not production Megatron DDP.
# Usage: bash run_tpr_dp2_empty_replica_validation.sh MANIFEST.json OUTPUT_DIR [SOURCE_STEP=6]
set -euo pipefail

if (( $# < 2 || $# > 3 )); then
  echo "Usage: $0 40K_MANIFEST.json OUTPUT_DIR [SOURCE_STEP=6]" >&2
  exit 2
fi
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../../.." && pwd -P)"
cd "$ROOT"
MANIFEST="$(realpath "$1")"
OUTPUT="$2"
STEP="${3:-6}"
[[ -f "$MANIFEST" ]] || { echo "Missing manifest: $MANIFEST" >&2; exit 2; }
[[ ! -e "$OUTPUT" ]] || { echo "Refusing to overwrite existing OUTPUT_DIR: $OUTPUT" >&2; exit 2; }
mkdir -p "$OUTPUT"
OUTPUT="$(realpath "$OUTPUT")"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export TPR_QWEN_4B_PATH="${TPR_QWEN_4B_PATH:-/workspace/hf_models/Qwen3-4B-128K}"
[[ -d "$TPR_QWEN_4B_PATH" ]] || { echo "Missing model: $TPR_QWEN_4B_PATH" >&2; exit 2; }
export TPR_OFFLINE_CAPTURE_MANIFEST="$MANIFEST"
export TPR_BENCH_OUTPUT_DIR="$OUTPUT"
export TPR_BENCH_SOURCE_STEP="$STEP"
export TPR_OFFLINE_MODE=tpr
export TPR_RUN_TP2CP2DP2_ONE_STEP=1
export TPR_EXPERIMENTAL_OFFLINE_DP2=1
export TPR_DP2_EMPTY_REPLICA=1
export TPR_OFFLINE_NATIVE_RECOMPUTE_LAYERS=0
export TPR_TREE_POLICY=qwen_turn
export TPR_TREE_BUILDER=lcp_sort
export TPR_MIN_SHARED_PREFIX_TOKENS="${TPR_MIN_SHARED_PREFIX_TOKENS:-1024}"
export TPR_BENCH_WARMUPS="${TPR_BENCH_WARMUPS:-0}"
export TPR_BENCH_REPEATS="${TPR_BENCH_REPEATS:-1}"
export TPR_BENCH_CAPTURE_LOGPROBS=0
export TPR_OFFLINE_GRAD_SYNC_DTYPE="${TPR_OFFLINE_GRAD_SYNC_DTYPE:-bf16}"
export TPR_PPO_LOGPROB_CHUNK_TOKENS="${TPR_PPO_LOGPROB_CHUNK_TOKENS:-512}"
export TPR_DP2_RECOMPUTE="${TPR_DP2_RECOMPUTE:-0}"
export TPR_DP2_OPTIMIZER_STEP="${TPR_DP2_OPTIMIZER_STEP:-0}"
export TPR_DP2_SAVE_FULL_GRADS="${TPR_DP2_SAVE_FULL_GRADS:-0}"
export TPR_EXEC_PROFILE=0
export PYTHONUNBUFFERED=1

echo "DP empty-replica audit: branch=$(git rev-parse --abbrev-ref HEAD) sha=$(git rev-parse --short HEAD)"
echo "TP2 CP2 DP2; DP0 response_mask=0, DP1 active; manifest=$MANIFEST step=$STEP"
set -o pipefail
torchrun --nproc_per_node=8 --master_addr=127.0.0.1 \
  --master_port="${TPR_BENCH_MASTER_PORT:-29794}" \
  -m pytest -q -s -x \
  tests/models/mcore/tpr/parallel/test_tpr_tp2cp2dp2_one_step_npu.py \
  2>&1 | tee "$OUTPUT/dp_empty_replica.log"
grep -q 'TPR_DP_EMPTY_FINALIZE_PASS' "$OUTPUT/dp_empty_replica.log" || {
  echo "Missing TPR_DP_EMPTY_FINALIZE_PASS marker" >&2
  exit 1
}
echo "TPR_DP_EMPTY_REPLICA_AUDIT_PASS output=$OUTPUT"

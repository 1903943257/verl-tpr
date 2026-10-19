#!/usr/bin/env bash
# Qwen3-4B, 2 NPU TP2 CP1 DP1, N8, P8192 S1024. F/B+finalize only.
# Full is a controlled whole-sequence square-CANN reference, NOT
# production Native Megatron/MindSpeed Full F/B.
# Offline performance-only capture retains original historical PPO labels.
set -euo pipefail
cd "$(dirname "$0")/../../../../.."

SOURCE="${TPR_CP1_N8_SOURCE:-/workspace/tpr_actor_capture_4b_n8_run1/actor_update_rank0_batch0.pt}"
ROOT="${TPR_CP1_N8_ROOT:-/workspace/tpr_experiments/cp1_n8_9k_8to1_v1}"
CAPTURE="$ROOT/cp1_n8_p8192_s1024.pt"
export TPR_QWEN_4B_PATH="${TPR_QWEN_4B_PATH:-/workspace/hf_models/Qwen3-4B-128K}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-8,9}"
if [[ "$ASCEND_RT_VISIBLE_DEVICES" != "8,9" ]]; then
  echo "Requires physical NPUs 8,9 for TP2 CP1 (two ranks)" >&2
  exit 2
fi
[[ -f "$SOURCE" ]] || { echo "Missing Actor PPO capture $SOURCE" >&2; exit 2; }
[[ -d "$TPR_QWEN_4B_PATH" ]] || { echo "Missing Qwen3-4B checkpoint" >&2; exit 2; }
export HCCL_DETERMINISTIC=true
export TPR_OFFLINE_GRAD_SYNC_DTYPE=bf16
export TPR_TREE_POLICY=qwen_turn
export TPR_TURN_MIN_SEGMENT_TOKENS=128
export TPR_PPO_LOGPROB_CHUNK_TOKENS=256
export TPR_ROOT_NATIVE_RING=0
export TPR_EXEC_PROFILE=0
export TPR_OFFLINE_NATIVE_RECOMPUTE_LAYERS=0
export TPR_BENCH_WARMUPS="${TPR_BENCH_WARMUPS:-1}"
export TPR_BENCH_REPEATS="${TPR_BENCH_REPEATS:-3}"
export TPR_RUN_CP1_N8_PERF=1
export TPR_CP1_N8_CAPTURE="$CAPTURE"
export TPR_CP1_N8_OUTPUT_DIR="$ROOT"
unset TPR_MEM_TRACE TPR_FIRST_LEAF_DIAG TPR_FIRST_LEAF_DIAG_SYNC || true

mkdir -p "$ROOT"
python -m py_compile \
  tests/models/mcore/tpr/parallel/prepare_tpr_cp1_n8_perf_capture.py \
  tests/models/mcore/tpr/parallel/test_tpr_cp1_n8_9k_perf_npu.py
# Fail fast on data/metric regression before allocating 4B model weights.
python -m pytest -q   tests/models/mcore/tpr/parallel/test_cp1_n8_perf_contract_cpu.py

if [[ ! -f "$CAPTURE" ]]; then
  python -m tests.models.mcore.tpr.parallel.prepare_tpr_cp1_n8_perf_capture \
    "$SOURCE" "$CAPTURE"
fi
[[ -f "${CAPTURE%.pt}.audit.json" ]] || {
  echo "Missing 8:1 capture audit; refusing an unverified benchmark" >&2
  exit 2
}
fail=0
for mode in full tpr; do
  export TPR_CP1_N8_MODE="$mode"
  echo "TPR_CP1_N8_RUN mode=$mode topology=TP2CP1 N=8 P8192 S1024"
  if ! torchrun --nproc_per_node=2 --master_port=29831 \
       -m pytest -q -s -x \
       tests/models/mcore/tpr/parallel/test_tpr_cp1_n8_9k_perf_npu.py \
       2>&1 | tee "$ROOT/${mode}.log"; then
    echo "TPR_CP1_N8_FAILED mode=$mode; do not interpret missing results as speedup" >&2
    fail=1
  fi
done
if [[ "$fail" == 0 ]]; then
  python -m tests.models.mcore.tpr.parallel.compare_tpr_cp1_n8_perf "$ROOT"
else
  exit 1
fi

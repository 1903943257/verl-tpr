#!/usr/bin/env bash
# Native Ascend Level1 profiling of TP2xCP4 Qwen3-4B controlled 17K:17K N8.
# Test-only: no production Ring/Executor/PPO changes. Does not overwrite results.
# Usage: bash tests/models/mcore/tpr/parallel/run_cp4_ring_comm_level1.sh
# Optional TPR_TRACE_RANKS=0 for one-rank smoke; default all eight for comm matrix.
set -euo pipefail
cd "$(dirname "$0")/../../../../.."

BASE="${TPR_VALIDATE_BASE:-/workspace/tpr_experiments/performance}"
CAP="${TPR_VALIDATE_CAPTURE:-$BASE/cp4_n8_merge_perf_none_v1/data/cp4_n8_p17440_s17440.pt}"
ROOT="${TPR_COMM_LEVEL1_ROOT:-$BASE/cp4_ring_comm_level1_v1}"
OUT="$ROOT/kv_query"
MODE="${TPR_COMM_MODE:-tpr}"
case "$MODE" in full|tpr) ;; *) echo "TPR_COMM_MODE must be full or tpr" >&2; exit 2;; esac
TEST=tests/models/mcore/tpr/parallel/test_tpr_fixed_shape_40k_benchmark_npu.py

[[ -f "$CAP" && -f "${CAP%.pt}.audit.json" ]] || {
  echo "Missing controlled 17K:17K Capture/audit: $CAP" >&2; exit 2;
}
[[ ! -e "$OUT/step0_${MODE}_fixed_shape.json" ]] || {
  echo "Refusing to overwrite existing output: $OUT/step0_${MODE}_fixed_shape.json" >&2; exit 2;
}
mkdir -p "$OUT"

export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-8,9,10,11,12,13,14,15}"
[[ "$ASCEND_RT_VISIBLE_DEVICES" == "8,9,10,11,12,13,14,15" ]] || {
  echo "Expected physical Ascend devices 8-15" >&2; exit 2;
}
export TPR_QWEN_4B_PATH="${TPR_QWEN_4B_PATH:-/workspace/hf_models/Qwen3-4B-128K}"
[[ -d "$TPR_QWEN_4B_PATH" ]] || { echo "Missing Qwen3-4B weights" >&2; exit 2; }
export PYTEST_ADDOPTS="-q -s -x"
export HCCL_DETERMINISTIC=true
export TPR_OFFLINE_GRAD_SYNC_DTYPE=bf16
export TPR_TREE_POLICY=qwen_turn
export TPR_TURN_MIN_SEGMENT_TOKENS=128
export TPR_RUN_FIXED_SHAPE_40K_BENCH=1
export TPR_OFFLINE_MODE="$MODE"
export TPR_BENCH_CP_SIZE=4
export TPR_BENCH_SOURCE_STEP=0
export TPR_BENCH_CONTROLLED_N8_AUDIT="${CAP%.pt}.audit.json"
export TPR_BENCH_WARMUPS=1
export TPR_BENCH_REPEATS=1
export TPR_BENCH_RECOMPUTE=0
export TPR_OFFLINE_NATIVE_RECOMPUTE_LAYERS=0
export TPR_BENCH_CAPTURE_LOGPROBS=0
export TPR_MERGE_TOKEN_DIAG=0
export TPR_BENCH_GRAD_WITNESS=0
export TPR_BENCH_MEM_AUDIT=0
export TPR_EXEC_PROFILE=0
export TPR_RING_BLOCK_STATS=0

# Preserve the best-performing attention configuration.
export TPR_RING_COALESCE_PREFIX_FULL=1
export TPR_RING_COALESCE_PREFIX_QUERY=1
export TPR_RING_COALESCE_PREFIX_PADDED=1

# Profile ONLY the measured iteration, after warmup.
export TPR_PHASE_DIAG=1
export TPR_OPERATOR_TRACE=1
export TPR_OPERATOR_TRACE_FORMAT=ascend_level1
export TPR_TRACE_RANKS="${TPR_TRACE_RANKS:-0,1,2,3,4,5,6,7}"
export TPR_BENCH_OUTPUT_DIR="$OUT"

echo "TPR_RING_COMM_LEVEL1_START mode=$TPR_OFFLINE_MODE out=$OUT trace_ranks=$TPR_TRACE_RANKS"
torchrun --nproc_per_node=8 --master_port="${TPR_COMM_LEVEL1_PORT:-29847}" \
  -m pytest "$TEST" 2>&1 | tee "$OUT/run.log"
echo "TPR_RING_COMM_LEVEL1_DONE out=$OUT"
find "$OUT/ascend_profiler" -type f \
  \( -name communication.json -o -name communication_matrix.json \
     -o -name trace_view.json -o -name step_trace_time.csv \
     -o -name kernel_details.csv -o -name op_summary.csv \) \
  -printf '%p  %s bytes\n' | sort

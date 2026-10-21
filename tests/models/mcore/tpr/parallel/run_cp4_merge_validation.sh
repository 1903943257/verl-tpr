#!/usr/bin/env bash
# Test-only Qwen3-4B TP2xCP4 N8: Attention-only / Native Full / TPR None,KV,KV+Query.
# Usage: bash tests/models/mcore/tpr/parallel/run_cp4_merge_validation.sh [attention|fourway|report|all]
# Never restarts Docker, changes production implementation, or overwrites results.
set -euo pipefail
cd "$(dirname "$0")/../../../../.."

BASE="${TPR_VALIDATE_BASE:-/workspace/tpr_experiments/performance}"
CAP="${TPR_VALIDATE_CAPTURE:-$BASE/cp4_n8_merge_perf_none_v1/data/cp4_n8_p17440_s17440.pt}"
ROOT="${TPR_VALIDATE_ROOT:-$BASE/cp4_merge_validation_v1}"
TEST=tests/models/mcore/tpr/parallel/test_tpr_fixed_shape_40k_benchmark_npu.py
ACTION="${1:-all}"
mkdir -p "$ROOT"

attention() {
  mkdir -p "$ROOT/attention"
  echo "CP4 Attention-only: fixed Q/K/V/dO and FP32 dense oracle (small shapes)"
  ASCEND_RT_VISIBLE_DEVICES=8,9,10,11 \
  TPR_RING_COALESCE_PREFIX_FULL=0 \
  TPR_RING_COALESCE_PREFIX_QUERY=0 \
  TPR_RING_COALESCE_PREFIX_PADDED=0 \
  PYTEST_ADDOPTS='-q -s -x' \
  torchrun --nproc_per_node=4 --master_port="${TPR_MERGE_ATTN_PORT:-29843}" -m pytest \
    tests/models/mcore/tpr/parallel/test_ring_cp_attention_npu.py \
    -k 'prefix_full_coalescing_off_on or prefix_query_coalescing_off_on or padded_prefix_full_coalescing_off_on or padded_prefix_query_kv_coalescing_off_on' \
    2>&1 | tee "$ROOT/attention/cp4_merge_attention.log"
}

check_saved_result() {
  python - "$1" <<'PY'
import json, sys
p = sys.argv[1]
d = json.load(open(p))
assert d.get("gradient_witness") is not None, f"{p}: no gradient witness"
measured = [r for r in d["runs"] if not r["warmup"]]
assert measured and all(r.get("token_logprobs") for r in measured), f"{p}: missing token diagnostic"
print("VALIDATION_RESULT_PRESENT", p)
PY
}

fourway() {
  [[ -f "$CAP" && -f "${CAP%.pt}.audit.json" ]] || {
    echo "Missing Capture or audit: $CAP" >&2
    exit 2
  }
  export ASCEND_RT_VISIBLE_DEVICES=8,9,10,11,12,13,14,15
  export TPR_QWEN_4B_PATH="${TPR_QWEN_4B_PATH:-/workspace/hf_models/Qwen3-4B-128K}"
  [[ -d "$TPR_QWEN_4B_PATH" ]] || { echo "Missing Qwen checkpoint" >&2; exit 2; }
  export PYTEST_ADDOPTS='-q -s -x'
  export HCCL_DETERMINISTIC=true
  export TPR_OFFLINE_GRAD_SYNC_DTYPE=bf16
  export TPR_TREE_POLICY=qwen_turn
  export TPR_TURN_MIN_SEGMENT_TOKENS=128
  export TPR_RUN_FIXED_SHAPE_40K_BENCH=1
  export TPR_BENCH_CP_SIZE=4
  export TPR_BENCH_SOURCE_STEP=0
  export TPR_BENCH_CONTROLLED_N8_AUDIT="${CAP%.pt}.audit.json"
  export TPR_BENCH_WARMUPS=1
  export TPR_BENCH_REPEATS=1
  export TPR_BENCH_RECOMPUTE=0
  export TPR_OFFLINE_NATIVE_RECOMPUTE_LAYERS=0
  export TPR_BENCH_CAPTURE_LOGPROBS=1
  export TPR_BENCH_GRAD_WITNESS=1
  export TPR_MERGE_TOKEN_DIAG=1
  export TPR_BENCH_MEM_AUDIT=0
  export TPR_EXEC_PROFILE=0
  export TPR_PHASE_DIAG=0
  export TPR_RING_BLOCK_STATS=0
  export TPR_OPERATOR_TRACE=0

  for mode in full none kv kv_query; do
    case "$mode" in
      full)     export TPR_OFFLINE_MODE=full; KV=0; QUERY=0; PADDED=0 ;;
      none)     export TPR_OFFLINE_MODE=tpr;  KV=0; QUERY=0; PADDED=0 ;;
      kv)       export TPR_OFFLINE_MODE=tpr;  KV=1; QUERY=0; PADDED=1 ;;
      kv_query) export TPR_OFFLINE_MODE=tpr;  KV=1; QUERY=1; PADDED=1 ;;
    esac
    export TPR_RING_COALESCE_PREFIX_FULL="$KV"
    export TPR_RING_COALESCE_PREFIX_QUERY="$QUERY"
    export TPR_RING_COALESCE_PREFIX_PADDED="$PADDED"
    export TPR_BENCH_OUTPUT_DIR="$ROOT/$mode"
    mkdir -p "$TPR_BENCH_OUTPUT_DIR"
    result="$TPR_BENCH_OUTPUT_DIR/step0_${TPR_OFFLINE_MODE}_fixed_shape.json"
    if [[ -f "$result" ]]; then
      check_saved_result "$result"
      echo "SKIP_COMPLETE mode=$mode"
      continue
    fi
    echo "RUN mode=$mode CP4 N8 merge KV=$KV QUERY=$QUERY PADDED=$PADDED"
    torchrun --nproc_per_node=8 --master_port="${TPR_MERGE_PPO_PORT:-29839}" \
      -m pytest "$TEST" 2>&1 | tee "$TPR_BENCH_OUTPUT_DIR/run.log"
    check_saved_result "$result"
  done
}

report() {
  TPR_VALIDATE_ROOT="$ROOT" TPR_VALIDATE_CAPTURE="$CAP" \
    python -m tests.models.mcore.tpr.parallel.compare_cp4_merge_validation \
    2>&1 | tee "$ROOT/summary.log"
}

case "$ACTION" in
  attention) attention ;;
  fourway) fourway ;;
  report) report ;;
  all) attention; fourway; report ;;
  *) echo "Usage: $0 [attention|fourway|report|all]" >&2; exit 2 ;;
esac

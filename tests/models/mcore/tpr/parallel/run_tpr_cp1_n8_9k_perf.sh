#!/usr/bin/env bash
# Qwen3-4B, 2 NPU TP2 CP1 DP1, N8, supported 9K shapes. F/B+finalize only.
# Full is a controlled whole-sequence square-CANN reference, NOT
# production Native Megatron/MindSpeed Full F/B.
# Offline performance-only capture retains original historical PPO labels.
set -euo pipefail
cd "$(dirname "$0")/../../../../.."

P="${TPR_CP1_N8_P:-8192}"
S="${TPR_CP1_N8_S:-1024}"
if [[ "$P:$S" != "8192:1024" && "$P:$S" != "4608:4608" ]]; then
  echo "Supported CP1 shapes: 8192/1024 or 4608/4608" >&2
  exit 2
fi
SOURCE="${TPR_CP1_N8_SOURCE:-/workspace/tpr_actor_capture_4b_n8_run2/actor_update_step000003_rank2_capture000002_batch0.pt}"
if [[ "$P:$S" == "8192:1024" ]]; then
  DEFAULT_ROOT="/workspace/tpr_experiments/performance/cp1_n8_9k_8to1_v1"
else
  DEFAULT_ROOT="/workspace/tpr_experiments/performance/cp1_n8_9k_1to1_v1"
fi
ROOT="${TPR_CP1_N8_ROOT:-$DEFAULT_ROOT}"
DATA_DIR="$ROOT/data"
RESULT_DIR="$ROOT/results"
LOG_DIR="$ROOT/logs"
CAPTURE="$DATA_DIR/cp1_n8_p${P}_s${S}.pt"
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
export TPR_CP1_N8_P="$P"
export TPR_CP1_N8_S="$S"
export TPR_CP1_N8_CAPTURE="$CAPTURE"
export TPR_CP1_N8_OUTPUT_DIR="$RESULT_DIR"
unset TPR_MEM_TRACE TPR_FIRST_LEAF_DIAG TPR_FIRST_LEAF_DIAG_SYNC || true
# Keep pytest short flags out of sys.argv: MindSpeed parses unknown -q/-s/-x
# as empty config field names and make_dataclass(TransformerConfig) then fails.
# PYTEST_ADDOPTS is consumed by pytest, not by MindSpeed argv parsing.
export PYTEST_ADDOPTS="-q -s -x"

# Runtime files stay below ONE versioned experiment directory, never the repo.
# Do not create Python __pycache__ files in the source checkout.
export PYTHONDONTWRITEBYTECODE=1
mkdir -p "$DATA_DIR" "$RESULT_DIR" "$LOG_DIR"
# CPU-only contract preflight uses the repository's isolated Megatron
# interface stubs: bare pytest imports MindSpeed/Megatron unnecessarily.
python tests/models/mcore/tpr/run_cpu_regressions.py --only-cp1-n8-perf \
  2>&1 | tee "$LOG_DIR/preflight.log"

if [[ ! -f "$CAPTURE" ]]; then
  python -m tests.models.mcore.tpr.parallel.prepare_tpr_cp1_n8_perf_capture \
    "$SOURCE" "$CAPTURE" --prefix "$P" --suffix "$S" 2>&1 | tee "$LOG_DIR/prepare.log"
fi
[[ -f "${CAPTURE%.pt}.audit.json" ]] || {
  echo "Missing capture audit; refusing an unverified benchmark" >&2
  exit 2
}
fail=0
for mode in full tpr; do
  export TPR_CP1_N8_MODE="$mode"
  echo "TPR_CP1_N8_RUN mode=$mode topology=TP2CP1 N=8 P=$P S=$S"
  if ! torchrun --nproc_per_node=2 --master_port=29831 \
       -m pytest \
       tests/models/mcore/tpr/parallel/test_tpr_cp1_n8_9k_perf_npu.py \
       2>&1 | tee "$LOG_DIR/${mode}.log"; then
    echo "TPR_CP1_N8_FAILED mode=$mode; do not interpret missing results as speedup" >&2
    fail=1
  fi
done
if [[ "$fail" == 0 ]]; then
  python -m tests.models.mcore.tpr.parallel.compare_tpr_cp1_n8_perf "$RESULT_DIR" \
    --prefix "$P" --suffix "$S" \
    2>&1 | tee "$LOG_DIR/compare.log"
else
  exit 1
fi

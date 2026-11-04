#!/usr/bin/env bash
# Observe dKV accumulation and Replay-root dtype for TPR-old vs TPR-FP32.
# N=8 real Actor capture, one NPU, TP1 x CP1 x DP1. No Full arm or grad dumps.
set -Eeuo pipefail
cd "$(dirname "$0")/../../../../.."

: "${ASCEND_RT_VISIBLE_DEVICES:?Select exactly one available physical NPU ID}"
[[ "$ASCEND_RT_VISIBLE_DEVICES" != *,* ]] || {
  echo "ERROR: diagnostics require exactly one physical NPU" >&2
  exit 2
}
ROOT="${TPR_CP1_DKV_DIAG_ROOT:-/workspace/tpr_experiments/cp1_n8_dkv_replay_diag}"
CAP="${TPR_CP1_N8_CAPTURE:-/workspace/tpr_experiments/cp1_n8_dkv_fp32_round1/data/n8_short.pt}"
MODEL="${TPR_QWEN_4B_PATH:-/workspace/hf_models/Qwen3-4B-128K}"
TEST="tests/models/mcore/tpr/parallel/test_real_actor_cp1_n8_fair_npu.py"
[[ ! -e "$ROOT" ]] || { echo "ERROR: output already exists: $ROOT" >&2; exit 2; }
[[ -f "$CAP" ]] || { echo "ERROR: missing N8 capture: $CAP (set TPR_CP1_N8_CAPTURE)" >&2; exit 2; }
[[ -d "$MODEL" ]] || { echo "ERROR: missing model checkpoint: $MODEL" >&2; exit 2; }
[[ -f "$TEST" ]] || { echo "ERROR: missing N8 Actor test: $TEST" >&2; exit 2; }

mkdir -p "$ROOT"
export TPR_CP1_N8_CAPTURE="$CAP"
export TPR_QWEN_4B_PATH="$MODEL"
# The N8 Actor test has an explicit opt-in pytest skip guard.
export TPR_RUN_CP1_N8_FAIR=1
export TPR_TREE_POLICY=qwen_turn
export TPR_LEAF_BATCH_SIZE=1
export TPR_CP1_N8_WARMUPS=0
export TPR_CP1_N8_REPEATS=1
export TPR_CP1_N8_STAGE_PROFILE=0
export TPR_CP1_N8_SAVE_FULL_GRADS=0
export TPR_EXEC_PROFILE=0
export TPR_OFFLINE_GRAD_SYNC_DTYPE=bf16
export TPR_PPO_LOGPROB_CHUNK_TOKENS=256
export TPR_ROOT_NATIVE_RING=0
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export HCCL_DETERMINISTIC=true
unset TPR_CP1_TRACE TPR_CP1_TRACE_DIR TPR_MEM_TRACE TPR_FIRST_LEAF_DIAG TPR_FIRST_LEAF_DIAG_SYNC || true

echo "TPR_DKV_REPLAY_DIAG capture=$CAP model=$MODEL"
echo "TPR_DKV_REPLAY_DIAG topology=TP1CP1DP1 rows=8 full_parameter_grad_dump=disabled"

for arm in tpr_old tpr_fp32; do
  mkdir -p "$ROOT/$arm"
  export TPR_CP1_N8_MODE=tpr
  export TPR_CP1_N8_OUTPUT="$ROOT/$arm/result.json"
  export TPR_DKV_REPLAY_DIAG_PATH="$ROOT/$arm/dkv.jsonl"
  case "$arm" in
    tpr_old) export TPR_EXPERIMENT_DELAYED_DKV_FP32=0 ;;
    tpr_fp32) export TPR_EXPERIMENT_DELAYED_DKV_FP32=1 ;;
  esac
  echo "===== DKV REPLAY DIAGNOSTICS arm=$arm delayed_dkv_fp32=$TPR_EXPERIMENT_DELAYED_DKV_FP32 ====="
  torchrun --master_addr=127.0.0.1 --master_port="${TPR_TEST_MASTER_PORT:-29869}" \
    --nproc_per_node=1 -m tests.models.mcore.tpr._mindspeed_first_pytest \
    -q -s -x -rs "$TEST" 2>&1 | tee "$ROOT/$arm/run.log"
  [[ -s "$ROOT/$arm/dkv.jsonl" ]] || {
    echo "ERROR: no dKV diagnostics written for $arm" >&2
    exit 1
  }
done

python tests/models/mcore/tpr/parallel/summarize_dkv_replay_diagnostics.py "$ROOT"
echo "TPR_DKV_REPLAY_DIAG_DONE output=$ROOT"

#!/usr/bin/env bash
# Run two isolated 8-NPU TPR topology checks on the same captured N8 PPO batch:
#   1) TP2 x RingCP2 x native DP2, recompute OFF
#   2) TP4 x CP1 x native DP2, TPR-aware full/uniform/1 recompute ON
#
# Usage:
#   ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
#   TPR_QWEN_4B_PATH=/workspace/hf_models/Qwen3-4B \
#   bash tests/models/mcore/tpr/parallel/run_tpr_tp4dp2_tp2cp2dp2.sh \
#     /workspace/tpr_experiments/n8_short_shapes/n8_p8192_s2048.pt \
#     /workspace/tpr_experiments/tp4dp2_vs_tp2cp2dp2_10k
#
# This is an isolated F/B + gradient/logprob witness, not a throughput benchmark.
set -euo pipefail
cd "$(dirname "$0")/../../../../.."

if [[ "$#" != 2 ]]; then
  echo "Usage: $0 N8_ACTOR_CAPTURE OUTPUT_ROOT" >&2
  exit 2
fi

CAPTURE="$(realpath "$1")"
ROOT="$2"
[[ -f "$CAPTURE" ]] || { echo "Missing N8 Actor Capture: $CAPTURE" >&2; exit 2; }

export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
IFS=',' read -r -a VISIBLE_DEVICES <<< "$ASCEND_RT_VISIBLE_DEVICES"
[[ "${#VISIBLE_DEVICES[@]}" == "8" ]] || {
  echo "Expected exactly 8 visible NPUs; got $ASCEND_RT_VISIBLE_DEVICES" >&2
  exit 2
}
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
export TPR_QWEN_4B_PATH="${TPR_QWEN_4B_PATH:-/workspace/hf_models/Qwen3-4B}"
[[ -d "$TPR_QWEN_4B_PATH" ]] || {
  echo "Missing Qwen3-4B checkpoint: $TPR_QWEN_4B_PATH" >&2
  exit 2
}

# These are deliberate opt-in guards. The wrapper sets all required flags so
# direct invocation cannot fail with "10K parity-only gate not enabled".
export TPR_EXPERIMENTAL_OFFLINE_DP2=1
export TPR_10K_PARITY_ONLY=1
export TPR_RUN_10K_PARALLEL_PARITY=1
export TPR_OFFLINE_GRAD_SYNC_DTYPE=bf16
export HCCL_DETERMINISTIC=true
export TPR_TREE_POLICY=qwen_turn
export TPR_TURN_MIN_SEGMENT_TOKENS=128
export TPR_PPO_LOGPROB_CHUNK_TOKENS=512
export TPR_EXEC_PROFILE=0
export TPR_10K_WARMUPS="${TPR_10K_WARMUPS:-1}"
export TPR_10K_REPEATS="${TPR_10K_REPEATS:-3}"
export PYTHONUNBUFFERED=1
export TPR_10K_CAPTURE="$CAPTURE"

mkdir -p "$ROOT"
ROOT="$(realpath "$ROOT")"

run_case() {
  local name="$1" tp="$2" cp="$3" dp="$4" recompute="$5"
  local out="$ROOT/$name"
  if [[ -f "$out/result.json" ]]; then
    echo "TPR_TOPOLOGY_CASE_REUSE name=$name result=$out/result.json"
    return
  fi
  if [[ -e "$out" ]]; then
    echo "Refusing partial output directory: $out" >&2
    exit 2
  fi

  export TPR_10K_TP="$tp"
  export TPR_10K_CP="$cp"
  export TPR_10K_DP="$dp"
  export TPR_10K_MODE=tpr
  export TPR_10K_RECOMPUTE="$recompute"
  export TPR_10K_OUTPUT="$out"

  echo "TPR_TOPOLOGY_CASE_START name=$name tp=$tp cp=$cp dp=$dp recompute=$recompute"
  torchrun --nproc_per_node=8 \
    --master_addr=127.0.0.1 \
    --master_port="${TPR_BENCH_MASTER_PORT:-29871}" \
    -m pytest -q -s -x \
    tests/models/mcore/tpr/parallel/test_tpr_10k_parallel_parity_npu.py \
    2>&1 | tee "$ROOT/$name.log"
}

# Reference configuration already exercised by the native-DP2 one-step path.
run_case tpr_tp2cp2dp2 2 2 2 0

# The new short-sequence candidate: remove CP, use TP4, retain native DP2.
# Recompute is enabled only for this case to address the prior TP4 OOM.
run_case tpr_tp4cp1dp2_recompute 4 1 2 1

echo "TPR_TOPOLOGY_CASES_COMPLETE root=$ROOT"

python - "$ROOT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
cases = (
    ("TP2xCP2xDP2", root / "tpr_tp2cp2dp2" / "result.json"),
    ("TP4xCP1xDP2 + Recompute", root / "tpr_tp4cp1dp2_recompute" / "result.json"),
)
rows = []
for name, path in cases:
    if not path.is_file():
        print(f"RESULT_MISSING {name}: {path}")
        continue
    data = json.loads(path.read_text())
    rows.append((
        name,
        data.get("median_fwd_bwd_seconds_max_rank"),
        data.get("memory_peak_allocated_gib_worst"),
        data.get("global_ppo_tokens"),
        data.get("global_tpr_reuse_audit", {}).get("global_reuse_ratio"),
    ))
print("TOPOLOGY\tMEDIAN_FWD_BWD_S\tWORST_PEAK_ALLOC_GIB\tPPO_TOKENS\tGLOBAL_REUSE_RATIO")
for row in rows:
    print("\t".join(str(x) if x is not None else "n/a" for x in row))
if len(rows) == 2 and rows[0][1] and rows[1][1]:
    print(f"TP4_VS_TP2CP2DP2_SPEEDUP={rows[0][1] / rows[1][1]:.4f}x")
PY

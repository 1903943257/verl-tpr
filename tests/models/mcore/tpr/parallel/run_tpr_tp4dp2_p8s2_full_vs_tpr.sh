#!/usr/bin/env bash
# Controlled TP4 x CP1 x DP2 replay for the N8 P8K/S2K Actor capture.
# "unshared" is the Full-like no-prefix-sharing control through the same
# isolated TPR executor; it is NOT native Megatron Full scheduling.
# Both cases use the same capture, TP/CP/DP, checkpoint, and recompute=ON (full/uniform/1).
set -euo pipefail
cd "$(dirname "$0")/../../../../.."

CAPTURE="${1:-/workspace/tpr_experiments/n8_short_shapes/n8_p8192_s2048.pt}"
ROOT="${2:-/workspace/tpr_experiments/tp4dp2_p8k_s2k_full_vs_tpr_recompute_20261028}"
CAPTURE="$(realpath "$CAPTURE")"
[[ -f "$CAPTURE" ]] || { echo "Missing N8 Actor Capture: $CAPTURE" >&2; exit 2; }
[[ ! -e "$ROOT" ]] || { echo "Refusing existing output root: $ROOT" >&2; exit 2; }
mkdir -p "$ROOT"
ROOT="$(realpath "$ROOT")"

export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
IFS=',' read -r -a DEVICES <<< "$ASCEND_RT_VISIBLE_DEVICES"
[[ "${#DEVICES[@]}" == "8" ]] || { echo "Need exactly 8 visible NPUs" >&2; exit 2; }
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
export TPR_RUN_10K_PARALLEL_PARITY=1
export TPR_10K_PARITY_ONLY=1
export TPR_EXPERIMENTAL_OFFLINE_DP2=1
export TPR_10K_TP=4
export TPR_10K_CP=1
export TPR_10K_DP=2
export TPR_10K_CAPTURE="$CAPTURE"
export TPR_QWEN_4B_PATH="${TPR_QWEN_4B_PATH:-/workspace/hf_models/Qwen3-4B}"
export HCCL_DETERMINISTIC=true
export TPR_OFFLINE_GRAD_SYNC_DTYPE=bf16
export TPR_TREE_POLICY=qwen_turn
export TPR_TURN_MIN_SEGMENT_TOKENS=128
export TPR_PPO_LOGPROB_CHUNK_TOKENS=512
export TPR_EXEC_PROFILE=0
export TPR_10K_WARMUPS="${TPR_10K_WARMUPS:-1}"
export TPR_10K_REPEATS="${TPR_10K_REPEATS:-3}"
export TPR_10K_SUMMARY_ONLY=1
export TPR_10K_RECOMPUTE="${TPR_10K_RECOMPUTE:-1}"
export PYTHONUNBUFFERED=1
unset TPR_MEM_TRACE TPR_FIRST_LEAF_DIAG TPR_FIRST_LEAF_DIAG_SYNC || true

[[ -d "$TPR_QWEN_4B_PATH" ]] || { echo "Missing checkpoint: $TPR_QWEN_4B_PATH" >&2; exit 2; }

RUNNER="$(mktemp /tmp/run_tpr_tp4dp2_p8s2_XXXXXX.py)"
trap 'rm -f "$RUNNER"' EXIT
cat > "$RUNNER" <<'PY'
import sys
sys.argv[:] = [sys.argv[0]]
from tests.models.mcore.tpr.parallel.test_tpr_10k_parallel_parity_npu import (
    test_10k_parallel_factorial_one_topology,
)
test_10k_parallel_factorial_one_topology()
PY

run_mode() {
  local mode="$1"
  export TPR_10K_MODE="$mode"
  export TPR_10K_OUTPUT="$ROOT/$mode"
  echo "TPR_P8S2_CASE_START mode=$mode tp=4 cp=1 dp=2 recompute=$TPR_10K_RECOMPUTE capture=$CAPTURE"
  torchrun --nproc_per_node=8 \
    --master_addr=127.0.0.1 \
    --master_port="${TPR_BENCH_MASTER_PORT:-29872}" \
    "$RUNNER" 2>&1 | tee "$ROOT/$mode.log"
}

run_mode unshared
run_mode tpr

python - "$ROOT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = []
for mode, label in (("unshared", "Unshared control (Full-like)"), ("tpr", "TPR")):
    path = root / mode / "result.json"
    if not path.is_file():
        raise SystemExit(f"Missing result: {path}")
    data = json.loads(path.read_text())
    audit = data.get("global_tpr_reuse_audit") or {}
    rows.append({
        "label": label,
        "mode": mode,
        "median_s": data.get("median_fwd_bwd_seconds_max_rank"),
        "samples_s": data.get("fwd_bwd_seconds_max_rank_samples"),
        "peak_gib": data.get("memory_peak_allocated_gib_worst"),
        "ppo_tokens": data.get("global_ppo_tokens"),
        "logical_tokens": audit.get("global_logical_tokens"),
        "unique_tree_tokens": audit.get("global_unique_tree_tokens"),
        "reuse_ratio": audit.get("global_reuse_ratio"),
        "execution_metrics_by_dp": data.get("execution_metrics_by_dp"),
        "input_sha256": data.get("input_sha256"),
        "recompute": data.get("recompute"),
        "summary_only": data.get("summary_only"),
    })
print("LABEL\\tMEDIAN_F_B_S\\tPEAK_ALLOC_GIB\\tPPO_TOKENS\\tEXEC_LOGICAL\\tEXEC_UNIQUE\\tEXEC_REUSE\\tPUSH_TOKENS\\tPOP_TOKENS\\tLEAF_TOKENS\\tEXEC_SEGMENTS\\tEXEC_FORWARD_CALLS")
for row in rows:
    by_dp = row.get("execution_metrics_by_dp") or []
    exec_logical = sum(x["logical_input_tokens"] for x in by_dp)
    exec_unique = sum(x["unique_tree_tokens"] for x in by_dp)
    exec_reuse = exec_logical / exec_unique if exec_unique else None
    exec_segments = sum(x["physical_segments"] for x in by_dp)
    exec_calls = sum(x["model_forward_calls"] for x in by_dp)
    push_tokens = sum(x["push_forward_tokens"] for x in by_dp)
    pop_tokens = sum(x["pop_forward_tokens"] for x in by_dp)
    leaf_tokens = sum(x["leaf_forward_tokens"] + x["grouped_leaf_forward_tokens"] for x in by_dp)
    row["executed_push_tokens"] = push_tokens
    row["executed_pop_tokens"] = pop_tokens
    row["executed_leaf_tokens"] = leaf_tokens
    row["executed_forward_query_tokens"] = push_tokens + pop_tokens + leaf_tokens
    row["executed_logical_tokens"] = exec_logical
    row["executed_unique_tree_tokens"] = exec_unique
    row["executed_reuse_ratio"] = exec_reuse
    row["executed_physical_segments"] = exec_segments
    row["executed_model_forward_calls"] = exec_calls
    print("\\t".join(str(v) if v is not None else "n/a" for v in (
        row["label"], row["median_s"], row["peak_gib"], row["ppo_tokens"],
        exec_logical, exec_unique,
        f"{exec_reuse:.4f}" if exec_reuse is not None else None,
        push_tokens, pop_tokens, leaf_tokens, exec_segments, exec_calls,
    )))
print("NOTE=EXEC_REUSE is computed from actual mode-specific per-DP forests. The legacy global reuse audit is over all 8 rows before DP partitioning and is not the executed-work ratio.")
if rows[0]["input_sha256"] != rows[1]["input_sha256"]:
    raise SystemExit("Input SHA mismatch between modes")
if rows[0]["median_s"] and rows[1]["median_s"]:
    print(f"UNSHARED_OVER_TPR_SPEEDUP={rows[0]['median_s'] / rows[1]['median_s']:.4f}x")
if rows[0]["peak_gib"] is not None and rows[1]["peak_gib"] is not None:
    print(f"TPR_PEAK_MEMORY_DELTA_GIB={rows[0]['peak_gib'] - rows[1]['peak_gib']:.3f}")
for row in rows:
    print(f"EXECUTION_METRICS_{row['mode'].upper()}={json.dumps(row['execution_metrics_by_dp'], separators=(',', ':'))}")
    print(f"EXECUTED_REUSE_{row['mode'].upper()}={row['executed_reuse_ratio']}")
    print(f"EXECUTED_SEGMENTS_{row['mode'].upper()}={row['executed_physical_segments']}")
    print(f"EXECUTED_FORWARD_CALLS_{row['mode'].upper()}={row['executed_model_forward_calls']}")
    print(f"EXECUTED_PUSH_POP_LEAF_TOKENS_{row['mode'].upper()}={row['executed_push_tokens']}/{row['executed_pop_tokens']}/{row['executed_leaf_tokens']}")
    print(f"EXECUTED_FORWARD_QUERY_TOKENS_{row['mode'].upper()}={row['executed_forward_query_tokens']}")
print(f"INPUT_SHA256={rows[0]['input_sha256']}")
print("NOTE=Unshared control uses TPR executor without shared-prefix reuse; it is not native Megatron Full scheduling. Both modes use the same TPR-aware full/uniform/1 recompute setting.")
PY

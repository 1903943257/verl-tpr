#!/usr/bin/env bash
# One-step real Qwen3-4B TP2xCP4 smoke for joint Prefix QKV Merge + Padding.
# Run from the Docker checkout; this script deliberately performs no Git work.
set -Eeuo pipefail

REPO=/workspace/verl-tpr-main
CAP=${TPR_REAL_ACTOR_MINIBATCH:-/workspace/tpr_experiments/performance/cp4_n8_merge_padding_synthetic_v1/cp4_n8_padding_synthetic.pt}
MODEL=${TPR_QWEN_4B_PATH:-/workspace/hf_models/Qwen3-4B-128K}
OUT=${TPR_QKV_PADDING_SMOKE_OUTPUT_DIR:-/workspace/tpr_experiments/performance/cp4_qkv_padding_smoke_20261103}
DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
ENTRY=tests/models/mcore/tpr/parallel/_fourway_pytest_entry.py
TEST=tests/models/mcore/tpr/parallel/test_real_actor_tp2cp4_multistep_npu.py

cd "$REPO"
[[ -f "$CAP" ]] || { echo "ERROR: synthetic Actor capture not found: $CAP" >&2; exit 2; }
[[ -d "$MODEL" ]] || { echo "ERROR: model directory not found: $MODEL" >&2; exit 2; }
[[ -f "$ENTRY" && -f "$TEST" ]] || { echo "ERROR: missing multistep pytest entry/test" >&2; exit 2; }
[[ $(awk -F, '{print NF}' <<< "$DEVICES") -eq 8 ]] || { echo "ERROR: exactly 8 NPUs required" >&2; exit 2; }
[[ ! -e "$OUT" ]] || { echo "ERROR: output already exists: $OUT; choose a fresh TPR_QKV_PADDING_SMOKE_OUTPUT_DIR" >&2; exit 2; }

mkdir -p "$OUT"
printf 'capture=%s\nmodel=%s\ndevices=%s\nmode=tpr\nsteps=1\nTPR_RING_COALESCE_PREFIX_FULL=1\nTPR_RING_COALESCE_PREFIX_QUERY=1\nTPR_RING_COALESCE_PREFIX_PADDED=1\n' \
  "$CAP" "$MODEL" "$DEVICES" > "$OUT/experiment.txt"

export ASCEND_RT_VISIBLE_DEVICES="$DEVICES"
export PYTHONUNBUFFERED=1
export HCCL_DETERMINISTIC=true
export TPR_RUN_REAL_ACTOR_TP2CP4_MULTISTEP=1
export TPR_REAL_ACTOR_MINIBATCH="$CAP"
export TPR_QWEN_4B_PATH="$MODEL"
export TPR_OFFLINE_ROWS=all
export TPR_OFFLINE_STEPS=1
export TPR_OFFLINE_ADAMW_LR=2e-5
export TPR_OFFLINE_ADAMW_CLIP_NORM=1.0
export TPR_OFFLINE_NATIVE_RECOMPUTE_LAYERS=0
export TPR_TREE_POLICY=token_lcp
export TPR_TURN_MIN_SEGMENT_TOKENS=128
export TPR_OFFLINE_MODE=tpr
export TPR_OFFLINE_OUTPUT_DIR="$OUT"
export TPR_RING_BLOCK_STATS=1
export TPR_RING_COALESCE_PREFIX_FULL=1
export TPR_RING_COALESCE_PREFIX_QUERY=1
export TPR_RING_COALESCE_PREFIX_PADDED=1
export TPR_EXEC_PROFILE=0
export TPR_OFFLINE_SAVE_FULL_GRADS=0
unset TPR_OFFLINE_CAPTURE_MANIFEST TPR_OFFLINE_FULL_GRAD_STEPS || true

echo "========== QKV + PADDING: one real Actor forward/backward/update =========="
TPR_FOURWAY_PYTEST_PHASE=multistep \
  torchrun --nproc_per_node=8 --master_addr=127.0.0.1 --master_port=29787 \
  "$ENTRY" 2>&1 | tee "$OUT/run.log"

for rank in {0..7}; do
  [[ -f "$OUT/tpr_step1_rank$rank.pt" ]] || {
    echo "ERROR: missing step-1 snapshot for rank $rank" >&2
    exit 3
  }
done

python - "$OUT/run.log" <<'PY'
import json
import sys

path = sys.argv[1]
records = []
for line in open(path, encoding="utf-8", errors="replace"):
    marker = "TPR_RING_BLOCK_STATS "
    if marker not in line:
        continue
    try:
        records.append(json.loads(line.split(marker, 1)[1].strip()))
    except json.JSONDecodeError:
        continue

forward = [item for item in records if item.get("phase") == "forward"]
if not forward:
    raise SystemExit("FAIL: no forward TPR_RING_BLOCK_STATS records")

padded_segments = sum(
    sum(a != b for a, b in zip(
        item.get("prefix_lengths", []),
        item.get("prefix_padded_lengths", []),
        strict=True,
    ))
    for item in forward
)
query_calls = sum(int(item.get("fa_prefix_query_merged", 0)) for item in forward)
requested = all(
    item.get("kv_merge_requested") is True
    and item.get("query_merge_requested") is True
    and item.get("padded_kv_merge_requested") is True
    for item in forward
)
print(
    f"QKV_PADDING_AUDIT forward_records={len(forward)} "
    f"padded_prefix_segments={padded_segments} "
    f"query_merged_fa_calls={query_calls} all_merge_flags_requested={requested}",
    flush=True,
)
if not requested:
    raise SystemExit("FAIL: QKV/Padding merge flags were not all observed")
if padded_segments < 1:
    raise SystemExit("FAIL: no Prefix segment had physical Ring padding")
if query_calls < 1:
    raise SystemExit("FAIL: joint Query+KV merged FA path was not executed")
print("PASS: joint Prefix QKV Merge + physical Padding path executed", flush=True)
PY

echo "QKV_PADDING_SMOKE_DONE output=$OUT"

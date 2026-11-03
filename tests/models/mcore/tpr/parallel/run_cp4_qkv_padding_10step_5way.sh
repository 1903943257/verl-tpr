#!/usr/bin/env bash
# Five-arm CP4 fixed-data PPO drift: Full-A / None / KV / QKV / Full-B.
# Docker execution is intentionally independent of Git metadata.
set -Eeuo pipefail

REPO=/workspace/verl-tpr-main
CAP=\${TPR_REAL_ACTOR_MINIBATCH:-/workspace/tpr_experiments/performance/cp4_n8_merge_padding_synthetic_v1/cp4_n8_padding_synthetic.pt}
CAP_REPORT=\${CAP%.pt}.json
MODEL=\${TPR_QWEN_4B_PATH:-/workspace/hf_models/Qwen3-4B-128K}
OUT=\${TPR_FIVEWAY_OUTPUT_DIR:-/workspace/tpr_experiments/performance/cp4_qkv_padding_10step_5way_20261103}
DEVICES=\${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
TEST=tests/models/mcore/tpr/parallel/test_real_actor_tp2cp4_multistep_npu.py
COMPARE=tests/models/mcore/tpr/parallel/compare_real_actor_tp2cp4_multistep.py
REPEAT_COMPARE=tests/models/mcore/tpr/parallel/compare_cp4_full_repeat.py
PYTEST_ENTRY=tests/models/mcore/tpr/parallel/_fourway_pytest_entry.py

cd "$REPO"
[[ -f "$TEST" && -f "$COMPARE" && -f "$REPEAT_COMPARE" && -f "$PYTEST_ENTRY" ]] || {
  echo "ERROR: missing multistep test/comparator/pytest entry" >&2
  exit 2
}
[[ -f "$CAP" && -f "$CAP_REPORT" && -d "$MODEL" ]] || {
  echo "ERROR: missing synthetic capture/report or model: $CAP $CAP_REPORT $MODEL" >&2
  exit 2
}
[[ ! -e "$OUT" ]] || {
  echo "ERROR: output already exists: $OUT; choose a fresh TPR_FIVEWAY_OUTPUT_DIR" >&2
  exit 2
}
[[ $(awk -F, '{print NF}' <<< "$DEVICES") -eq 8 ]] || {
  echo "ERROR: exactly 8 NPUs required" >&2
  exit 2
}

python - "$CAP_REPORT" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1]))
assert report.get("protocol") == "synthetic_cp4_ring_padding_actor_capture_v1"
assert report.get("rows") == 8
shared = int(report["expected_shared_prefix_length"])
padded = int(report["expected_padded_prefix_length"])
padding = int(report["expected_prefix_padding_tokens"])
assert shared % 8 != 0 and padded % 8 == 0 and padding > 0
print(
    f"CAPTURE_SHAPE rows=8 shared_prefix={shared} padded_prefix={padded} "
    f"physical_padding_tokens={padding}",
    flush=True,
)
PY

python - "$CAP" <<'PY'
import sys
import torch

raw = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
assert raw.get("capture_stage") == "actor_update_mini_batch", "Not an Actor PPO capture"
assert len(raw["keys"]) == 8, "Expected N=8"
td = raw["tensordict"]
needed = (
    "input_ids", "prompts", "responses", "response_mask",
    "loss_mask", "advantages", "old_log_probs", "temperature",
)
assert all(key in td for key in needed), "Missing original Actor PPO fields"
masks = [item.bool() for item in td["response_mask"].unbind(0)]
tokens = sum(int(mask.sum()) for mask in masks)
active_advantages = sum(
    int(torch.count_nonzero(adv[mask]))
    for adv, mask in zip(td["advantages"].unbind(0), masks, strict=True)
)
print(
    f"PREFLIGHT rows=8 ppo_tokens={tokens} "
    f"nonzero_active_advantages={active_advantages}",
    flush=True,
)
assert tokens > 0 and active_advantages > 0, "Zero active PPO tokens/advantages"
PY

mkdir -p "$OUT"
CAP_SHA256=$(sha256sum "$CAP" | awk '{print $1}')
{
  printf 'repo=%s\ncapture=%s\ncapture_sha256=%s\nmodel=%s\ndevices=%s\n' \
    "$REPO" "$CAP" "$CAP_SHA256" "$MODEL" "$DEVICES"
  printf 'steps=10\noptimizer=CPU_FP32_master_AdamW\n'
  printf 'protocol=Full-A, None, KV, QKV, Full-B\n'
  printf 'data=repeated_fixed_synthetic_PPO_capture_NOT_on_policy\n'
  printf 'note=all_five_arms_use_the_same_derived_non_aligned_capture\n'
} > "$OUT/experiment.txt"

export ASCEND_RT_VISIBLE_DEVICES="$DEVICES"
export PYTHONUNBUFFERED=1
export HCCL_DETERMINISTIC=true
export TPR_RUN_REAL_ACTOR_TP2CP4_MULTISTEP=1
export TPR_REAL_ACTOR_MINIBATCH="$CAP"
export TPR_QWEN_4B_PATH="$MODEL"
export TPR_OFFLINE_ROWS=all
export TPR_OFFLINE_STEPS=10
export TPR_OFFLINE_ADAMW_LR=2e-5
export TPR_OFFLINE_ADAMW_CLIP_NORM=1.0
export TPR_OFFLINE_NATIVE_RECOMPUTE_LAYERS=0
export TPR_TREE_POLICY=token_lcp
export TPR_TURN_MIN_SEGMENT_TOKENS=128
export TPR_RING_BLOCK_STATS=1
export TPR_OFFLINE_SAVE_FULL_GRADS=0
export TPR_EXEC_PROFILE=0
unset TPR_OFFLINE_FULL_GRAD_STEPS TPR_OFFLINE_CAPTURE_MANIFEST || true

# Keep pytest CLI flags out of process sys.argv before MindSpeed is imported.
TPR_FOURWAY_PYTEST_PHASE=preflight python "$PYTEST_ENTRY" \
  2>&1 | tee "$OUT/preflight.log"

run_arm() {
  local label=$1 mode=$2 kv=$3 query=$4 padded=$5
  local dir="$OUT/$label"
  mkdir "$dir"
  export TPR_OFFLINE_MODE="$mode"
  export TPR_OFFLINE_OUTPUT_DIR="$dir"
  export TPR_RING_COALESCE_PREFIX_FULL="$kv"
  export TPR_RING_COALESCE_PREFIX_QUERY="$query"
  export TPR_RING_COALESCE_PREFIX_PADDED="$padded"
  printf 'mode=%s kv=%s query=%s padded=%s\n' "$mode" "$kv" "$query" "$padded" \
    > "$dir/settings.txt"

  echo "========== $label: 10 optimizer updates =========="
  TPR_FOURWAY_PYTEST_PHASE=multistep \
    torchrun --nproc_per_node=8 --master_addr=127.0.0.1 --master_port=29786 \
    "$PYTEST_ENTRY" 2>&1 | tee "$dir/run.log"

  for step in 1 2 5 10; do
    for rank in {0..7}; do
      [[ -f "$dir/\${mode}_step\${step}_rank\${rank}.pt" ]] || {
        echo "ERROR: missing $label snapshot step=$step rank=$rank" >&2
        exit 3
      }
    done
  done

  if [[ "$mode" == "tpr" ]]; then
    python - "$dir/run.log" "$label" <<'PY'
import json
import sys

path, arm = sys.argv[1:]
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
    raise SystemExit(f"FAIL: {arm}: no forward TPR_RING_BLOCK_STATS records")
kv = sum(int(item.get("fa_prefix_kv_merged", 0)) for item in forward)
query = sum(int(item.get("fa_prefix_query_merged", 0)) for item in forward)
padded = sum(
    sum(a != b for a, b in zip(
        item.get("prefix_lengths", []),
        item.get("prefix_padded_lengths", []),
        strict=True,
    ))
    for item in forward
)
requested = all(
    item.get("kv_merge_requested") is (arm in ("kv", "qkv"))
    and item.get("query_merge_requested") is (arm == "qkv")
    and item.get("padded_kv_merge_requested") is (arm in ("kv", "qkv"))
    for item in forward
)
print(
    f"MERGE_AUDIT arm={arm} forward_records={len(forward)} "
    f"padded_prefix_segments={padded} kv_calls={kv} query_calls={query} "
    f"all_merge_flags_requested={requested}",
    flush=True,
)
if not requested:
    raise SystemExit(f"FAIL: {arm}: requested Merge flags do not match arm")
if arm == "none" and (kv or query):
    raise SystemExit("FAIL: None unexpectedly executed merged FA")
if arm == "kv" and (kv < 1 or query != 0 or padded < 1):
    raise SystemExit("FAIL: KV+Padding path did not execute")
if arm == "qkv" and (query < 1 or padded < 1):
    raise SystemExit("FAIL: QKV+Padding path did not execute")
PY
  fi
}

# Full at both ends measures repeatability/noise of the Native Full baseline.
run_arm full_a full 0 0 0
run_arm none tpr 0 0 0
run_arm kv tpr 1 0 1
run_arm qkv tpr 1 1 1
run_arm full_b full 0 0 0

# Compare each TPR arm independently against BOTH Full endpoints.
for arm in none kv qkv; do
  for baseline in full_a full_b; do
    target="$OUT/compare_\${arm}_vs_\${baseline}"
    mkdir "$target"
    for step in 1 2 5 10; do
      for rank in {0..7}; do
        ln -s "$OUT/$baseline/full_step\${step}_rank\${rank}.pt" \
          "$target/full_step\${step}_rank\${rank}.pt"
        ln -s "$OUT/$arm/tpr_step\${step}_rank\${rank}.pt" \
          "$target/tpr_step\${step}_rank\${rank}.pt"
      done
    done
    python "$COMPARE" "$target" --steps 10 2>&1 | tee "$target/compare.log"
  done
done

python "$REPEAT_COMPARE" "$OUT/full_a" "$OUT/full_b" \
  --steps 10 --output "$OUT/full_repeat_comparison.json" \
  2>&1 | tee "$OUT/full_repeat_compare.log"

python - "$OUT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = []
repeat = json.loads((root / "full_repeat_comparison.json").read_text())
for item in repeat["milestones"]:
    rows.append((
        item["step"], "full_a_vs_full_b", item["loss_gap_b_minus_a"],
        item["logprob"]["rel_l2"], item["logprob"]["mean_abs"],
        item["grad_norm_relative_gap"], item["sampled_grad_relative_l2"],
        item["master_update"]["master_update_rel_l2"],
        item["replica_numerical_gate_ok"],
    ))

for arm in ("none", "kv", "qkv"):
    for baseline in ("full_a", "full_b"):
        report = json.loads(
            (root / f"compare_{arm}_vs_{baseline}" / "multistep_comparison.json").read_text()
        )
        for item in report["milestones"]:
            lp = item["logprob"]
            rows.append((
                item["step"], f"{arm}_vs_{baseline}", item["loss_gap"],
                lp["rel_l2"], lp["mean_abs"], item["grad_norm_relative_gap"],
                item["sampled_grad_relative_l2"],
                item["master"]["master_update_rel_l2"],
                item["replica_numerical_gate_ok"],
            ))

rows.sort(key=lambda row: (row[0], row[1]))
header = (
    "step\tcomparison\tloss_gap\tlp_rel_l2\tlp_mae\tgradnorm_relative_gap\t"
    "sampled_grad_rel_l2\tmaster_update_rel_l2\treplica_ok"
)
(root / "summary.tsv").write_text(
    header + "\n" + "".join("\t".join(map(str, row)) + "\n" for row in rows)
)
print(header)
for row in rows:
    print("\t".join(map(str, row)))
print("SAVED", root / "summary.tsv")
PY

echo "FIVEWAY_DONE output=$OUT"

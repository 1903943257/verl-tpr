#!/usr/bin/env bash
# Four-arm CP4, 10-step fixed-data PPO drift: Native Full / TPR None / KV / KV+Query.
# Runs against the existing multistep NPU test; does not patch Runtime or tests.
set -Eeuo pipefail

REPO=/workspace/verl-tpr-main
CAP=${TPR_REAL_ACTOR_MINIBATCH:-/workspace/tpr_experiments/performance/cp4_n8_merge_perf_none_v1/data/cp4_n8_p17440_s17440.pt}
MODEL=${TPR_QWEN_4B_PATH:-/workspace/hf_models/Qwen3-4B-128K}
OUT=${TPR_FOURWAY_OUTPUT_DIR:-/workspace/tpr_experiments/performance/cp4_qkv_padding_10step_20261103}
DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-8,9,10,11,12,13,14,15}
TEST=tests/models/mcore/tpr/parallel/test_real_actor_tp2cp4_multistep_npu.py
COMPARE=tests/models/mcore/tpr/parallel/compare_real_actor_tp2cp4_multistep.py

cd "$REPO"
[[ -f "$TEST" && -f "$COMPARE" ]] || { echo 'ERROR: missing existing multistep test/comparator' >&2; exit 2; }
[[ -f "$CAP" && -d "$MODEL" ]] || { echo "ERROR: missing capture or model: $CAP $MODEL" >&2; exit 2; }
[[ ! -e "$OUT" ]] || { echo "ERROR: output already exists: $OUT; use a fresh TPR_FOURWAY_OUTPUT_DIR" >&2; exit 2; }
[[ $(awk -F, '{print NF}' <<< "$DEVICES") -eq 8 ]] || { echo "ERROR: exactly 8 NPUs required" >&2; exit 2; }

# Original Actor PPO labels, at least one nonzero active advantage. Fixed-data
# replay intentionally reuses one minibatch for all ten independent updates.
python - "$CAP" <<'PY'
import sys
import torch

raw = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
assert raw.get("capture_stage") == "actor_update_mini_batch", "Not an Actor PPO capture"
assert len(raw["keys"]) == 8, "Expected N=8"
td = raw["tensordict"]
needed = ("input_ids", "prompts", "responses", "response_mask",
          "loss_mask", "advantages", "old_log_probs", "temperature")
assert all(k in td for k in needed), "Missing original PPO fields"
masks = [x.bool() for x in td["response_mask"].unbind(0)]
count = sum(int(m.sum()) for m in masks)
active_adv = sum(
    int(torch.count_nonzero(a[m]))
    for a, m in zip(td["advantages"].unbind(0), masks, strict=True)
)
print(f"PREFLIGHT rows=8 ppo_tokens={count} nonzero_active_advantages={active_adv}", flush=True)
assert count > 0 and active_adv > 0, "Zero active PPO tokens/advantages"
PY

mkdir -p "$OUT"
printf 'repo=%s\ncapture=%s\nmodel=%s\ndevices=%s\nsteps=10\noptimizer=CPU_FP32_master_AdamW\ndata=repeated_fixed_PPO_capture_NOT_online\n' \
  "$REPO" "$CAP" "$MODEL" "$DEVICES" > "$OUT/experiment.txt"

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

python -m pytest -q \
  tests/models/mcore/tpr/parallel/test_native_full_dynamic_microbatches_cpu.py \
  tests/models/mcore/tpr/parallel/test_multistep_artifact_recovery_cpu.py \
  tests/models/mcore/tpr/parallel/test_cp4_multistep_replica_audit_cpu.py \
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
  printf 'mode=%s kv=%s query=%s padded=%s\n' "$mode" "$kv" "$query" "$padded" > "$dir/settings.txt"
  echo "======== $label: ten optimizer steps ========"
  torchrun --nproc_per_node=8 --master_addr=127.0.0.1 --master_port=29786 \
    -m pytest -vv -s -x "$TEST" 2>&1 | tee "$dir/run.log"

  for step in 1 2 5 10; do
    for rank in {0..7}; do
      [[ -f "$dir/${mode}_step${step}_rank${rank}.pt" ]] || {
        echo "ERROR: missing $label snapshot step=$step rank=$rank" >&2
        exit 3
      }
    done
  done

  if [[ "$mode" == tpr ]]; then
    # Assert actual FA merge calls. With Query Merge enabled, Query-path FA
    # supersedes KV-specific FA calls, so KV count need not be positive there.
    python - "$dir/run.log" "$label" <<'PY'
import json
import sys

path, arm = sys.argv[1:]
records = []
for line in open(path, errors="replace"):
    if "TPR_RING_BLOCK_STATS " not in line:
        continue
    try:
        records.append(json.loads(line.split("TPR_RING_BLOCK_STATS ", 1)[1].strip()))
    except json.JSONDecodeError:
        pass
forward = [x for x in records if x.get("phase") == "forward"]
kv = sum(int(x.get("fa_prefix_kv_merged", 0)) for x in forward)
query = sum(int(x.get("fa_prefix_query_merged", 0)) for x in forward)
padded = sum(
    any(a != b for a, b in zip(
        x.get("prefix_lengths", []), x.get("prefix_padded_lengths", []), strict=True
    ))
    for x in forward
)
print(f"MERGE_AUDIT arm={arm} forward_records={len(forward)} padded_prefix_records={padded} "
      f"kv_calls={kv} query_calls={query}", flush=True)
assert forward, "No actual Ring FA block records"
if arm == "none":
    assert kv == 0 and query == 0, "None unexpectedly ran merged FA"
elif arm == "kv":
    assert kv > 0 and query == 0, "KV mode did not take KV-only merged FA path"
    assert padded > 0, "No padded prefix: not a Padding regression test"
elif arm == "kv_query":
    assert query > 0, "KV+Query mode did not take Query merged FA path"
    assert padded > 0, "No padded prefix: not a Padding regression test"
PY
  fi
}

run_arm full full 0 0 0
run_arm none tpr 0 0 0
run_arm kv tpr 1 0 1
run_arm kv_query tpr 1 1 1

# Comparator takes exactly one full and one tpr in a single directory.
# Use symlinks to preserve immutable mode snapshots.
for arm in none kv kv_query; do
  target="$OUT/compare_$arm"
  mkdir "$target"
  for step in 1 2 5 10; do
    for rank in {0..7}; do
      ln -s "$OUT/full/full_step${step}_rank${rank}.pt" "$target/full_step${step}_rank${rank}.pt"
      ln -s "$OUT/$arm/tpr_step${step}_rank${rank}.pt" "$target/tpr_step${step}_rank${rank}.pt"
    done
  done
  python "$COMPARE" "$target" --steps 10 2>&1 | tee "$target/compare.log"
done

python - "$OUT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = []
for arm in ("none", "kv", "kv_query"):
    report = json.loads((root / f"compare_{arm}" / "multistep_comparison.json").read_text())
    for item in report["milestones"]:
        lp = item["logprob"]
        rows.append((
            item["step"], arm, item["loss_gap"], lp["rel_l2"], lp["mean_abs"],
            item["grad_norm_relative_gap"], item["sampled_grad_relative_l2"],
            item["master"]["master_update_rel_l2"], item["replica_numerical_gate_ok"],
        ))
rows.sort()
header = "step\tarm\tloss_gap\tlp_rel_l2\tlp_mae\tgradnorm_relative_gap\tsampled_grad_rel_l2\tmaster_update_rel_l2\treplica_ok"
(root / "summary.tsv").write_text(header + "\n" + "".join(
    "\t".join(map(str, row)) + "\n" for row in rows
))
print(header)
for row in rows:
    print("\t".join(map(str, row)))
print("SAVED", root / "summary.tsv")
PY

echo "FOURWAY_DONE output=$OUT"

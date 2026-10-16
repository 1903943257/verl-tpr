#!/usr/bin/env bash
# Verify the refactored TPR against an independent pre-refactor checkout.
# This is a REAL Qwen3-4B PPO F/B + gradient-finalize gate, not a Full-vs-TPR
# speedup test. It intentionally does not change the production Ring runtime.
# Usage: bash this_script.sh OLD_ROOT NEW_ROOT 40K_MANIFEST OUTPUT_ROOT [STEP=6]
set -euo pipefail
if (( $# < 4 || $# > 5 )); then
  echo "Usage: $0 OLD_ROOT NEW_ROOT 40K_MANIFEST OUTPUT_ROOT [STEP=6]" >&2
  exit 2
fi
OLD="$(realpath "$1")"
NEW="$(realpath "$2")"
MANIFEST="$(realpath "$3")"
OUT="$4"
STEP="${5:-6}"
[[ -d "$OLD/.git" || -f "$OLD/.git" ]] || { echo "Missing old checkout: $OLD" >&2; exit 2; }
[[ -d "$NEW/.git" || -f "$NEW/.git" ]] || { echo "Missing new checkout: $NEW" >&2; exit 2; }
[[ "$OLD" != "$NEW" ]] || { echo "Old and new checkouts must differ" >&2; exit 2; }
[[ -f "$MANIFEST" ]] || { echo "Missing 40K manifest: $MANIFEST" >&2; exit 2; }
[[ ! -e "$OUT" ]] || { echo "Output exists: $OUT" >&2; exit 2; }
mkdir -p "$OUT"
OUT="$(realpath "$OUT")"
git -C "$OLD" rev-parse HEAD > "$OUT/old.sha"
git -C "$NEW" rev-parse HEAD > "$OUT/new.sha"
echo "TPR_OLD_NEW_DP2_REVISIONS old=$(cat "$OUT/old.sha") new=$(cat "$OUT/new.sha")"

# Hold all experimental knobs equal. No Fixed Tile and no Ring redesign here.
export TPR_BENCH_WARMUPS=0
export TPR_BENCH_REPEATS=1
export TPR_BENCH_CAPTURE_LOGPROBS=1
export TPR_BENCH_GRAD_WITNESS=1
export TPR_BENCH_GRAD_REPLICA_POLICY=diagnostic
export TPR_PPO_LOGPROB_CHUNK_TOKENS=512
export TPR_RING_COALESCE_PREFIX_FULL=0
export TPR_RING_COALESCE_PREFIX_QUERY=0
export TPR_EXEC_PROFILE=0
export HCCL_DETERMINISTIC=true
export TPR_OFFLINE_GRAD_SYNC_DTYPE=bf16
export PYTHONUNBUFFERED=1

for version in old new; do
  if [[ "$version" == old ]]; then root="$OLD"; else root="$NEW"; fi
  echo "TPR_OLD_NEW_DP2_RUN version=$version checkout=$root step=$STEP"
  (
    cd "$root"
    export PYTHONPATH="$root:${PYTHONPATH:-}"
    bash tests/models/mcore/tpr/parallel/run_tpr_tp2cp2dp2_one_step.sh \
      "$MANIFEST" "$OUT/$version" "$STEP"
  ) 2>&1 | tee "$OUT/$version.log"
done

python - "$OUT/old/step${STEP}_tpr_tp2cp2dp2.json" \
         "$OUT/new/step${STEP}_tpr_tp2cp2dp2.json" "$OUT/comparison.json" <<'PY'
import json, math, os, pathlib, sys
old_path, new_path, output_path = map(pathlib.Path, sys.argv[1:])
old, new = (json.loads(path.read_text()) for path in (old_path, new_path))
invariants = (
    "source_step", "mode", "topology", "effective_sequence_cap",
    "source_capture", "checkpoint", "input_sha256", "global_ppo_tokens",
    "placement_policy", "dp_partitions", "dp_tree_token_costs",
    "global_tree_tokens", "duplicated_tree_tokens",
    "warmups", "repeats", "gradient_sync_reduce_dtype",
    "ppo_logprob_chunk_tokens", "recompute", "updated_weights",
)
for key in invariants:
    if old.get(key) != new.get(key):
        raise AssertionError(f"old/new DP2 invariant {key} changed: {old.get(key)!r} != {new.get(key)!r}")
if old["topology"] != {"tp": 2, "cp": 2, "dp": 2, "world": 8}:
    raise AssertionError("Expected TP2CP2DP2")
if old["warmups"] != 0 or old["repeats"] != 1 or old["updated_weights"] is not False:
    raise AssertionError("Expected exactly one fixed-weight PPO F/B step per revision")
a, b = old["runs"][-1], new["runs"][-1]
loss_gap = abs(float(a["loss"]) - float(b["loss"]))
rank_losses_a = [float(r["local_loss_dp_scaled"]) for r in a["ranks"]]
rank_losses_b = [float(r["local_loss_dp_scaled"]) for r in b["ranks"]]
if len(rank_losses_a) != len(rank_losses_b) or len(rank_losses_a) != 8:
    raise AssertionError("Expected eight rank-local PPO loss records")
rank_loss_gap = max(abs(x - y) for x, y in zip(rank_losses_a, rank_losses_b))
ga, gb = old.get("gradient_witness"), new.get("gradient_witness")
if not ga or not gb:
    raise AssertionError("Missing old/new gradient witness; do not accept loss-only PPO parity")
for key in ("names", "shapes", "samples_per_name", "samples_per_parameter"):
    if ga[key] != gb[key]:
        raise AssertionError(f"Gradient witness coordinates differ: {key}")
va = [float(v) for block in ga["tp_shards"] for v in block]
vb = [float(v) for block in gb["tp_shards"] for v in block]
if not va or len(va) != len(vb) or not all(map(math.isfinite, va + vb)):
    raise AssertionError("Missing, mismatched, or non-finite sampled gradients")
norm = math.sqrt(sum(x*x for x in va))
delta = math.sqrt(sum((x-y)**2 for x, y in zip(va, vb)))
grad_rel_l2 = delta / norm if norm else (0.0 if delta == 0 else math.inf)
replica_abs = max(
    float(ga.get("replica_sample_max_abs", math.inf)),
    float(gb.get("replica_sample_max_abs", math.inf)),
)
limits = {
    "loss_abs": float(os.getenv("TPR_ACCEPT_LOSS_ABS", "0.001")),
    "rank_loss_abs": float(os.getenv("TPR_ACCEPT_RANK_LOSS_ABS", "0.001")),
    "sampled_grad_rel_l2": float(os.getenv("TPR_ACCEPT_GRAD_REL_L2", "0.02")),
    "replica_sample_max_abs": float(os.getenv("TPR_ACCEPT_REPLICA_ABS", "0.00003")),
}
measured = {
    "loss_abs": loss_gap, "rank_loss_abs": rank_loss_gap,
    "sampled_grad_rel_l2": grad_rel_l2,
    "replica_sample_max_abs": replica_abs,
}
checks = {k: math.isfinite(v) and v <= limits[k] for k, v in measured.items()}
report = {
    "old_capture": str(old_path), "new_capture": str(new_path),
    "input_sha256": old["input_sha256"], "source_step": old["source_step"],
    "measured": measured, "limits": limits, "checks": checks,
    "passed": all(checks.values()),
    "scope": "Fixed-weight one-step PPO, same capture, sampled gradient witness; does not certify full gradients, optimizer updates, or tokenwise logprobs",
}
output_path.write_text(json.dumps(report, indent=2, allow_nan=False))
print("TPR_OLD_NEW_TP2CP2DP2_REPORT " + json.dumps(report, allow_nan=False), flush=True)
if not report["passed"]:
    raise AssertionError("Old/new TP2CP2DP2 PPO acceptance failed")
print("TPR_OLD_NEW_TP2CP2DP2_PASS", flush=True)
PY

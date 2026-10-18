#!/usr/bin/env bash
# Verify the refactored TPR against an independent pre-refactor checkout.
# This is a REAL Qwen3-4B PPO F/B + gradient-finalize gate, not a Full-vs-TPR
# speedup test. It intentionally does not change the production Ring runtime.
# Usage: bash this_script.sh OLD_ROOT NEW_ROOT CAPTURE_MANIFEST OUTPUT_ROOT [STEP=6]
set -euo pipefail
if (( $# < 4 || $# > 5 )); then
  echo "Usage: $0 OLD_ROOT NEW_ROOT CAPTURE_MANIFEST OUTPUT_ROOT [STEP=6]" >&2
  exit 2
fi
OLD="$(realpath "$1")"
NEW="$(realpath "$2")"
MANIFEST="$(realpath "$3")"
OUT="$4"
STEP="${5:-6}"
[[ -f "$OLD/tests/models/mcore/tpr/parallel/run_tpr_tp2cp2dp2_one_step.sh" ]] || {
  echo "Missing old test runner: $OLD" >&2; exit 2;
}
[[ -f "$NEW/tests/models/mcore/tpr/parallel/run_tpr_tp2cp2dp2_one_step.sh" ]] || {
  echo "Missing new test runner: $NEW" >&2; exit 2;
}
[[ "$OLD" != "$NEW" ]] || { echo "Old and new checkouts must differ" >&2; exit 2; }
[[ -f "$MANIFEST" ]] || { echo "Missing capture manifest: $MANIFEST" >&2; exit 2; }
[[ ! -e "$OUT" ]] || { echo "Output exists: $OUT" >&2; exit 2; }
mkdir -p "$OUT"
OUT="$(realpath "$OUT")"
# Git is exclusively a host-side operation. Docker records the SHAs
# explicitly verified on the host; it never invokes Git.
for version in old new; do
  if [[ "$version" == old ]]; then sha="${TPR_OLD_GIT_SHA:-}"; else sha="${TPR_NEW_GIT_SHA:-}"; fi
  if [[ ! "$sha" =~ ^[0-9a-f]{40}$ ]]; then
    echo "Set TPR_${version^^}_GIT_SHA to the full host-verified commit SHA" >&2
    exit 2
  fi
  printf '%s\n' "$sha" > "$OUT/$version.sha"
  echo "TPR_DP2_REVISION version=$version sha=$sha source=host_verified"
done

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
    "placement_policy",
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
# Turn-aware planning may legitimately change DP assignment. Rank-local
# contributions must therefore agree WITHIN each run, not across revisions.
# The total global PPO objective and post-finalize gradients remain comparable.
def validate_run(name, data, run):
    partitions = data["dp_partitions"]
    if (len(partitions) != 2 or any(not p for p in partitions)
            or sorted(row for p in partitions for row in p) != list(range(8))):
        raise AssertionError(f"{name}: DP placement must cover all eight rows exactly once")
    ranks = run["ranks"]
    if len(ranks) != 8 or sorted(int(r["rank"]) for r in ranks) != list(range(8)):
        raise AssertionError(f"{name}: expected eight distinct rank-local reports")
    for rank in ranks:
        if rank["rows"] != partitions[int(rank["dp_rank"])]:
            raise AssertionError(f"{name}: rank-local rows disagree with DP placement")
    report = run["loss_report"]
    if not math.isfinite(float(run["loss"])):
        raise AssertionError(f"{name}: nonfinite global loss")
    gaps = {
        "tp_local_max_gap": float(report["tp_local_max_gap"]),
        "cp_replica_max_gap": float(report["cp_replica_max_gap"]),
        "global_reporting_max_gap": float(report["global_reporting_max_gap"]),
    }
    if any(not math.isfinite(v) or v > 0.001 for v in gaps.values()):
        raise AssertionError(f"{name}: internal TP/CP/global loss report disagrees: {gaps}")
    return gaps

old_internal = validate_run("old", old, a)
new_internal = validate_run("new", new, b)
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
    "internal_loss_abs": float(os.getenv("TPR_ACCEPT_RANK_LOSS_ABS", "0.001")),
    "sampled_grad_rel_l2": float(os.getenv("TPR_ACCEPT_GRAD_REL_L2", "0.02")),
    "replica_sample_max_abs": float(os.getenv("TPR_ACCEPT_REPLICA_ABS", "0.00003")),
}
measured = {
    "loss_abs": loss_gap,
    "internal_loss_abs": max(*old_internal.values(), *new_internal.values()),
    "sampled_grad_rel_l2": grad_rel_l2,
    "replica_sample_max_abs": replica_abs,
}
checks = {k: math.isfinite(v) and v <= limits[k] for k, v in measured.items()}
report = {
    "old_capture": str(old_path), "new_capture": str(new_path),
    "input_sha256": old["input_sha256"], "source_step": old["source_step"],
    "placement_old": {
        "partitions": old["dp_partitions"], "cost_model": old.get("placement_cost_model"),
        "tree_costs": old["dp_tree_token_costs"], "duplicated_tokens": old["duplicated_tree_tokens"],
    },
    "placement_new": {
        "partitions": new["dp_partitions"], "cost_model": new.get("placement_cost_model"),
        "tree_costs": new["dp_tree_token_costs"], "duplicated_tokens": new["duplicated_tree_tokens"],
    },
    "old_internal_loss_gaps": old_internal, "new_internal_loss_gaps": new_internal,
    "measured": measured, "limits": limits, "checks": checks,
    "passed": all(checks.values()),
    "scope": "Fixed-weight one-step PPO, same capture, globally comparable loss and sampled gradients across potentially different legal DP placements; does not certify full gradients, optimizer updates, or tokenwise logprobs",
}
output_path.write_text(json.dumps(report, indent=2, allow_nan=False))
print("TPR_OLD_NEW_TP2CP2DP2_REPORT " + json.dumps(report, allow_nan=False), flush=True)
if not report["passed"]:
    raise AssertionError("Old/new TP2CP2DP2 PPO acceptance failed")
print("TPR_OLD_NEW_TP2CP2DP2_PASS", flush=True)
PY

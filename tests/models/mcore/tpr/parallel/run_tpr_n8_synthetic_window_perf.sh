#!/usr/bin/env bash
# Two controlled SYNTHETIC-WINDOW performance experiments from one Run2 batch.
# Six true long response prefixes + two distinct, later windows from long replies.
# TP2 CP2 P9216/S9216 and TP2 CP4 P17440/S17440; never PPO parity.
set -euo pipefail
cd "$(dirname "$0")/../../../../.."

SOURCE="${TPR_SYNTHETIC_WINDOW_SOURCE:-/workspace/tpr_actor_capture_4b_n8_run2/actor_update_step000003_rank2_capture000002_batch0.pt}"
BASE=/workspace/tpr_experiments/performance
selection="${1:-both}"
case "$selection" in
  both) shapes=("cp2" "cp4") ;;
  cp2|cp4) shapes=("$selection") ;;
  *) echo "Usage: bash $0 [both|cp2|cp4]" >&2; exit 2 ;;
esac
[[ -f "$SOURCE" ]] || { echo "Missing original Actor source: $SOURCE" >&2; exit 2; }
mkdir -p "$BASE"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
echo "TPR_SYNTHETIC_WORKLOAD_WARNING: performance only, cloned source rows with shifted response windows; PPO validity NOT claimed"
python tests/models/mcore/tpr/run_cpu_regressions.py --only-cp1-n8-perf

status=0
for shape in "${shapes[@]}"; do
  if [[ "$shape" == "cp2" ]]; then
    P=9216; S=9216; RATIO=9k_9k; DEVICES="8,9,10,11"
    ROOT="$BASE/cp2_n8_18k_1to1_synthetic_windows_v1"
  else
    P=17440; S=17440; RATIO=17k_17k; DEVICES="8,9,10,11,12,13,14,15"
    ROOT="$BASE/cp4_n8_34k_1to1_synthetic_windows_v1"
  fi
  echo "=== SYNTHETIC N8 $shape P=$P S=$S ==="
  if [[ -f "$ROOT/results/comparison.json" ]]; then
    echo "TPR_SYNTHETIC_ALREADY_COMPLETE $ROOT/results/comparison.json"
    continue
  fi
  STAGE="$ROOT/data/window_source_p${P}_s${S}.pt"
  if [[ ! -f "$STAGE" ]]; then
    mkdir -p "$ROOT/data" "$ROOT/logs" "$ROOT/results"
    if ! python -m tests.models.mcore.tpr.parallel.prepare_tpr_n8_synthetic_windows \
      "$SOURCE" "$STAGE" --prefix "$P" --suffix "$S" \
      2>&1 | tee "$ROOT/logs/synthetic_prepare.log"; then
      echo "TPR_SYNTHETIC_PREPARE_FAILED shape=$shape" >&2
      status=1
      continue
    fi
  fi
  # Never silently reuse synthetic source staged from a different original file.
  if ! python - "$STAGE" "$SOURCE" "$P" "$S" <<'PY'
import json, pathlib, sys
stage, source = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
p,s=int(sys.argv[3]),int(sys.argv[4])
audit = json.loads(stage.with_suffix(".windows.json").read_text())
assert audit["synthetic_response_windows"] and not audit["on_policy_valid"]
assert pathlib.Path(audit["original_capture"]).resolve() == source.resolve()
assert (audit["prefix"], audit["suffix"], audit["n"]) == (p,s,8)
assert len(audit["source_rows"]) == 8
print("TPR_SYNTHETIC_SOURCE_AUDIT_PASS",stage, flush=True)
PY
  then
    status=1
    continue
  fi
  if ! ASCEND_RT_VISIBLE_DEVICES="$DEVICES" TPR_CP_N8_ROOT="$ROOT" \
    bash tests/models/mcore/tpr/parallel/run_tpr_cp4_n8_27k_perf.sh \
    "$RATIO" "$STAGE"; then
    echo "TPR_SYNTHETIC_PERF_FAILED shape=$shape; inspect $ROOT/logs" >&2
    status=1
    continue
  fi
done

echo "=== Synthetic-window performance results ==="
for root in "$BASE/cp2_n8_18k_1to1_synthetic_windows_v1" \
            "$BASE/cp4_n8_34k_1to1_synthetic_windows_v1"; do
  [[ -f "$root/results/comparison.json" ]] && echo "COMPLETE $root/results/comparison.json" \
    || echo "PENDING $root/results/comparison.json"
done
exit "$status"

#!/usr/bin/env bash
# Test-only CP4 Native Full vs TPR Ring Level1 trace, same audited N8 capture.
# Usage: bash tests/models/mcore/tpr/parallel/run_cp4_ring_timeline_compare.sh [both|full|tpr|compare]
# Existing outputs are never overwritten. Run each mode as a separate 8-NPU job.
set -euo pipefail
cd "$(dirname "$0")/../../../../.."

ROOT="${TPR_COMM_COMPARE_ROOT:-/workspace/tpr_experiments/performance/cp4_ring_timeline_v1}"
RANK="${TPR_TIMELINE_RANK:-0}"
SELECT="${1:-both}"
case "$SELECT" in both|full|tpr|compare) ;; *) echo "Usage: $0 [both|full|tpr|compare]" >&2; exit 2;; esac
[[ "$RANK" =~ ^[0-7]$ ]] || { echo "TPR_TIMELINE_RANK must be in 0..7" >&2; exit 2; }

if [[ "$SELECT" != compare ]]; then
  for mode in full tpr; do
    [[ "$SELECT" == both || "$SELECT" == "$mode" ]] || continue
    dst="$ROOT/$mode/kv_query"
    marker="$dst/step0_${mode}_fixed_shape.json"
    if [[ -e "$marker" ]]; then
      echo "CP4_TIMELINE_MODE_ALREADY_COMPLETE mode=$mode file=$marker"
      continue
    fi
    # Trace ranks and capture can be controlled externally via existing Level1 runner.
    TPR_COMM_MODE="$mode" TPR_COMM_LEVEL1_ROOT="$ROOT/$mode" \
      bash tests/models/mcore/tpr/parallel/run_cp4_ring_comm_level1.sh
  done
fi

FULL_RESULT="$ROOT/full/kv_query/step0_full_fixed_shape.json"
TPR_RESULT="$ROOT/tpr/kv_query/step0_tpr_fixed_shape.json"
if [[ ! -f "$FULL_RESULT" || ! -f "$TPR_RESULT" ]]; then
  echo "CP4_TIMELINE_NEEDS_BOTH_MODES full=$FULL_RESULT tpr=$TPR_RESULT" >&2
  exit 0
fi

python - "$FULL_RESULT" "$TPR_RESULT" <<'PY'
import json, sys
f, t = (json.load(open(path)) for path in sys.argv[1:])
for key in ("checkpoint", "topology", "input_sha256", "ppo_tokens"):
    if f.get(key) != t.get(key) or f.get(key) is None:
        raise SystemExit(f"CP4 profile mismatch {key}: {f.get(key)!r} != {t.get(key)!r}")
if f.get("mode") != "full" or t.get("mode") != "tpr":
    raise SystemExit("Mode mismatch between CP4 reference and TPR")
print("CP4_TIMELINE_INPUTS_MATCH sha256=" + str(f["input_sha256"]))
PY

FULL_TRACE="$ROOT/full/kv_query/ascend_profiler/full_iter1_rank$RANK"
TPR_TRACE="$ROOT/tpr/kv_query/ascend_profiler/tpr_iter1_rank$RANK"
[[ -d "$FULL_TRACE" && -d "$TPR_TRACE" ]] || {
  echo "Missing matching rank-$RANK traces (check TPR_TRACE_RANKS): $FULL_TRACE | $TPR_TRACE" >&2; exit 2;
}
options=()
# Giant trace_view.json may exhaust host RAM; CSV-only is the safe default.
# Set TPR_TIMELINE_PARSE_TRACE=1 to enable CPU Host markers from trace_view.json.
if [[ "${TPR_TIMELINE_PARSE_TRACE:-0}" != 1 ]]; then
  options+=(--no-trace)
fi
output="$ROOT/cp4_rank${RANK}_full_tpr_timeline.json"
[[ ! -e "$output" ]] || { echo "Refusing to overwrite $output" >&2; exit 2; }
python -m tests.models.mcore.tpr.parallel.compare_cp4_ring_timeline \
  --full "$FULL_TRACE" --tpr "$TPR_TRACE" \
  "${options[@]}" --json-out "$output" \
  | tee "$ROOT/cp4_rank${RANK}_timeline.log"
echo "CP4_RING_TIMELINE_DONE $output"

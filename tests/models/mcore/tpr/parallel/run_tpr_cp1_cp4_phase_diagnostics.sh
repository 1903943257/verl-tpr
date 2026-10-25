#!/usr/bin/env bash
# Test-only diagnostics: CP1 N8 4608/4608 + CP4 N8 17440/17440.
# No production changes. Exactly 1 warmup + 1 measured profiled F/B.
# Results are separate from historical unprofiled performance runs.
set -euo pipefail
cd "$(dirname "$0")/../../../../.."

BASE=/workspace/tpr_experiments/performance
CP1_REF="$BASE/cp1_n8_9k_1to1_v1/results"
CP1_SOURCE="/workspace/tpr_actor_capture_4b_n8_run2/actor_update_step000003_rank2_capture000002_batch0.pt"
CP4_REF="$BASE/cp4_n8_34k_1to1_synthetic_windows_v1/results"
CP1_ROOT="$BASE/cp1_n8_9k_1to1_phase_v1"
CP4_ROOT="$BASE/cp4_n8_34k_1to1_phase_v1"
SOURCE_CP4="$BASE/cp4_n8_34k_1to1_synthetic_windows_v1/data/window_source_p17440_s17440.pt"
SUMMARY="$BASE/cp1_cp4_phase_comparison_v1.json"

for file in "$CP1_SOURCE" "$CP1_REF/full.json" "$CP1_REF/tpr.json" \
            "$CP4_REF/step0_full_fixed_shape.json" \
            "$CP4_REF/step0_tpr_fixed_shape.json" "$SOURCE_CP4"; do
  [[ -f "$file" ]] || { echo "Missing reference $file" >&2; exit 2; }
done
export TPR_PHASE_DIAG=1
export TPR_OPERATOR_TRACE="${TPR_OPERATOR_TRACE:-1}"
export TPR_TRACE_RANKS="${TPR_TRACE_RANKS:-0}"
export TPR_BENCH_WARMUPS=1
export TPR_BENCH_REPEATS=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
mkdir -p "$BASE"

selection="${1:-both}"
case "$selection" in
  cp1|cp4|both|summary) ;;
  *) echo "Usage: bash $0 [both|cp1|cp4|summary]" >&2; exit 2 ;;
esac

if [[ "$selection" == "cp1" || "$selection" == "both" ]]; then
  if [[ -f "$CP1_ROOT/results/full.json" && -f "$CP1_ROOT/results/tpr.json" ]]; then
    echo "CP1_PHASE_ALREADY_COMPLETE $CP1_ROOT/results"
  else
    echo "=== CP1 4608/4608 controlled Full vs TPR stage profile ==="
    ASCEND_RT_VISIBLE_DEVICES=8,9 \
      TPR_CP1_N8_SOURCE="$CP1_SOURCE" \
      TPR_CP1_N8_P=4608 TPR_CP1_N8_S=4608 TPR_CP1_N8_ROOT="$CP1_ROOT" \
      bash tests/models/mcore/tpr/parallel/run_tpr_cp1_n8_9k_perf.sh \
      2>&1 | tee "$BASE/cp1_1to1_phase_v1.log"
  fi
fi

if [[ "$selection" == "cp4" || "$selection" == "both" ]]; then
  if [[ -f "$CP4_ROOT/results/step0_full_fixed_shape.json" && \
        -f "$CP4_ROOT/results/step0_tpr_fixed_shape.json" ]]; then
    echo "CP4_PHASE_ALREADY_COMPLETE $CP4_ROOT/results"
  else
    echo "=== CP4 17440/17440 native Full vs Ring TPR stage profile ==="
    ASCEND_RT_VISIBLE_DEVICES=8,9,10,11,12,13,14,15 \
      TPR_CP_N8_ROOT="$CP4_ROOT" \
      bash tests/models/mcore/tpr/parallel/run_tpr_cp4_n8_27k_perf.sh \
      17k_17k "$SOURCE_CP4" \
      2>&1 | tee "$BASE/cp4_1to1_phase_v1.log"
  fi
fi

if [[ -f "$CP1_ROOT/results/full.json" && -f "$CP1_ROOT/results/tpr.json" && \
      -f "$CP4_ROOT/results/step0_full_fixed_shape.json" && \
      -f "$CP4_ROOT/results/step0_tpr_fixed_shape.json" ]]; then
  if [[ -f "$SUMMARY" ]]; then
    echo "PHASE_SUMMARY_ALREADY_COMPLETE $SUMMARY"
    exit 0
  fi
  python -m tests.models.mcore.tpr.parallel.compare_tpr_cp1_cp4_phase_diagnostics \
    --cp1 "$CP1_ROOT/results" --cp4 "$CP4_ROOT/results" \
    --cp1-reference "$CP1_REF" --cp4-reference "$CP4_REF" \
    --output "$SUMMARY" \
    2>&1 | tee "$BASE/cp1_cp4_phase_comparison_v1.log"
else
  echo "PENDING: provide both modes of both CP1 and CP4 before cross-case stage table." >&2
fi

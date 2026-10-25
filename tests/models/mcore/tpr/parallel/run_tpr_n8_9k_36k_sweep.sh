#!/usr/bin/env bash
# Qwen3-4B N=8 three-shape sweep: CP1 4608/4608, CP4 18432/18432, 32768/4096.
# Sequential, strict real-label capture gates; no recompute or optimizer.
set -euo pipefail
cd "$(dirname "$0")/../../../../.."

BASE=/workspace/tpr_experiments/performance
CP1_DIR="$BASE/cp1_n8_9k_1to1_v1"
echo "=== [1/3] CP1 N8 P4608 S4608: controlled Square-CANN Full vs TPR ==="
status=0
if [[ -f "$CP1_DIR/results/comparison.json" ]]; then
  echo "TPR_N8_ALREADY_COMPLETE $CP1_DIR/results/comparison.json"
elif ! TPR_CP1_N8_P=4608 TPR_CP1_N8_S=4608 TPR_CP1_N8_ROOT="$CP1_DIR" \
    bash tests/models/mcore/tpr/parallel/run_tpr_cp1_n8_9k_perf.sh; then
  echo "TPR_N8_SWEEP_FAILED CP1 P4608 S4608" >&2
  status=1
fi

for shape in 18k_18k 32k_4k; do
  if [[ "$shape" == "18k_18k" ]]; then
    P=18432; S=18432; TAG="36k_1to1"
    SOURCE="${TPR_CP4_N8_SOURCE_18K_18K:-${TPR_CP4_N8_SOURCE:-}}"
  else
    P=32768; S=4096; TAG="36k_8to1"
    SOURCE="${TPR_CP4_N8_SOURCE_32K_4K:-${TPR_CP4_N8_SOURCE:-}}"
  fi
  ROOT="$BASE/cp4_n8_${TAG}_v1"
  echo "=== CP4 N8 P=$P S=$S: native Full vs TPR Ring ==="
  if [[ -f "$ROOT/results/comparison.json" ]]; then
    echo "TPR_N8_ALREADY_COMPLETE $ROOT/results/comparison.json"
    continue
  fi
  if [[ -z "$SOURCE" ]]; then
    if ! SOURCE="$(python -m tests.models.mcore.tpr.parallel.select_tpr_n8_perf_source \
      --prefix "$P" --suffix "$S")"; then
      echo "TPR_N8_SWEEP_NO_SOURCE P=$P S=$S; preserved other cases" >&2
      status=1
      continue
    fi
  fi
  if [[ ! -f "$SOURCE" ]]; then
    echo "TPR_N8_SWEEP_MISSING_SOURCE $SOURCE" >&2
    status=1
    continue
  fi
  if ! TPR_CP4_N8_ROOT="$ROOT" \
      bash tests/models/mcore/tpr/parallel/run_tpr_cp4_n8_27k_perf.sh \
      "$shape" "$SOURCE"; then
    echo "TPR_N8_SWEEP_FAILED CP4 P=$P S=$S source=$SOURCE" >&2
    status=1
  fi
done

echo "=== Results ==="
for f in "$CP1_DIR/results/comparison.json" \
  "$BASE/cp4_n8_36k_1to1_v1/results/comparison.json" \
  "$BASE/cp4_n8_36k_8to1_v1/results/comparison.json"; do
  if [[ -f "$f" ]]; then
    echo "COMPLETE $f"
  else
    echo "PENDING $f"
  fi
done
exit "$status"

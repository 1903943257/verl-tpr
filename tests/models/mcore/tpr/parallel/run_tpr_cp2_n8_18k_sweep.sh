#!/usr/bin/env bash
# Sequential CP2 N=8 TP2: P16384/S2048 (8:1), P9216/S9216 (1:1).
# Qwen3-4B BF16, native Megatron/MindSpeed Full vs TPR Ring, no recompute.
set -euo pipefail
cd "$(dirname "$0")/../../../../.."
export PYTHONDONTWRITEBYTECODE=1

BASE=/workspace/tpr_experiments/performance
mkdir -p "$BASE"
echo "=== CP2 N8 18K CPU contract preflight ==="
python tests/models/mcore/tpr/run_cpu_regressions.py --only-cp1-n8-perf

status=0
for shape in 16k_2k 9k_9k; do
  if [[ "$shape" == "16k_2k" ]]; then
    P=16384; S=2048; TAG="18k_8to1"
    SOURCE="${TPR_CP2_N8_SOURCE_16K_2K:-${TPR_CP2_N8_SOURCE:-}}"
  else
    P=9216; S=9216; TAG="18k_1to1"
    SOURCE="${TPR_CP2_N8_SOURCE_9K_9K:-${TPR_CP2_N8_SOURCE:-}}"
  fi
  ROOT="$BASE/cp2_n8_${TAG}_v1"
  echo "=== CP2 N8 P=$P S=$S (TP2 CP2 DP1): Native Full / TPR Ring ==="
  if [[ -f "$ROOT/results/comparison.json" ]]; then
    echo "TPR_CP2_ALREADY_COMPLETE $ROOT/results/comparison.json"
    continue
  fi
  if [[ -z "$SOURCE" ]]; then
    if ! SOURCE="$(python -m tests.models.mcore.tpr.parallel.select_tpr_n8_perf_source \
      --prefix "$P" --suffix "$S")"; then
      echo "TPR_CP2_N8_NO_SOURCE P=$P S=$S" >&2
      status=1
      continue
    fi
  fi
  if [[ ! -f "$SOURCE" ]]; then
    echo "TPR_CP2_N8_MISSING_SOURCE $SOURCE" >&2
    status=1
    continue
  fi
  if ! TPR_CP_N8_ROOT="$ROOT" \
      bash tests/models/mcore/tpr/parallel/run_tpr_cp4_n8_27k_perf.sh \
      "$shape" "$SOURCE"; then
    echo "TPR_CP2_N8_FAILED P=$P S=$S source=$SOURCE" >&2
    status=1
  fi
done

echo "=== CP2 results ==="
for root in "$BASE/cp2_n8_18k_8to1_v1" "$BASE/cp2_n8_18k_1to1_v1"; do
  if [[ -f "$root/results/comparison.json" ]]; then
    echo "COMPLETE $root/results/comparison.json"
  else
    echo "PENDING $root/results/comparison.json"
  fi
done
exit "$status"

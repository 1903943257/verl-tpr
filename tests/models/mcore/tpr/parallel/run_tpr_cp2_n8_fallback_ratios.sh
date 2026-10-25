#!/usr/bin/env bash
# N8 CP2 source-preserving substitutions for unavailable P9216/S9216.
# Default source: authentic Run2 PPO 8-row capture (min response 6026).
# Both run native Megatron/MindSpeed Full vs production TPR Ring (TP2 CP2).
set -euo pipefail
cd "$(dirname "$0")/../../../../.."
export PYTHONDONTWRITEBYTECODE=1

SOURCE="${TPR_CP2_N8_SOURCE:-/workspace/tpr_actor_capture_4b_n8_run2/actor_update_step000003_rank2_capture000002_batch0.pt}"
BASE="/workspace/tpr_experiments/performance"
mkdir -p "$BASE"
if [[ ! -f "$SOURCE" ]]; then
  echo "Missing original 8-row Actor PPO capture: $SOURCE" >&2
  exit 2
fi
selector="${1:-both}"
case "$selector" in
  both) shapes=("6k_6k" "12k_6k") ;;
  6k_6k|12k_6k|16k_2k) shapes=("$selector") ;;
  *) echo "Usage: bash $0 [both|6k_6k|12k_6k|16k_2k]" >&2; exit 2 ;;
esac

echo "=== CPU contract preflight ==="
python tests/models/mcore/tpr/run_cpu_regressions.py --only-cp1-n8-perf
status=0
for shape in "${shapes[@]}"; do
  case "$shape" in
    6k_6k)  tag="12k_1to1" ;;
    12k_6k) tag="18k_2to1" ;;
    16k_2k) tag="18k_8to1_run2matched" ;;
  esac
  root="$BASE/cp2_n8_${tag}_v1"
  echo "=== CP2 N8 $shape source=$SOURCE ==="
  if [[ -f "$root/results/comparison.json" ]]; then
    echo "TPR_CP2_ALREADY_COMPLETE $root/results/comparison.json"
    continue
  fi
  if ! ASCEND_RT_VISIBLE_DEVICES=8,9,10,11 TPR_CP_N8_ROOT="$root" \
      bash tests/models/mcore/tpr/parallel/run_tpr_cp4_n8_27k_perf.sh \
      "$shape" "$SOURCE"; then
    echo "TPR_CP2_FAILED $shape; see $root/logs" >&2
    status=1
  fi
done
echo "=== Selected CP2 results ==="
for shape in "${shapes[@]}"; do
  case "$shape" in
    6k_6k)  tag="12k_1to1" ;;
    12k_6k) tag="18k_2to1" ;;
    16k_2k) tag="18k_8to1_run2matched" ;;
  esac
  file="$BASE/cp2_n8_${tag}_v1/results/comparison.json"
  if [[ -f "$file" ]]; then
    echo "COMPLETE $file"
  else
    echo "PENDING $file"
  fi
done
exit "$status"

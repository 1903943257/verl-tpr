#!/usr/bin/env bash
# Maintained correctness entry. NPU modes require a real NPU-enabled runtime.
# Usage: bash tests/models/mcore/tpr/run_correctness.sh [cp1|ring|tp2|qwen|actor-cp4]
set -euo pipefail
cd "$(dirname "$0")/../../../.."
HERE=tests/models/mcore/tpr
mode="${1:-cp1}"
[[ "$#" -le 1 ]] || { echo "Usage: $0 [cp1|ring|tp2|qwen|actor-cp4]" >&2; exit 2; }
case "$mode" in
  cp1) python "$HERE/run_suite.py" run npu-core --ack-npu ;;
  ring) python "$HERE/run_suite.py" run npu-ring --ack-npu ;;
  tp2) python "$HERE/run_suite.py" run npu-tp --ack-npu ;;
  qwen) python "$HERE/run_suite.py" run npu-qwen --ack-npu ;;
  actor-cp4)
    : "${TPR_REAL_ACTOR_MINIBATCH:?Set path to saved real actor PPO minibatch}"
    : "${TPR_OFFLINE_OUTPUT_DIR:?Set a fresh output directory}"
    [[ ! -e "$TPR_OFFLINE_OUTPUT_DIR" ]] || {
      echo "Output already exists: $TPR_OFFLINE_OUTPUT_DIR" >&2; exit 2;
    }
    export TPR_OFFLINE_RUN_MODES=both
    bash "$HERE/parallel/run_real_actor_tp2cp4_offline_ppo.sh"
    ;;
  *) echo "Unknown correctness scope: $mode" >&2; exit 2 ;;
esac

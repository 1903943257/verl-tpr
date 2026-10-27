#!/usr/bin/env bash
# Recompute entry: tiny contract first, then real-model OFF/ON.
# Usage: bash tests/models/mcore/tpr/run_recompute.sh [cpu|primitive|negative|actor-cp4|dp2] [args]
set -euo pipefail
cd "$(dirname "$0")/../../../.."
HERE=tests/models/mcore/tpr
mode="${1:-cpu}"
shift || :
case "$mode" in
  cpu)
    [[ "$#" -eq 0 ]] || exit 2
    python "$HERE/run_suite.py" run cpu-recompute
    ;;
  primitive)
    [[ "$#" -eq 0 ]] || exit 2
    python "$HERE/run_suite.py" run npu-recompute-primitive --ack-npu
    ;;
  negative)
    # Intentionally incompatible config-only baseline; NOT a green acceptance gate.
    [[ "$#" -eq 0 ]] || exit 2
    python "$HERE/run_suite.py" show npu-recompute-negative
    echo "Negative probe is diagnostic-only; launch directly only for deliberate failure analysis."
    ;;
  actor-cp4)
    [[ "$#" -eq 2 ]] || {
      echo "Usage: $0 actor-cp4 CAPTURE.pt NEW_OUTPUT_ROOT" >&2; exit 2;
    }
    capture="$1"; output="$2"
    [[ -f "$capture" && ! -e "$output" ]] || {
      echo "Missing capture or output already exists" >&2; exit 2;
    }
    mkdir -p "$output"
    output="$(realpath "$output")"
    export TPR_REAL_ACTOR_MINIBATCH="$(realpath "$capture")"
    export TPR_OFFLINE_ROWS="${TPR_OFFLINE_ROWS:-4,6}"
    export TPR_OFFLINE_SAVE_FULL_GRADS=1
    export TPR_OFFLINE_RUN_MODES=tpr
    export TPR_RECOMPUTE_LOG_DIR="$output/logs"
    for setting in 0 1; do
      export TPR_OFFLINE_RECOMPUTE="$setting"
      if [[ "$setting" == 0 ]]; then label=off; else label=on; fi
      export TPR_OFFLINE_OUTPUT_DIR="$output/$label"
      bash "$HERE/parallel/run_real_actor_tp2cp4_offline_ppo.sh"
    done
    python "$HERE/parallel/compare_tpr_recompute_off_on.py" \
      "$output/off" "$output/on" --require-full-grads
    ;;
  dp2)
    [[ "$#" -ge 2 && "$#" -le 3 ]] || {
      echo "Usage: $0 dp2 40K_MANIFEST NEW_OUTPUT_DIR [STEP=6]" >&2; exit 2;
    }
    bash "$HERE/parallel/run_tpr_dp2_recompute_validation.sh" "$@"
    ;;
  *) echo "Unknown recompute scope: $mode" >&2; exit 2 ;;
esac

#!/usr/bin/env bash
# Controlled profiling; neither --help nor merely running this script certifies numerics.
# Usage: bash tests/models/mcore/tpr/run_performance.sh [ring|fixed40k] [args]
set -euo pipefail
cd "$(dirname "$0")/../../../.."
HERE=tests/models/mcore/tpr
mode="${1:-ring}"
shift || :
case "$mode" in
  ring)
    [[ "$#" -eq 0 ]] || exit 2
    python "$HERE/run_suite.py" run npu-performance --ack-npu
    ;;
  fixed40k)
    [[ "$#" -eq 2 ]] || {
      echo "Usage: $0 fixed40k 40K_MANIFEST NEW_OUTPUT_DIR" >&2; exit 2;
    }
    bash "$HERE/parallel/run_tpr_fixed_shape_40k_benchmark.sh" "$@"
    ;;
  *) echo "Unknown performance scope: $mode" >&2; exit 2 ;;
esac

#!/usr/bin/env bash
# Maintained CPU entry point: bash tests/models/mcore/tpr/run_unit.sh [core|recompute|parallel|all]
set -euo pipefail
cd "$(dirname "$0")/../../../.."
HERE=tests/models/mcore/tpr
mode="${1:-core}"
[[ "$#" -le 1 ]] || { echo "Usage: $0 [core|recompute|parallel|all]" >&2; exit 2; }
python "$HERE/run_suite.py" check
case "$mode" in
  core) python "$HERE/run_suite.py" run cpu-core ;;
  recompute) python "$HERE/run_suite.py" run cpu-recompute ;;
  parallel) python "$HERE/run_suite.py" run cpu-parallel ;;
  all)
    python "$HERE/run_suite.py" run cpu-core
    python "$HERE/run_suite.py" run cpu-recompute
    python "$HERE/run_suite.py" run cpu-parallel
    ;;
  *) echo "Unknown CPU scope: $mode" >&2; exit 2 ;;
esac

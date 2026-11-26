#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
OPS_ROOT="$PWD"
if [ "$#" -ge 1 ]; then
  OPS_ROOT="$1"
  shift
fi

python3 "$SCRIPT_DIR/apply_stateful_gdn.py" "$OPS_ROOT" "$@"

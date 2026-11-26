#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
OPS_ROOT="$PWD"
if [ "$#" -ge 1 ]; then
  OPS_ROOT="$1"
fi

python3 "$SCRIPT_DIR/verify_stateful_gdn.py" "$OPS_ROOT"

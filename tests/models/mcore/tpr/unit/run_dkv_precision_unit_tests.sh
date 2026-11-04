#!/usr/bin/env bash
# Run dKV precision/diagnostic CPU unit tests through the repository's
# MindSpeed-first bootstrap. Direct pytest collection imports verl.workers.engine,
# whose package initializer eagerly imports MindSpeed and parses process argv.
set -Eeuo pipefail
cd "$(dirname "$0")/../../../../.."

PYTHON_BIN="${PYTHON_BIN:-python}"
exec "$PYTHON_BIN" -m tests.models.mcore.tpr._mindspeed_first_pytest \
  -q -s -x \
  tests/models/mcore/tpr/unit/test_delayed_dkv_fp32_experiment.py \
  tests/models/mcore/tpr/unit/test_dkv_replay_diagnostics.py

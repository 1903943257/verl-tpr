#!/usr/bin/env bash
# Lightweight TP1/CP1/DP2 smoke on physical Ascend NPUs 2 and 3.
# No Qwen checkpoint, no 30K manifest, no need for functional git metadata.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../../.." && pwd -P)"
cd "$ROOT"
export ASCEND_RT_VISIBLE_DEVICES=2,3
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
echo "TPR 2-NPU DP empty-replica smoke: physical cards 2,3; TP1 CP1 DP2"
torchrun --standalone --nproc_per_node=2 \
  tests/models/mcore/tpr/parallel/run_tpr_dp2_empty_2npu.py

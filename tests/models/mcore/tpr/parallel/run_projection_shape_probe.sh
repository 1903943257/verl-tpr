#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$0")/../../../../.."
: "${TPR_PROJECTION_OUTPUT:?Set a fresh output directory}"
: "${TPR_QWEN_4B_PATH:?Set the real checkpoint directory}"
: "${ASCEND_RT_VISIBLE_DEVICES:?Select one physical NPU}"
[[ "$ASCEND_RT_VISIBLE_DEVICES" != *,* && -d "$TPR_QWEN_4B_PATH" ]] || { echo 'Expected one NPU and an existing checkpoint' >&2; exit 2; }
[[ ! -e "$TPR_PROJECTION_OUTPUT" && ! -e "${TPR_PROJECTION_OUTPUT}.log" ]] || { echo 'Output already exists' >&2; exit 2; }
mkdir -p "$(dirname "$TPR_PROJECTION_OUTPUT")"
export TPR_PROJECTION_OUTPUT="$(cd "$(dirname "$TPR_PROJECTION_OUTPUT")" && pwd -P)/$(basename "$TPR_PROJECTION_OUTPUT")"
export TPR_RUN_PROJECTION_PROBE=1
export TPR_ROOT_NATIVE_RING=0 TPR_LEAF_BATCH_SIZE=1 TPR_EXEC_PROFILE=0
export HCCL_DETERMINISTIC=true PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
unset TPR_DKV_REPLAY_DIAG_PATH TPR_MEM_TRACE TPR_FIRST_LEAF_DIAG TPR_FIRST_LEAF_DIAG_SYNC || true
"${PYTHON_BIN:-python}" -m torch.distributed.run --nnodes=1 --node_rank=0 \
  --nproc_per_node=1 --master_addr=127.0.0.1 --master_port="${TPR_PROJECTION_PORT:-29695}" \
  -m tests.models.mcore.tpr._mindspeed_first_pytest -q -s -x -rs \
  tests/models/mcore/tpr/parallel/test_projection_shape_probe_npu.py \
  2>&1 | tee "${TPR_PROJECTION_OUTPUT}.log"
[[ -s "$TPR_PROJECTION_OUTPUT/report.json" ]] || { echo 'Projection probe did not produce report.json' >&2; exit 1; }

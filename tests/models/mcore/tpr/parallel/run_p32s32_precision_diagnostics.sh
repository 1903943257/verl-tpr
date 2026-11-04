#!/usr/bin/env bash
# Small fixed sequence: sequential attention ablations, then BF16 CP2 Ring parity.
set -Eeuo pipefail
cd "$(dirname "$0")/../../../../.."
: "${TPR_P32S32_DIAG_OUTPUT:?Set a fresh output directory}"
: "${TPR_QWEN_4B_PATH:?Set the same real Qwen checkpoint as before}"
: "${TPR_P32S32_DEVICES:?Select two distinct physical NPUs, e.g. 0,1}"
[[ ! -e "$TPR_P32S32_DIAG_OUTPUT" ]] || { echo 'Output already exists' >&2; exit 2; }
[[ -d "$TPR_QWEN_4B_PATH" ]] || { echo 'Checkpoint directory not found' >&2; exit 2; }
IFS=, read -r DEVICE0 DEVICE1 EXTRA <<< "$TPR_P32S32_DEVICES"
if [[ -z "$DEVICE0" || -z "$DEVICE1" || -n "${EXTRA:-}" || "$DEVICE0" == "$DEVICE1" ]]; then
  echo 'Select exactly two distinct NPUs' >&2; exit 2;
fi
mkdir -p "$TPR_P32S32_DIAG_OUTPUT"
export TPR_P32S32_DIAG_OUTPUT="$(cd "$TPR_P32S32_DIAG_OUTPUT" && pwd -P)"
export TPR_EXPERIMENT_DELAYED_DKV_FP32="${TPR_EXPERIMENT_DELAYED_DKV_FP32:-1}"
export TPR_P32S32_PROFILE="${TPR_P32S32_PROFILE:-4b}"
export ASCEND_RT_VISIBLE_DEVICES="$DEVICE0"
export TPR_SPLIT_BF16_DIAGNOSTICS=1
export TPR_SPLIT_BF16_PREFIX=32 TPR_SPLIT_BF16_SUFFIX=32
export TPR_SPLIT_BF16_PROFILE="$TPR_P32S32_PROFILE"
export TPR_SPLIT_BF16_OUTPUT="$TPR_P32S32_DIAG_OUTPUT/attention"
bash tests/models/mcore/tpr/parallel/run_split_bf16_quick.sh

export TPR_P32S32_OUTPUT="$TPR_P32S32_DIAG_OUTPUT/cp2"
bash tests/models/mcore/tpr/parallel/run_split_p32s32_cp2_parity.sh
echo "P32S32_PRECISION_DONE attention=$TPR_SPLIT_BF16_OUTPUT/report.json cp2=$TPR_P32S32_OUTPUT/comparison.json"

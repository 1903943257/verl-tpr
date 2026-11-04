#!/usr/bin/env bash
# Only the new checks; do not repeat the three full-model attention ablations.
set -Eeuo pipefail
cd "$(dirname "$0")/../../../../.."
: "${TPR_P32S32_CHECKS_OUTPUT:?Set a fresh output directory}"
: "${TPR_QWEN_4B_PATH:?Set the same checkpoint as the previous run}"
: "${TPR_P32S32_DEVICES:?Select two distinct NPUs, e.g. 0,1}"
[[ ! -e "$TPR_P32S32_CHECKS_OUTPUT" && -d "$TPR_QWEN_4B_PATH" ]] || { echo 'Expected fresh output and an existing checkpoint' >&2; exit 2; }
IFS=, read -r DEVICE0 DEVICE1 EXTRA <<< "$TPR_P32S32_DEVICES"
if [[ -z "$DEVICE0" || -z "$DEVICE1" || -n "${EXTRA:-}" || "$DEVICE0" == "$DEVICE1" ]]; then
  echo 'Select exactly two distinct NPUs' >&2; exit 2;
fi
mkdir -p "$TPR_P32S32_CHECKS_OUTPUT"
export TPR_P32S32_CHECKS_OUTPUT="$(cd "$TPR_P32S32_CHECKS_OUTPUT" && pwd -P)"
export TPR_P32S32_PROFILE="${TPR_P32S32_PROFILE:-4b}"
export TPR_PROJECTION_OUTPUT="$TPR_P32S32_CHECKS_OUTPUT/projection"
export ASCEND_RT_VISIBLE_DEVICES="$DEVICE0"
bash tests/models/mcore/tpr/parallel/run_projection_shape_probe.sh
export TPR_P32S32_NATIVE_RING=1
export TPR_P32S32_OUTPUT="$TPR_P32S32_CHECKS_OUTPUT/cp2"
bash tests/models/mcore/tpr/parallel/run_split_p32s32_cp2_parity.sh
echo "PROJECTION_NATIVE_DONE projection=$TPR_PROJECTION_OUTPUT/report.json native=$TPR_P32S32_OUTPUT/comparison.json"

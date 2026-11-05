#!/usr/bin/env bash
# Reuse the completed four routes; run only Base CP1 repeat and TPR CP2 QKV merge.
set -Eeuo pipefail
cd "$(dirname "$0")/../../../../.."
: "${TPR_TRAIN10_SOURCE:?Set the completed original four-route experiment}"
: "${TPR_TRAIN10_OUTPUT:?Set a NEW combined supplemental output directory}"
: "${TPR_TRAIN_MODEL_PATH:?Use the same checkpoint as the original experiment}"
: "${TPR_TRAIN_DEVICES:?Set the same two physical NPU IDs}"
[[ -d "$TPR_TRAIN_MODEL_PATH" ]] || { echo 'ERROR: checkpoint missing' >&2; exit 2; }
IFS=, read -r DEVICE0 DEVICE1 EXTRA <<< "$TPR_TRAIN_DEVICES"
[[ "$DEVICE0" =~ ^[0-9]+$ && "$DEVICE1" =~ ^[0-9]+$ && "$DEVICE0" != "$DEVICE1" && -z "${EXTRA:-}" ]] || {
    echo 'ERROR: set exactly two distinct physical NPU IDs' >&2; exit 2;
}
export TPR_TRAIN10_REPEAT_DEVICE="$DEVICE0"
export TPR_TRAIN10_REPEAT_SOURCE="$TPR_TRAIN10_SOURCE"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
[[ ! -e "$TPR_TRAIN10_OUTPUT" ]] || { echo 'ERROR: combined output already exists' >&2; exit 2; }
TPR_TRAIN10_SIX_PREFLIGHT=1 "${PYTHON_BIN:-python}" -m tests.models.mcore.tpr.parallel.training_multistep_sixway
mkdir -p "$TPR_TRAIN10_OUTPUT"
six_output="$(cd "$TPR_TRAIN10_OUTPUT" && pwd -P)"
export TPR_TRAIN10_OUTPUT="$six_output"
TPR_TRAIN10_OUTPUT="$six_output/base_cp1_repeat_run" \
    bash tests/models/mcore/tpr/parallel/run_training_base_cp1_repeat.sh
TPR_TRAIN10_OUTPUT="$six_output/qkvmerge_run" \
    bash tests/models/mcore/tpr/parallel/run_training_cp2_qkvmerge.sh
"${PYTHON_BIN:-python}" -m tests.models.mcore.tpr.parallel.training_multistep_sixway

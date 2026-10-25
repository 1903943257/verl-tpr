#!/usr/bin/env bash
# One bounded N8 experiment to separate sequence-split, prefix-sharing, and TreeBatch deltas.
# Arms: Full MB1 -> split_no_share -> TPR K1 -> TPR K2.
set -uo pipefail
cd "$(git rev-parse --show-toplevel)"

REF_ROOT="${TPR_CP1_N8_REFERENCE_ROOT:-/workspace/tpr_experiments/cp1_n8_full_mb2_vs_leaf_k_v1}"
ROOT="${TPR_CP1_N8_ATTRIBUTION_ROOT:-/workspace/tpr_experiments/cp1_n8_split_attribution_v1}"
REF_RESULT="$REF_ROOT/tpr_k1/result.json"

if [ -e "$ROOT" ]; then
    stamp=$(date +%Y%m%d_%H%M%S)
    mv "$ROOT" "${ROOT}.previous_${stamp}"
    echo "Existing attribution output moved to ${ROOT}.previous_${stamp}"
fi
mkdir -p "$ROOT"

CAPTURE="${TPR_CP1_N8_CAPTURE:-$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["source_capture"])' "$REF_RESULT" 2>/dev/null)}"
CKPT="${TPR_QWEN_4B_PATH:-$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["checkpoint"])' "$REF_RESULT" 2>/dev/null)}"
OVERALL=0

if [ ! -f "$CAPTURE" ]; then
    echo "ERROR: N8 capture missing: '$CAPTURE'; set TPR_CP1_N8_CAPTURE."
    OVERALL=1
fi
if [ ! -d "$CKPT" ]; then
    echo "ERROR: Qwen3-4B checkpoint missing: '$CKPT'; set TPR_QWEN_4B_PATH."
    OVERALL=1
fi

if [ "$OVERALL" -eq 0 ]; then
    export TPR_CP1_N8_CAPTURE="$CAPTURE"
    export TPR_QWEN_4B_PATH="$CKPT"
    export TPR_RUN_CP1_N8_FAIR=1
    export TPR_TREE_POLICY=qwen_turn
    export TPR_CP1_N8_WARMUPS=0
    export TPR_CP1_N8_REPEATS=1
    export TPR_CP1_N8_STAGE_PROFILE=0
    export TPR_EXEC_PROFILE=0
    export TPR_CP1_N8_SAVE_FULL_GRADS=1
    export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-8}"
    PORT_BASE="${TPR_CP1_N8_MASTER_PORT:-29711}"

    run_arm() {
        local arm="$1"
        local mode="$2"
        local k="$3"
        local out="$ROOT/$arm/result.json"
        local log="$ROOT/$arm.console.log"
        local port="$((PORT_BASE + $4))"
        mkdir -p "$ROOT/$arm"
        echo
        echo "===== SPLIT ATTRIBUTION ARM=$arm mode=$mode K=$k START ====="
        echo "capture=$TPR_CP1_N8_CAPTURE"
        echo "checkpoint=$TPR_QWEN_4B_PATH"
        echo "output=$out"
        TPR_CP1_N8_MODE="$mode" \
        TPR_LEAF_BATCH_SIZE="$k" \
        TPR_CP1_N8_OUTPUT="$out" \
        torchrun --nproc_per_node=1 --master_addr=127.0.0.1 --master_port="$port" \
            -m pytest -q -s \
            tests/models/mcore/tpr/parallel/test_real_actor_cp1_n8_fair_npu.py::test_real_actor_cp1_n8_full_mb2_vs_tpr \
            2>&1 | tee "$log"
        local rc=${PIPESTATUS[0]}
        echo "===== SPLIT ATTRIBUTION ARM=$arm rc=$rc ====="
        tail -n 80 "$log"
        if [ "$rc" -ne 0 ]; then
            OVERALL=1
        fi
    }

    run_arm full_mb1 full_mb1 1 1
    run_arm split_no_share split_no_share 1 2
    run_arm tpr_k1 tpr 1 3
    run_arm tpr_k2 tpr 2 4

    echo
    echo "===== SPLIT ATTRIBUTION COMPARISON ====="
    python -m tests.models.mcore.tpr.parallel.compare_cp1_n8_split_attribution \
        "$ROOT" 2>&1 | tee "$ROOT/comparison.console.log"
    compare_rc=${PIPESTATUS[0]}
    echo "===== SPLIT ATTRIBUTION COMPARATOR rc=$compare_rc ====="
    tail -n 120 "$ROOT/comparison.console.log"
    if [ "$compare_rc" -ne 0 ]; then
        OVERALL=1
    fi
fi

echo
echo "===== FINAL SPLIT ATTRIBUTION RESULTS / ERRORS ====="
echo "root=$ROOT"
echo "capture=$CAPTURE"
echo "checkpoint=$CKPT"
for f in "$ROOT"/*.console.log; do
    [ -f "$f" ] || continue
    echo "--- $f: key results/errors ---"
    grep -E 'TPR_CP1_N8_(TREE|SPLIT_NO_SHARE|RESULT|FULL_GRADS)|SPLIT ATTRIBUTION|grad_rel_l2=|error_vector_cosine=|ERROR|Error|Traceback|AssertionError|RuntimeError|out of memory|FAILED|passed' "$f" | tail -n 100 || true
done
if [ "$OVERALL" -eq 0 ]; then
    echo "N8 SPLIT ATTRIBUTION FINISHED: inspect split_attribution.json; diagnostic only, not a pass/fail gate."
else
    echo "N8 SPLIT ATTRIBUTION FAILED: inspect logs above."
    false
fi

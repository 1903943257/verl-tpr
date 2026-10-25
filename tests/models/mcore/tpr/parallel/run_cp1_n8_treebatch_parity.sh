#!/usr/bin/env bash
# Direct N=8 TreeBatch correctness gate: actual TPR K=1 vs K=2.
# Run from the repo in the NPU container. Full-gradient dumps are opt-in here
# and deliberately happen after the timed F/B iterations.
set -uo pipefail

cd "$(git rev-parse --show-toplevel)"
REF_ROOT="${TPR_CP1_N8_REFERENCE_ROOT:-/workspace/tpr_experiments/cp1_n8_full_mb2_vs_leaf_k_v1}"
ROOT="${TPR_CP1_N8_TREEBATCH_ROOT:-/workspace/tpr_experiments/treebatch_k1_k2_parity_v1}"
REF_RESULT="$REF_ROOT/tpr_k1/result.json"

if [ -e "$ROOT" ]; then
    stamp=$(date +%Y%m%d_%H%M%S)
    mv "$ROOT" "${ROOT}.previous_${stamp}"
    echo "Existing output moved to ${ROOT}.previous_${stamp}"
fi
mkdir -p "$ROOT"

CAPTURE="${TPR_CP1_N8_CAPTURE:-$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["source_capture"])' "$REF_RESULT" 2>/dev/null)}"
CKPT="${TPR_QWEN_4B_PATH:-$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["checkpoint"])' "$REF_RESULT" 2>/dev/null)}"
OVERALL=0

if [ ! -f "$CAPTURE" ]; then
    echo "ERROR: N=8 capture not found: '$CAPTURE'"
    echo "Set TPR_CP1_N8_CAPTURE explicitly, or restore $REF_RESULT."
    OVERALL=1
fi
if [ ! -d "$CKPT" ]; then
    echo "ERROR: Qwen3-4B checkpoint not found: '$CKPT'"
    echo "Set TPR_QWEN_4B_PATH explicitly."
    OVERALL=1
fi

if [ "$OVERALL" -eq 0 ]; then
    export TPR_CP1_N8_CAPTURE="$CAPTURE"
    export TPR_QWEN_4B_PATH="$CKPT"
    export TPR_RUN_CP1_N8_FAIR=1
    export TPR_TREE_POLICY=qwen_turn
    export TPR_CP1_N8_WARMUPS="${TPR_CP1_N8_WARMUPS:-1}"
    export TPR_CP1_N8_REPEATS="${TPR_CP1_N8_REPEATS:-3}"
    export TPR_CP1_N8_STAGE_PROFILE=0
    export TPR_EXEC_PROFILE=0
    export TPR_CP1_N8_SAVE_FULL_GRADS=1
    export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-8}"
    PORT="${TPR_CP1_N8_MASTER_PORT:-29681}"

    run_arm() {
        local k="$1"
        local log="$ROOT/tpr_k${k}.console.log"
        local out="$ROOT/tpr_k${k}/result.json"
        echo
        echo "===== TREEBATCH ARM K=${k} START ====="
        echo "capture=$TPR_CP1_N8_CAPTURE"
        echo "checkpoint=$TPR_QWEN_4B_PATH"
        echo "output=$out"
        TPR_CP1_N8_MODE=tpr \
        TPR_LEAF_BATCH_SIZE="$k" \
        TPR_CP1_N8_OUTPUT="$out" \
        torchrun --nproc_per_node=1 --master_addr=127.0.0.1 --master_port="$PORT" \
            -m pytest -q -s \
            tests/models/mcore/tpr/parallel/test_real_actor_cp1_n8_fair_npu.py::test_real_actor_cp1_n8_full_mb2_vs_tpr \
            2>&1 | tee "$log"
        local rc=${PIPESTATUS[0]}
        echo "===== TREEBATCH ARM K=${k} RESULT rc=${rc} ====="
        echo "log=$log"
        tail -n 60 "$log"
        if [ "$rc" -ne 0 ]; then
            OVERALL=1
        fi
    }

    run_arm 1
    run_arm 2

    echo
    echo "===== TREEBATCH K1/K2 OFFLINE COMPARISON ====="
    python -m tests.models.mcore.tpr.parallel.compare_cp1_n8_treebatch \
        "$ROOT" \
        --max-loss-gap 0.02 \
        --max-logprob-rel-l2 0.05 \
        --max-full-grad-rel-l2 0.05 \
        --min-full-grad-cosine 0.995 \
        2>&1 | tee "$ROOT/comparison.console.log"
    compare_rc=${PIPESTATUS[0]}
    echo "===== TREEBATCH COMPARISON RESULT rc=${compare_rc} ====="
    tail -n 100 "$ROOT/comparison.console.log"
    if [ "$compare_rc" -ne 0 ]; then
        OVERALL=1
    fi
fi

echo
echo "===== TREEBATCH K1/K2 FINAL RESULTS / ERRORS ====="
echo "root=$ROOT"
echo "capture=$CAPTURE"
echo "checkpoint=$CKPT"
for f in "$ROOT"/tpr_k*.console.log "$ROOT"/comparison.console.log; do
    [ -f "$f" ] || continue
    echo "--- $f : key results/errors ---"
    grep -E 'TPR_CP1_N8_(TREE|ITER|RESULT|FULL_GRADS)|TREEBATCH_PARITY=|GATES:|PASS|FAIL|ERROR|Error|Traceback|AssertionError|RuntimeError|out of memory' "$f" | tail -n 100 || true
done
if [ "$OVERALL" -eq 0 ]; then
    echo "TREEBATCH K1/K2 FINAL STATUS: PASS"
else
    echo "TREEBATCH K1/K2 FINAL STATUS: FAIL — inspect logs above"
    false
fi

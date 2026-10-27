#!/usr/bin/env bash
# Single prompt, N=1 or N=4. No PPO/Actor training.
# Usage:
#   PROBE_GROUP=20261027_1 bash examples/tpr/run_rollout_probe.sh 1
#   PROBE_GROUP=20261027_1 bash examples/tpr/run_rollout_probe.sh 4
set -euo pipefail
CHECKOUT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)"
cd "$CHECKOUT"

N="${1:-}"
case "$N" in 1|4) ;; *) echo "Usage: bash $0 1|4" >&2; exit 2;; esac

UNIAGENT_ROOT="${UNIAGENT_ROOT:-/workspace/uni-agent}"
MODEL_PATH="${MODEL_PATH:-/workspace/hf_models/Qwen3-4B-128K}"
TRAIN_FILE="${TRAIN_FILE:-/workspace/tpr_experiments/shortfix/pelita875_hint_h2_arm64_train.parquet}"
TASK_CONFIG="${TASK_CONFIG:-${CHECKOUT}/examples/tpr/task_config_swe_rebench_docker.yaml}"
PROBE_GROUP="${PROBE_GROUP:-$(date +%Y%m%d_%H%M%S)}"
PROBE_BASE="${PROBE_BASE:-/workspace/tpr_experiments/rollout_probes/${PROBE_GROUP}}"
RUN="$PROBE_BASE/n$N"
CONCURRENCY=4
PROMPT=16384
RESPONSE=8192
PROBE_TIMEOUT_S="${PROBE_TIMEOUT_S:-1200}"

export UNIAGENT_ROOT
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
export RAY_DEDUP_LOGS=0
export VLLM_USE_V1=1
export TASK_QUEUE_ENABLE=1
export VLLM_ASCEND_ENABLE_TOPK_OPTIMIZE=1
export PYTHONUNBUFFERED=1
export TQ_SIMPLE_STORAGE_SEND_RECV_TIMEOUT=2400

VLLM_ASCEND_SRC=""
if [[ -d /workspace/vllm-ascend ]]; then
  VLLM_ASCEND_SRC=/workspace/vllm-ascend:
elif [[ -d /vllm-ascend ]]; then
  VLLM_ASCEND_SRC=/vllm-ascend:
fi
export PYTHONPATH="$CHECKOUT:$UNIAGENT_ROOT:${VLLM_ASCEND_SRC}${PYTHONPATH:-}"

[[ -f "$TRAIN_FILE" ]] || { echo "ERROR missing dataset $TRAIN_FILE" >&2; exit 2; }
[[ -f "$TASK_CONFIG" ]] || { echo "ERROR missing config $TASK_CONFIG" >&2; exit 2; }
[[ -f "$MODEL_PATH/config.json" ]] || { echo "ERROR missing model $MODEL_PATH" >&2; exit 2; }
[[ ! -e "$RUN" ]] || { echo "ERROR: output already exists $RUN" >&2; exit 2; }
mkdir -p "$RUN"

# Fail rather than silently use an unpatched UniAgent. Apply before starting
# a *new* Ray job; never execute while another experiment is running.
python patches/apply_uniagent_session_hardcap.py "$UNIAGENT_ROOT" \
  > "$RUN/hardcap_apply.log" 2>&1 || {
    cat "$RUN/hardcap_apply.log" >&2
    exit 2
  }
grep -q '^APPLIED:' "$RUN/hardcap_apply.log" || {
  cat "$RUN/hardcap_apply.log" >&2
  exit 2
}

TRAIN_FILE="$TRAIN_FILE" python - <<'PY_CHECK'
import json, os
import pandas as pd
df = pd.read_parquet(os.environ["TRAIN_FILE"])
if len(df) != 1:
    raise SystemExit(f"Expected one Pelita task row, got {len(df)}")
t = df.iloc[0]["extra_info"]
if isinstance(t, str): t = json.loads(t)
task = t["tools_kwargs"]["task"]
assert task["metadata"]["instance_id"] == "ASPP__pelita-875", task["metadata"]["instance_id"]
assert (task.get("sandbox") or {}).get("image") == "uniagent-swe-pelita-875:latest"
print("ROLLOUT_PROBE_DATA_OK: one local-image Pelita-875 task", flush=True)
PY_CHECK

echo "ROLLOUT_PROBE_START n=$N concurrency=$CONCURRENCY TP=2 model=$MODEL_PATH output=$RUN"
echo "max_total_tokens=$((PROMPT + RESPONSE)) hard_stop=1 timeout_s=$PROBE_TIMEOUT_S"

# The standalone inference launcher calls ray.init, and should own a clean
# Ray runtime. Do not use on a host shared with an active PPO run.
ray stop --force >/dev/null 2>&1 || true

set +e
timeout --signal=TERM --kill-after=30s "$PROBE_TIMEOUT_S" \
  python examples/tpr/rollout_probe_entry.py \
  --data-path "$TRAIN_FILE" \
  --model-path "$MODEL_PATH" \
  --served-model-name Qwen3-4B \
  --task-config "$TASK_CONFIG" \
  --limit 1 --n "$N" --concurrency "$CONCURRENCY" \
  --nnodes 1 --n-gpus-per-node 8 --tp 2 \
  --engine vllm --gateway-count 1 --enforce-eager \
  --gpu-memory-utilization 0.60 \
  --tool-parser hermes \
  --temperature 0.6 --top-p 0.95 --top-k 20 \
  --prompt-length "$PROMPT" --response-length "$RESPONSE" \
  --log-dir "$RUN/agent" --result-path "$RUN/result.json" \
  2>&1 | tee "$RUN/launcher.log"
rc="${PIPESTATUS[0]}"
set -e

# Explicitly reject a "successful" inference run that still materializes
# multiple trajectories per Session or exceeds the 24K capacity.
if (( rc == 0 )); then
  if ! RUN="$RUN" N="$N" python - <<'PY_CAP_AUDIT'
import json
import os
from pathlib import Path

run = Path(os.environ["RUN"])
n = int(os.environ["N"])
paths = sorted((run / "agent").glob("session-*/trajectory.json"))
if len(paths) != n:
    raise SystemExit(f"STRICT_CAP_FAIL expected {n} session results; found {len(paths)}")
for path in paths:
    rows = json.loads(path.read_text())["trajectories"]
    if len(rows) != 1:
        raise SystemExit(f"STRICT_CAP_FAIL {path}: trajectories={len(rows)}")
    tokens = int(rows[0]["prompt_len"]) + int(rows[0]["response_len"])
    if tokens > 24576:
        raise SystemExit(f"STRICT_CAP_FAIL {path}: tokens={tokens}>24576")
print(f"STRICT_CAP_PASS sessions={n} max_tokens=24576", flush=True)
PY_CAP_AUDIT
  then
    rc=3
  fi
fi
echo "$rc" > "$RUN/exit_code.txt"

# Preserve engine stats before ray stop / next test replaces session_latest.
grep -RhE \
  'Avg prompt throughput|Avg generation throughput|GPU KV cache usage|Running: [0-9]+ reqs|Waiting: [0-9]+ reqs|preemption|num_requests_running|num_requests_waiting' \
  /tmp/ray/session_latest/logs \
  --include='*.out' --include='*.err' 2>/dev/null \
  > "$RUN/vllm_stats.log" || true

if [[ -d "$RUN/agent" ]]; then
  python examples/tpr/analyze_rollout_stages.py "$RUN/agent" \
    | tee "$RUN/stages.txt" || true
fi
echo "ROLLOUT_PROBE_END n=$N rc=$rc run=$RUN"
if (( rc != 0 )); then
  echo "Probe failed/timed out. Do not compare its partial wall time as a completed run." >&2
fi
exit "$rc"

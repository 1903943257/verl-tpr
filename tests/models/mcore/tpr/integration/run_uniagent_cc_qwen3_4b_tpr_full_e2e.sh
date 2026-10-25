#!/usr/bin/env bash
# One-command Uni-Agent E2E + offline Full/TPR replay.
# Run TPR online first, then Full regardless of TPR outcome. Preserve both
# Actor captures, TQ dumps and logs. Replay the same captured batches through
# genuine VERL Full and TPR engines; original trajectories are not cropped.
set -o pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../../../.." && pwd)"
ONLINE_SCRIPT="$REPO_ROOT/tests/models/mcore/tpr/integration/run_uniagent_cc_qwen17_swe_tpr_ppo_cp2.sh"
MANIFEST_BUILDER="$REPO_ROOT/tests/models/mcore/tpr/parallel/prepare_real_actor_20step_manifest.py"
OFFLINE_RUNNER="$REPO_ROOT/tests/models/mcore/tpr/parallel/run_verl_engine_training_replay.sh"
if [[ -z "$MODEL_PATH" ]]; then MODEL_PATH=/workspace/hf_models/Qwen3-4B-128K; fi
TRAIN_FILE=/workspace/data/uniagent-cc/django_11163_smoke_py311.parquet
TASK_CONFIG=/workspace/uni-agent-cc/examples/claude_code_swe_task/task_config_claude_code_django11163.yaml
STEPS=10
if [[ -z "$RUN_ROOT" ]]; then RUN_ROOT="/workspace/tpr_full_e2e_10step_$(date +%Y%m%d_%H%M%S)"; fi

if [[ -e "$RUN_ROOT" ]]; then
  echo "ERROR: refusing to overwrite existing run directory: $RUN_ROOT" >&2
  exit 2
fi
mkdir -p "$RUN_ROOT"/{logs,full,tpr,offline_sources} || exit 2
printf '%s\n' "$RUN_ROOT" > /workspace/tpr_full_e2e_latest.txt

# Use the exact source tree containing this orchestrator for online/offline
# engine imports. Uni-Agent application/data remain under /workspace/uni-agent-cc.
export VERL_CODE_ROOT="$REPO_ROOT"
export PYTHONUNBUFFERED=1
if [[ -z "$ASCEND_RT_VISIBLE_DEVICES" ]]; then
  export ASCEND_RT_VISIBLE_DEVICES=8,9,10,11,12,13,14,15
fi
if [[ "$ASCEND_RT_VISIBLE_DEVICES" != "8,9,10,11,12,13,14,15" ]]; then
  echo "ERROR: this recipe requires validated physical NPUs 8..15." >&2
  echo "ASCEND_RT_VISIBLE_DEVICES=$ASCEND_RT_VISIBLE_DEVICES" >&2
  exit 2
fi
export N=8
export N_GPUS_PER_NODE=8
export TOTAL_TRAINING_STEPS="$STEPS"
export TOTAL_EPOCHS="$STEPS"
export TRAIN_PROMPT_BSZ=1
export GEN_BATCH_SIZE=1
export PPO_MINI_BATCH_SIZE=1
export SESSION_HARD_CAP=True
export NATIVE_RECOMPUTE=False
export MODEL_PATH
export TPR_TREE_POLICY=qwen_turn
export TPR_TURN_MIN_SEGMENT_TOKENS=128
export TPR_EXEC_PROFILE=0

if [[ -f /workspace/.venvs/uniagent-cc/bin/activate ]]; then
  source /workspace/.venvs/uniagent-cc/bin/activate
fi
export PYTHONPATH="$REPO_ROOT:/workspace/uni-agent-cc:/workspace/vllm-ascend:$PYTHONPATH"

if [[ -f "$REPO_ROOT/patches/apply_uniagent_session_hardcap.py" ]]; then
  export UNIAGENT_SESSION_PATCHER="$REPO_ROOT/patches/apply_uniagent_session_hardcap.py"
elif [[ -f /workspace/bridge/patches/apply_uniagent_session_hardcap.py ]]; then
  export UNIAGENT_SESSION_PATCHER=/workspace/bridge/patches/apply_uniagent_session_hardcap.py
else
  echo "ERROR: session hard-cap patcher missing in repo and /workspace/bridge." >&2
  exit 2
fi

for required in \
  "$ONLINE_SCRIPT" "$MANIFEST_BUILDER" "$OFFLINE_RUNNER" \
  "$MODEL_PATH/config.json" "$TRAIN_FILE" "$TASK_CONFIG" \
  "$REPO_ROOT/verl/workers/config/engine.py" \
  "$REPO_ROOT/verl/workers/engine/megatron/transformer_impl.py" \
  "$REPO_ROOT/verl/workers/engine/megatron/tpr/adapter.py" \
  "$REPO_ROOT/verl/workers/engine_workers.py" \
  "$REPO_ROOT/verl/experimental/tpr/actor_capture.py"; do
  if [[ ! -e "$required" ]]; then
    echo "ERROR: required input/source missing: $required" >&2
    exit 2
  fi
done

grep -Fq 'tpr_enabled: bool' "$REPO_ROOT/verl/workers/config/engine.py" || {
  echo "ERROR: branch McoreEngineConfig lacks tpr_enabled." >&2; exit 2;
}
grep -Fq 'run_tpr_forward_backward_batch' "$REPO_ROOT/verl/workers/engine/megatron/transformer_impl.py" || {
  echo "ERROR: branch Megatron Engine lacks the TPR PPO route." >&2; exit 2;
}
grep -Fq 'TPR PPO supports TP1 CP1/2 DP1, TP2 CP1/2/4 DP1' "$REPO_ROOT/verl/workers/engine/megatron/tpr/adapter.py" || {
  echo "ERROR: branch TPR adapter does not advertise TP2/CP4/DP1 support." >&2; exit 2;
}
grep -Fq 'TPR P0 real actor-update capture' "$REPO_ROOT/verl/workers/engine_workers.py" || {
  echo "ERROR: selected branch is missing the real Actor minibatch capture hook." >&2; exit 2;
}
echo "TPR_E2E_PREFLIGHT_PASS code_root=$REPO_ROOT topology=TP2xRingCP4xDP1 NPU=8"

git -C "$REPO_ROOT" rev-parse HEAD > "$RUN_ROOT/git_commit.txt" 2>/dev/null || true
git -C "$REPO_ROOT" status --short > "$RUN_ROOT/git_status.txt" 2>/dev/null || true
echo "TPR_FULL_E2E_RUN_ROOT=$RUN_ROOT"
echo "CODE_ROOT=$REPO_ROOT MODEL_PATH=$MODEL_PATH STEPS_PER_MODE=$STEPS"

run_online() {
  local mode="$1"
  local enabled="$2"
  local start end rc
  export TPR_ENABLED="$enabled"
  export RUN_NAME="$(basename "$RUN_ROOT")-$mode"
  export TQ_DUMP_DIR="$RUN_ROOT/$mode/tq_dump"
  export TPR_CAPTURE_ACTOR_MINIBATCH_DIR="$RUN_ROOT/$mode/actor_capture"
  mkdir -p "$TQ_DUMP_DIR" "$TPR_CAPTURE_ACTOR_MINIBATCH_DIR"

  local log="$RUN_ROOT/logs/$mode.log"
  echo
  echo "START ONLINE mode=$mode TPR_ENABLED=$TPR_ENABLED steps=$STEPS"
  echo "ACTOR_CAPTURE=$TPR_CAPTURE_ACTOR_MINIBATCH_DIR"
  echo "TQ_DUMP=$TQ_DUMP_DIR LOG=$log"
  start=$(date +%s)
  rc=0
  bash "$ONLINE_SCRIPT" 2>&1 | tee "$log" || rc=$?
  end=$(date +%s)
  printf '%s\n' "$((end - start))" > "$RUN_ROOT/$mode""_wall_seconds.txt"
  printf '%s\n' "$rc" > "$RUN_ROOT/$mode""_exit_code.txt"
  echo "END ONLINE mode=$mode exit=$rc wall_seconds=$((end - start))"
  ray stop --force > "$RUN_ROOT/logs/$mode""_ray_cleanup.log" 2>&1 || true
  return "$rc"
}

# TPR first validates the enabled route. Full always runs if TPR exits nonzero.
tpr_rc=0
run_online tpr True || tpr_rc=$?
full_rc=0
run_online full False || full_rc=$?

# Build manifests from each run if possible. Prefer Full for the shared replay
# source; use TPR captures only if Full did not produce a complete manifest.
replay_source=""
replay_source_mode=none
for mode in full tpr; do
  source_dir="$RUN_ROOT/$mode/actor_capture"
  manifest_dir="$RUN_ROOT/offline_sources/$mode"
  mkdir -p "$manifest_dir"
  manifest_log="$RUN_ROOT/logs/manifest_$mode.log"
  echo
  echo "BUILD OFFLINE MANIFEST source=$mode capture_dir=$source_dir"
  if python "$MANIFEST_BUILDER" "$source_dir" "$manifest_dir/capture_manifest.json" \
      --steps "$STEPS" --max-seq-len 65536 --expected-sessions 8 \
      2>&1 | tee "$manifest_log"; then
    echo "OFFLINE_MANIFEST_PASS source=$mode path=$manifest_dir/capture_manifest.json"
    if [[ "$replay_source_mode" == "none" ]]; then
      replay_source="$manifest_dir"
      replay_source_mode="$mode"
    fi
  else
    echo "OFFLINE_MANIFEST_UNAVAILABLE source=$mode; raw captures retained at $source_dir" >&2
  fi
done

offline_rc=125
if [[ "$replay_source_mode" != "none" ]]; then
  offline_root="$RUN_ROOT/offline_replay"
  echo
  echo "START OFFLINE Full-vs-TPR replay source=$replay_source_mode steps=$STEPS"
  echo "SOURCE_MANIFEST=$replay_source/capture_manifest.json"
  echo "Sequence policy: original full trajectories, no controlled cropping"
  export TPR_REPLAY_SHORT_CAP=0
  export TPR_REPLAY_MAX_SEQ_LEN=65536
  export TPR_REPLAY_REQUIRE_SHARED_PREFIX=1
  export TPR_OFFLINE_NATIVE_RECOMPUTE_LAYERS=0
  export TPR_OFFLINE_SAVE_FULL_GRADS=0
  if [[ -z "$TPR_OFFLINE_ADAMW_LR" ]]; then export TPR_OFFLINE_ADAMW_LR=2e-5; fi
  if [[ -z "$TPR_OFFLINE_ADAMW_CLIP_NORM" ]]; then export TPR_OFFLINE_ADAMW_CLIP_NORM=1.0; fi
  offline_rc=0
  bash "$OFFLINE_RUNNER" "$replay_source" "$STEPS" both "$offline_root" \
    2>&1 | tee "$RUN_ROOT/logs/offline_replay.log" || offline_rc=$?
  echo "END OFFLINE replay exit=$offline_rc"
else
  echo "OFFLINE_REPLAY_SKIPPED: neither online mode produced a complete $STEPS-step, 8-session manifest." \
    | tee "$RUN_ROOT/logs/offline_replay_skipped.log"
fi

# Durable inventory: tomorrow's acceptance can start from exact paths and counts.
python - "$RUN_ROOT" "$STEPS" "$tpr_rc" "$full_rc" "$offline_rc" "$replay_source_mode" <<'PY'
import json
import os
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
steps = int(sys.argv[2])
tpr_rc, full_rc, offline_rc = map(int, sys.argv[3:6])
source_mode = sys.argv[6]

def files_under(rel, pattern="*"):
    base = root / rel
    return sorted(p for p in base.rglob(pattern) if p.is_file()) if base.exists() else []

artifacts = []
for p in sorted(root.rglob("*")):
    if p.is_file() and p.name not in {"artifact_manifest.json", "summary.json"}:
        artifacts.append({"path": str(p.relative_to(root)), "size_bytes": p.stat().st_size})

counts = {
    "full_actor_capture_files": len(files_under("full/actor_capture", "actor_update_step*.pt")),
    "tpr_actor_capture_files": len(files_under("tpr/actor_capture", "actor_update_step*.pt")),
    "full_tq_pt_files": len(files_under("full/tq_dump", "*.pt")),
    "tpr_tq_pt_files": len(files_under("tpr/tq_dump", "*.pt")),
    "offline_snapshot_files": len(files_under("offline_replay/offline", "*.pt")),
    "offline_source_full_manifest": int((root / "offline_sources/full/capture_manifest.json").is_file()),
    "offline_source_tpr_manifest": int((root / "offline_sources/tpr/capture_manifest.json").is_file()),
}
summary = {
    "run_root": str(root),
    "code_root": os.environ.get("VERL_CODE_ROOT"),
    "git_commit": (root / "git_commit.txt").read_text().strip() if (root / "git_commit.txt").exists() else None,
    "steps_per_online_mode": steps,
    "online_tpr_exit_code": tpr_rc,
    "online_full_exit_code": full_rc,
    "offline_replay_exit_code": offline_rc,
    "offline_replay_source_mode": source_mode,
    "counts": counts,
    "important_paths": {
        "tpr_online_log": str(root / "logs/tpr.log"),
        "full_online_log": str(root / "logs/full.log"),
        "offline_replay_log": str(root / "logs/offline_replay.log"),
        "offline_comparison": str(root / "offline_replay/logs/comparison.log"),
        "full_actor_captures": str(root / "full/actor_capture"),
        "tpr_actor_captures": str(root / "tpr/actor_capture"),
        "full_tq_dump": str(root / "full/tq_dump"),
        "tpr_tq_dump": str(root / "tpr/tq_dump"),
        "offline_source_manifest": (
            str(root / "offline_sources" / source_mode / "capture_manifest.json")
            if source_mode != "none" else None
        ),
    },
}
(root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
(root / "artifact_manifest.json").write_text(json.dumps(artifacts, indent=2) + "\n")
print(json.dumps(summary, indent=2))
print(f"ARTIFACT_MANIFEST={root / 'artifact_manifest.json'}")
PY

echo
echo "TPR_FULL_E2E_FINAL run_root=$RUN_ROOT"
echo "Summary: $RUN_ROOT/summary.json"
echo "Artifacts: $RUN_ROOT/artifact_manifest.json"
echo "Online TPR log: $RUN_ROOT/logs/tpr.log"
echo "Online Full log: $RUN_ROOT/logs/full.log"
echo "Offline replay log: $RUN_ROOT/logs/offline_replay.log"
echo "Offline comparison log: $RUN_ROOT/offline_replay/logs/comparison.log"

if [[ "$tpr_rc" != "0" || "$full_rc" != "0" || "$offline_rc" != "0" ]]; then
  echo "TPR_FULL_E2E_FAILED tpr_exit=$tpr_rc full_exit=$full_rc offline_exit=$offline_rc; artifacts are retained." >&2
  exit 1
fi
echo "TPR_FULL_E2E_PASS online_tpr=0 online_full=0 offline_replay=0"

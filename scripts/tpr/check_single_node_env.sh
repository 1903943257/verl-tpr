#!/usr/bin/env bash
# Read-only single-node E2E environment inventory. Never launches Ray or allocates an NPU tensor.
# Usage: bash scripts/tpr/check_single_node_env.sh
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
TPR_ROOT="$(cd -- "$HERE/../.." && pwd -P)"
UNIAGENT_ROOT="${UNIAGENT_ROOT:-/workspace/uni-agent}"
MODEL_PATH="${MODEL_PATH:-/workspace/hf_models/Qwen3-4B-128K}"
TRAIN_FILE="${TRAIN_FILE:-/workspace/tpr_experiments/shortfix/shortfix_candidates_task.parquet}"
TASK_CONFIG="${TASK_CONFIG:-$TPR_ROOT/examples/tpr/task_config_swe_rebench_docker.yaml}"
VLLM_ASCEND_SRC=""
if [[ -d /vllm-ascend ]]; then
  VLLM_ASCEND_SRC="/vllm-ascend:"
elif [[ -d /workspace/vllm-ascend ]]; then
  VLLM_ASCEND_SRC="/workspace/vllm-ascend:"
fi
export PYTHONPATH="$TPR_ROOT:$UNIAGENT_ROOT:${VLLM_ASCEND_SRC}${PYTHONPATH:-}"
export UNIAGENT_ROOT TPR_ROOT MODEL_PATH TRAIN_FILE TASK_CONFIG
cd "$TPR_ROOT"
echo "=== TPR single-node inventory; NO Ray, model load or NPU tensor allocation ==="
echo "Hostname: $(hostname) | Python: $(command -v python || echo missing)"
echo "TPR=$TPR_ROOT | UniAgent=$UNIAGENT_ROOT"
for d in "$TPR_ROOT" "$UNIAGENT_ROOT" /Megatron-LM /MindSpeed /vllm-ascend /Megatron-Bridge /verl; do
  [[ -d "$d" ]] && echo "DIR OK: $d" || echo "DIR MISSING: $d"
done
for d in "$TPR_ROOT" "$UNIAGENT_ROOT"; do
  if git -C "$d" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "GIT $(git -C "$d" rev-parse --short HEAD) $(git -C "$d" branch --show-current): $d"
    [[ -z "$(git -C "$d" status --porcelain)" ]] || echo "GIT WARNING: uncommitted changes in $d"
  else
    echo "GIT MISSING: $d"
  fi
done
if command -v npu-smi >/dev/null 2>&1; then
  echo "--- npu-smi info (read-only) ---"
  npu-smi info || echo "NPU-SMI ERROR"
else
  echo "NPU-SMI MISSING: check Ascend runtime or device mounts"
fi
echo "--- Files (only existence, not valid contents) ---"
for f in "$MODEL_PATH/config.json" "$TRAIN_FILE" "$TASK_CONFIG" "$TPR_ROOT/patches/apply_uniagent_session_hardcap.py"; do
  [[ -f "$f" ]] && echo "FILE PRESENT: $f" || echo "FILE MISSING: $f"
done
if command -v docker >/dev/null 2>&1; then
  docker info >/dev/null 2>&1 && echo "DOCKER OK: daemon reachable" || echo "DOCKER MISSING: CLI present, daemon inaccessible"
else
  echo "DOCKER MISSING: CLI not available inside container"
fi
echo "--- Python package versions and import locations (no imports of heavy NPU modules) ---"
python - <<'PY'
import importlib.metadata as md
import importlib.util
import os
from pathlib import Path
packages = ("torch", "torch-npu", "ray", "vllm", "vllm-ascend",
            "megatron-core", "mbridge", "transformers", "swebench", "pandas", "pyarrow")
for name in packages:
    try:
        print(f"VERSION {name}: {md.version(name)}")
    except md.PackageNotFoundError:
        print(f"VERSION {name}: NOT INSTALLED (metadata missing)")
for module, expected in (("verl", "TPR_ROOT"), ("uni_agent", "UNIAGENT_ROOT")):
    try:
        spec = importlib.util.find_spec(module)
        origin = getattr(spec, "origin", None)
        path = Path(origin).resolve() if origin else None
        root = Path(os.environ[expected]).resolve()
        if path and path.is_relative_to(root):
            print(f"IMPORT PASS {module}: {path}")
        else:
            print(f"IMPORT FAIL {module}: {path} (expected under {root})")
    except Exception as exc:
        print(f"IMPORT FAIL {module}: {type(exc).__name__}: {exc}")
for module in ("torch_npu", "vllm", "vllm_ascend", "ray", "mbridge", "megatron"):
    try:
        spec = importlib.util.find_spec(module)
        print(f"MODULE {module}: {getattr(spec, 'origin', None)}")
    except Exception as exc:
        print(f"MODULE {module}: ERROR {exc}")
PY
echo "=== Inventory complete. Missing files/modules are NOT silently installed. ==="

#!/usr/bin/env bash
# Reproduce the 2026-10-13 Ascend development container layout, using uniagent:new.
# Run on the HOST after cloning verl-tpr-e2e. Git remains on the host.
set -euo pipefail
IMAGE="${IMAGE:-uniagent:new}"
NAME="${CONTAINER_NAME:-uniagent_16npu}"
WS="${HOST_WORKSPACE:-/home/j00972288/project/uniagent_linear}"

command -v docker >/dev/null || { echo "ERROR: Docker CLI missing on host" >&2; exit 2; }
[[ -S /var/run/docker.sock ]] || { echo "ERROR: Docker socket unavailable" >&2; exit 2; }
docker image inspect "$IMAGE" >/dev/null || { echo "ERROR: Docker image not installed: $IMAGE" >&2; exit 2; }
# This launcher is for ASCEND, not CUDA: fail rather than silently start the wrong runtime.
[[ -c /dev/davinci_manager ]] || {
  echo "ERROR: no Ascend /dev/davinci_manager; this may be an NVIDIA machine. Do not run NPU E2E here." >&2
  exit 2
}
[[ -d /usr/local/Ascend/driver && -d /usr/local/dcmi ]] || {
  echo "ERROR: expected Ascend driver and dcmi on host" >&2; exit 2;
}
NPU_SMI="$(command -v npu-smi || true)"
[[ -n "$NPU_SMI" && -f "$NPU_SMI" ]] || { echo "ERROR: host npu-smi missing" >&2; exit 2; }
if docker container inspect "$NAME" >/dev/null 2>&1; then
  echo "ERROR: container $NAME already exists; inspect before changing or deleting it." >&2
  exit 2
fi
mkdir -p "$WS"
# Preserve the old development contract: host network, privileged, Docker-in-Docker via
# host socket, workspace -> /workspace. These grants are powerful; use trusted hosts only.
args=(
  run -itd --name "$NAME" --privileged --network host
  --shm-size=128g --ulimit memlock=-1 --ulimit stack=67108864
  -v "/usr/local/Ascend/driver:/usr/local/Ascend/driver"
  -v "/usr/local/dcmi:/usr/local/dcmi"
  -v "$NPU_SMI:/usr/local/bin/npu-smi:ro"
  -v "/var/run/docker.sock:/var/run/docker.sock"
  -v "$WS:/workspace"
  -w /workspace
  --entrypoint /bin/bash
)
# The Docker sandbox backend executes "docker run" from inside this container.
DOCKER_BIN="$(command -v docker)"
args+=(-v "$DOCKER_BIN:/usr/local/bin/docker:ro")
[[ ! -d /var/log/npu ]] || args+=(-v "/var/log/npu:/usr/slog")
echo "Starting $IMAGE as $NAME; workspace $WS -> /workspace"
# The explicit entrypoint prevents /workspace bind-mount masking the image's
# original /workspace/entrypoint.sh (observed failure in the former server).
docker "${args[@]}" "$IMAGE" -lc 'exec sleep infinity'
echo "Enter with: docker exec -it $NAME bash"
echo "NEXT: Check docs/tpr/GETTING_STARTED_UNIAGENT_NEW.md before attempting E2E."

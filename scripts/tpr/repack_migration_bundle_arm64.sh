#!/usr/bin/env bash
# Replace the AMD64 Pelita image in the 2026-10-26 migration bundle with the
# locally verified ARM64 + Claude Code image, preserving the original bundle.
# Run inside uniagent:qwen36, where /workspace is the shared host workspace.
set -euo pipefail

ROOT="${ROOT:-/workspace}"
OLD="$ROOT/tpr_migration_bundle_20261026"
NEW="$ROOT/tpr_migration_bundle_20261027_arm64"
OUT="$ROOT/tpr_migration_bundle_20261027_arm64.tar"
TAG="uniagent-swe-pelita-875:latest"
DATA="$ROOT/tpr_experiments/shortfix/pelita875_hint_h2_noinstall.parquet"
BASE_COMMIT=68af15d8d4199882d32bb4ede363195e2c5b5a99

for cmd in docker python tar gzip sha256sum; do
  command -v "$cmd" >/dev/null 2>&1 || { echo "Missing command: $cmd" >&2; exit 2; }
done
[[ -d "$OLD" ]] || { echo "Missing original staged bundle: $OLD" >&2; exit 2; }
[[ -f "$OLD/sandbox/pelita875.zip" ]] || { echo "Missing original Pelita ZIP" >&2; exit 2; }
[[ -f "$DATA" ]] || { echo "Missing validated no-install Parquet: $DATA" >&2; exit 2; }
for path in "$NEW" "$OUT" "$OUT.sha256" "$OUT.part"; do
  [[ ! -e "$path" ]] || { echo "Refusing to overwrite existing output: $path" >&2; exit 2; }
done

actual="$(docker image inspect "$TAG" --format '{{.Os}}/{{.Architecture}}')"
image_id="$(docker image inspect "$TAG" --format '{{.Id}}')"
[[ "$actual" == linux/arm64 ]] || { echo "Wrong image architecture: $actual" >&2; exit 2; }
claude_version="$(docker run --rm --pull never --network none --entrypoint bash "$TAG" -lc \
  "test \"\$(uname -m)\" = aarch64 && command -v claude >/dev/null && claude --version && test \"\$(git -C /testbed cat-file -t $BASE_COMMIT)\" = commit")"
[[ -n "$claude_version" ]] || { echo "Claude Code smoke test returned empty output" >&2; exit 2; }
echo "Using $TAG $actual $image_id (Claude: $claude_version)"

# The staged model is hardlinked, not copied, so only the new Sandbox ZIP costs
# significant extra space until the final uncompressed TAR is created.
old_bytes="$(du -sb "$OLD" | awk '{print $1}')"
free_bytes="$(df -PB1 "$ROOT" | awk 'NR==2 {print $4}')"
reserve=$(( old_bytes + 4 * 1024 * 1024 * 1024 ))
if (( free_bytes < reserve )); then
  echo "Insufficient space: free=$free_bytes bytes, need at least $reserve bytes." >&2
  echo "No files were deleted. Use a larger filesystem if needed." >&2
  exit 2
fi

TMP="$(mktemp -d "$ROOT/.tpr_arm64_export.XXXXXXXX")"
trap 'rm -rf -- "$TMP"' EXIT

echo "Hardlinking staged assets without overwriting the old bundle..."
cp -al -- "$OLD" "$NEW"
# The ZIP was hardlinked from OLD. Unlink in NEW before writing its replacement.
rm -- "$NEW/sandbox/pelita875.zip"
cp -p -- "$DATA" "$NEW/data/shortfix/pelita875_hint_h2_noinstall.parquet"

echo "Exporting the actual ARM64 + Claude Code image..."
docker save "$TAG" | gzip -1 > "$TMP/image.tar.gz"
(cd "$TMP" && sha256sum image.tar.gz > image.tar.gz.sha256)
printf '%s\n' "$TAG" > "$TMP/IMAGE.txt"
printf '%s\n' "$image_id" > "$TMP/IMAGE_ID.txt"
printf '%s\n' linux/arm64 > "$TMP/PLATFORM.txt"
printf '%s\n' "$claude_version" > "$TMP/CLAUDE_VERSION.txt"
printf '%s\n' 'Native ARM64 Pelita-875 Docker sandbox with Claude Code preinstalled' > "$TMP/SOURCE_IMAGE.txt"
python - "$TMP" "$NEW/sandbox/pelita875.zip" <<'PY'
from pathlib import Path
import sys
from zipfile import ZipFile, ZIP_STORED
src, dest = map(Path, sys.argv[1:])
with ZipFile(dest, 'w', compression=ZIP_STORED, allowZip64=True) as z:
    for name in ('image.tar.gz', 'image.tar.gz.sha256', 'IMAGE.txt', 'IMAGE_ID.txt',
                 'PLATFORM.txt', 'CLAUDE_VERSION.txt', 'SOURCE_IMAGE.txt'):
        z.write(src / name, arcname=name)
print('Created:', dest, 'bytes=', dest.stat().st_size)
PY

# Replace hardlinked text/scripts via rename, never truncate them in place.
cat > "$TMP/import_pelita_image.sh" <<'IMPORT'
#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd -- "$(dirname -- "$0")" && pwd -P)"
ZIP="$HERE/../sandbox/pelita875.zip"
TAG=uniagent-swe-pelita-875:latest
BASE_COMMIT=68af15d8d4199882d32bb4ede363195e2c5b5a99
python - "$ZIP" <<'PY' | docker load
import hashlib, shutil, sys, zipfile
p = sys.argv[1]
with zipfile.ZipFile(p) as z:
    manifest = z.read('image.tar.gz.sha256').decode().split()[0]
    h = hashlib.sha256()
    with z.open('image.tar.gz') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    if h.hexdigest() != manifest:
        print('ERROR: embedded image SHA256 mismatch', file=sys.stderr)
        raise SystemExit(2)
    print('Verified image.tar.gz SHA256:', manifest, file=sys.stderr)
    with z.open('image.tar.gz') as stream:
        shutil.copyfileobj(stream, sys.stdout.buffer, 8 * 1024 * 1024)
PY
expected="$(python - "$ZIP" <<'PY'
import sys, zipfile
with zipfile.ZipFile(sys.argv[1]) as z:
    print(z.read('IMAGE_ID.txt').decode().strip())
PY
)"
loaded="$(docker image inspect "$TAG" --format '{{.Id}}')"
platform="$(docker image inspect "$TAG" --format '{{.Os}}/{{.Architecture}}')"
[[ "$expected" == "$loaded" ]] || { echo "ERROR: image ID mismatch ($loaded != $expected)" >&2; exit 3; }
[[ "$platform" == linux/arm64 ]] || { echo "ERROR: wrong platform: $platform" >&2; exit 3; }
docker run --rm --pull never --network none --entrypoint bash "$TAG" -lc \
  "test \"\$(uname -m)\" = aarch64 && command -v claude && claude --version && test \"\$(git -C /testbed cat-file -t $BASE_COMMIT)\" = commit"
echo "PASS: imported $TAG ($loaded, $platform); Claude Code ready."
IMPORT
chmod 755 "$TMP/import_pelita_image.sh"
mv -f -- "$TMP/import_pelita_image.sh" "$NEW/tools/import_pelita_image.sh"

sed '/^Pelita image ID:/d' "$OLD/SOURCE_INFO.txt" > "$TMP/SOURCE_INFO.txt"
{
  printf 'Migration bundle updated: %s\n' "$(date -Is)"
  printf 'Pelita image ID: %s\n' "$image_id"
  printf 'Pelita image name: %s\n' "$TAG"
  printf 'Pelita image architecture: %s\n' "$actual"
  printf 'Preinstalled Claude Code: %s\n' "$claude_version"
  printf 'Gold patch offline reward validated: baseline=0, gold=1; no install needed\n'
} >> "$TMP/SOURCE_INFO.txt"
mv -f -- "$TMP/SOURCE_INFO.txt" "$NEW/SOURCE_INFO.txt"

cat > "$TMP/README.md" <<'README'
# TPR / UniAgent migration bundle — ARM64 (2026-10-27)

Updated from tpr_migration_bundle_20261026. The original AMD64 migration archive
is unchanged. This bundle is for the ARM64 Ascend NPU servers.

Contents:
- models/Qwen3-4B/: all Qwen3-4B model shards, tokenizer and base config.
- models/Qwen3-4B-128K-config/: 131072-token YaRN config, no duplicated weights.
- sandbox/pelita875.zip: *ARM64* Pelita-875 Docker image with Claude Code preloaded.
- data/shortfix/pelita875_hint_h2_noinstall.parquet: preferred offline reward
  dataset (no pip install -e .). Gold/baseline offline reward validated 1/0.
- Other data/shortfix/ files, migration_extras/, docs/, and tools/ retained.

The Django image, training checkpoints, duplicate 128K weights, and source
repositories are not included. Runtime code and images must be prepared on
the target node separately.

On each target server, unpack under /workspace:
    sha256sum -c tpr_migration_bundle_20261027_arm64.tar.sha256
    tar -xf tpr_migration_bundle_20261027_arm64.tar -C /workspace
    cd /workspace/tpr_migration_bundle_20261027_arm64
    sha256sum -c SHA256SUMS

Import image into the Docker daemon used to launch sibling task sandboxes:
    bash tools/import_pelita_image.sh
This verifies the embedded archive digest, actual image ID, ARM64 architecture,
Claude Code CLI, and Pelita-875 base commit.
The required image tag is uniagent-swe-pelita-875:latest. Pull policy: never.

Install the task data and model without overwriting existing target assets:
    mkdir -p /workspace/tpr_experiments/shortfix /workspace/hf_models
    cp -an data/shortfix/. /workspace/tpr_experiments/shortfix/
    # Only if Qwen3-4B does not already exist and paths share a filesystem:
    cp -al models/Qwen3-4B /workspace/hf_models/Qwen3-4B

If the model directory exists, compare checksums instead of overwriting it.
To create the long-context 128K YaRN model sharing the model weights:
    bash tools/materialize_qwen3_4b_128k.sh
Use pelita875_hint_h2_noinstall.parquet for future inference/e2e experiments;
its metadata omits runtime pip installation. The other Parquet files are
retained as reference datasets.
README
mv -f -- "$TMP/README.md" "$NEW/README.md"

echo "Recomputing full bundle checksums..."
(cd "$NEW" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum) > "$TMP/SHA256SUMS"
mv -f -- "$TMP/SHA256SUMS" "$NEW/SHA256SUMS"
(cd "$NEW" && sha256sum -c SHA256SUMS >/dev/null)

echo "Creating a separate ARM64 bundle TAR..."
tar -cf "$OUT.part" -C "$ROOT" "$(basename "$NEW")"
mv -- "$OUT.part" "$OUT"
(cd "$ROOT" && sha256sum "$(basename "$OUT")" > "$(basename "$OUT").sha256")
(cd "$ROOT" && sha256sum -c "$(basename "$OUT").sha256")
ls -lh "$OUT" "$OUT.sha256" "$NEW/sandbox/pelita875.zip"
echo "READY: $OUT"

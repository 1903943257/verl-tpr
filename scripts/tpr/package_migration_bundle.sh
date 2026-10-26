#!/usr/bin/env bash
# Run inside the current Docker container:
# bash /workspace/verl-tpr-e2e/scripts/tpr/package_migration_bundle.sh
set -euo pipefail
ROOT="${ROOT:-/workspace}"
NAME=tpr_migration_bundle_20261026
STAGE="$ROOT/$NAME"
OUT="$ROOT/$NAME.tar"
MODEL="$ROOT/hf_models/Qwen3-4B"
DATA="$ROOT/tpr_experiments/shortfix"
EXTRAS="$ROOT/tpr_experiments/migration_extras_20261026"
ZIP="$ROOT/swe-rebench-swerebench-sweb.eval.x86_64.aspp_1776_pelita-875-latest.zip"
TPR="$ROOT/verl-tpr-e2e"
UNI="$ROOT/uni-agent"

for f in "$MODEL/config.json" "$MODEL/model.safetensors.index.json" \
 "$DATA/shortfix_candidates_task.parquet" "$DATA/pelita875_hint_h2.parquet" \
 "$DATA/task_config_claude_code_pelita_docker.yaml" \
 "$EXTRAS/model_128k/config.json" "$EXTRAS/model_weight_sha256.txt" "$ZIP"; do
  [[ -f "$f" ]] || { echo "ERROR missing: $f" >&2; exit 2; }
done
shopt -s nullglob
shards=("$MODEL"/model-*.safetensors)
if [[ "${#shards[@]}" -ne 3 ]]; then
  echo "ERROR: expected 3 model shards, found ${#shards[@]}" >&2; exit 2
fi
if [[ -e "$STAGE" || -e "$OUT" || -e "$OUT.sha256" ]]; then
  echo "ERROR: output exists; move it aside before rebuilding: $STAGE / $OUT" >&2; exit 2
fi

estimated=$(du -sb "$MODEL" "$ZIP" "$EXTRAS" "$DATA" | awk '{s+=$1} END {print s+0}')
free=$(df -PB1 "$ROOT" | awk 'NR==2 {print $4}')
if (( free < estimated + estimated / 10 )); then
  echo "ERROR: need about $((estimated/1024/1024/1024)) GiB plus 10% free space." >&2
  echo "No files were deleted. Free space safely or package assets separately." >&2; exit 2
fi

mkdir -p "$STAGE"/{data/shortfix,models/Qwen3-4B,models/Qwen3-4B-128K-config,sandbox,migration_extras,docs,tools}
link_or_copy() { ln "$1" "$2" 2>/dev/null || cp -p "$1" "$2"; }
while IFS= read -r -d '' f; do link_or_copy "$f" "$STAGE/models/Qwen3-4B/$(basename "$f")"; done \
  < <(find "$MODEL" -maxdepth 1 -type f -print0)
cp -a "$EXTRAS/model_128k/." "$STAGE/models/Qwen3-4B-128K-config/"
cp -p "$DATA/shortfix_candidates_task.parquet" "$DATA/pelita875_hint_h2.parquet" \
  "$DATA/task_config_claude_code_pelita_docker.yaml" "$STAGE/data/shortfix/"
cp -a "$EXTRAS/." "$STAGE/migration_extras/"
link_or_copy "$ZIP" "$STAGE/sandbox/pelita875.zip"
[[ ! -f "$TPR/docs/tpr/GETTING_STARTED_UNIAGENT_NEW.md" ]] || \
  cp -p "$TPR/docs/tpr/GETTING_STARTED_UNIAGENT_NEW.md" "$STAGE/docs/"
[[ ! -f "$TPR/docs/tpr/MULTI_NODE_MIGRATION_BUNDLE_20261026.md" ]] || \
  cp -p "$TPR/docs/tpr/MULTI_NODE_MIGRATION_BUNDLE_20261026.md" "$STAGE/docs/"
[[ ! -f "$TPR/tests/models/mcore/tpr/integration/REBENCH_E2E.md" ]] || \
  cp -p "$TPR/tests/models/mcore/tpr/integration/REBENCH_E2E.md" "$STAGE/docs/"

cat > "$STAGE/tools/import_pelita_image.sh" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd -- "$(dirname -- "$0")" && pwd -P)"
ZIP="$HERE/../sandbox/pelita875.zip"
TAG=swerebench/sweb.eval.x86_64.aspp_1776_pelita-875:latest
EXPECTED=sha256:933103ac2f63e4d3ffcc8ffe769dfa38c7d4e85efc6ccfc3dcdba0b93617eaa6
set -o pipefail
python - "$ZIP" <<'PY' | docker load
import hashlib, shutil, sys, zipfile
p=sys.argv[1]
with zipfile.ZipFile(p) as z:
    expected=z.read("image.tar.gz.sha256").decode().split()[0]
    h=hashlib.sha256()
    with z.open("image.tar.gz") as f:
        for b in iter(lambda:f.read(8*1024*1024),b""): h.update(b)
    if h.hexdigest()!=expected:
        print(f"SHA256 mismatch: {h.hexdigest()} != {expected}",file=sys.stderr); raise SystemExit(2)
    print("Verified image.tar.gz SHA256:",h.hexdigest(),file=sys.stderr)
    with z.open("image.tar.gz") as f: shutil.copyfileobj(f,sys.stdout.buffer,8*1024*1024)
PY
id="$(docker image inspect "$TAG" --format '{{.Id}}')"
echo "$TAG -> $id"
[[ "$id" == "$EXPECTED" ]] || { echo "ERROR: unexpected image ID" >&2; exit 3; }
echo "PASS: pelita-875 image ID matches migration record."
SH
chmod +x "$STAGE/tools/import_pelita_image.sh"

cat > "$STAGE/tools/materialize_qwen3_4b_128k.sh" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd -- "$(dirname -- "$0")" && pwd -P)"
BUNDLE="$(cd -- "$HERE/.." && pwd -P)"
BASE="${MODEL_BASE:-/workspace/hf_models/Qwen3-4B}"
TARGET="${MODEL_128K_TARGET:-/workspace/hf_models/Qwen3-4B-128K}"
CFG="$BUNDLE/models/Qwen3-4B-128K-config"
[[ -d "$BASE" && -f "$CFG/config.json" ]] || { echo "Missing base model or 128K config" >&2; exit 2; }
[[ ! -e "$TARGET" ]] || { echo "Refusing to overwrite existing $TARGET" >&2; exit 2; }
mkdir -p "$(dirname "$TARGET")"
cp -al "$BASE" "$TARGET"
for f in "$CFG"/*; do
  [[ -f "$f" ]] || continue
  n="$(basename "$f")"
  rm -f "$TARGET/$n"
  cp -p "$f" "$TARGET/$n"
done
python - "$TARGET/config.json" <<'PY'
import json,sys
c=json.load(open(sys.argv[1]))
print("max_position_embeddings:",c.get("max_position_embeddings"))
print("rope_scaling:",c.get("rope_scaling"))
assert c.get("max_position_embeddings")==131072
assert (c.get("rope_scaling") or {}).get("rope_type")=="yarn"
print("PASS: 128K YaRN config")
PY
SH
chmod +x "$STAGE/tools/materialize_qwen3_4b_128k.sh"

{
  echo "Created: $(date -Is)"
  echo "Source host: $(hostname)"
  echo "TPR repo: $TPR | commit=$(git -C "$TPR" rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "UniAgent repo: $UNI | commit=$(git -C "$UNI" rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "Qwen3-4B and Qwen3-4B-128K: all 3 safetensors shards SHA256-matched on source."
  echo "Pelita image ID: sha256:933103ac2f63e4d3ffcc8ffe769dfa38c7d4e85efc6ccfc3dcdba0b93617eaa6"
  echo "Django image and training checkpoints intentionally excluded."
} > "$STAGE/SOURCE_INFO.txt"

cat > "$STAGE/README.md" <<'MD'
# TPR / UniAgent migration bundle — 2026-10-26

Contents:
- models/Qwen3-4B/: complete model shards, tokenizer and base config.
- models/Qwen3-4B-128K-config/: separate 131072-token YaRN config/tokenizer metadata; no duplicate weight shards.
- sandbox/pelita875.zip: source ZIP with image.tar.gz and its SHA256 sidecar.
- data/shortfix/: primary task Parquet files and actual pelita Docker task config.
- migration_extras/: supplementary Parquet/CSV/JSON, old logs, old Hard Cap source backup and weight SHA256 manifest.
- docs/, SOURCE_INFO.txt, SHA256SUMS and helper scripts under tools/.

The 36.7 GB Django image, training checkpoints and duplicate 128K weights are excluded.

Verify after extracting the outer TAR:
    sha256sum -c SHA256SUMS

Load pelita-875 on each host whose Docker daemon launches SWE sandbox containers:
    bash tools/import_pelita_image.sh
The script verifies the embedded image.tar.gz SHA256, streams it to docker load, and checks the expected image ID.

Extract the outer bundle under /workspace so this directory is /workspace/tpr_migration_bundle_20261026. Then install the primary task files and model:
    mkdir -p /workspace/tpr_experiments/shortfix /workspace/hf_models
    cp -an data/shortfix/. /workspace/tpr_experiments/shortfix/
    cp -an models/Qwen3-4B/. /workspace/hf_models/Qwen3-4B/

If the target model already exists, compare hashes before replacing anything.
To create the 128K YaRN model directory while sharing identical weight inodes:
    bash tools/materialize_qwen3_4b_128k.sh
This refuses to overwrite an existing target and requires base and target on the same filesystem.

This bundle contains migration assets, not the base Docker image or source repositories. Install/checkout the matching runtime separately on each node.
MD

{
  find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum
} > "$STAGE/SHA256SUMS"

echo "Creating uncompressed TAR (model shards and sandbox image are already compressed)..."
tar -cf "$OUT.tmp" -C "$ROOT" "$NAME"
mv "$OUT.tmp" "$OUT"
(cd "$ROOT" && sha256sum "$(basename "$OUT")" > "$(basename "$OUT").sha256")
echo
ls -lh "$OUT" "$OUT.sha256"
echo "Bundle directory: $STAGE"
echo "On target: tar -xf $(basename "$OUT") -C /workspace"
echo "Then: cd /workspace/$NAME && sha256sum -c SHA256SUMS"

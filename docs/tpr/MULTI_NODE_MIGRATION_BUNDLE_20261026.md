# Multi-node migration bundle: 2026-10-26

This runbook packages the assets already migrated to the first new host into one portable archive for the other three hosts.

## Source facts

- New host workspace bind mount: host /ssd_3/jyb/project/uniagent_linear -> Docker /workspace.
- Current runtime checkout: /workspace/verl-tpr-e2e; UniAgent checkout: /workspace/uni-agent.
- Model: all three Qwen3-4B safetensors shards were verified SHA256-identical to the old Qwen3-4B-128K shards. Keep the 128K YaRN config separate; do not overwrite the base model config.
- Sandbox image: swerebench/sweb.eval.x86_64.aspp_1776_pelita-875:latest, expected ID sha256:933103ac2f63e4d3ffcc8ffe769dfa38c7d4e85efc6ccfc3dcdba0b93617eaa6.
- Excluded intentionally: swebench/sweb.eval.x86_64.django_1776_django-11163:latest (36.7 GB), model duplicate shards, and training checkpoints.

## 1. Build the bundle on the first new host

Pull this branch in the existing container, then run:

```bash
cd /workspace/verl-tpr-e2e
git pull --ff-only
bash scripts/tpr/package_migration_bundle.sh
```

The script checks required source files and free disk space. It hard-links large local files into staging where possible, so staging does not make a second full copy of model weights. The output is:

- /workspace/tpr_migration_bundle_20261026/ — inspectable staging directory
- /workspace/tpr_migration_bundle_20261026.tar — one transferable bundle
- /workspace/tpr_migration_bundle_20261026.tar.sha256 — outer archive checksum

The archive includes the model weights, separate 128K config files, primary Parquet/YAML, supplementary migration data/logs/Hard Cap backup, and pelita ZIP. It does not include source repositories or the base UniAgent Docker image; those are provisioned separately.

## 2. Transfer the single archive to Windows

The host directory /ssd_3/jyb/project/uniagent_linear maps to Docker /workspace. On the new host, the archive is therefore also at:

```text
/ssd_3/jyb/project/uniagent_linear/tpr_migration_bundle_20261026.tar
/ssd_3/jyb/project/uniagent_linear/tpr_migration_bundle_20261026.tar.sha256
```

Use Windows Git Bash with the SSH route that already worked for the first new host. Download the archive and checksum, then upload the same pair to each of the other three hosts:

```bash
mkdir -p "$HOME/Downloads/tpr-transfer"
scp FIRST_HOST_SSH_ALIAS:/ssd_3/jyb/project/uniagent_linear/tpr_migration_bundle_20261026.tar "$HOME/Downloads/tpr-transfer/"
scp FIRST_HOST_SSH_ALIAS:/ssd_3/jyb/project/uniagent_linear/tpr_migration_bundle_20261026.tar.sha256 "$HOME/Downloads/tpr-transfer/"
```

Replace FIRST_HOST_SSH_ALIAS with the actual connection name. Keep the original archive until all destination hosts have passed checksums.

## 3. Install on each destination host

Place the archive and checksum in the destination host's bind-mounted workspace, then in its Docker container run:

```bash
cd /workspace
sha256sum -c tpr_migration_bundle_20261026.tar.sha256
tar -xf tpr_migration_bundle_20261026.tar -C /workspace
cd /workspace/tpr_migration_bundle_20261026
sha256sum -c SHA256SUMS
```

Then import the sandbox image into that host's Docker daemon:

```bash
bash tools/import_pelita_image.sh
```

This helper verifies the inner image.tar.gz SHA256 from the ZIP before streaming it to docker load, then verifies the exact image ID. Run it on every host whose Docker daemon will create SWE sandbox containers.

Copy task data and model files only after checking whether the destination already has them:

```bash
mkdir -p /workspace/tpr_experiments/shortfix /workspace/hf_models
cp -an data/shortfix/. /workspace/tpr_experiments/shortfix/
cp -an models/Qwen3-4B/. /workspace/hf_models/Qwen3-4B/
```

cp -an avoids overwriting existing files. Compare SHA256 values if a destination already has Qwen3-4B.

To create /workspace/hf_models/Qwen3-4B-128K with the same weight inodes and the separate YaRN config, run:

```bash
bash tools/materialize_qwen3_4b_128k.sh
```

The helper refuses to overwrite an existing target. It requires the base model and target directory to be on the same filesystem.

## 4. Resource planning

The model shards are approximately 8.1 GB and the pelita ZIP approximately 1.3 GB, so the outer archive is expected to be around 9–10 GB. Confirm adequate host disk space on each node. Do not extract the model separately if it already exists and matches the hashes.

This bundle contains migrated assets and provenance, not the runtime image or source repositories. The runtime checkout and UniAgent/VERL versions must be aligned separately on all nodes before distributed E2E.

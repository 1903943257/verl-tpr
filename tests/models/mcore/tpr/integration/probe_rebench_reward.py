#!/usr/bin/env python3
"""CPU/Docker-only SWE-reBench Reward probe: no Ray, NPU, or Agent.

Runs the installed uni_agent.tasks.swe_rebench.reward.compute_reward against
a fresh container from the image recorded in one normalized dataset row.
Use for timing diagnosis, not for verifying an Agent's submitted patch.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from types import SimpleNamespace


def cli(args, *, timeout=60, check=True):
    return subprocess.run(args, capture_output=True, text=True, timeout=timeout, check=check)


class DockerSandbox:
    def __init__(self, container, output_dir):
        self.container = container
        self.output_dir = output_dir

    async def write_file(self, path, content):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False) as f:
            f.write(content)
            tmp = Path(f.name)
        try:
            await asyncio.to_thread(
                cli, ["docker", "cp", str(tmp), f"{self.container}:{path}"]
            )
            (self.output_dir / "generated_eval_script.sh").write_text(content)
        finally:
            tmp.unlink(missing_ok=True)

    async def exec_shell(self, command, *, workdir, timeout):
        def run():
            result = cli(
                ["docker", "exec", "-w", workdir, self.container, "bash", "-c", command],
                timeout=timeout + 30, check=False,
            )
            output = result.stdout + result.stderr
            (self.output_dir / "eval_raw.log").write_text(output)
            return SimpleNamespace(stdout=output, exit_code=result.returncode)
        return await asyncio.to_thread(run)


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--row", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=1200.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    out = Path(args.output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(out / "probe.log")],
        force=True,
    )
    logger = logging.getLogger("reward_probe")

    import pandas as pd
    from uni_agent.tasks.swe_rebench import reward

    df = pd.read_parquet(args.data_path)
    if not 0 <= args.row < len(df):
        raise SystemExit(f"invalid row={args.row} for {len(df)} rows")
    extra = df.iloc[args.row]["extra_info"]
    if isinstance(extra, str):
        extra = json.loads(extra)
    task = extra["tools_kwargs"]["task"]
    metadata = task["metadata"]
    image = task["sandbox"]["image"]
    required = ["instance_id", "repo", "test_cmd", "base_commit", "test_patch"]
    missing = [key for key in required if key not in metadata]
    if missing:
        raise SystemExit(f"missing metadata keys {missing}; available={list(metadata)}")
    logger.info("instance=%s image=%s", metadata["instance_id"], image)
    logger.info("test_cmd=%s", str(metadata["test_cmd"])[:300])
    logger.info("install=%s", str(metadata.get("install", ""))[:500])
    logger.info("eval_commands=%s", str(metadata.get("eval_commands", ""))[:500])
    logger.info("reward_module=%s", reward.__file__)

    if not hasattr(reward, "_REWARD_PHASE_FUNCTIONS"):
        raise SystemExit("Actual reward.py has no phase timing; apply reward_phase_timing.patch first")

    specs = {
        "test_cmd": metadata["test_cmd"],
        "install": metadata.get("install", ""),
        "eval_commands": metadata.get("eval_commands", ""),
    }
    commands = reward._make_eval_script_list(
        metadata, specs, "testbed", "/testbed",
        metadata["base_commit"], metadata["test_patch"], instrument=True,
    )
    script = "\n".join([
        "#!/bin/bash", "set -uxo pipefail", reward._REWARD_PHASE_FUNCTIONS, *commands
    ]) + "\n"
    (out / "generated_eval_script.sh").write_text(script)
    logger.info("generated script at %s", out / "generated_eval_script.sh")

    if args.dry_run:
        logger.info("DRY_RUN: no Docker container started")
        return

    inspected = cli(["docker", "image", "inspect", image], check=False)
    if inspected.returncode:
        raise SystemExit(
            f"Docker image not local: {image}\n{inspected.stderr.strip()}\n"
            "No download was attempted."
        )

    name = "rebench-reward-probe-" + uuid.uuid4().hex[:12]
    logger.info("starting temporary Docker container: %s", name)
    cid = None
    try:
        created = cli(
            ["docker", "run", "-d", "--rm", "--name", name,
             "--entrypoint", "/bin/sh", image, "-c", f"sleep {int(args.timeout + 180)}"],
            timeout=120,
        )
        cid = created.stdout.strip()
        logger.info("container_id=%s", cid)
        start = time.perf_counter()
        result = await reward.compute_reward(
            metadata, DockerSandbox(name, out), eval_timeout=args.timeout
        )
        logger.info("reward_total_call_s=%.3f", time.perf_counter() - start)
        logger.info(
            "resolved=%s eval_execution_time=%s",
            result["resolved"], result["eval_execution_time"],
        )
        (out / "result.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False, default=str)
        )
    finally:
        if cid:
            cli(["docker", "rm", "-f", name], timeout=60, check=False)
            logger.info("removed temporary Docker container: %s", name)
    logger.info("OUTPUT_DIR=%s", out)


if __name__ == "__main__":
    asyncio.run(main())

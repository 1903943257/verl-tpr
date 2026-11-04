# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Optional source provenance for experiments mounted into containers."""

import os
import subprocess
from pathlib import Path


def source_metadata(repo_dir=None):
    """Git metadata must never invalidate a completed numerical experiment.

    A host can pass TPR_SOURCE_HEAD when a mounted worktree's .git pointer
    cannot resolve inside the container. Such a value is caller-supplied,
    not a revision verified against the mounted source.
    """
    supplied = os.getenv("TPR_SOURCE_HEAD", "").strip()
    if supplied:
        return {"source_head": supplied, "source_head_origin": "environment", "source_head_error": None}
    directory = Path(repo_dir) if repo_dir is not None else Path(__file__).resolve().parents[5]
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=directory,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=3,
            check=False,
        )
        revision = result.stdout.strip()
        if result.returncode == 0 and revision:
            return {"source_head": revision, "source_head_origin": "git", "source_head_error": None}
        reason = result.stderr.strip() or f"git returned {result.returncode} with no HEAD"
    except (OSError, subprocess.SubprocessError) as error:
        reason = f"{type(error).__name__}: {error}"
    return {"source_head": None, "source_head_origin": "unavailable", "source_head_error": reason[:1024]}

"""Pure-stdlib test-suite registry guard (does not import NPU or Megatron)."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


TPR_DIR = Path(__file__).resolve().parents[1]
SCRIPT = TPR_DIR / "run_suite.py"


def _call(*args: str):
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        text=True, capture_output=True, check=False,
    )


def test_tpr_suite_registry_has_no_missing_paths():
    result = _call("check")
    assert result.returncode == 0, result.stderr
    assert "PASS:" in result.stdout


def test_tpr_suite_listing_and_inventory_do_not_import_megatron():
    suites = _call("list")
    inv = _call("inventory")
    assert suites.returncode == 0, suites.stderr
    assert "npu-recompute-negative" in suites.stdout
    assert inv.returncode == 0, inv.stderr
    assert "total pytest files:" in inv.stdout


def test_tpr_suite_show_keeps_native_recompute_as_negative_control():
    negative = _call("show", "npu-recompute-negative")
    assert negative.returncode == 0
    assert "EXPECTED FAILURE" in negative.stdout
    positive = _call("show", "npu-recompute-primitive")
    assert positive.returncode == 0
    assert "TPR_RUN_KV_CHECKPOINT_REPLAY=1" in positive.stdout


def test_tpr_suite_never_runs_npu_without_opt_in():
    result = _call("run", "npu-ring")
    assert result.returncode == 2
    assert "--ack-npu" in result.stderr

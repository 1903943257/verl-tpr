"""Entry point to keep pytest CLI flags out of MindSpeed's import-time parser.

Usage: torchrun --nproc_per_node=8 .../run_cp4_pytest_argv_safe.py
Only the CP4 recorded-actor replay test is selected. A separate pytest arg
vector is supplied to pytest.main; sys.argv stays clean for MindSpeed imports.
"""
import sys

sys.argv[:] = [sys.argv[0]]

import pytest

if __name__ == "__main__":
    raise SystemExit(
        pytest.main([
            "-vv",
            "-s",
            "-x",
            "tests/models/mcore/tpr/parallel/test_real_actor_tp2cp4_offline_ppo_npu.py",
        ])
    )

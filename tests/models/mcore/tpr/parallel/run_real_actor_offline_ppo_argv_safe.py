"""Pytest entry point for the fixed-capture real-Actor offline PPO replay.

Keep torchrun/MindSpeed argument parsing isolated from pytest arguments.
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

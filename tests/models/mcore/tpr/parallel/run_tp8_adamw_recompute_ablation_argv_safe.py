"""Safe torchrun entrypoint for the TP8 AdamW recompute-ablation replay."""
import sys
sys.argv[:] = [sys.argv[0]]
import pytest

if __name__ == "__main__":
    raise SystemExit(pytest.main([
        "-vv", "-s", "-x",
        "tests/models/mcore/tpr/parallel/test_real_actor_tp8_adamw_recompute_ablation_npu.py",
    ]))

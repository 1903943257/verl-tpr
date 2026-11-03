"""Isolated pytest entrypoint for the CP4 four-way replay runner.

MindSpeed parses the global process sys.argv when its adaptor is imported.
Passing pytest CLI flags such as -q/-s/-x directly through torchrun can make
MindSpeed synthesize an invalid empty-name TransformerConfig field. Instead,
keep the process argv clean and supply explicit arguments to pytest.main().
This entrypoint performs no repository or Git operations.
"""

from __future__ import annotations

import os
import sys


def main() -> int:
    # The torchrun worker receives only the entrypoint path. Sanitize even if
    # a launcher injects additional flags before any pytest/plugin imports.
    sys.argv[:] = [sys.argv[0]]
    sys.path.insert(0, os.getcwd())

    phase = os.environ.get("TPR_FOURWAY_PYTEST_PHASE")
    if phase == "preflight":
        pytest_args = [
            "-q",
            "tests/models/mcore/tpr/parallel/test_native_full_dynamic_microbatches_cpu.py",
            "tests/models/mcore/tpr/parallel/test_multistep_artifact_recovery_cpu.py",
            "tests/models/mcore/tpr/parallel/test_cp4_multistep_replica_audit_cpu.py",
        ]
    elif phase == "multistep":
        pytest_args = [
            "-vv", "-s", "-x",
            "tests/models/mcore/tpr/parallel/test_real_actor_tp2cp4_multistep_npu.py",
        ]
    else:
        raise ValueError(
            "TPR_FOURWAY_PYTEST_PHASE must be preflight or multistep"
        )

    import pytest

    return pytest.main(pytest_args)


if __name__ == "__main__":
    raise SystemExit(main())

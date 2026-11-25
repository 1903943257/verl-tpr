"""Shared NPU runtime for the migrated GDN tests."""

from __future__ import annotations

import pytest


@pytest.fixture(scope="session")
def runtime():
    from ._gdn_npu_test_utils import destroy_npu_runtime, initialize_npu_runtime

    value = initialize_npu_runtime(world_size=1)
    yield value
    destroy_npu_runtime(value)

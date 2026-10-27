"""Initialize MindSpeed before collecting NPU/Megatron pytest modules.

Keep pytest short flags out of MindSpeed's argument parser: an unknown -q/-s
can be turned into an invalid empty TransformerConfig dataclass field.
"""

import sys


def main() -> int:
    pytest_args = sys.argv[1:]
    sys.argv[:] = [sys.argv[0]]

    import mindspeed.megatron_adaptor  # noqa: F401
    import pytest

    print("MINDSPEED_FIRST_BOOTSTRAP_OK", flush=True)
    return int(pytest.main(pytest_args))


if __name__ == "__main__":
    raise SystemExit(main())

"""Start isolated pytest/NPU replay with the same MindSpeed-first import order as online E2E.

MBridge imports get_gpt_decoder_block_spec by value. If that happens before
MindSpeed's import-time feature patches, Megatron caches HAVE_TE=False and
never defines TENorm, causing NameError at GPT construction. The original
online E2E initializes MindSpeed before importing MBridge.

torchrun ... _mindspeed_first_pytest.py -vv -s -x test_file.py
"""
from __future__ import annotations

import sys


def main() -> int:
    # torchrun passes pytest flags (e.g. -vv -s -x test_file.py) as this
    # script's argv. MindSpeed parses argv during import/repatch and can turn
    # pytest options into malformed dynamic TransformerConfig dataclass fields.
    # Preserve pytest's arguments separately and keep argv clean for every
    # MindSpeed/Megatron/MBridge import in this bootstrap.
    pytest_args = sys.argv[1:]
    sys.argv[:] = [sys.argv[0]]

    # Fail explicitly if a test bootstrap or eager plugin already imported
    # GPT specs. An in-process "repatch" is not equivalent to a fresh import.
    polluted = [
        name for name in (
            "mbridge.core.llm_bridge",
            "megatron.core.models.gpt.gpt_layer_specs",
        ) if name in sys.modules
    ]
    if polluted:
        raise RuntimeError(
            "MindSpeed bootstrap must run BEFORE MBridge/GPT layer-spec import; "
            f"already imported: {polluted}"
        )

    # This has NPU-specific import-time feature registrations and must be
    # strictly earlier than pytest and the VERL replay test modules.
    import mindspeed.megatron_adaptor  # noqa: F401
    from megatron.core.models.gpt import gpt_layer_specs

    if not getattr(gpt_layer_specs, "HAVE_TE", False) or not hasattr(
        gpt_layer_specs, "TENorm"
    ):
        raise RuntimeError(
            "MindSpeed-first import did not register expected Megatron GPT "
            "Spec compatibility. Refuse to build with missing TENorm. "
            "Check installed MindSpeed/Megatron versions and patch state."
        )

    from mbridge.core import llm_bridge
    for name in ("get_gpt_decoder_block_spec", "get_gpt_decoder_layer_specs"):
        mbridge_spec = getattr(llm_bridge, name, None)
        megatron_spec = getattr(gpt_layer_specs, name, None)
        if mbridge_spec is not None and mbridge_spec is not megatron_spec:
            raise RuntimeError(
                f"MindSpeed-first bootstrap has stale MBridge {name} binding"
            )
    print(
        "REAL_ENGINE_MINDSPEED_FIRST_PASS "
        f"have_te={gpt_layer_specs.HAVE_TE} "
        f"tenorm_defined={hasattr(gpt_layer_specs, 'TENorm')} "
        f"mbridge_spec_same=True",
        flush=True,
    )

    # Lightweight executable check: verify the exact worker bootstrap without
    # initializing NPU distributed or collecting pytest plugins.
    if pytest_args == ["--bootstrap-only"]:
        return 0

    import pytest

    return int(pytest.main(pytest_args))


if __name__ == "__main__":
    raise SystemExit(main())

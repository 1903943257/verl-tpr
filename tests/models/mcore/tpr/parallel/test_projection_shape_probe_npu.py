# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Opt-in real Qwen first-projection capture and matched 64-vs-32-row probe."""

import hashlib
import json
import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    os.getenv("TPR_RUN_PROJECTION_PROBE") != "1", reason="Opt in to one-NPU projection probe"
)


def test_real_qwen_projection_shape_probe():
    import torch
    import torch.distributed as dist
    import torch_npu
    from megatron.core import parallel_state

    from ._qwen3_tp_checkpoint import make_real_qwen3_tp_model
    from .experiment_source import source_metadata
    from .projection_shape_probe import capture_projection_sample, run_projection_probe
    from .test_split_p32s32_cp2_parity_npu import _tokens
    from .test_tpr_10k_parallel_parity_npu import _runtime

    output = Path(os.environ["TPR_PROJECTION_OUTPUT"])
    if output.exists():
        raise FileExistsError(output)
    source = source_metadata()
    runtime = _runtime(1, 1, 1)
    try:
        torch.set_float32_matmul_precision("highest")
        if not hasattr(torch_npu.npu.matmul, "allow_hf32"):
            raise RuntimeError("Cannot verify FP32 matmul precision")
        torch_npu.npu.matmul.allow_hf32 = False
        if torch_npu.npu.matmul.allow_hf32:
            raise RuntimeError("HF32 remained enabled")
        aclnn = getattr(torch_npu.npu, "aclnn", None)
        if aclnn is not None and hasattr(aclnn, "allow_hf32"):
            aclnn.allow_hf32 = False
        model, _ = make_real_qwen3_tp_model(
            runtime,
            model_path=Path(os.environ["TPR_QWEN_4B_PATH"]),
            tp_size=1,
            cp_size=1,
            profile=os.getenv("TPR_P32S32_PROFILE", "4b"),
            params_dtype=torch.bfloat16,
        )
        if model.config.gradient_accumulation_fusion or model.config.recompute_granularity is not None:
            raise AssertionError("Probe requires direct parameter.grad and no recomputation")
        model.config.grad_scale_func = None
        tokens = _tokens()
        with torch.autocast(device_type="npu", enabled=False):
            module, sample = capture_projection_sample(model, tokens.to(runtime.device))
            report = run_projection_probe(module, sample)
        report["environment"] = {
            **source,
            "checkpoint": os.environ["TPR_QWEN_4B_PATH"],
            "profile": os.getenv("TPR_P32S32_PROFILE", "4b"),
            "capture_loss": sample["loss"],
            "torch": torch.__version__,
            "torch_npu": torch_npu.__version__,
            "matmul_allow_hf32": bool(torch_npu.npu.matmul.allow_hf32),
            "module_class": f"{type(module).__module__}.{type(module).__qualname__}",
            "input_sha256": hashlib.sha256(tokens.numpy().tobytes()).hexdigest(),
            "token_ids": tokens.tolist(),
            "seed": 20261104,
        }
        output.mkdir(parents=True, exist_ok=False)
        (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        for mode, result in report["modes"].items():
            metrics = result["full_vs_split_connected"]
            print(
                f"PROJECTION_SHAPE mode={mode} 64_vs_32x2: "
                + " ".join(
                    f"{name}_rel_l2={metrics[name]['relative_l2']}"
                    for name in ("output", "input_gradient", "weight_gradient")
                )
                + f" sequential_dW_rel_l2={result['full_vs_split_sequential']['weight_gradient']['relative_l2']}"
                + f" repeat_output_max_abs={result['full_vs_repeat_full']['output']['max_abs']}",
                flush=True,
            )
        print(f"PROJECTION_SHAPE_DONE report={output / 'report.json'}", flush=True)
    finally:
        if parallel_state.model_parallel_is_initialized():
            parallel_state.destroy_model_parallel()
        if dist.is_initialized():
            dist.destroy_process_group()

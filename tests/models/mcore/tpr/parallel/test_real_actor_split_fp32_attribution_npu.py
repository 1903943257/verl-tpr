# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Single synthetic trajectory, actual Qwen weights, FP32 three-way attribution."""

import json
import os
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    os.getenv("TPR_RUN_SPLIT_FP32_ATTRIBUTION") != "1",
    reason="Explicitly opt into one-NPU FP32 Full/connected/replay attribution",
)


def test_real_qwen_split_fp32_attribution():
    import torch
    import torch.distributed as dist
    import torch_npu
    from megatron.core import parallel_state

    from ._qwen3_tp_checkpoint import make_real_qwen3_tp_model
    from .split_fp32_attribution import run_three_way
    from .test_tpr_10k_parallel_parity_npu import _runtime

    output = Path(os.environ["TPR_SPLIT_FP32_OUTPUT"])
    checkpoint = Path(os.environ["TPR_QWEN_4B_PATH"])
    prefix = int(os.getenv("TPR_SPLIT_FP32_PREFIX", "32"))
    suffix = int(os.getenv("TPR_SPLIT_FP32_SUFFIX", "32"))
    profile = os.getenv("TPR_SPLIT_FP32_PROFILE", "4b")
    if prefix < 1 or suffix < 2 or prefix + suffix > 512:
        raise ValueError("Short diagnostic requires P>=1, S>=2, total<=512")
    if output.exists():
        raise FileExistsError(output)
    if int(os.getenv("WORLD_SIZE", "1")) != 1:
        raise ValueError("Requires exactly TP1/CP1/DP1")
    runtime = _runtime(1, 1, 1)
    try:
        torch.set_float32_matmul_precision("highest")
        matmul = getattr(torch_npu.npu, "matmul", None)
        if matmul is None or not hasattr(matmul, "allow_hf32"):
            raise RuntimeError("Cannot verify FP32 matmul: this torch_npu lacks allow_hf32")
        matmul.allow_hf32 = False
        precision = {"matmul_allow_hf32": bool(matmul.allow_hf32)}
        aclnn = getattr(torch_npu.npu, "aclnn", None)
        if aclnn is not None and hasattr(aclnn, "allow_hf32"):
            aclnn.allow_hf32 = False
            precision["aclnn_allow_hf32"] = bool(aclnn.allow_hf32)
        if any(precision.values()):
            raise RuntimeError(f"HF32 remained enabled: {precision}")
        model, hf = make_real_qwen3_tp_model(
            runtime,
            model_path=checkpoint,
            tp_size=1,
            cp_size=1,
            sequence_parallel=False,
            profile=profile,
            native_attention=False,
            params_dtype=torch.float32,
        )
        config = model.config
        if config.bf16 or config.fp16 or config.recompute_granularity is not None:
            raise AssertionError("Expected actual FP32 model with recompute disabled")
        for name in (
            "gradient_accumulation_fusion",
            "bias_activation_fusion",
            "bias_dropout_fusion",
            "masked_softmax_fusion",
            "persist_layer_norm",
            "memory_efficient_layer_norm",
        ):
            if getattr(config, name, False):
                raise AssertionError(f"Diagnostic requires {name}=False")
        config.grad_scale_func = None
        # CPU RNG makes the synthetic input independent of NPU execution order.
        generator = torch.Generator(device="cpu").manual_seed(20261104)
        tokens = torch.randint(256, hf.vocab_size - 256, (prefix + suffix,), generator=generator)
        tokens = tokens.to(runtime.device)
        autocast = torch.autocast
        with autocast(device_type="npu", enabled=False):
            report = run_three_way(
                model,
                tokens,
                prefix,
                output,
                production_attention=True,
                atol=float(os.getenv("TPR_SPLIT_FP32_ATOL", "1e-5")),
                rtol=float(os.getenv("TPR_SPLIT_FP32_RTOL", "1e-4")),
            )
        torch.npu.synchronize()
        report["environment"] = {
            "torch": torch.__version__,
            "torch_npu": torch_npu.__version__,
            "checkpoint": str(checkpoint),
            "profile": profile,
            "layers": len(model.decoder.layers),
            "topology": {"tp": 1, "cp": 1, "dp": 1},
            "precision": precision,
            "autocast": False,
            "optimizer_step": False,
            "synthetic_input": True,
            "same_model_no_weight_update": True,
            "source_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        }
        (output / "report.json").write_text(json.dumps(report, indent=2))
        for name, pair in report["pairs"].items():
            print(
                f"SPLIT_FP32 {name}: grad_rel_l2={pair['gradient_relative_l2']:.9g} "
                f"cosine={pair['gradient_cosine']} max_abs={pair['gradient_max_abs']:.9g} "
                f"loss_gap={pair['loss_abs_gap']:.9g}",
                flush=True,
            )
        print("SPLIT_FP32_FIRST_EXCEEDANCE " + json.dumps(report["layer_summary"], sort_keys=True), flush=True)
        print(f"SPLIT_FP32_ATTRIBUTION_DONE report={output / 'report.json'}", flush=True)
    finally:
        if parallel_state.model_parallel_is_initialized():
            parallel_state.destroy_model_parallel()
        if dist.is_initialized():
            dist.destroy_process_group()

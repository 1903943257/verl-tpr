# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Opt-in single-NPU BF16 short three-arm summary; no tensor dumps."""

import json
import os
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    os.getenv("TPR_RUN_SPLIT_BF16_QUICK") != "1",
    reason="Opt in to summary-only BF16 Full/connected/Replay comparison",
)


def test_real_qwen_split_bf16_quick():
    import torch
    import torch.distributed as dist
    import torch_npu
    from megatron.core import parallel_state

    from ._qwen3_tp_checkpoint import make_real_qwen3_tp_model
    from .split_bf16_quick import run_bf16_quick
    from .test_tpr_10k_parallel_parity_npu import _runtime

    output = Path(os.environ["TPR_SPLIT_BF16_OUTPUT"])
    checkpoint = Path(os.environ["TPR_QWEN_4B_PATH"])
    prefix = int(os.getenv("TPR_SPLIT_BF16_PREFIX", "32"))
    suffix = int(os.getenv("TPR_SPLIT_BF16_SUFFIX", "32"))
    profile = os.getenv("TPR_SPLIT_BF16_PROFILE", "4b")
    if prefix < 1 or suffix < 2 or prefix + suffix > 512:
        raise ValueError("Short diagnostic requires P>=1, S>=2, total<=512")
    if output.exists():
        raise FileExistsError(output)
    if int(os.getenv("WORLD_SIZE", "1")) != 1:
        raise ValueError("Requires exactly TP1/CP1/DP1")
    runtime = _runtime(1, 1, 1)
    try:
        model, hf = make_real_qwen3_tp_model(
            runtime,
            model_path=checkpoint,
            tp_size=1,
            cp_size=1,
            sequence_parallel=False,
            profile=profile,
            native_attention=False,
            params_dtype=torch.bfloat16,
        )
        config = model.config
        if not config.bf16 or config.fp16 or config.recompute_granularity is not None:
            raise AssertionError("Expected BF16 model with recompute disabled")
        if config.gradient_accumulation_fusion:
            raise AssertionError("Quick diagnostic collects parameter.grad without a main_grad wrapper")
        config.grad_scale_func = None
        generator = torch.Generator(device="cpu").manual_seed(20261104)
        tokens = torch.randint(256, hf.vocab_size - 256, (prefix + suffix,), generator=generator).to(runtime.device)
        with torch.autocast(device_type="npu", enabled=False):
            report = run_bf16_quick(model, tokens, prefix, output)
        torch.npu.synchronize()
        report["environment"] = {
            "torch": torch.__version__,
            "torch_npu": torch_npu.__version__,
            "checkpoint": str(checkpoint),
            "profile": profile,
            "topology": {"tp": 1, "cp": 1, "dp": 1},
            "seed": 20261104,
            "optimizer_step": False,
            "autocast": False,
            "config": {
                name: getattr(config, name, None)
                for name in (
                    "bf16",
                    "fp16",
                    "gradient_accumulation_fusion",
                    "persist_layer_norm",
                    "memory_efficient_layer_norm",
                    "bias_activation_fusion",
                    "masked_softmax_fusion",
                )
            },
            "source_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        }
        (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        for name, pair in report["pairs"].items():
            print(
                f"SPLIT_BF16 {name}: rel_l2={pair['gradient_relative_l2']:.9g} "
                f"cosine={pair['gradient_cosine']:.12g} max_abs={pair['gradient_max_abs']:.9g} "
                f"loss_gap={pair['loss_abs_gap']:.9g} elements={pair['elements']}",
                flush=True,
            )
        print(f"SPLIT_BF16_QUICK_DONE report={output / 'report.json'}", flush=True)
    finally:
        if parallel_state.model_parallel_is_initialized():
            parallel_state.destroy_model_parallel()
        if dist.is_initialized():
            dist.destroy_process_group()

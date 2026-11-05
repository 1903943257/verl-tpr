# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""CPU route-observer fixtures, never substitutes for an NPU training test."""

import linecache
import sys
import types

import pytest
import torch
from torch import nn

from tests.models.mcore.tpr.parallel.training_execution_witness import ExecutionWitness


def _module(monkeypatch, name, source, **extra):
    filename = "/env/site-packages/" + name.replace(".", "/") + ".py"
    module = types.ModuleType(name)
    module.__file__ = filename
    module.__dict__.update(nn=nn, torch=torch, **extra)
    monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setitem(linecache.cache, filename, (len(source), None, source.splitlines(True), filename))
    exec(compile(source, filename, "exec"), module.__dict__)
    return module


def _fixture(monkeypatch, *, core_path="mindspeed.te.pytorch.module.TE_dot_product_attention", ring=False, fused=True):
    primitive = _module(
        monkeypatch,
        "torch_npu.testing_primitive",
        "def npu_fusion_attention(value):\n    return value * 2\n",
    ).npu_fusion_attention
    ring_entry = _module(
        monkeypatch,
        "mindspeed.core.context_parallel.ring_context_parallel.ring_context_parallel",
        "def ringattn_context_parallel(value):\n    return value\n",
    ).ringattn_context_parallel
    body = "        value = ring_entry(value)\n" if ring else ""
    body += "        return primitive(value)\n" if fused else "        return value * 2\n"
    module = _module(
        monkeypatch,
        core_path,
        "class TEDotProductAttention(nn.Module):\n    def forward(self, value):\n" + body,
        primitive=primitive,
        ring_entry=ring_entry,
    )
    schedule = _module(
        monkeypatch,
        "megatron.core.pipeline_parallel.schedules",
        "def finalize():\n    pass\n\n"
        "def forward_backward_no_pipelining(core, value):\n"
        "    output = core(value)\n    finalize()\n    return output\n",
    )
    return module.TEDotProductAttention(), schedule, primitive


@pytest.mark.parametrize(
    "path",
    [
        "mindspeed.te.pytorch.module.TE_dot_product_attention",
        "mindspeed.te.pytorch.attention.dot_product_attention.dot_product_attention",
        "megatron.core.transformer.dot_product_attention",
    ],
)
def test_native_te_method_named_forward_is_observed_without_changing_results(monkeypatch, path):
    core, schedule, _primitive = _fixture(monkeypatch, core_path=path)
    value = torch.tensor([1.0, -3.0], requires_grad=True)
    previous = sys.getprofile()
    with ExecutionWitness(schedule.finalize, [core]) as witness:
        result = schedule.forward_backward_no_pipelining(core, value)
    witness.validate(tpr=False, cp=1, fusion=True)
    assert witness.counts["native_attention"] == 1
    assert witness.counts["native_fused_attention"] > 0
    assert witness.snapshot()["core_attention"][0]["forward"]["name"].endswith(".forward")
    assert witness.snapshot()["observed_entrypoints"]
    assert sys.getprofile() is previous
    assert not core._forward_pre_hooks and not core._forward_hooks
    torch.testing.assert_close(result, value * 2, rtol=0, atol=0)
    result.sum().backward()
    torch.testing.assert_close(value.grad, torch.full_like(value, 2), rtol=0, atol=0)


def test_replacement_adapter_is_rejected_even_if_it_calls_native_fusion(monkeypatch):
    core, schedule, _primitive = _fixture(
        monkeypatch,
        core_path="tests.models.mcore.tpr.profiling._engine_profile_fixture",
    )
    with ExecutionWitness(schedule.finalize, [core]) as witness:
        schedule.forward_backward_no_pipelining(core, torch.ones(2))
    with pytest.raises(AssertionError, match="registered native core"):
        witness.validate(tpr=False, cp=1, fusion=True)
    assert not witness.snapshot()["core_attention"][0]["native"]


def test_unrelated_fusion_does_not_qualify_an_unfused_core(monkeypatch):
    core, schedule, primitive = _fixture(monkeypatch, fused=False)
    with ExecutionWitness(schedule.finalize, [core]) as witness:
        schedule.forward_backward_no_pipelining(core, torch.ones(2))
        primitive(torch.ones(2))
    assert witness.counts["native_attention"] == 1
    assert not witness.counts["native_fused_attention"]
    with pytest.raises(AssertionError, match="No native fused attention"):
        witness.validate(tpr=False, cp=1, fusion=True)


def test_cp2_still_requires_native_ring_called_inside_the_core(monkeypatch):
    core, schedule, _primitive = _fixture(monkeypatch)
    with ExecutionWitness(schedule.finalize, [core]) as witness:
        schedule.forward_backward_no_pipelining(core, torch.ones(2))
    with pytest.raises(AssertionError, match="did not enter native MindSpeed Ring"):
        witness.validate(tpr=False, cp=2, fusion=True)
    core, schedule, _primitive = _fixture(monkeypatch, ring=True)
    with ExecutionWitness(schedule.finalize, [core]) as witness:
        schedule.forward_backward_no_pipelining(core, torch.ones(2))
    witness.validate(tpr=False, cp=2, fusion=True)
    assert witness.counts["native_ring"] == 1


def test_every_registered_core_must_execute(monkeypatch):
    core, schedule, _primitive = _fixture(monkeypatch)
    missing_core = type(core)()
    with ExecutionWitness(schedule.finalize, [core, missing_core]) as witness:
        schedule.forward_backward_no_pipelining(core, torch.ones(2))
    with pytest.raises(AssertionError, match="Not every registered native core"):
        witness.validate(tpr=False, cp=1, fusion=True)


def test_dispatcher_packet_fusion_is_recognized_without_python_api_wrapper(monkeypatch):
    core, schedule, _primitive = _fixture(monkeypatch)
    # Execute a packet with the Torch dispatcher's source path. Keep the real
    # torch._ops module in sys.modules intact for normal nn.Module execution.
    namespace = dict(__name__="torch._ops")
    exec(
        compile(
            "class Packet:\n    _qualified_op_name = 'npu::npu_fusion_attention'\n"
            "    def __call__(self, value):\n        return value * 2\n",
            "/env/site-packages/torch/_ops.py",
            "exec",
        ),
        namespace,
    )
    sys.modules[type(core).__module__].primitive = namespace["Packet"]()
    with ExecutionWitness(schedule.finalize, [core]) as witness:
        schedule.forward_backward_no_pipelining(core, torch.ones(2))
    witness.validate(tpr=False, cp=1, fusion=True)
    assert witness.counts["native_fused_attention"] == 1


def test_failure_restores_profile_and_preserves_existing_model_hooks(monkeypatch):
    core, schedule, _primitive = _fixture(monkeypatch)
    seen = []
    existing = core.register_forward_pre_hook(lambda *_args: seen.append(True))
    sys.modules[type(core).__module__].primitive = lambda _value: (_ for _ in ()).throw(ValueError("backend failed"))
    previous = sys.getprofile()
    with pytest.raises(ValueError, match="backend failed"):
        with ExecutionWitness(schedule.finalize, [core]) as witness:
            schedule.forward_backward_no_pipelining(core, torch.ones(2))
    assert sys.getprofile() is previous
    assert witness.active_native_cores == 0
    assert len(core._forward_pre_hooks) == 1 and not core._forward_hooks
    assert seen == [True]
    existing.remove()

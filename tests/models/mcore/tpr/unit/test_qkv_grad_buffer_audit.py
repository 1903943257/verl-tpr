"""CPU-only contract: QKV audit must not change gradients or allocate FP32 buffers."""

import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from verl.workers.engine.megatron.tpr.grad_buffer_audit import QKVGradBufferAudit


class _Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attention = nn.Module()
        self.self_attention.layer_number = 1
        self.self_attention.linear_qkv = nn.Linear(4, 6, bias=False)


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = nn.Module()
        self.decoder.layers = nn.ModuleList([_Layer()])


def _events(capsys):
    return [
        json.loads(line.split("TPR_QKV_GRAD_AUDIT ", 1)[1])
        for line in capsys.readouterr().out.splitlines()
        if line.startswith("TPR_QKV_GRAD_AUDIT ")
    ]


def test_qkv_grad_audit_is_metadata_only(capsys):
    model = _Model()
    weight = model.decoder.layers[0].self_attention.linear_qkv.weight
    # Main grad storage may be allocated by Megatron DDP before backward.
    weight.main_grad = torch.zeros_like(weight, dtype=torch.float32)
    wrapper = SimpleNamespace(ddp_config=SimpleNamespace(grad_reduce_in_fp32=True))
    before_weight = weight.detach().clone()
    before_main = weight.main_grad.detach().clone()
    audit = QKVGradBufferAudit(model, wrapper).install()
    try:
        output = model.decoder.layers[0].self_attention.linear_qkv(torch.ones((2, 4)))
        output.sum().backward()
        audit.snapshot("AFTER_TREE", tree_index=0)
        audit.snapshot("BEFORE_FINALIZE")
        audit.snapshot("AFTER_FINALIZE")
    finally:
        audit.close()

    assert torch.equal(before_weight, weight)
    assert torch.equal(before_main, weight.main_grad)
    assert torch.equal(weight.grad, torch.full_like(weight, 2.0))
    events = _events(capsys)
    assert "PRE_ACCUM" in {event["stage"] for event in events}
    assert "POST_ACCUM" in {event["stage"] for event in events}
    assert [event["stage"] for event in events][-2:] == [
        "BEFORE_FINALIZE", "AFTER_FINALIZE",
    ]
    after = next(event for event in events if event["stage"] == "AFTER_TREE")
    assert after["incoming_dw"] is None
    assert after["param_grad"]["dtype"] == "torch.float32"
    assert after["main_grad"]["dtype"] == "torch.float32"
    assert after["main_grad"]["data_ptr"] == weight.main_grad.data_ptr()
    assert after["grad_reduce_in_fp32"] is True
    assert after["tree_index"] == 0


def test_audit_rejects_missing_layer():
    with pytest.raises(ValueError, match="missing layers"):
        QKVGradBufferAudit(_Model(), object(), layer_numbers=(2,))

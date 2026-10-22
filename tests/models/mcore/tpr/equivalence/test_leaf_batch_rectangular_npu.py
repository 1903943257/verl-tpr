"""NPU comparison: grouped rectangular attention vs independent leaves.

Run explicitly on Ascend (not part of the isolated CPU suite):
    pytest -q -s tests/models/mcore/tpr/equivalence/test_leaf_batch_rectangular_npu.py
"""

import pytest
import torch

pytest.importorskip("torch_npu")
if not torch.npu.is_available():
    pytest.skip("Ascend NPU required", allow_module_level=True)

from verl.models.mcore.tpr.rectangular_attention import rectangular_causal_attention


def test_two_siblings_share_prefix_and_accumulate_exact_gradient_owner():
    torch.manual_seed(141)
    device, dtype = "npu", torch.bfloat16
    p, s, b, heads, head_dim = 64, 32, 2, 4, 64
    base = {
        "q": torch.randn(s, b, heads, head_dim, device=device, dtype=dtype),
        "k": torch.randn(s, b, heads, head_dim, device=device, dtype=dtype),
        "v": torch.randn(s, b, heads, head_dim, device=device, dtype=dtype),
        "pk": torch.randn(p, 1, heads, head_dim, device=device, dtype=dtype),
        "pv": torch.randn(p, 1, heads, head_dim, device=device, dtype=dtype),
    }

    def execute(grouped):
        tensors = {name: value.detach().clone().requires_grad_(True) for name, value in base.items()}
        q, k, v, pk, pv = (tensors[key] for key in ("q", "k", "v", "pk", "pv"))
        if grouped:
            out = rectangular_causal_attention(
                q, torch.cat([pk.expand(-1, b, -1, -1), k]),
                torch.cat([pv.expand(-1, b, -1, -1), v]),
            )
        else:
            out = torch.cat([
                rectangular_causal_attention(
                    q[:, i:i+1], torch.cat([pk, k[:, i:i+1]]),
                    torch.cat([pv, v[:, i:i+1]]),
                )
                for i in range(b)
            ], dim=1)
        out.float().square().sum().backward()
        return out.detach(), {key: value.grad.detach() for key, value in tensors.items()}

    grouped_out, grouped_grads = execute(grouped=True)
    separate_out, separate_grads = execute(grouped=False)
    torch.testing.assert_close(grouped_out, separate_out, atol=3e-2, rtol=3e-2)
    for key in base:
        assert torch.isfinite(grouped_grads[key]).all(), key
        torch.testing.assert_close(grouped_grads[key], separate_grads[key], atol=5e-2, rtol=5e-2)

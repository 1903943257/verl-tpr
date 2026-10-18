"""Production TPR: zero-loss CP ranks must preserve graph connectivity cheaply."""

import torch
from unittest.mock import patch

from verl.workers.engine.megatron.tpr.executor import SegmentExecutor
from verl.workers.engine.megatron.tpr.tree import SegmentSpec


def _segment():
    return SegmentSpec(
        segment_id=0,
        parent_id=None,
        token_ids=torch.tensor([2, 3, 4], dtype=torch.long, device="cpu"),
        position_start=0,
        prefix_length=0,
    )


def _executor_for_empty_loss():
    executor = object.__new__(SegmentExecutor)
    executor.segment_loss_term_counts = {0: 0}
    executor.segment_loss_fn = lambda *_: (_ for _ in ()).throw(
        AssertionError("No PPO objective on a rank without owned tokens")
    )
    executor.segment_loss_metrics = []
    executor.cp_size = 4
    executor.loss_scale_func = None
    return executor


def test_empty_objective_is_scalar_connected_zero_for_all_logits():
    logits = torch.randn(1, 3, 8, dtype=torch.float32, requires_grad=True, device="cpu")
    executor = _executor_for_empty_loss()
    loss = executor._compute_loss(_segment(), logits)
    backward_loss = executor._prepare_backward_loss(
        loss, logits=logits, has_owned_loss=False
    )
    assert loss.shape == backward_loss.shape == ()
    assert loss.requires_grad and backward_loss.requires_grad
    torch.autograd.backward(backward_loss)
    assert logits.grad is not None
    torch.testing.assert_close(logits.grad, torch.zeros_like(logits))


def test_empty_objective_does_not_convert_whole_logits_to_float():
    # The old zero path upcast all logits to FP32, which allocates ~1.63 GiB
    # for a sufficiently large real CP4 logits shard. Verify no full-matrix
    # dtype conversion, while retaining a scalar autograd link.
    x = torch.randn(1, 3, 16, dtype=torch.bfloat16, requires_grad=True, device="cpu")
    executor = _executor_for_empty_loss()
    # torch.autograd.profiler is broken on some torch/torch_npu binary
    # combinations. Intercept only the Python .float() boundary instead:
    # any full-sized upcast (the original 1.63-GiB allocation) is rejected.
    real_float = torch.Tensor.float
    def scalar_float_only(tensor):
        if tensor.numel() != 1:
            raise AssertionError("zero objective upcast a non-scalar tensor")
        return real_float(tensor)
    with patch.object(torch.Tensor, "float", scalar_float_only):
        loss = executor._compute_loss(_segment(), x)
        backward_loss = executor._prepare_backward_loss(
            loss, logits=x, has_owned_loss=False
        )
    assert loss.dtype == backward_loss.dtype == torch.float32
    backward_loss.backward()
    torch.testing.assert_close(x.grad, torch.zeros_like(x))

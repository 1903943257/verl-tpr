"""Production TPR: zero-loss CP ranks must preserve graph connectivity cheaply."""

import torch

from verl.workers.engine.megatron.tpr.executor import SegmentExecutor
from verl.workers.engine.megatron.tpr.tree import SegmentSpec


def _segment():
    return SegmentSpec(
        segment_id=0,
        parent_id=None,
        token_ids=torch.tensor([2, 3, 4], dtype=torch.long),
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
    logits = torch.randn(1, 3, 8, dtype=torch.float32, requires_grad=True)
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
    x = torch.randn(1, 3, 16, dtype=torch.bfloat16, requires_grad=True)
    executor = _executor_for_empty_loss()
    with torch.autograd.profiler.profile(use_cpu=True) as profile:
        loss = executor._compute_loss(_segment(), x)
        backward_loss = executor._prepare_backward_loss(
            loss, logits=x, has_owned_loss=False
        )
    assert loss.dtype == backward_loss.dtype == torch.float32
    converted = [
        e for e in profile.function_events
        if e.name in ("aten::to", "aten::_to_copy")
        and any(dim == list(x.shape) for dim in e.input_shapes)
    ]
    assert not converted, "empty loss must not upcast the full logits tensor"
    backward_loss.backward()
    torch.testing.assert_close(x.grad, torch.zeros_like(x))

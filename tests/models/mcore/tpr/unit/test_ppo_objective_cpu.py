"""Compare the production PPO boundary with the frozen pre-refactor adapter."""

import sys
import types

import pytest
import torch
from tensordict import TensorDict

from verl.experimental.tpr.legacy.objective_adapter import SegmentPPOObjectiveAdapter as ReferenceObjective
from verl.utils import tensordict_utils as tu
from verl.workers.engine.megatron.tpr.objective import SegmentPPOObjectiveAdapter, _logit_parts
from verl.workers.engine.megatron.tpr.tree import (
    SegmentObjectiveRef,
    SegmentPlan,
    SegmentSpec,
    TreeExecutionPlan,
)
from verl.workers.utils.padding import no_padding_2_padding


def _log_probs(logits, labels):
    return logits.log_softmax(-1).gather(-1, labels[:, None]).squeeze(-1)


def _entropy(logits):
    log_probs = logits.log_softmax(-1)
    return -(log_probs.exp() * log_probs).sum(-1)


def _batch(temperature, extras):
    batch = TensorDict({
        "old_log_probs": torch.tensor([[-1.2, -0.9], [-1.4, 0.0]]),
        "advantages": torch.tensor([[1.0, -2.0], [0.5, 0.0]]),
        "response_mask": torch.tensor([[1, 1], [1, 0]], dtype=torch.bool),
    }, batch_size=[2])
    tu.assign_non_tensor_data(batch, "temperature", temperature)
    tu.assign_non_tensor(batch, batch_num_tokens=13,
                         dp_size=2, global_batch_size=5, tpr_capture_log_probs=True,
                         tpr_capture_logit_parts=extras)
    if extras:
        batch["ref_log_prob"] = torch.tensor([[-1.1, -1.3], [-1.7, 0.0]])
        batch["rollout_is_weights"] = torch.tensor([[0.8, 1.1], [1.2, 0.0]])
    return batch


def _loss_callback(captured):
    def loss_fn(*, model_output, data, dp_group):
        captured.append((model_output, data, dp_group))
        new = no_padding_2_padding(model_output["log_probs"], data)
        ratios = (new - data["old_log_probs"]).exp()
        advantages = data["advantages"]
        per_token = torch.maximum(-advantages * ratios, -advantages * ratios.clamp(0.8, 1.2))
        if "rollout_is_weights" in data:
            per_token = per_token * data["rollout_is_weights"]
        if "ref_log_prob" in data:
            per_token = per_token + 0.1 * (new - data["ref_log_prob"])
        if "entropy" in model_output:
            per_token = per_token - 0.05 * no_padding_2_padding(model_output["entropy"], data)
        loss = (per_token * data["response_mask"]).sum() / data["batch_num_tokens"] * data["dp_size"]
        return loss, {"native_callback": True}
    return loss_fn


@pytest.mark.parametrize("temperature", [1.0, [0.7, 1.3], torch.tensor([0.7, 1.3]), torch.tensor(0.8)])
@pytest.mark.parametrize("extras", [False, True])
@pytest.mark.parametrize("chunk_size", [None, 1])
def test_compact_ppo_fields_loss_and_gradients_match_reference(temperature, extras, chunk_size):
    torch.manual_seed(113)
    initial = torch.randn(1, 2, 7, dtype=torch.float64)
    segment = SegmentSpec(0, None, torch.tensor([3, 4]), 0, 0)
    # Unsorted logical rows, repeated physical query, and two branch targets.
    refs = (SegmentObjectiveRef(0, 0, 2, 1, 0), SegmentObjectiveRef(0, 1, 4, 0, 1),
            SegmentObjectiveRef(0, 0, 3, 0, 0))
    results = []
    group = object()
    for cls in (ReferenceObjective, SegmentPPOObjectiveAdapter):
        captured = []
        logits = initial.clone().requires_grad_()
        adapter = cls(_batch(temperature, extras), _loss_callback(captured),
                      log_prob_fn=_log_probs, entropy_fn=_entropy, calculate_entropy=extras,
                      dp_group=group, log_prob_chunk_size=chunk_size)
        loss, metrics = adapter.compute_loss(segment, logits, refs)
        loss.backward()
        results.append((loss, logits.grad, captured[0], adapter, metrics))
    expected, actual = results
    torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
    torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)
    for key in expected[2][0]:
        torch.testing.assert_close(actual[2][0][key], expected[2][0][key], rtol=0, atol=0)
    for key in expected[2][1].keys():
        left, right = actual[2][1][key], expected[2][1][key]
        if isinstance(left, torch.Tensor):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        else:
            assert tu.get_non_tensor_data(actual[2][1], key, None) == tu.get_non_tensor_data(expected[2][1], key, None)
    assert actual[2][2] is expected[2][2] is group
    assert actual[3].debug_new_log_probs == expected[3].debug_new_log_probs
    assert actual[3].debug_logit_parts == expected[3].debug_logit_parts
    assert actual[4] == expected[4] == {"native_callback": True}


@pytest.mark.parametrize("temperature", [0, -1, float("nan"), float("inf"), [1, 0]])
def test_invalid_temperature_rejected(temperature):
    adapter = SegmentPPOObjectiveAdapter(_batch(temperature, False), _loss_callback([]))
    refs = (SegmentObjectiveRef(0, 0, 1, 0, 0), SegmentObjectiveRef(0, 0, 1, 1, 0))
    with pytest.raises(ValueError, match="finite and positive"):
        adapter._temperatures(refs)


class _DestructiveLogProbs(torch.autograd.Function):
    """Mimic native FP32 CE reusing its input storage for softmax."""

    @staticmethod
    def forward(ctx, logits, targets):
        result = _log_probs(logits, targets)
        logits.copy_(logits.softmax(-1))
        ctx.save_for_backward(logits, targets)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        probabilities, targets = ctx.saved_tensors
        gradient = -probabilities.clone()
        gradient.scatter_add_(1, targets[:, None], torch.ones_like(targets[:, None], dtype=gradient.dtype))
        return gradient * grad_output[:, None], None


class _DestructiveEntropy(torch.autograd.Function):
    """Native entropy backward temporarily modifies its saved input logits."""

    @staticmethod
    def forward(ctx, logits):
        probabilities = logits.softmax(-1)
        mean = (probabilities * logits).sum(-1, keepdim=True)
        ctx.save_for_backward(logits, probabilities, mean)
        return _entropy(logits)

    @staticmethod
    def backward(ctx, grad_output):
        logits, probabilities, mean = ctx.saved_tensors
        logits.sub_(mean)
        gradient = -probabilities * logits * grad_output[:, None]
        logits.add_(mean)
        return gradient


@pytest.mark.parametrize("capture_parts", [False, True])
@pytest.mark.parametrize("calculate_entropy", [False, True])
def test_native_inplace_logprob_and_entropy_preserve_loss_gradients_and_probes(
    capture_parts, calculate_entropy,
):
    initial = torch.tensor([[[2.0, -1.0, 4.0, 0.5], [-0.7, 1.1, 0.2, 3.5]]])
    segment = SegmentSpec(0, None, torch.tensor([3, 4]), 0, 0)
    refs = (SegmentObjectiveRef(0, 0, 2, 1, 0), SegmentObjectiveRef(0, 1, 0, 0, 1),
            SegmentObjectiveRef(0, 0, 3, 0, 0))
    results = []
    for destructive in (False, True):
        captured = []
        logits = initial.clone().requires_grad_()
        batch = _batch([0.7, 1.3], False)
        tu.assign_non_tensor(batch, tpr_capture_logit_parts=capture_parts)
        adapter = SegmentPPOObjectiveAdapter(
            batch, _loss_callback(captured), calculate_entropy=calculate_entropy,
            log_prob_fn=_DestructiveLogProbs.apply if destructive else _log_probs,
            entropy_fn=_DestructiveEntropy.apply if destructive else _entropy,
            log_prob_chunk_size=1,
        )
        loss, _ = adapter.compute_loss(segment, logits, refs)
        loss.backward()
        results.append((loss, logits.grad, captured[0][0], adapter))
    expected, actual = results
    torch.testing.assert_close(actual[0], expected[0])
    torch.testing.assert_close(actual[1], expected[1])
    for key in expected[2]:
        torch.testing.assert_close(actual[2][key], expected[2][key])
    assert actual[3].debug_new_log_probs == expected[3].debug_new_log_probs
    assert actual[3].debug_logit_parts == expected[3].debug_logit_parts


@pytest.mark.parametrize("rank", [0, 1])
def test_logit_parts_reduce_global_vocabulary_and_targets(monkeypatch, rank):
    full = torch.tensor([[2.0, -1.0, 4.0, 0.5, 7.0, -2.0, 1.0, 0.0],
                         [-0.7, 1.1, 0.2, 3.5, -8.0, 9.0, 2.0, -1.0]])
    targets = torch.tensor([5, 0])
    group = object()
    parallel_state = types.SimpleNamespace(
        get_tensor_model_parallel_group=lambda: group,
    )
    monkeypatch.setitem(sys.modules, "megatron.core.parallel_state", parallel_state)
    if "megatron.core" in sys.modules:
        monkeypatch.setattr(sys.modules["megatron.core"], "parallel_state", parallel_state, raising=False)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)

    def get_rank(supplied_group):
        assert supplied_group is group
        return rank

    monkeypatch.setattr(torch.distributed, "get_rank", get_rank)
    expected_targets = full.gather(1, targets[:, None]).squeeze(1)
    maximum = full.amax(-1)
    exp_sum = (full - maximum[:, None]).exp().sum(-1)
    calls = []

    def all_reduce(value, op=torch.distributed.ReduceOp.SUM, **kwargs):
        assert kwargs["group"] is group
        if not calls:
            owned = (targets >= rank * 4) & (targets < (rank + 1) * 4)
            torch.testing.assert_close(value, expected_targets.masked_fill(~owned, 0))
            assert op == torch.distributed.ReduceOp.SUM
            value.copy_(expected_targets)
        elif len(calls) == 1:
            assert op == torch.distributed.ReduceOp.MAX
            torch.testing.assert_close(value, full[:, rank * 4:(rank + 1) * 4].amax(-1))
            value.copy_(maximum)
        else:
            assert op == torch.distributed.ReduceOp.SUM
            local_exp_sum = (full[:, rank * 4:(rank + 1) * 4] - maximum[:, None]).exp().sum(-1)
            torch.testing.assert_close(value, local_exp_sum)
            value.copy_(exp_sum)
        calls.append(op)

    monkeypatch.setattr(torch.distributed, "all_reduce", all_reduce)
    target_logits, lse = _logit_parts(full[:, rank * 4:(rank + 1) * 4], targets)
    torch.testing.assert_close(target_logits, expected_targets)
    torch.testing.assert_close(lse, full.logsumexp(-1))
    assert len(calls) == 3


def test_bind_empty_supervision_tree_returns_zero_counts():
    root = SegmentSpec(0, None, torch.tensor([1, 2]), 0, 0)
    child = SegmentSpec(1, 0, torch.tensor([3]), 2, 2)
    tree = TreeExecutionPlan("zero", 0, (0,), SegmentPlan((root, child), 0), ())
    adapter = SegmentPPOObjectiveAdapter(_batch(1.0, False), _loss_callback([]))
    callback, counts = adapter.bind_tree(tree)
    assert callable(callback)
    assert counts == {0: 0, 1: 0}


def test_logit_parts_reject_missing_tp_group_without_using_global_dp_group(monkeypatch):
    parallel_state = types.SimpleNamespace(get_tensor_model_parallel_group=lambda: None)
    monkeypatch.setitem(sys.modules, "megatron.core.parallel_state", parallel_state)
    if "megatron.core" in sys.modules:
        monkeypatch.setattr(sys.modules["megatron.core"], "parallel_state", parallel_state, raising=False)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda *args, **kwargs: pytest.fail("entered global group"))
    with pytest.raises(RuntimeError, match="native Megatron TP group"):
        _logit_parts(torch.ones(2, 4), torch.tensor([0, 3]))

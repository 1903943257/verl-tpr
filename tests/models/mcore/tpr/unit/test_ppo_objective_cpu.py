"""Compare the production PPO boundary with the frozen pre-refactor adapter."""

import pytest
import torch
from tensordict import TensorDict

from verl.experimental.tpr.legacy.objective_adapter import SegmentPPOObjectiveAdapter as ReferenceObjective
from verl.utils import tensordict_utils as tu
from verl.workers.engine.megatron.tpr.objective import SegmentPPOObjectiveAdapter
from verl.workers.engine.megatron.tpr.tree import SegmentObjectiveRef, SegmentSpec
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

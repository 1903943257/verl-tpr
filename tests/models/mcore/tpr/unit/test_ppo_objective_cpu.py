"""Check the native PPO boundary, FP32 output contract and frozen FP64 reference."""

import ast
import sys
import types
from functools import partial
from pathlib import Path

import pytest
import torch
from tensordict import TensorDict

from verl.experimental.tpr.legacy.objective_adapter import SegmentPPOObjectiveAdapter as ReferenceObjective
from verl.utils import tensordict_utils as tu
from verl.workers.engine.megatron.tpr import objective as objective_module
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


@pytest.mark.parametrize("chunk_size", [None, 2])
def test_physical_dedup_keeps_first_seen_order_and_every_logical_occurrence(chunk_size):
    initial = torch.arange(21, dtype=torch.float64).reshape(1, 3, 7) / 9
    segment = SegmentSpec(0, None, torch.tensor([3, 4, 5]), 0, 0)
    refs = (
        SegmentObjectiveRef(0, 2, 3, 2, 1), SegmentObjectiveRef(0, 0, 1, 0, 0),
        SegmentObjectiveRef(0, 0, 1, 1, 0), SegmentObjectiveRef(0, 0, 1, 2, 0),
        SegmentObjectiveRef(0, 2, 4, 0, 1), SegmentObjectiveRef(0, 2, 3, 1, 1),
    )
    positions = torch.tensor([2, 0, 0, 2, 2])
    targets = torch.tensor([3, 1, 1, 4, 3])
    temperatures = torch.tensor([1, 1, 2, 1, 2], dtype=initial.dtype)
    scaled = initial[0].index_select(0, positions) / temperatures[:, None]
    unique_log_probs = _log_probs(scaled, targets)
    expected_packed = torch.stack([
        unique_log_probs[1], unique_log_probs[3], unique_log_probs.new_zeros(()),
        unique_log_probs[2], unique_log_probs[4], unique_log_probs.new_zeros(()),
        unique_log_probs[1], unique_log_probs[0], unique_log_probs.new_zeros(()),
    ])
    results = []
    for cls in (ReferenceObjective, SegmentPPOObjectiveAdapter):
        batch = TensorDict({
            "old_log_probs": torch.full((3, 2), -1.2),
            "advantages": torch.tensor([[1., -2.], [0.5, -0.8], [1.3, -0.2]]),
            "response_mask": torch.ones(3, 2, dtype=torch.bool),
        }, batch_size=[3])
        tu.assign_non_tensor_data(batch, "temperature", [1., 2., 1.])
        tu.assign_non_tensor(batch, batch_num_tokens=13, dp_size=2)
        traced, captured = [], []

        def recorded_log_probs(logits, labels):
            traced.append((logits.detach().clone(), labels.clone()))
            return _log_probs(logits, labels)

        logits = initial.clone().requires_grad_()
        adapter = cls(batch, _loss_callback(captured), log_prob_fn=recorded_log_probs,
                      log_prob_chunk_size=chunk_size)
        loss, _ = adapter.compute_loss(segment, logits, refs)
        assert len(traced) == (1 if chunk_size is None else 3)
        torch.testing.assert_close(torch.cat([call[0] for call in traced]), scaled, rtol=0, atol=0)
        torch.testing.assert_close(torch.cat([call[1] for call in traced]), targets, rtol=0, atol=0)
        torch.testing.assert_close(captured[0][0]["log_probs"], expected_packed, rtol=0, atol=0)
        assert captured[0][1]["response_mask"].sum().item() == len(refs)
        loss.backward()
        results.append((loss.detach(), logits.grad))
    torch.testing.assert_close(results[0][0], results[1][0], rtol=0, atol=0)
    torch.testing.assert_close(results[0][1], results[1][1], rtol=0, atol=0)


@pytest.mark.parametrize("temperature", [0, -1, float("nan"), float("inf"), [1, 0]])
def test_invalid_temperature_rejected(temperature):
    adapter = SegmentPPOObjectiveAdapter(_batch(temperature, False), _loss_callback([]))
    refs = (SegmentObjectiveRef(0, 0, 1, 0, 0), SegmentObjectiveRef(0, 0, 1, 1, 0))
    with pytest.raises(ValueError, match="finite and positive"):
        adapter._temperatures(refs)


@pytest.mark.parametrize("storage", ["stack", "data", "tuple", "tensor", "scalar_tensor"])
def test_temperature_metadata_is_prepared_once_per_sample_not_per_ref(monkeypatch, storage):
    batch = _batch(1.0, False)
    if storage == "stack":
        # This is VERL's normal assign_non_tensor(list) representation.
        tu.assign_non_tensor(batch, temperature=[0.7, 1.3])
    elif storage == "data":
        tu.assign_non_tensor_data(batch, "temperature", [0.7, 1.3])
    elif storage == "tuple":
        tu.assign_non_tensor(batch, temperature=(0.7, 1.3))
    elif storage == "tensor":
        batch["temperature"] = torch.tensor([0.7, 1.3])
    else:
        tu.assign_non_tensor_data(batch, "temperature", torch.tensor(0.8))
    adapter = SegmentPPOObjectiveAdapter(batch, _loss_callback([]))
    refs = tuple(SegmentObjectiveRef(0, 0, 1, row, 0) for row in [0, 1] * 5000)
    calls = []
    original_metadata = objective_module._metadata

    def metadata(*args, **kwargs):
        calls.append(args[1])
        return original_metadata(*args, **kwargs)

    monkeypatch.setattr(objective_module, "_metadata", metadata)
    monkeypatch.setattr(torch.Tensor, "__float__", lambda self: pytest.fail("per-token Tensor scalar read"))
    values = adapter._temperatures(refs)
    assert adapter._temperatures(refs) == values
    assert calls == ["temperature"]
    expected = [0.8, 0.8] if storage == "scalar_tensor" else [0.7, 1.3]
    assert values[:2] == pytest.approx(expected)
    assert values == values[:2] * 5000


@pytest.mark.parametrize("storage", ["scalar", "list", "tuple", "tensor", "wrapped", "stack", "missing"])
def test_plain_dict_metadata_matches_tensordict_loss_and_gradients(storage):
    temperatures = 1.0 if storage in ("scalar", "missing") else [0.7, 1.3]
    source = _batch(temperatures, True)
    tu.assign_non_tensor(source, calculate_entropy=True)
    plain = {key: source[key] for key in
             ("old_log_probs", "advantages", "response_mask", "ref_log_prob", "rollout_is_weights")}
    plain.update(batch_num_tokens=13, dp_size=2, global_batch_size=5,
                 calculate_entropy=True, tpr_capture_log_probs=True, tpr_capture_logit_parts=True)
    if storage == "wrapped":
        plain["temperature"] = tu.NonTensorData(temperatures)
        plain["dp_size"] = tu.NonTensorData(2)
    elif storage == "stack":
        tu.assign_non_tensor(source, temperature=temperatures)
        plain["temperature"] = source.get("temperature")
    elif storage != "missing":
        plain["temperature"] = (
            torch.tensor(temperatures) if storage == "tensor"
            else tuple(temperatures) if storage == "tuple" else temperatures
        )
    segment = SegmentSpec(0, None, torch.tensor([3, 4]), 0, 0)
    refs = (SegmentObjectiveRef(0, 0, 2, 1, 0), SegmentObjectiveRef(0, 1, 4, 0, 1),
            SegmentObjectiveRef(0, 0, 3, 0, 0))
    initial = torch.arange(14, dtype=torch.float64).reshape(1, 2, 7) / 9
    results = []
    for data in (source, plain):
        logits = initial.clone().requires_grad_()
        adapter = SegmentPPOObjectiveAdapter(data, _loss_callback([]),
                                             log_prob_fn=_log_probs, entropy_fn=_entropy)
        loss, _ = adapter.compute_loss(segment, logits, refs)
        loss.backward()
        results.append((loss.detach(), logits.grad, adapter.debug_new_log_probs))
    torch.testing.assert_close(results[0][0], results[1][0], rtol=0, atol=0)
    torch.testing.assert_close(results[0][1], results[1][1], rtol=0, atol=0)
    assert results[0][2] == results[1][2]


@pytest.mark.parametrize("temperature", [[], [1.0], [1.0, 2.0, 3.0], torch.empty(0)])
def test_temperature_metadata_rejects_wrong_sample_count(temperature):
    adapter = SegmentPPOObjectiveAdapter(_batch(temperature, False), _loss_callback([]))
    with pytest.raises(ValueError, match="one value per PPO sample"):
        adapter._temperatures((SegmentObjectiveRef(0, 0, 1, 0, 0),))


@pytest.mark.parametrize("row", [-1, 2])
def test_temperature_metadata_rejects_out_of_range_logical_sample(row):
    adapter = SegmentPPOObjectiveAdapter(_batch(1.0, False), _loss_callback([]))
    with pytest.raises(ValueError, match="sample_row"):
        adapter._temperatures((SegmentObjectiveRef(0, 0, 1, row, 0),))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("extras", [False, True])
@pytest.mark.parametrize("chunk_size", [None, 1])
def test_selected_low_precision_logits_match_native_fp32_output_contract(dtype, extras, chunk_size):
    torch.manual_seed(517)
    # Only three unique physical (query,target,temp) combinations are needed,
    # rather than casting all five physical vocab rows as Float16Module does.
    initial = torch.randn(1, 5, 7).to(dtype)
    segment = SegmentSpec(0, None, torch.tensor([3, 4, 5, 6, 7]), 0, 0)
    refs = (SegmentObjectiveRef(0, 0, 2, 1, 0), SegmentObjectiveRef(0, 1, 4, 0, 1),
            SegmentObjectiveRef(0, 0, 3, 0, 0))
    results = []
    for native_output in (True, False):
        captured, inputs = [], []
        logits = (initial.float() if native_output else initial.clone()).requires_grad_()

        def recorded_log_probs(scaled, targets):
            assert scaled.dtype == torch.float32
            assert scaled.shape[0] <= 3
            inputs.append((targets.detach().clone(), scaled.detach().clone()))
            return _log_probs(scaled, targets)

        def recorded_entropy(scaled):
            assert scaled.dtype == torch.float32
            return _entropy(scaled)

        cls = ReferenceObjective if native_output else SegmentPPOObjectiveAdapter
        adapter = cls(_batch([0.7, 1.3], extras), _loss_callback(captured),
                      log_prob_fn=recorded_log_probs, entropy_fn=recorded_entropy,
                      calculate_entropy=extras, log_prob_chunk_size=None if native_output else chunk_size)
        loss, _ = adapter.compute_loss(segment, logits, refs)
        labels = torch.cat([item[0] for item in inputs])
        forward_inputs = torch.cat([item[1] for item in inputs]).index_select(0, labels.argsort())
        loss.backward()
        results.append((loss.detach(), logits.grad, captured[0][0], forward_inputs))
    expected, actual = results
    torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
    torch.testing.assert_close(actual[3], expected[3], rtol=0, atol=0)
    for key in expected[2]:
        torch.testing.assert_close(actual[2][key], expected[2][key], rtol=0, atol=0)
    # Shared queries accumulate every target/temp contribution in FP32 before
    # one cast back, including when one query exceeds the soft chunk-size limit.
    torch.testing.assert_close(actual[1], expected[1].to(dtype), rtol=0, atol=0)
    assert not actual[1][:, 2:].count_nonzero()


def test_batch_rows_and_host_masks_are_cached_across_segments(monkeypatch):
    batch = _batch([0.7, 1.3], True)
    tu.assign_non_tensor(batch, tpr_capture_log_probs=False, tpr_capture_logit_parts=False)
    captured, calls = [], []
    adapter = SegmentPPOObjectiveAdapter(batch, _loss_callback(captured),
                                         log_prob_fn=_log_probs, calculate_entropy=False)
    original_rows = objective_module._rows

    def rows(batch, key):
        calls.append(key)
        return original_rows(batch, key)

    monkeypatch.setattr(objective_module, "_rows", rows)
    segment = SegmentSpec(0, None, torch.tensor([3, 4]), 0, 0)
    refs = (SegmentObjectiveRef(0, 0, 2, 1, 0), SegmentObjectiveRef(0, 1, 4, 0, 1),
            SegmentObjectiveRef(0, 0, 3, 0, 0))
    for _ in range(2):
        logits = torch.randn(1, 2, 7, requires_grad=True)
        loss, _ = adapter.compute_loss(segment, logits, refs)
        loss.backward()
    assert calls == ["response_mask", "old_log_probs", "advantages", "ref_log_prob", "rollout_is_weights"]
    assert all(mask.device.type == "cpu" and mask.dtype == torch.bool for mask in adapter._response_rows)
    for key in ("ref_log_prob", "rollout_is_weights", "old_log_probs", "advantages"):
        expected = batch[key][0, [0, 1]]
        torch.testing.assert_close(captured[0][1][key][0], expected)


def test_batch_row_preparation_rejects_mismatched_response_fields():
    batch = _batch(1.0, False)
    batch["advantages"] = torch.ones(2, 1)
    adapter = SegmentPPOObjectiveAdapter(batch, _loss_callback([]))
    with pytest.raises(ValueError, match="advantages.*response_mask"):
        adapter._prepare_batch_rows()


def _native_ppo_loss():
    """Load the real loss bodies without unrelated distributed/config imports."""
    import verl.utils.torch_functional as verl_F
    from verl.utils.metric import AggregationType, Metric

    root = Path(__file__).resolve().parents[5]
    namespace = dict(torch=torch, tu=tu, no_padding_2_padding=no_padding_2_padding,
                     verl_F=verl_F, Metric=Metric, AggregationType=AggregationType,
                     AlgoConfig=type("AlgoConfig", (), {}))
    for relative, names in (
        ("verl/trainer/ppo/core_algos.py",
         {"agg_loss", "compute_policy_loss_vanilla", "kl_penalty", "kl_penalty_forward"}),
        ("verl/workers/utils/losses.py", {"ppo_loss"}),
    ):
        path = root / relative
        tree = ast.parse(path.read_text(encoding="utf-8"))
        tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
        for node in tree.body:
            node.decorator_list = []
        tree.body.insert(0, ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0))
        ast.fix_missing_locations(tree)
        exec(compile(tree, str(path), "exec"), namespace)

    def get_policy_loss_fn(mode):
        assert mode == "vanilla"
        return namespace["compute_policy_loss_vanilla"]

    namespace["get_policy_loss_fn"] = get_policy_loss_fn
    return namespace["ppo_loss"]


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("extras", [False, True])
@pytest.mark.parametrize("chunk_size", [None, 1])
def test_low_precision_contract_uses_real_native_ppo_entropy_kl_and_is(dtype, extras, chunk_size):
    class ActorConfig:
        loss_agg_mode = "token-mean"
        policy_loss = {"loss_mode": "vanilla"}
        clip_ratio, clip_ratio_low, clip_ratio_high = 0.2, None, None
        entropy_coeff, kl_loss_coef, kl_loss_type = 0.05, 0.1, "low_var_kl"
        loss_scale_factor = None
        use_kl_loss = extras

        def __init__(self):
            self.global_batch_info = {}

        def get(self, key, default=None):
            return getattr(self, key, default)

    torch.manual_seed(519)
    initial = torch.randn(1, 3, 7).to(dtype)
    segment = SegmentSpec(0, None, torch.tensor([3, 4, 5]), 0, 0)
    refs = (SegmentObjectiveRef(0, 0, 2, 1, 0), SegmentObjectiveRef(0, 1, 4, 0, 1),
            SegmentObjectiveRef(0, 0, 3, 0, 0))
    native_loss = _native_ppo_loss()
    results = []
    for native_output in (True, False):
        logits = (initial.float() if native_output else initial.clone()).requires_grad_()
        cls = ReferenceObjective if native_output else SegmentPPOObjectiveAdapter
        adapter = cls(
            _batch([0.7, 1.3], extras), partial(native_loss, config=ActorConfig()),
            log_prob_fn=_log_probs if native_output else _DestructiveLogProbs.apply,
            entropy_fn=_entropy if native_output else _DestructiveEntropy.apply,
            calculate_entropy=extras, log_prob_chunk_size=None if native_output else chunk_size,
        )
        loss, metrics = adapter.compute_loss(segment, logits, refs)
        assert "actor/pg_loss" in metrics
        assert ("actor/entropy_loss" in metrics) is extras
        assert ("kl_loss" in metrics) is extras
        loss.backward()
        results.append((loss.detach(), logits.grad))
    torch.testing.assert_close(results[1][0], results[0][0], rtol=0, atol=0)
    torch.testing.assert_close(results[1][1], results[0][1].to(dtype), rtol=0, atol=0)


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
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_native_inplace_logprob_and_entropy_preserve_loss_gradients_and_probes(
    capture_parts, calculate_entropy, dtype,
):
    initial = torch.tensor([[[2.0, -1.0, 4.0, 0.5], [-0.7, 1.1, 0.2, 3.5]]], dtype=dtype)
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

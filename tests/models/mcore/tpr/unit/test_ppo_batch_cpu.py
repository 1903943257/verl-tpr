"""CPU checks for adapter-owned forest finalization and weighted PPO metrics.

Load these classes without the unrelated Megatron model-construction imports.
The code under test is compiled directly from the production adapter source.
"""

import ast
from collections.abc import Callable
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import Tensor

from verl.utils.metric import AggregationType, Metric
from verl.workers.engine.megatron.tpr.tree import ForestExecutionPlan, TreeExecutionPlan, build_tree_execution_plans


@dataclass
class _ModuleSpec:
    submodules: Any = None


def _adapter_components():
    source = Path(__file__).resolve().parents[5] / "verl/workers/engine/megatron/tpr/adapter.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    names = {"run_forest", "_PPOMetrics", "_get_layer_specs", "_assert_finite_tree_loss"}
    tree.body = [
        node for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names
    ]
    namespace = dict(globals())
    namespace["ModuleSpec"] = _ModuleSpec
    exec(compile(tree, str(source), "exec"), namespace)
    return tuple(namespace[name] for name in ("run_forest", "_PPOMetrics", "_get_layer_specs", "_assert_finite_tree_loss"))


run_forest, PPOMetrics, get_layer_specs, assert_finite_tree_loss = _adapter_components()


def _forest():
    return build_tree_execution_plans(["u_s_0", "u_s_1", "v_s_0"], {
        "input_ids": [torch.tensor([1, 151645, 2]), torch.tensor([1, 151645, 3]),
                      torch.tensor([7, 151645, 4])],
        "response_mask": [torch.ones(1, dtype=torch.bool) for _ in range(3)],
    }, turn_min_segment_tokens=1)


def test_forest_finalizes_once_after_no_sync():
    calls = []

    @contextmanager
    def no_sync():
        calls.append("enter")
        yield
        calls.append("exit")

    def run_tree(plan):
        calls.append("tree")
        return torch.tensor(0.25 * plan.logical_loss_tokens)

    forest = _forest()
    result = run_forest(forest, run_tree, lambda: calls.append("finalize"), no_sync)
    assert calls == ["enter", *(["tree"] * len(forest.trees)), "exit", "finalize"]
    assert result.item() == 0.75


@pytest.mark.parametrize("failure", [RuntimeError("tree failed"), 1.0])
def test_failed_tree_never_finalizes(failure):
    finalized = []

    def run_tree(plan):
        if isinstance(failure, Exception):
            raise failure
        return failure

    with pytest.raises((RuntimeError, TypeError)):
        run_forest(_forest(), run_tree, lambda: finalized.append(True))
    assert not finalized


def test_forest_skips_unsupervised_trees_without_changing_loss_denominator():
    forest = _forest()
    empty = replace(forest.trees[0], objective_refs=())
    forest = ForestExecutionPlan((empty, forest.trees[1]), forest.trees[1].logical_loss_tokens)
    called, finalized = [], []

    def run_tree(tree):
        called.append(tree)
        return torch.tensor(0.25 * tree.logical_loss_tokens)

    loss = run_forest(forest, run_tree, lambda: finalized.append(True))
    assert called == [forest.trees[1]] and finalized == [True]
    assert loss.item() == 0.25


def test_all_unsupervised_forest_never_finalizes():
    empty = replace(_forest().trees[0], objective_refs=())
    finalized = []
    with pytest.raises(ValueError, match="no supervised"):
        run_forest(ForestExecutionPlan((empty,), 0), lambda _: pytest.fail("ran empty tree"),
                   lambda: finalized.append(True))
    assert not finalized


def test_metrics_weight_means_preserve_sums_and_numeric_lists():
    metrics = PPOMetrics()
    sum_metric = Metric(AggregationType.SUM, 0.25)
    sum_metric.append(0.5)
    mean_metric = Metric(AggregationType.MEAN, 1.0)
    mean_metric.append(3.0)
    metrics.add({"mean": mean_metric,
                 "sum": sum_metric, "numeric": 7}, 2 / 13)
    metrics.add({"mean": Metric(AggregationType.MEAN, 10.0),
                 "sum": Metric(AggregationType.SUM, 0.5), "numeric": 3}, 8 / 13)
    # SUM objects retain native accumulation behavior until the forest finishes.
    sum_metric.append(0.125)
    result = metrics.scalar_lists()
    assert result["mean"] == [(2 * 2 / 13) + (10 * 8 / 13)]
    assert result["sum"] == [1.375]
    assert result["numeric"] == [7.0, 3.0]


def test_metric_type_mix_and_unsupported_reductions_fail_clearly():
    metrics = PPOMetrics()
    metrics.add({"mixed": Metric(AggregationType.SUM, 1)}, 1.0)
    metrics.add({"mixed": 2.0}, 1.0)
    with pytest.raises(TypeError, match="mixes Metric and numeric"):
        metrics.scalar_lists()
    with pytest.raises(NotImplementedError, match="unsupported"):
        metrics.add({"max": Metric(AggregationType.MAX, 1)}, 1.0)


@pytest.mark.parametrize("empty_rank", [False, True])
def test_cp_metrics_include_all_owned_tokens_even_with_an_empty_rank(monkeypatch, empty_rank):
    runtime = SimpleNamespace(enabled=True, size=2, group=object())
    metrics = PPOMetrics()
    if not empty_rank:
        metrics.add({"kl": Metric(AggregationType.MEAN, 3),
                     "loss": Metric(AggregationType.SUM, 0.2), "coef": 0.1}, 2 / 5)

    def gather(peers, local, group):
        assert group is runtime.group
        peers[:] = [local, {"kl": (True, [4 * 3 / 5]),
                           "loss": (True, [0.3]), "coef": (False, [0.1])}]

    monkeypatch.setattr(torch.distributed, "all_gather_object", gather)
    result = metrics.scalar_lists(runtime)
    assert result["kl"] == pytest.approx([2.4 if empty_rank else 3.6])
    assert result["loss"] == pytest.approx([0.3 if empty_rank else 0.5])
    assert result["coef"] == ([0.1] if empty_rank else [0.1, 0.1])


def test_cp_metrics_reject_type_mismatch_on_every_rank(monkeypatch):
    def gather(peers, local, group):
        peers[:] = [{"x": (True, [1.0])}, {"x": (False, [1.0])}]

    monkeypatch.setattr(torch.distributed, "all_gather_object", gather)
    with pytest.raises(TypeError, match="across CP"):
        PPOMetrics().scalar_lists(SimpleNamespace(enabled=True, size=2, group=object()))


@pytest.mark.parametrize("tp_size,cp_size", [(2, 1), (1, 2), (2, 2)])
def test_nonfinite_peer_loss_is_collectively_rejected_before_finalization(monkeypatch, tp_size, cp_size):
    from megatron.core import parallel_state

    group = object()
    monkeypatch.setattr(parallel_state, "get_tensor_model_parallel_group", lambda: group)
    monkeypatch.setattr(parallel_state, "get_tensor_and_context_parallel_group", lambda: group, raising=False)

    def reduce(finite, op, group):
        assert finite.item() == 1
        assert op is torch.distributed.ReduceOp.MIN
        finite.zero_()  # A peer had a non-finite loss.

    monkeypatch.setattr(torch.distributed, "all_reduce", reduce)
    finalized = []

    def run_tree(tree):
        loss = torch.tensor(0.25)
        assert_finite_tree_loss(loss, tp_size=tp_size, cp_size=cp_size)
        return loss

    with pytest.raises(FloatingPointError, match="not be finalized"):
        run_forest(_forest(), run_tree, lambda: finalized.append(True))
    assert not finalized


def test_finite_tree_loss_without_parallelism():
    assert_finite_tree_loss(torch.tensor(0.25), tp_size=1, cp_size=1)
    with pytest.raises(FloatingPointError):
        assert_finite_tree_loss(torch.tensor(float("inf")), tp_size=1, cp_size=1)


def test_layer_spec_normalization_preserves_single_and_block_forms():
    layer = _ModuleSpec(SimpleNamespace(self_attention=object()))
    layers = [layer, layer]
    block = SimpleNamespace(layer_specs=layers)
    assert get_layer_specs(layer) == [layer]
    assert get_layer_specs(block) is layers
    assert get_layer_specs(_ModuleSpec(block)) is layers
    # Block dispatch takes precedence if a provider exposes both attributes.
    assert get_layer_specs(_ModuleSpec(SimpleNamespace(layer_specs=layers, self_attention=object()))) is layers
    with pytest.raises(ValueError, match="must describe either"):
        get_layer_specs(_ModuleSpec())
    with pytest.raises(ValueError, match="non-empty list"):
        get_layer_specs(SimpleNamespace(layer_specs=[]))
    with pytest.raises(TypeError, match="containing layer_specs"):
        get_layer_specs(object())

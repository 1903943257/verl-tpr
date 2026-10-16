"""CPU checks for adapter-owned forest finalization and weighted PPO metrics.

Load these classes without the unrelated Megatron model-construction imports.
The code under test is compiled directly from the production adapter source.
"""

import ast
from collections.abc import Callable
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
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
    names = {"run_forest", "_PPOMetrics", "_get_layer_specs"}
    tree.body = [
        node for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names
    ]
    namespace = dict(globals())
    namespace["ModuleSpec"] = _ModuleSpec
    exec(compile(tree, str(source), "exec"), namespace)
    return namespace["run_forest"], namespace["_PPOMetrics"], namespace["_get_layer_specs"]


run_forest, PPOMetrics, get_layer_specs = _adapter_components()


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

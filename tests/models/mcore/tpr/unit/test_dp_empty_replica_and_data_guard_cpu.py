"""Contracts for filtered batch sizing and empty-local-DP TPR forest sync.

AST-load the production functions so CPU CI does not require Megatron/NPU.
An 8-NPU Actor replay is still required to validate real gradient finalization.
"""

import ast
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from collections.abc import Callable

import pytest
import torch
from torch import Tensor


ROOT = Path(__file__).resolve().parents[5]


def _load_function(path: str, name: str, namespace: dict):
    source = ROOT / path
    tree = ast.parse(source.read_text(encoding="utf-8"))
    fn = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(source), "exec"), namespace)
    return namespace[name]


validate = _load_function(
    "verl/trainer/ppo/v1/trainer_base.py",
    "_validate_filtered_train_batch_size",
    {},
)
run_forest = _load_function(
    "verl/workers/engine/megatron/tpr/adapter.py",
    "run_forest",
    {"Tensor": Tensor, "nullcontext": nullcontext, "Callable": Callable, "Any": Any,
     "ForestExecutionPlan": object, "TreeExecutionPlan": object},
)


@pytest.mark.parametrize("filtered,bs", [(0, 1), (1, 2), (1, 4), (2, 4)])
def test_short_filtered_dataset_fails_early(filtered, bs):
    with pytest.raises(ValueError, match="AFTER filtering"):
        validate(filtered, bs)


@pytest.mark.parametrize("filtered,bs", [(1, 1), (2, 2), (3, 2), (500, 4)])
def test_one_complete_train_batch_is_enough(filtered, bs):
    validate(filtered, bs)


@pytest.mark.parametrize("invalid", [0, -1, True])
def test_bad_batch_size_is_rejected(invalid):
    with pytest.raises(ValueError, match="data.train_batch_size"):
        validate(100, invalid)


def test_filtered_check_runs_after_dataset_construction_before_model_workers():
    source = (ROOT / "verl/trainer/ppo/v1/trainer_base.py").read_text(encoding="utf-8")
    start = source.index("    def _init_dataloader(self):")
    end = source.index("    def _init_resource_pool_mgr(self):", start)
    body = source[start:end]
    assert body.index("self.train_dataset = create_rl_dataset(") < body.index(
        "_validate_filtered_train_batch_size("
    ) < body.index("self.train_dataloader = StatefulDataLoader(")


def _forest(empty: bool):
    return SimpleNamespace(
        trees=(SimpleNamespace(logical_loss_tokens=0 if empty else 2),),
        logical_loss_tokens=0 if empty else 2,
    )


def test_empty_local_forest_calls_finalize_outside_no_sync_and_returns_zero():
    events = []

    @contextmanager
    def no_sync():
        events.append("enter_no_sync")
        yield
        events.append("leave_no_sync")

    loss = run_forest(
        _forest(True),
        lambda _: pytest.fail("empty forest must not execute model"),
        lambda: events.append("finalize"),
        no_sync,
        empty_loss=torch.zeros((), dtype=torch.float32),
    )
    assert loss.item() == 0
    assert events == ["enter_no_sync", "leave_no_sync", "finalize"]


def test_empty_forest_without_global_positive_gate_does_not_finalize():
    calls = []
    with pytest.raises(ValueError, match="no supervised"):
        run_forest(_forest(True), lambda _: None, lambda: calls.append("finalize"))
    assert not calls


def test_empty_loss_must_not_create_backward_graph():
    with pytest.raises(TypeError, match="detached scalar"):
        run_forest(
            _forest(True), lambda _: None, lambda: pytest.fail("bad loss finalized"),
            empty_loss=torch.ones((), requires_grad=True),
        )


def test_supervised_dp_behavior_unmodified():
    events = []
    loss = run_forest(
        _forest(False),
        lambda _: torch.tensor(2.5),
        lambda: events.append("finalize"),
    )
    assert loss.item() == 2.5
    assert events == ["finalize"]


def test_failed_tree_never_syncs_other_dp_replicas():
    events = []

    def fail(_):
        raise RuntimeError("forward/backward failed")

    with pytest.raises(RuntimeError, match="failed"):
        run_forest(_forest(False), fail, lambda: events.append("finalize"))
    assert not events

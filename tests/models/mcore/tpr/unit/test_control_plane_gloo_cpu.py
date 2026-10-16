"""Exercise TPR metric/schema exchange and plan rejection on two real CPU ranks.

No model kernels or Megatron implementation are simulated by this check;
only Megatron's group accessor is bound to the real two-rank Gloo group.
"""

import ast
import runpy
import sys
import types
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist


ROOT = Path(__file__).resolve().parents[5]


def _worker(rank, rendezvous):
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2,
                            timeout=timedelta(seconds=25))
    try:
        namespace = runpy.run_path(str(ROOT / "verl/utils/metric/utils.py"))
        source = ROOT / "verl/workers/engine/megatron/tpr/adapter.py"
        syntax = ast.parse(source.read_text(encoding="utf-8"))
        syntax.body = [node for node in syntax.body
                       if isinstance(node, (ast.ClassDef, ast.FunctionDef))
                       and node.name in ("_PPOMetrics", "_assert_finite_tree_loss")]
        namespace["Any"] = object
        namespace["Tensor"] = torch.Tensor
        exec(compile(syntax, str(source), "exec"), namespace)
        metric, kind, accumulator = (namespace[name] for name in ("Metric", "AggregationType", "_PPOMetrics"))
        runtime = types.SimpleNamespace(enabled=True, size=2, group=dist.group.WORLD)
        metrics = accumulator()
        metrics.add({"mean": metric(kind.MEAN, 3 + rank),
                     "loss": metric(kind.SUM, 0.2 + rank * 0.1), "coef": 0.1}, (2 + rank) / 5)
        result = metrics.scalar_lists(runtime)
        assert result["mean"] == pytest.approx([3.6])
        assert result["loss"] == pytest.approx([0.5])
        assert result["coef"] == [0.1, 0.1]

        metrics = accumulator()
        if rank == 1:
            metrics.add({"loss": metric(kind.SUM, 0.75)}, 1)
        assert metrics.scalar_lists(runtime) == {"loss": [0.75]}

        for name in ("megatron", "megatron.core", "megatron.core.parallel_state"):
            sys.modules[name] = types.ModuleType(name)
        sys.modules["megatron.core.parallel_state"].get_tensor_and_context_parallel_group = lambda: dist.group.WORLD
        preflight = runpy.run_path(str(ROOT / "verl/workers/engine/megatron/tpr/tree.py"))["assert_tp_plan_agreement"]
        preflight("0" * 64, tp_size=1, cp_size=2)
        with pytest.raises(RuntimeError, match="TP×CP schedule mismatch"):
            preflight(str(rank) * 64, tp_size=1, cp_size=2)
        with pytest.raises(FloatingPointError):
            namespace["_assert_finite_tree_loss"](
                torch.tensor(0.25 if rank == 0 else float("inf")), tp_size=1, cp_size=2,
            )
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(), reason="Gloo unavailable")
def test_two_rank_metrics_and_collective_plan_rejection(tmp_path):
    try:
        dist.ProcessGroupGloo.create_default_device()
    except RuntimeError as error:
        if sys.platform == "win32" and "unsupported gloo device" in str(error):
            pytest.skip(f"This Windows torch build cannot create a Gloo device: {error}")
        raise
    torch.multiprocessing.spawn(_worker, args=((tmp_path / "rendezvous").as_uri(),), nprocs=2, join=True)

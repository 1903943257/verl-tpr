"""CPU-only structural contract for the production TPR modules.

No NPU or Megatron import is needed: parse code as Python source. These checks
protect separation of the Ring-only PPO runtime from optional experiments.
"""
from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[5]
ENGINE = ROOT / "verl/workers/engine/megatron/tpr"
MODEL = ROOT / "verl/models/mcore/tpr"
PRODUCTION = ("adapter", "tree", "objective", "executor", "state", "ring", "placement")


def _source(name: str) -> str:
    return (ENGINE / f"{name}.py").read_text(encoding="utf-8")


def test_production_modules_parse() -> None:
    for name in PRODUCTION:
        ast.parse(_source(name), filename=str(ENGINE / f"{name}.py"))


def test_no_experimental_backend_in_production_imports() -> None:
    forbidden = (
        "verl.experimental.tpr",
        "verl.models.mcore.tpr.parallel.backend",
        "verl.models.mcore.tpr.parallel.allgather_attention",
        "verl.models.mcore.tpr.parallel.hybrid_attention",
        "verl.models.mcore.tpr.parallel.ulysses_attention",
    )
    for name in PRODUCTION:
        tree = ast.parse(_source(name))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                assert not any(node.module.startswith(v) for v in forbidden), (name, node.module)
            elif isinstance(node, ast.Import):
                assert not any(alias.name.startswith(v) for alias in node.names for v in forbidden), name


def test_turn_tree_is_only_production_builder() -> None:
    source = _source("tree")
    assert 'tree_policy: str = "qwen_turn"' in source
    assert "def build_trajectory_trees_turn(" in source
    assert "class _TrieNode" not in source
    assert "def build_trajectory_trees(" not in source
    assert "class SegmentObjectiveRef" in source


def test_ppo_executor_no_legacy_ce_gdn_paths() -> None:
    source = _source("executor")
    assert "class SegmentExecutor:" in source
    assert "class FixedTopologyScheduler:" in source
    assert "_ChunkedLanguageModelOutput" not in source
    assert "SegmentLossTerm" not in source
    assert "with_activation_offload" not in source
    assert "self.gdn_states" not in source


def test_single_ring_module_alias() -> None:
    source = (MODEL / "parallel/ring_attention.py").read_text(encoding="utf-8")
    assert "_sys.modules[__name__] = _ring" in source
    assert "class _RingTPRAttention" not in source


def test_engine_routes_to_new_adapter() -> None:
    engine = (ROOT / "verl/workers/engine/megatron/transformer_impl.py").read_text(encoding="utf-8")
    assert "from verl.workers.engine.megatron.tpr.adapter import" in engine

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


def test_single_sequence_shard_implementation() -> None:
    from pathlib import Path

    source = (MODEL / "shard.py").read_text(encoding="utf-8")
    ring = _source("ring")
    assert 'tpr.ring' in source
    assert 'class PaddedSequenceShard:' in ring
    assert 'class RangeSequenceShard:' in ring
    assert 'class PrefixShard:' in ring
    assert "from .state import KVStack" not in ring



def test_dense_prefix_state_is_engine_owned() -> None:
    state = _source("state")
    compat = (MODEL / "prefix_state.py").read_text(encoding="utf-8")
    assert "class KVPrefixState:" in state
    assert "class PrefixStateStack:" in state
    assert "GDNPrefixState" not in state
    assert "def __getattr__(name):" in compat



def test_compatibility_aliases_have_live_targets() -> None:
    """Moved Python module aliases must refer to files in this checkout."""
    paths = [
        MODEL / "shard.py",
        MODEL / "segment_plan.py",
        MODEL / "rope.py",
        MODEL / "kv_stack.py",
        MODEL / "prefix_state.py",
        MODEL / "megatron_adapter.py",
        MODEL / "segment_executor.py",
        MODEL / "objective_adapter.py",
        MODEL / "fixed_topology_scheduler.py",
        MODEL / "tree_plan_builder.py",
        MODEL / "trajectory_tree.py",
        MODEL / "trajectory_tree_turn.py",
        MODEL / "tpr_batch_runner.py",
        MODEL / "engine_adapter.py",
        MODEL / "parallel/backend.py",
        MODEL / "parallel/execution_context.py",
        MODEL / "parallel/engine_runtime.py",
        MODEL / "parallel/ring_attention.py",
        MODEL / "parallel/allgather_attention.py",
        MODEL / "parallel/ulysses_attention.py",
        MODEL / "parallel/hybrid_attention.py",
    ]
    for path in paths:
        source = path.read_text(encoding="utf-8")
        assert len(source.splitlines()) < 100, f"Compatibility file should be thin: {path}"
        tree = ast.parse(source)
        targets = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "_import_module":
                if node.args and isinstance(node.args[0], ast.Constant):
                    targets.append(node.args[0].value)
        for module in targets:
            candidate = ROOT / (module.replace(".", "/") + ".py")
            assert candidate.is_file(), f"{path} points to nonexistent {module}"


def test_ring_retains_qkvmerge_and_padding_contract() -> None:
    source = _source("ring")
    assert "def _can_coalesce_prefix_query(" in source
    assert "def _prefix_full_slices(" in source
    assert "def _physical_block_attention_mask(" in source
    assert "class PaddedSequenceShard:" in source
    assert "def iter_valid_sequence_slices(" in source
    assert "class RingCPBackend:" in source


def test_optional_cp_factories_outside_model_implementation() -> None:
    old_parallel = MODEL / "parallel"
    for name in ("backend", "engine_runtime", "execution_context"):
        source = (old_parallel / f"{name}.py").read_text(encoding="utf-8")
        assert len(source.splitlines()) <= 20, name
        assert "experimental.tpr.legacy.cp." in source


def test_context_is_thin_and_has_no_eager_engine_imports() -> None:
    source = (MODEL / "context.py").read_text(encoding="utf-8")
    assert len(source.splitlines()) <= 230
    assert "class TPRAttentionContext:" in source
    assert "def use_tpr_attention_context(" in source
    assert "def get_past_kv(" in source
    assert "def set_new_kv(" in source
    assert "def assert_new_kv_layers(" in source
    assert "from verl.workers" not in source
    assert "from megatron" not in source


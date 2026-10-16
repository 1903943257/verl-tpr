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
    assert "def build_tree_execution_plans(" in source
    assert "class TrajectoryTree" not in source
    assert "class SegmentLossTerm" not in source
    assert "class SegmentEvent" not in source
    assert "class _TrieNode" not in source
    assert "def build_trajectory_trees(" not in source
    assert "class SegmentObjectiveRef" in source


def test_ppo_executor_no_legacy_ce_gdn_paths() -> None:
    source = _source("executor")
    assert "class SegmentExecutor:" in source
    assert "class FixedTopologyScheduler:" not in source
    assert "class SchedulerState" not in source
    assert "class PhysicalExecution" not in source
    assert "def run(self," in source
    assert "_ChunkedLanguageModelOutput" not in source
    assert "SegmentLossTerm" not in source
    assert "with_activation_offload" not in source
    assert "self.gdn_states" not in source


def test_model_dir_is_only_attention_context_cann() -> None:
    expected = {"__init__.py", "attention.py", "context.py", "rectangular_attention.py"}
    assert {f.name for f in MODEL.glob("*.py")} == expected
    assert not (MODEL / "parallel").exists()
    assert "class RingCPBackend:" in _source("ring")


def test_engine_routes_to_new_adapter() -> None:
    engine = (ROOT / "verl/workers/engine/megatron/transformer_impl.py").read_text(encoding="utf-8")
    assert "from verl.workers.engine.megatron.tpr.adapter import" in engine


def test_engine_owns_shard_tree_state() -> None:
    ring = _source("ring")
    state = _source("state")
    tree = _source("tree")
    ring_classes = {node.name for node in ast.parse(ring).body if isinstance(node, ast.ClassDef)}
    assert "SequenceShard" in ring_classes
    assert not {"PaddedSequenceShard", "RangeSequenceShard", "PrefixShard"} & ring_classes
    eager_state_imports = [node for node in ast.parse(ring).body
                           if isinstance(node, ast.ImportFrom) and node.module == "state"]
    assert not eager_state_imports
    assert "class KVPrefixState:" in state
    assert "class KVStack:" in state
    assert "class PrefixStateStack:" not in state
    assert "class ShardedPastKVAnchors" not in state
    assert "GDNPrefixState" not in state
    assert "class SegmentPlan:" in tree
    assert "class SegmentObjectiveRef:" in tree
    assert "def digest_forest_plan(" in tree


def test_no_obsolete_model_submodule_imports() -> None:
    allowed = {"attention", "context", "rectangular_attention"}
    prefix = "verl.models.mcore.tpr."
    for base in (ROOT / "verl", ROOT / "tests/models/mcore/tpr"):
        for f in base.rglob("*.py"):
            syntax = ast.parse(f.read_text(encoding="utf-8"), filename=str(f))
            for item in ast.walk(syntax):
                mods = []
                if isinstance(item, ast.ImportFrom) and item.module:
                    mods.append(item.module)
                if isinstance(item, ast.Import):
                    mods += [alias.name for alias in item.names]
                for name in mods:
                    if name.startswith(prefix):
                        part = name[len(prefix):].split(".")[0]
                        assert part in allowed, f"{f} still imports deleted {name}"


def test_lazy_exports_resolve_to_real_modules() -> None:
    for path in (MODEL / "__init__.py", ROOT / "verl/experimental/tpr/__init__.py"):
        syntax = ast.parse(path.read_text(encoding="utf-8"))
        assignments = {}
        for n in syntax.body:
            if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
                if n.targets[0].id in ("_PRODUCTION_EXPORTS", "_LEGACY_EXPORTS", "_MODULES"):
                    assignments[n.targets[0].id] = ast.literal_eval(n.value)
                elif n.targets[0].id == "_EXPORTS" and path != MODEL / "__init__.py":
                    assignments[n.targets[0].id] = ast.literal_eval(n.value)
        for entries in assignments.values():
            for symbol, module in entries.items():
                candidate = ROOT / (module.replace(".", "/") + ".py")
                assert candidate.is_file() or (candidate.parent / candidate.stem / "__init__.py").is_file(), (
                    f"{path} export {symbol} points to missing {module}"
                )


def test_ring_retains_qkvmerge_and_padding_contract() -> None:
    source = _source("ring")
    assert "def _can_coalesce_prefix_query(" in source
    assert "def _prefix_full_slices(" in source
    assert "def _physical_block_attention_mask(" in source
    assert "padded_length" in source
    assert "physical_global_ranges" in source
    assert "def iter_valid_sequence_slices(" in source
    assert "class RingCPBackend:" in source


def test_optional_cp_factories_are_experimental() -> None:
    cp = ROOT / "verl/experimental/tpr/legacy/cp"
    for name in ("backend", "engine_runtime", "execution_context"):
        assert (cp / f"{name}.py").is_file()
    assert not (MODEL / "parallel").exists()


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



def test_internal_relative_imports_resolve_after_relocation() -> None:
    """AST-only smoke check: no migrated module imports a vanished sibling."""
    for base in (
        ROOT / "verl/models/mcore/tpr",
        ROOT / "verl/workers/engine/megatron/tpr",
        ROOT / "verl/experimental/tpr",
    ):
        for file in base.rglob("*.py"):
            syntax = ast.parse(file.read_text(encoding="utf-8"))
            for item in ast.walk(syntax):
                if not isinstance(item, ast.ImportFrom) or item.level == 0 or not item.module:
                    continue
                ancestor = file.parent
                for _ in range(item.level - 1):
                    ancestor = ancestor.parent
                target = ancestor.joinpath(*item.module.split("."))
                assert target.with_suffix(".py").is_file() or (target / "__init__.py").is_file(), (
                    f"{file} has unresolved relative import: level={item.level} module={item.module}"
                )

def test_public_dense_symbols_resolve_to_canonical_engine() -> None:
    """No exported production executor or Ring type may point to legacy copies."""
    import ast

    syntax = ast.parse((MODEL / "__init__.py").read_text(encoding="utf-8"))
    exports = next(ast.literal_eval(node.value) for node in syntax.body if
                   isinstance(node, ast.Assign) and
                   any(isinstance(target, ast.Name) and target.id == "_PRODUCTION_EXPORTS" for target in node.targets))
    engine_expected = {
        "RingCPBackend": "ring",
        "SegmentExecutor": "executor",
        "PastKVAnchors": "state",
        "KVStack": "state",
        "SequenceShard": "ring",
    }
    for name, module in engine_expected.items():
        assert exports[name] == "verl.workers.engine.megatron.tpr." + module, name



def test_model_public_api_excludes_legacy_aliases() -> None:
    """Production model exports stay canonical while old explicit imports transition."""
    syntax = ast.parse((MODEL / "__init__.py").read_text(encoding="utf-8"))
    maps = {
        target.id: ast.literal_eval(node.value)
        for node in syntax.body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name) and target.id in ("_PRODUCTION_EXPORTS", "_LEGACY_EXPORTS")
    }
    public = maps["_PRODUCTION_EXPORTS"]
    legacy = maps["_LEGACY_EXPORTS"]
    assert not (public.keys() & legacy.keys())
    assert all(not module.startswith("verl.experimental.tpr") for module in public.values())
    assert all(module.startswith("verl.experimental.tpr") for module in legacy.values())
    assert public["RingCPBackend"] == "verl.workers.engine.megatron.tpr.ring"
    assert public["SegmentExecutor"] == "verl.workers.engine.megatron.tpr.executor"
    assert "SegmentPlan" not in public and "SegmentSpec" not in public
    assert legacy["SegmentPlan"] == "verl.experimental.tpr.legacy.plan"
    assert legacy["SegmentSpec"] == "verl.experimental.tpr.legacy.plan"
    assert "parallel" not in public

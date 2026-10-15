"""Keep TP2CP4 replay's MindSpeed repatch ahead of Engine package imports.

AST-only guard: this catches regression without launching distributed NPU.
"""
from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "parallel"


def _function(tree, name):
    return next(node for node in ast.walk(tree)
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name)


def _engine_imports(nodes):
    return [
        node for node in nodes
        if isinstance(node, ast.ImportFrom) and
        (node.module or "").startswith("verl.workers.engine.megatron.tpr.")
    ]


def test_replay_module_does_not_import_engine_before_repatch():
    tree = ast.parse((ROOT / "test_real_actor_tp2cp4_offline_ppo_npu.py").read_text())
    assert not _engine_imports(tree.body), "Engine import at module scope runs before MindSpeed repatch"
    runtime = _function(tree, "_runtime")
    assert any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and
               node.func.id == "repatch" for node in ast.walk(runtime))
    runner = _function(tree, "test_recorded_full_vs_tpr_one_step")
    imports = _engine_imports(list(ast.walk(runner)))
    assert any(node.module == "verl.workers.engine.megatron.tpr.adapter" for node in imports)


def test_tp_checkpoint_helper_has_no_module_level_engine_imports():
    tree = ast.parse((ROOT / "_qwen3_tp_checkpoint.py").read_text())
    assert not _engine_imports(tree.body), "Checkpoint helper imported before MindSpeed repatch"
    maker = _function(tree, "make_real_qwen3_tp_model")
    assert any(node.module == "verl.workers.engine.megatron.tpr.adapter"
               for node in _engine_imports(list(ast.walk(maker))))

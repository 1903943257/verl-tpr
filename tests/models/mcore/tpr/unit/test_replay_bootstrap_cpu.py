# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Keep the demonstrated launcher compatibility fix, not historical routing."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest


PARALLEL = Path(__file__).resolve().parents[1] / "parallel"


def _sanitize(args):
    # Exercise the real stdlib-only function without importing the NPU runtime.
    path = PARALLEL / "test_real_actor_tp2cp4_offline_ppo_npu.py"
    syntax = ast.parse(path.read_text(encoding="utf-8"))
    syntax.body = [node for node in syntax.body
                   if isinstance(node, ast.FunctionDef) and node.name == "_sanitize_mindspeed_repatch_args"]
    assert len(syntax.body) == 1
    namespace = {}
    exec(compile(syntax, str(path), "exec"), namespace)
    namespace["_sanitize_mindspeed_repatch_args"](args)


@pytest.mark.parametrize("artifact", [False, True])
def test_launcher_compatibility_keeps_valid_arguments(artifact):
    original = {"context_parallel_algo": "megatron_cp_algo", "sequence_parallel": False, "seed": 261013}
    args = SimpleNamespace(**original)
    if artifact:
        vars(args)[""] = "pytest"
    _sanitize(args)
    assert vars(args) == original


@pytest.mark.parametrize("key,value", [("", 12), ("not-a-field", "pytest"), (1, "pytest")])
def test_launcher_compatibility_never_drops_other_malformed_fields(key, value):
    args = SimpleNamespace(context_parallel_algo="megatron_cp_algo")
    vars(args)[key] = value
    with pytest.raises(RuntimeError, match="Unexpected MindSpeed|Invalid MindSpeed"):
        _sanitize(args)


def test_actor_replay_and_comparison_use_only_the_canonical_turn_path():
    actor = (PARALLEL / "test_real_actor_tp2cp4_offline_ppo_npu.py").read_text(encoding="utf-8")
    script = (PARALLEL / "run_old_new_tpr_tp2cp4_ppo.sh").read_text(encoding="utf-8")
    assert 'os.getenv("TPR_TREE_POLICY", "qwen_turn")' in actor
    assert 'os.getenv("TPR_TREE_POLICY", "exact")' not in actor
    assert "verl.models.mcore.tpr.megatron_adapter" not in script
    assert "verl.workers.engine.megatron.tpr.adapter" in script
    assert '${TPR_ACCEPT_GRAD_REL_L2:-0.02}' in script
    assert 'TPR_OFFLINE_SAVE_FULL_GRADS:-1' in script and 'compare_args+=(--require-full-grads)' in script

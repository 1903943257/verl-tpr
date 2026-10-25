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

"""Run isolated TPR CPU contracts without installing Ray, Megatron or NPU.

Usage: python tests/models/mcore/tpr/run_cpu_regressions.py

Only package initializers and unavailable Megatron interfaces are stubbed.
TPR implementations, torch autograd and TensorDict are real. This suite is
not a substitute for native Megatron, distributed kernels or NPU parity gates.
"""

import sys
import types
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[4]
UNIT = (
    "turn_tree_cpu", "ppo_objective_cpu", "ppo_batch_cpu", "ppo_executor_cpu",
    "sequence_shard_cpu", "unified_kv_state_cpu", "leaf_microbatch_cpu", "prefix_state", "kv_stack",
    "shard", "tpr_self_attention", "context", "rectangular_attention_contract",
    "ring_streaming_transport", "ring_coalescing_schedule", "production_architecture",
    "tpr_production_dp2_gate_cpu",
    "segment_plan", "tree_plan_builder", "trajectory_tree_turn", "trajectory_tree",
    "segment_executor", "fixed_topology_scheduler", "segment_external_objective",
    "tp_validation", "rope", "gdn_prefix_state", "tpr_batch_runner",
    "control_plane_gloo_cpu",
    "replay_bootstrap_cpu",
    "dp_placement", "gdn_context",
)
PARALLEL = (
    "ring_cp_attention_contract", "ring_numerics_cpu",
    "cp_padding_contract", "cp_backend_contract",
    "cp_segment_execution_contract", "allgather_cp_attention_contract",
    "hybrid_cp_attention_contract", "ulysses_cp_attention_contract",
    "cp1_n8_perf_contract_cpu", "cp4_ring_timeline_cpu",
)


def main():
    # Narrow performance-data preflight without collecting Megatron/NPU tests.
    cp1_perf_only = "--only-cp1-n8-perf" in sys.argv[1:]
    if cp1_perf_only:
        sys.argv.remove("--only-cp1-n8-perf")
    sys.path.insert(0, str(ROOT))
    for name in (
        "verl", "verl.utils", "verl.models", "verl.models.mcore", "verl.workers",
        "verl.workers.engine", "verl.workers.engine.megatron", "megatron",
        "megatron.core", "megatron.core.inference", "megatron.core.models",
        "megatron.core.models.common", "megatron.core.models.common.embeddings",
        "megatron.core.transformer",
    ):
        package = types.ModuleType(name)
        package.__path__ = [str(ROOT.joinpath(*name.split(".")))]
        sys.modules[name] = package
    interfaces = {
        "megatron.core.inference.contexts": {"BaseInferenceContext": object},
        "megatron.core.models.common.embeddings.rope_utils": {
            "apply_rotary_pos_emb": lambda tensor, *args, **kwargs: tensor,
        },
        "megatron.core.packed_seq_params": {"PackedSeqParams": object},
        "megatron.core.transformer.attention": {
            "SelfAttention": type("SelfAttention", (torch.nn.Module,), {
                "_apply_output_gate": lambda self, tensor, gate: tensor,
            }),
        },
        "megatron.core.typed_torch": {"apply_module": lambda module: module},
        "megatron.core.parallel_state": {"get_tensor_model_parallel_group": lambda: None},
    }
    for name, values in interfaces.items():
        module = types.ModuleType(name)
        module.__dict__.update(values)
        sys.modules[name] = module
    if cp1_perf_only:
        paths = [str(ROOT / "tests/models/mcore/tpr/parallel/test_cp1_n8_perf_contract_cpu.py")]
    else:
        paths = [str(ROOT / f"tests/models/mcore/tpr/unit/test_{name}.py") for name in UNIT]
        paths += [str(ROOT / f"tests/models/mcore/tpr/parallel/test_{name}.py") for name in PARALLEL]
    return pytest.main(["-q", "-p", "no:cacheprovider", *paths, *sys.argv[1:]])


if __name__ == "__main__":
    raise SystemExit(main())

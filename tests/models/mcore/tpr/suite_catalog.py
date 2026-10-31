"""Curated, side-effect-free TPR regression gate registry.

The registry deliberately does not enumerate every historical diagnostic in
parallel/.  An explicit gate is a maintained command, not evidence that every
test in the directory is production-ready.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Gate:
    path: str
    kind: str = "pytest"  # pytest, torchrun, or isolated
    ranks: int = 1
    opt_in: tuple[tuple[str, str], ...] = ()
    required_env: tuple[str, ...] = ()
    expected_failure: bool = False
    note: str = ""
    bootstrap: bool = False


GATES: dict[str, Gate] = {
    "cpu-isolated": Gate(
        "tests/models/mcore/tpr/run_cpu_regressions.py", kind="isolated",
        note="Curated CPU contracts with Megatron interface stubs; not an NPU gate.",
    ),
    "cpu-recompute": Gate(
        "tests/models/mcore/tpr/unit/test_tpr_full_recompute_cpu.py",
        bootstrap=True,
        note="Requires a real Megatron import plus PyTorch; tests a fake block.",
    ),
    "cpu-parallel-report": Gate(
        "tests/models/mcore/tpr/parallel/test_cp_padding_contract.py",
        bootstrap=True,
        note="CPU contract; run under the actual project environment.",
    ),
    "cpu-tree-transport": Gate(
        "tests/models/mcore/tpr/tree_transport/test_tree_transport_cpu.py",
        bootstrap=True,
        note="Session Tree schema, NonTensorStack serialization, Leaf reconstruction and TPR plan parity.",
    ),
    "npu-attention": Gate(
        "tests/models/mcore/tpr/equivalence/test_rectangular_attention_npu.py",
        kind="torchrun", ranks=1,
        note="Rectangular causal attention forward/backward equivalence.",
    ),
    "npu-tree": Gate(
        "tests/models/mcore/tpr/equivalence/test_segment_push_pop_npu.py",
        kind="torchrun", ranks=1,
        note="Shared-prefix push/visit/pop gradients.",
    ),
    "npu-ring": Gate(
        "tests/models/mcore/tpr/parallel/test_ring_cp_attention_npu.py",
        kind="torchrun", ranks=2,
        note="Ring CP kernel correctness; confirm fixture topology before running.",
    ),
    "npu-tp": Gate(
        "tests/models/mcore/tpr/parallel/test_tpr_tp2_npu.py",
        kind="torchrun", ranks=2,
        opt_in=(("TPR_RUN_TP2", "1"),),
        note="Megatron native tensor parallel TPR gate.",
    ),
    "npu-recompute-kv": Gate(
        "tests/models/mcore/tpr/correctness/test_tpr_kv_checkpoint_replay_npu.py",
        kind="torchrun", ranks=1,
        opt_in=(("TPR_RUN_KV_CHECKPOINT_REPLAY", "1"),),
        note="Native checkpoint primitive: explicit KV roots and replay-local context only.",
    ),
    "npu-recompute-negative": Gate(
        "tests/models/mcore/tpr/correctness/test_tpr_native_recompute_contract_npu.py",
        kind="torchrun", ranks=1,
        opt_in=(("TPR_RUN_NATIVE_RECOMPUTE_CONTRACT", "1"),),
        expected_failure=True,
        note="Intentional negative control: naive native recompute is incompatible.",
    ),
    "npu-qwen": Gate(
        "tests/models/mcore/tpr/correctness/test_tpr_qwen3_compatibility_npu.py",
        kind="torchrun", ranks=1,
        opt_in=(("TPR_RUN_QWEN_COMPAT", "1"),),
        note="Qwen3 model compatibility; model assets and NPU are required.",
    ),
    "npu-ring-profile": Gate(
        "tests/models/mcore/tpr/profiling/test_tpr_qwen3_ring_cp_profile_npu.py",
        kind="torchrun", ranks=4,
        opt_in=(("TPR_RUN_QWEN_RING_CP_PROFILE", "1"),),
        note="OPT-IN profiling, not a correctness gate; inspect its case switches.",
    ),
}

SUITES: dict[str, tuple[str, ...]] = {
    "cpu-core": ("cpu-isolated",),
    "cpu-recompute": ("cpu-recompute",),
    "cpu-parallel": ("cpu-parallel-report",),
    "cpu-tree-transport": ("cpu-tree-transport",),
    "npu-core": ("npu-attention", "npu-tree"),
    "npu-ring": ("npu-ring",),
    "npu-tp": ("npu-tp",),
    "npu-recompute-primitive": ("npu-recompute-kv",),
    "npu-recompute-negative": ("npu-recompute-negative",),
    "npu-qwen": ("npu-qwen",),
    "npu-performance": ("npu-ring-profile",),
}

SUITE_DESCRIPTIONS: dict[str, str] = {
    "cpu-core": "Default lightweight CPU regression (existing stubbed runner).",
    "cpu-recompute": "Real Megatron import / fake TransformerBlock recompute contract.",
    "cpu-parallel": "Selected CPU parallel-shape contract.",
    "cpu-tree-transport": "Session Tree transport and Leaf reconstruction correctness (CPU).",
    "npu-core": "Rectangular attention and tree gradients.",
    "npu-ring": "Ring CP numerical contract.",
    "npu-tp": "Megatron TP2 correctness.",
    "npu-recompute-primitive": "KV-aware native checkpoint primitive, NOT full-model acceptance.",
    "npu-recompute-negative": "EXPECTED FAILURE diagnostic; not a green CI gate.",
    "npu-qwen": "Real Qwen3 compatibility.",
    "npu-performance": "Profiling only; not performance acceptance without controlled setup.",
}

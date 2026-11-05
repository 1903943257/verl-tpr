# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Supplement only the two missing routes and audit actual joint merge execution."""

import sys
from types import SimpleNamespace

import pytest

from tests.models.mcore.tpr.parallel import training_multistep_repeat as repeat
from tests.models.mcore.tpr.parallel import training_multistep_runtime as runtime
from tests.models.mcore.tpr.parallel import training_multistep_sixway as sixway
from tests.models.mcore.tpr.parallel.training_multistep_report import finish as finish_original
from tests.models.mcore.tpr.parallel.training_precision_contract import MODES
from tests.models.mcore.tpr.parallel.training_precision_io import write_json
from tests.models.mcore.tpr.parallel.training_qkvmerge_witness import QKVMergeWitness
from tests.models.mcore.tpr.unit.test_training_base_cp1_repeat_cpu import repeated_run, source_run
from tests.models.mcore.tpr.unit.test_training_multistep_cpu import write_route


@pytest.mark.parametrize(
    "mode,expected",
    [
        ("base_cp1", (1, False)),
        ("base_cp2", (2, False)),
        ("tpr_cp1", (1, True)),
        ("tpr_cp2", (2, True)),
        (runtime.BASE_CP1_REPEAT, (1, False)),
        (runtime.TPR_CP2_QKVMERGE, (2, True)),
    ],
)
def test_six_modes_have_explicit_topology(mode, expected):
    assert runtime.mode_topology(mode) == expected


@pytest.mark.parametrize("backward", [False, True])
def test_joint_merge_witness_requires_both_phases_and_restores_hook(monkeypatch, backward):
    module = SimpleNamespace(_trace_ring_block=None)
    monkeypatch.setitem(sys.modules, "verl.workers.engine.megatron.tpr.ring_attention", module)
    event = dict(
        rank=0,
        query_length=512,
        kv_length=512,
        query_chunk=(0, 3),
        kv_chunks=(0, 3),
        query_coalesced=True,
        coalesced=True,
        fa_called=True,
    )
    with QKVMergeWitness() as witness:
        module._trace_ring_block(phase="forward", **event)
        if backward:
            module._trace_ring_block(phase="backward", **event)
            witness.validate()
        else:
            with pytest.raises(AssertionError, match="both Ring"):
                witness.validate()
    assert module._trace_ring_block is None
    with pytest.raises(RuntimeError, match="original hook"):
        with QKVMergeWitness():
            raise RuntimeError("original hook cleanup")
    assert module._trace_ring_block is None


def test_sixway_reuses_cleaned_up_history_and_compares_both_cp2_ranks(tmp_path):
    source, output = tmp_path / "source", tmp_path / "extra"
    fixture, baseline = source_run(source, keep_full=True)
    for mode, factor in zip(MODES[1:], (1.1, 1.02, 1.12), strict=True):
        write_route(source, mode, factor, fixture["controls"], input_sha256=fixture["sha256"])
    for rank in range(2):
        write_json(
            source / "tpr_cp2" / f"runtime_audit_rank{rank}.json",
            dict(status="ten_updates_validated", optimizer=dict(completed_updates=10), precision={}),
        )
    finish_original(source, keep_updates=False)
    before = {str(path.relative_to(source)): path.read_bytes() for path in source.rglob("*") if path.is_file()}
    sixway.preflight(source, output, packages={})
    for mode, subfolder, factor in (
        (runtime.BASE_CP1_REPEAT, "base_cp1_repeat_run", 1),
        (runtime.TPR_CP2_QKVMERGE, "qkvmerge_run", 1.13),
    ):
        root = output / subfolder
        context = repeat.check_source(source, output=root, packages={}, mode=mode)
        assert not context["full_update_available"]
        repeated_run(root, context, factor)
        result = repeat.complete_repeat(root, context)
        assert result["full_update_comparison"] is None
    report = sixway.finish(source, output)
    assert len(report["comparisons_to_base_cp1"]) == 6
    assert report["final_sampled_error_ratio_to_native_cp2"][runtime.BASE_CP1_REPEAT] == 0
    assert report["final_sampled_error_ratio_to_native_cp2"][runtime.TPR_CP2_QKVMERGE] == pytest.approx(1.3, abs=0.005)
    assert report["tpr_cp2_merge_off_vs_on"][-1]["sampled_update"]["update_relative_l2"] == pytest.approx(
        0.01 / 1.12, abs=0.0003
    )
    assert len((output / "summary.md").read_text().splitlines()) < 35
    after = {str(path.relative_to(source)): path.read_bytes() for path in source.rglob("*") if path.is_file()}
    assert before == after
    audit_path = output / "qkvmerge_run/tpr_cp2_qkvmerge/runtime_audit_rank1.json"
    audit = repeat.read(audit_path)
    audit["qkv_merge_execution"]["joint_query_kv_calls"]["backward"] = 0
    write_json(audit_path, audit)
    with pytest.raises(AssertionError, match="joint forward/backward"):
        repeat.complete_repeat(output / "qkvmerge_run", repeat.check_source(source, mode=runtime.TPR_CP2_QKVMERGE))

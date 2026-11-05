# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Numerical and ownership gates for the real-engine experiment observer."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from tests.models.mcore.tpr.parallel import training_precision_io as io
from tests.models.mcore.tpr.parallel.training_precision_contract import (
    PREFIX,
    SUFFIX,
    actor_batch,
    audit_wgrad_completion,
    synthetic_tokens,
    validate_forest,
)
from tests.models.mcore.tpr.parallel.training_precision_report import _logprobs, complete_mode, finish


def test_real_production_builder_retains_1k_boundary_and_first_response_loss():
    from verl.workers.engine.megatron.tpr.tree import build_tree_execution_plans

    tokens = synthetic_tokens(151936, 17)
    labels = torch.zeros(2, SUFFIX)
    batch = actor_batch(tokens, labels, torch.ones_like(labels))
    assert torch.equal(tokens[0, :PREFIX], tokens[1, :PREFIX])
    assert tokens[0, PREFIX] != tokens[1, PREFIX]
    assert validate_forest(batch)["lengths"] == [1024, 1024, 1024]
    forest = build_tree_execution_plans(("precision_0_0", "precision_0_1"), batch)
    refs = forest.trees[0].objective_refs
    boundary = [ref for ref in refs if ref.response_offset == 0]
    assert len(refs) == 2048
    assert {(ref.sample_row, ref.segment_id, ref.query_offset) for ref in boundary} == {
        (0, 0, PREFIX - 1),
        (1, 0, PREFIX - 1),
    }
    assert {ref.target_token_id for ref in boundary} == set(tokens[:, PREFIX].tolist())


class _Model(nn.Module):
    def __init__(self, gradients):
        super().__init__()
        for index, gradient in enumerate(gradients):
            parameter = nn.Parameter(torch.zeros_like(gradient, dtype=torch.bfloat16))
            parameter.main_grad = gradient.clone()
            self.register_parameter(f"p{index}", parameter)


class _Owner:
    def __init__(self, model, spans):
        self.model_param_gbuf_map = {
            parameter: span for parameter, span in zip(model.parameters(), spans, strict=False) if span
        }

    def _get_model_param_range_map(self, parameter):
        start, end = self.model_param_gbuf_map[parameter]
        return {"param": SimpleNamespace(start=start, end=end)}


def test_reduce_scatter_unowned_tails_never_enter_comparison(tmp_path, monkeypatch):
    monkeypatch.setattr(io, "BLOCK", 4)
    expected = [torch.arange(1, 12).float(), torch.arange(2, 7).float()]
    baseline = _Model(expected)
    full = [(0, 11), (0, 5)]
    reference = tmp_path / "base"
    saved = io.observe_gradients(baseline, _Owner(baseline, full), reference, 0, {}, save_reference=True)
    io.index_reference(reference, [saved])
    candidate_values = [gradient * 1.01 + 0.03 for gradient in expected]
    reports = []
    for rank, spans in enumerate(([(0, 6), None], [(6, 11), (0, 5)])):
        candidate = _Model([torch.full_like(value, float("nan")) for value in expected])
        for parameter, values, span in zip(candidate.parameters(), candidate_values, spans, strict=True):
            if span:
                parameter.main_grad[span[0] : span[1]] = values[span[0] : span[1]]
        reports.append(
            io.observe_gradients(
                candidate,
                _Owner(candidate, spans),
                tmp_path / "cp2",
                rank,
                {"base": reference},
                save_reference=True,
            )
        )
    io.index_reference(tmp_path / "cp2", reports)
    metrics = io.merge_stats([report["stats"]["base"] for report in reports], 16)
    a, b = (torch.cat(values).double().numpy() for values in (expected, candidate_values))
    assert metrics["gradient_relative_l2"] == pytest.approx(np.linalg.norm(b - a) / np.linalg.norm(a), rel=1e-14)
    assert metrics["gradient_cosine"] == pytest.approx(np.dot(a, b) / np.linalg.norm(a) / np.linalg.norm(b))
    # Read across both the saved block boundary and CP ownership boundary.
    reader = io.GradientReference(tmp_path / "cp2", io.parameter_layout(baseline))
    torch.testing.assert_close(reader.read(0, 3, 9), candidate_values[0][3:9], rtol=0, atol=0)
    # Export is read-only, including the deliberately poisoned unowned part.
    assert torch.isnan(candidate.p0.main_grad[:6]).all()


def test_bf16_grad_cannot_be_promoted_and_reported_as_native_fp32(tmp_path):
    model = _Model([torch.ones(4)])
    model.p0.main_grad = model.p0.main_grad.bfloat16()
    with pytest.raises(AssertionError, match="FP32 main_grad"):
        io.observe_gradients(model, _Owner(model, [(0, 4)]), tmp_path, 0, {}, save_reference=False)


@pytest.mark.parametrize("second", [(2, 5), (4, 5)])
def test_duplicate_or_missing_optimizer_owned_elements_are_rejected(second):
    layout = [dict(name="p", shape=[5], elements=5)]
    reports = [dict(layout=layout, owned=[dict(index=0, name="p", start=a, end=b)]) for a, b in [(0, 3), second]]
    with pytest.raises(AssertionError, match="overlaps or has gaps"):
        io.validate_owned_coverage(layout, reports)


def test_reference_index_rejects_missing_saved_blocks(tmp_path):
    layout = [dict(name="p", shape=[5], elements=5)]
    reports = [
        dict(
            layout=layout,
            owned=[dict(index=0, name="p", start=0, end=5)],
            parts=[dict(index=0, start=0, end=4, file="partial.pt")],
        )
    ]
    with pytest.raises(AssertionError, match="Incomplete saved reference"):
        io.index_reference(tmp_path, reports)
    assert not (tmp_path / "index.json").exists()


def test_gradient_serialization_uses_string_filenames_for_npu_mmap(tmp_path, monkeypatch):
    original_load, original_save = torch.load, torch.save
    calls = []

    def npu_load(filename, *args, **kwargs):
        if not isinstance(filename, str):
            raise TypeError("f must be a string filename in order to use mmap argument")
        calls.append(kwargs.get("mmap"))
        return original_load(filename, *args, **kwargs)

    def npu_save(value, filename, *args, **kwargs):
        assert isinstance(filename, str)
        return original_save(value, filename, *args, **kwargs)

    monkeypatch.setattr(torch, "load", npu_load)
    monkeypatch.setattr(torch, "save", npu_save)
    model = _Model([torch.arange(1, 6).float()])
    owner = _Owner(model, [(0, 5)])
    saved = io.observe_gradients(model, owner, tmp_path / "base", 0, {}, save_reference=True)
    io.index_reference(tmp_path / "base", [saved])
    observed = io.observe_gradients(
        model,
        owner,
        tmp_path / "candidate",
        0,
        {"base": tmp_path / "base"},
        save_reference=False,
    )
    assert calls == [True]
    assert io.merge_stats([observed["stats"]["base"]], 5)["gradient_relative_l2"] == 0


def test_native_fused_wgrad_requires_backward_completion_flags():
    module = nn.Linear(3, 2, bias=False)
    module.gradient_accumulation_fusion = True
    with pytest.raises(AssertionError, match="did not mark"):
        audit_wgrad_completion(module, fusion=True)
    module.weight.grad_added_to_main_grad = True
    assert audit_wgrad_completion(module, fusion=True)["native_fused_wgrad_completed"]


def test_cp_logprob_observer_rejects_double_counting():
    values = [[row, offset, -12.0] for row in range(2) for offset in range(SUFFIX)]
    first = {"logprobs": values[:SUFFIX]}
    second = {"logprobs": values[SUFFIX:]}
    assert np.asarray(_logprobs([first, second])).shape == (2, SUFFIX)
    with pytest.raises(AssertionError, match="Duplicate"):
        _logprobs([first, second, first])


def test_weight_witness_hashes_values_outside_a_sampled_prefix():
    model = _Model([torch.ones(100)])
    before = io.weight_digest(model)
    with torch.no_grad():
        model.p0[-1] = 7
    assert io.weight_digest(model) != before
    # Also ensure diagnostic metadata is strict JSON (no NaN placeholders).
    assert json.loads(json.dumps(io.parameter_layout(model)))[0]["elements"] == 100


def test_four_mode_report_compares_owned_shards_and_cleans_only_manifest_files(tmp_path, monkeypatch):
    monkeypatch.setattr(io, "BLOCK", 4)
    common = dict(
        prefix_length=PREFIX,
        suffix_length=SUFFIX,
        input_sha256="same-input",
        initial_weights_sha256="same-weights",
        objective="same-native-PPO",
        controls={"fusion": True},
        packages={"megatron-core": "test"},
        precision=dict(
            main_grad_dtype="torch.float32",
            gradient_accumulation_fusion=True,
            grad_reduce_in_fp32=True,
            use_distributed_optimizer=True,
        ),
    )
    initial = torch.arange(1, 10).float()
    for index, (mode, cp) in enumerate((("base_cp1", 1), ("base_cp2", 2), ("tpr_cp1", 1), ("tpr_cp2", 2))):
        folder = tmp_path / mode
        folder.mkdir()
        references = {name: tmp_path / name / "grads" for name in ("base_cp1", "base_cp2") if name != mode}
        if index == 0:
            references = {}
        for rank in range(cp):
            model = _Model([initial * (1 + index * 0.01)])
            spans = [(0, 9)] if cp == 1 else [(0, 4)] if rank == 0 else [(4, 9)]
            gradients = io.observe_gradients(
                model,
                _Owner(model, spans),
                folder / "grads",
                rank,
                references,
                save_reference=mode.startswith("base"),
            )
            values = [[row, offset, -12.0 - index * 0.001] for row in range(2) for offset in range(SUFFIX)]
            if rank != 0:
                values = []
            io.write_json(
                folder / f"rank{rank}.json",
                {
                    **common,
                    "mode": mode,
                    "cp": cp,
                    "rank": rank,
                    "loss": -0.5 + index * 0.001,
                    "gradients": gradients,
                    "logprobs": values,
                    "calls": {},
                    "backward": {},
                    "peak_allocated_bytes": 17,
                },
            )
        complete_mode(tmp_path, mode, cp, io.parameter_layout(model))
    unrelated = tmp_path / "base_cp1/grads/unrelated.pt"
    unrelated.write_bytes(b"not an experiment reference")
    report = finish(tmp_path, keep_gradients=False)
    assert len(report["pairs"]) == 5
    assert report["error_ratio_to_native_cp2"]["tpr_cp1"] == pytest.approx(2, rel=1e-5)
    assert report["error_ratio_to_native_cp2"]["tpr_cp2"] == pytest.approx(3, rel=1e-5)
    assert unrelated.read_bytes() == b"not an experiment reference"
    assert not list((tmp_path / "base_cp2/grads").glob("rank*/*.pt"))
    assert (tmp_path / "comparison.json").is_file()

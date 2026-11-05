# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Numerical oracles for native master ownership, real updates and drift reports."""

import json
import sys
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from tests.models.mcore.tpr.parallel import training_multistep_io as io
from tests.models.mcore.tpr.parallel.training_multistep_npu import controls_from_environment
from tests.models.mcore.tpr.parallel.training_multistep_report import complete_mode, finish, probe_metrics
from tests.models.mcore.tpr.parallel.training_multistep_resume import prepare_resume
from tests.models.mcore.tpr.parallel.training_multistep_runtime import (
    CHECKPOINTS,
    STEPS,
    fixture_digest,
    input_item,
    native_update,
)
from tests.models.mcore.tpr.parallel.training_precision_contract import MODES, SUFFIX
from tests.models.mcore.tpr.parallel.training_precision_io import (
    index_reference,
    merge_stats,
    weight_digest,
    write_json,
)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.p0 = nn.Parameter(torch.ones(11, dtype=torch.bfloat16))
        self.p1 = nn.Parameter(torch.arange(1, 6, dtype=torch.bfloat16))
        for parameter in self.parameters():
            # Poison the model's .grad/main_grad: the observer must read native masters.
            parameter.grad = torch.full_like(parameter, float("nan"))
            parameter.main_grad = torch.full_like(parameter, float("nan"), dtype=torch.float32)


class Owner:
    def __init__(self, model, spans):
        parameters = list(model.parameters())
        self.model_param_gbuf_map = {p: span for p, span in zip(parameters, spans, strict=True) if span}
        masters = [p.detach().flatten()[lo:hi].float().clone() for p, (lo, hi) in self.model_param_gbuf_map.items()]
        self.optimizer = torch.optim.AdamW(masters, lr=1e-3)
        self.model_param_group_index_map = {p: (0, i) for i, p in enumerate(self.model_param_gbuf_map)}

    def _get_model_param_range_map(self, parameter):
        start, end = self.model_param_gbuf_map[parameter]
        return {"param": SimpleNamespace(start=start, end=end)}


def advance(model, owner, factor):
    for index, _name, start, end, master in io.owned_master_ranges(model, owner):
        # All updates below BF16 resolution for these initial weights.
        master.add_(torch.arange(start + 1, end + 1).float() * (index + 1) * factor * 1e-5)


def test_full_master_update_uses_true_optimizer_shards_and_sub_bf16_updates(tmp_path, monkeypatch):
    monkeypatch.setattr(io, "BLOCK", 4)
    model = Model()
    owner = Owner(model, [(0, 11), (0, 5)])
    initial = io.InitialWeights(model, owner)
    assert initial.digest == weight_digest(model)
    advance(model, owner, 1)
    reference = tmp_path / "base"
    saved = initial.full_update(model, owner, reference, 0, {}, save_reference=True)
    index_reference(reference, [saved])
    candidates, samples = [], []
    for rank, spans in enumerate(([(0, 6), None], [(6, 11), (0, 5)])):
        candidate, other = Model(), None
        other = Owner(candidate, spans)
        start = io.InitialWeights(candidate, other)
        advance(candidate, other, 1.2)
        samples.append(start.samples(candidate, other))
        candidates.append(
            start.full_update(candidate, other, tmp_path / "candidate", rank, {"base": reference}, save_reference=False)
        )
    values = io.merge_samples(initial.layout, samples)
    original = io.merge_samples(initial.layout, [initial.samples(model, owner)])
    assert torch.equal(model.p0, torch.ones_like(model.p0))
    assert values.norm() > 0 and not torch.equal(original, values)
    merged = merge_stats([item["stats"]["base"] for item in candidates], 16)
    expected = np.linalg.norm((values - original).double().numpy()) / np.linalg.norm(original.double().numpy())
    assert merged["gradient_relative_l2"] == pytest.approx(expected, rel=1e-12)
    assert not (tmp_path / "candidate").exists()


def test_global_sampling_rejects_duplicate_or_missing_ownership(monkeypatch):
    monkeypatch.setattr(io, "SAMPLES_PER_PARAMETER", 3)
    model = Model()
    full = Owner(model, [(0, 11), (0, 5)])
    initial = io.InitialWeights(model, full)
    advance(model, full, 1)
    single = initial.samples(model, full)
    distributed = []
    for spans in ([(0, 4), (0, 3)], [(4, 11), (3, 5)]):
        candidate = Model()
        owner = Owner(candidate, spans)
        start = io.InitialWeights(candidate, owner)
        advance(candidate, owner, 1)
        distributed.append(start.samples(candidate, owner))
    assert torch.equal(io.merge_samples(initial.layout, [single]), io.merge_samples(initial.layout, distributed))
    with pytest.raises(AssertionError, match="missing/duplicate"):
        io.merge_samples(initial.layout, [single, single])
    with pytest.raises(AssertionError):
        io.merge_samples(initial.layout, distributed[:1])


def test_initial_master_must_match_checkpoint_and_optimizer_must_be_fresh():
    model = Model()
    owner = Owner(model, [(0, 11), (0, 5)])
    master = owner.optimizer.param_groups[0]["params"][0]
    master[0] += 1e-5
    with pytest.raises(AssertionError, match="differs"):
        io.InitialWeights(model, owner)
    master[0] = 1
    owner.optimizer.param_groups[0]["step"] = 1
    with pytest.raises(AssertionError, match="fresh"):
        io.InitialWeights(model, owner)


def test_adam_state_audit_supports_parameter_and_mindspeed_group_counters():
    model = Model()
    owner = Owner(model, [(0, 11), (0, 5)])
    for _ in range(STEPS):
        for master in owner.optimizer.param_groups[0]["params"]:
            master.grad = torch.ones_like(master)
        owner.optimizer.step()
    assert io.audit_adam_updates(owner, STEPS)["moment_dtype"] == "torch.float32"
    for state in owner.optimizer.state.values():
        state.pop("step")
    owner.optimizer.param_groups[0]["step"] = torch.tensor(STEPS)
    io.audit_adam_updates(owner, STEPS)
    owner.optimizer.param_groups[0]["step"] = STEPS - 1
    with pytest.raises(AssertionError, match="counter"):
        io.audit_adam_updates(owner, STEPS)
    owner.optimizer.param_groups[0]["step"] = STEPS
    next(iter(owner.optimizer.state.values()))["exp_avg"] = torch.zeros(11, dtype=torch.bfloat16)
    with pytest.raises(AssertionError, match="FP32 Adam"):
        io.audit_adam_updates(owner, STEPS)


class Engine:
    """CPU training stand-in verifies calls/order against independent native AdamW."""

    def __init__(self):
        self.weight = nn.Parameter(torch.tensor([0.2, -0.4]))
        self.optimizer = torch.optim.AdamW([self.weight], lr=1e-3)
        self.optimizer_config = SimpleNamespace(clip_grad=1.0)
        self.calls = []

    def train_mode(self):
        return nullcontext()

    def optimizer_zero_grad(self):
        self.calls.append("zero")
        self.optimizer.zero_grad()

    def forward_backward_batch(self, batch, loss_function, *, forward_only):
        assert not forward_only
        self.calls.append("backward")
        loss = ((self.weight - batch) ** 2).sum()
        loss.backward()
        return {"loss": loss.detach()}

    def optimizer_step(self):
        self.calls.append("step")
        norm = torch.nn.utils.clip_grad_norm_([self.weight], self.optimizer_config.clip_grad)
        self.optimizer.step()
        return norm

    def lr_scheduler_step(self):
        self.calls.append("scheduler")


def test_ten_updates_preserve_native_clipping_and_adam_order():
    engine = Engine()
    reference = nn.Parameter(engine.weight.detach().clone())
    optimizer = torch.optim.AdamW([reference], lr=1e-3)
    for step in range(STEPS):
        batch = torch.tensor([step * 0.1, 0.5])
        optimizer.zero_grad()
        ((reference - batch) ** 2).sum().backward()
        norm = torch.nn.utils.clip_grad_norm_([reference], 1.0)
        optimizer.step()

        def observer():
            assert engine.calls[-1] == "backward"
            assert engine.weight.grad is not None

        _, metrics = native_update(engine, batch, None, lambda: None, after_backward=observer)
        assert metrics["grad_norm"] == pytest.approx(float(norm))
        assert metrics["clip_coefficient_from_native_norm"] <= 1
        assert torch.equal(engine.weight, reference)
    assert engine.calls == ["zero", "backward", "step", "scheduler"] * STEPS
    assert float(engine.optimizer.state[engine.weight]["step"]) == STEPS


def test_fixture_is_reproducible_varied_and_hashes_frozen_actor_labels():
    items = [input_item(151936, 17, ordinal) for ordinal in range(STEPS + 1)]
    for ordinal, item in enumerate(items):
        assert torch.equal(item["tokens"], input_item(151936, 17, ordinal)["tokens"])
        assert torch.equal(item["tokens"][0, :1024], item["tokens"][1, :1024])
        assert item["tokens"][0, 1024] != item["tokens"][1, 1024]
        item["old_log_probs"] = torch.zeros(2, SUFFIX)
    assert not torch.equal(items[0]["tokens"], items[1]["tokens"])
    fixture = dict(controls={}, initial_weights_sha256="weights", batches=items[:STEPS], probe=items[-1])
    before = fixture_digest(fixture)
    items[-1]["old_log_probs"][0, 0] += 0.01
    assert fixture_digest(fixture) != before


def write_route(root, mode, factor, controls):
    folder = root / mode
    folder.mkdir()
    cp = 2 if mode.endswith("cp2") else 1
    for rank in range(cp):
        model = Model()
        spans = [(0, 11), (0, 5)] if cp == 1 else ([(0, 6), None] if rank == 0 else [(6, 11), (0, 5)])
        owner = Owner(model, spans)
        initial = io.InitialWeights(model, owner)
        samples = {}
        for step in (1, 5, 10):
            for index, _name, lo, hi, master in io.owned_master_ranges(model, owner):
                master.copy_(initial.values[f"p{index}"][lo:hi].float())
            advance(model, owner, factor * step)
            samples[step] = initial.samples(model, owner)
        torch.save(samples, str(folder / f"samples_rank{rank}.pt"))
        references = {name: root / name / "updates" for name in ("base_cp1", "base_cp2") if name != mode}
        if mode == "base_cp1":
            references = {}
        update = initial.full_update(
            model, owner, folder / "updates", rank, references, save_reference=mode.startswith("base")
        )
        report = dict(
            mode=mode,
            cp=cp,
            controls=controls,
            input_sha256="fixture",
            initial_weights_sha256="weights",
            packages={},
            source={},
            full_update=update,
            steps=[
                dict(
                    step=step,
                    loss=-0.5,
                    grad_norm=0.1,
                    learning_rates_used=[1e-6],
                    peak_allocated_bytes=1024,
                    forward_backward_seconds=2.0,
                    optimizer_seconds=0.5,
                )
                for step in range(1, STEPS + 1)
            ],
            timings=dict(total_seconds=60),
        )
        write_json(folder / f"rank{rank}.json", report)
    probes = {step: torch.full((2, SUFFIX), -10.0 + (cp - 1) * 0.03 + factor * step * 0.001) for step in CHECKPOINTS}
    torch.save(probes, str(folder / "probes.pt"))
    complete_mode(root, mode, cp)


def test_four_route_report_and_safe_final_cleanup(tmp_path):
    for mode, factor in zip(MODES, (1, 1.1, 1.02, 1.12), strict=True):
        write_route(tmp_path, mode, factor, {})
    unrelated = tmp_path / "base_cp1/updates/unrelated.pt"
    unrelated.write_bytes(b"preserve")
    report = finish(tmp_path, keep_updates=False)
    assert len(report["pairs"]) == 5
    assert report["final_update_error_ratio_to_native_cp2"]["tpr_cp1"] == pytest.approx(0.2, abs=0.002)
    assert report["final_update_error_ratio_to_native_cp2"]["tpr_cp2"] == pytest.approx(1.2, abs=0.002)
    assert report["modes"]["base_cp1"]["training_seconds"] == 25
    assert unrelated.read_bytes() == b"preserve"
    assert not list((tmp_path / "base_cp1/updates/rank0").glob("*.pt"))
    assert (tmp_path / "base_cp1/samples.pt").is_file()
    assert len((tmp_path / "comparison.json").read_text().splitlines()) < 400


def test_cleanup_rejects_escaped_manifest_before_deleting_any_reference(tmp_path):
    for mode, factor in zip(MODES, (1, 1.1, 1.02, 1.12), strict=True):
        write_route(tmp_path, mode, factor, {})
    outside = tmp_path / "outside.pt"
    outside.write_bytes(b"preserve")
    manifest = tmp_path / "base_cp2/updates/index.json"
    value = json.loads(manifest.read_text())
    value["parts"][0][0]["file"] = str(outside)
    write_json(manifest, value)
    with pytest.raises(ValueError, match="Unsafe"):
        finish(tmp_path, keep_updates=False)
    assert outside.read_bytes() == b"preserve"
    assert list((tmp_path / "base_cp1/updates/rank0").glob("*.pt"))


def test_probe_change_does_not_mislabel_initial_cp_layout_gap_as_training_drift():
    base = {step: torch.full((2, SUFFIX), -10.0 + step * 0.001) for step in CHECKPOINTS}
    other = {step: value + 0.5 for step, value in base.items()}
    metrics = probe_metrics(base, other, 10)
    assert metrics["logprob_mean_abs"] == pytest.approx(0.5)
    assert metrics["logprob_change_mean_abs"] == 0


def test_resume_validates_before_archiving_and_preserves_completed_route(tmp_path, monkeypatch):
    monkeypatch.setenv("TPR_TRAIN10_LR", "1e-6")
    controls = controls_from_environment()
    item = input_item(151936, controls["seed"], 0)
    item["old_log_probs"] = torch.zeros(2, SUFFIX)
    fixture = dict(controls=controls, initial_weights_sha256="weights", batches=[item] * STEPS, probe=item)
    fixture["sha256"] = fixture_digest(fixture)
    torch.save(fixture, str(tmp_path / "actor_fixture.pt"))
    write_route(tmp_path, "base_cp1", 1, controls)
    result_path = tmp_path / "base_cp1/result.json"
    result = json.loads(result_path.read_text())
    result["input_sha256"] = fixture["sha256"]
    write_json(result_path, result)
    original = result_path.read_bytes()
    failed = tmp_path / "base_cp2"
    failed.mkdir()
    (failed / "run.log").write_text("original log")
    monkeypatch.setenv("TPR_TRAIN10_LR", "2e-6")
    with pytest.raises(AssertionError, match="controls"):
        prepare_resume(tmp_path)
    assert (failed / "run.log").is_file()
    monkeypatch.setenv("TPR_TRAIN10_LR", "1e-6")
    completed, pending = prepare_resume(tmp_path)
    assert completed == ["base_cp1"] and pending == list(MODES[1:])
    assert not failed.exists()
    assert next(tmp_path.glob("base_cp2.failed-*/run.log")).read_text() == "original log"
    assert result_path.read_bytes() == original


@pytest.mark.parametrize("failure", [None, "skipped_core", "forward_error", "leaked_context"])
def test_whole_sequence_probe_observes_native_cores_and_cleans_up_hooks(monkeypatch, failure):
    from tests.models.mcore.tpr.parallel import training_multistep_runtime as runtime

    monkeypatch.setitem(
        sys.modules,
        "verl.models.mcore.tpr.context",
        SimpleNamespace(get_tpr_attention_context=lambda: object() if failure == "leaked_context" else None),
    )
    monkeypatch.setitem(sys.modules, "verl.utils.megatron_utils", SimpleNamespace(unwrap_model=lambda model: model))
    cores = [nn.Identity(), nn.Identity()]
    model = SimpleNamespace(
        decoder=SimpleNamespace(
            layers=[SimpleNamespace(self_attention=SimpleNamespace(core_attention=c)) for c in cores]
        )
    )

    class ProbeEngine:
        module = [model]

        def eval_mode(self):
            return nullcontext()

        def forward_backward_batch(self, _batch, _loss, *, forward_only):
            assert forward_only
            for core in cores[:1] if failure == "skipped_core" else cores:
                core(torch.ones(1))
            if failure == "forward_error":
                raise RuntimeError("forward failed")
            return {}

    monkeypatch.setattr(runtime, "item_batch", lambda item: item)
    monkeypatch.setattr(runtime, "_native_response_logprobs", lambda *_: torch.zeros(2, SUFFIX))
    if failure is None:
        assert len(runtime.native_probe(ProbeEngine(), {}, 0)) == 2
    else:
        with pytest.raises((AssertionError, RuntimeError)):
            runtime.native_probe(ProbeEngine(), {}, 0)
    assert all(not core._forward_pre_hooks for core in cores)

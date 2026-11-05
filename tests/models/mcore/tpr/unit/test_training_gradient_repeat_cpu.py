# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Full-vector oracle, native-call ordering and safe temporary-file lifecycle."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tests.models.mcore.tpr.parallel import training_gradient_repeat_io as repeat
from tests.models.mcore.tpr.parallel import training_precision_io as io
from tests.models.mcore.tpr.parallel.training_gradient_repeat_npu import native_gradient_getters, native_norms
from tests.models.mcore.tpr.unit.test_training_base_cp1_repeat_cpu import source_run
from tests.models.mcore.tpr.unit.test_training_path_precision_cpu import _Model, _Owner


def snapshots(tmp_path, monkeypatch, *, identical=False):
    monkeypatch.setattr(io, "BLOCK", 3)
    monkeypatch.setattr(repeat, "BLOCK", 3)
    first = [torch.tensor([1.0, -2, 0, 0, 1e-8, -1e-8, 5, 1]), torch.zeros(2)]
    second = first if identical else [torch.tensor([-1.0, -2, 1, 0, -1e-8, 1e-8, 6, 0]), torch.zeros(2)]
    models = [_Model(values) for values in (first, second)]
    owners = [_Owner(model, [(0, 8), (0, 2)]) for model in models]
    folder = tmp_path / "first/grads"
    saved = io.observe_gradients(models[0], owners[0], folder, 0, {}, save_reference=True)
    io.index_reference(folder, [saved])
    detail = repeat.compare_gradients(models[1], owners[1], folder, 1e-6)
    io.write_json(tmp_path / "detail.json", detail)
    return first, second, detail


@pytest.mark.parametrize("identical", [False, True])
def test_full_element_statistics_match_independent_vector_oracle(tmp_path, monkeypatch, identical):
    first, second, detail = snapshots(tmp_path, monkeypatch, identical=identical)
    result = repeat.summarize_gradients(detail, 1e-6)
    x, y = [torch.cat(values).double().numpy() for values in (first, second)]
    assert result["elements"] == len(x)
    assert result["gradient_relative_l2"] == pytest.approx(np.linalg.norm(y - x) / np.linalg.norm(x), abs=1e-15)
    assert result["gradient_cosine"] == pytest.approx(np.dot(x, y) / np.linalg.norm(x) / np.linalg.norm(y))
    assert result["gradient_max_abs"] == np.abs(y - x).max()
    signs = result["signs"]
    assert signs["opposite_sign_elements"] == (0 if identical else 3)
    assert signs["zero_transition_elements"] == (0 if identical else 2)
    assert signs["changed_elements"] == (0 if identical else 6)
    if not identical:
        assert signs["both_nonzero_elements"] == 5
        assert signs["small_reference_elements"] == 6  # Includes the all-zero second parameter.
        assert signs["small_reference_opposite_sign_elements"] == 2
        assert signs["small_reference_zero_transition_elements"] == 1
        assert signs["opposite_sign_fraction_both_nonzero"] == pytest.approx(3 / 5)


def test_corrupt_gradient_and_missing_owned_coordinates_are_rejected(tmp_path, monkeypatch):
    snapshots(tmp_path, monkeypatch)
    model = _Model([torch.ones(8), torch.zeros(2)])
    owner = _Owner(model, [(0, 8), (0, 2)])
    model.p0.main_grad[0] = float("nan")
    with pytest.raises(FloatingPointError, match="Nonfinite"):
        repeat.compare_gradients(model, owner, tmp_path / "first/grads", 1e-6)
    model.p0.main_grad[0] = 1
    with pytest.raises(AssertionError, match="Missing owned gradient"):
        repeat.compare_gradients(model, _Owner(model, [(0, 7), (0, 2)]), tmp_path / "first/grads", 1e-6)


@pytest.mark.parametrize("keep", [False, True])
@pytest.mark.parametrize("corrupt", [None, "input_sha256", "native_norm_elements", "optimizer_step", "audit_status"])
def test_completed_report_retains_scalars_and_only_removes_own_manifest_blocks(tmp_path, monkeypatch, keep, corrupt):
    _a, _b, detail = snapshots(tmp_path, monkeypatch)
    source = tmp_path / "historical.pt"
    source.write_bytes(b"do not touch")
    unrelated = tmp_path / "first/grads/unrelated.pt"
    unrelated.write_bytes(b"also keep")
    paths = repeat.reference_paths(tmp_path / "first/grads")
    precision = {key: True for key in repeat.PRECISION_KEYS}
    precision["topology_tp_pp_cp_dp"] = [1] * 4
    precision["transformer_config"] = {"params_dtype": "torch.bfloat16"}
    for run in ("first", "second"):
        folder = tmp_path / run
        folder.mkdir(exist_ok=True)
        io.write_json(
            folder / "result.json",
            dict(
                run=run,
                controls={},
                input_sha256="inputs",
                initial_weights_sha256="weights",
                packages={},
                precision=precision,
                execution_environment={},
                small_gradient_threshold=1e-6,
                native_norm_elements=10,
                model_elements=10,
                optimizer_step=False,
                gradient_stage="after_native_finalize_and_prepare_before_clip",
                native_norms=[5.5, 5.5],
                loss=0.2,
            ),
        )
        io.write_json(
            folder / "runtime_audit_rank0.json", dict(status="validated_before_clip_no_update", optimizer_step=False)
        )
        torch.save(
            dict(probe0=torch.ones(2, 1024), training_log_probs=torch.zeros(2, 1024)), str(folder / "forwards.pt")
        )
    io.write_json(tmp_path / "second/parameter_differences.json", detail)
    if corrupt is not None:
        path = tmp_path / "second" / ("runtime_audit_rank0.json" if corrupt == "audit_status" else "result.json")
        invalid = repeat.read(path)
        invalid["status" if corrupt == "audit_status" else corrupt] = "corrupt"
        io.write_json(path, invalid)
        with pytest.raises(AssertionError):
            repeat.finish(tmp_path, keep_gradients=keep)
        assert all(path.exists() for path in paths)
        assert not (tmp_path / "comparison.json").exists()
        return
    result = repeat.finish(tmp_path, keep_gradients=keep)
    assert result["gradients"]["gradient_relative_l2"] > 0
    assert result["forwards"]["training_log_probs"]["logprob_max_abs"] == 0
    assert result["temporary_gradients_retained"] == keep
    assert all(path.exists() == keep for path in paths)
    assert source.read_bytes() == b"do not touch"
    assert unrelated.read_bytes() == b"also keep"
    assert (tmp_path / "comparison.json").is_file()
    assert (tmp_path / "summary.md").is_file()


def test_cleanup_rejects_path_escape_before_any_delete(tmp_path):
    folder = tmp_path / "grads"
    folder.mkdir()
    victim = tmp_path / "victim.pt"
    victim.write_bytes(b"keep")
    io.write_json(folder / "index.json", dict(parts=[[dict(file="../victim.pt")]]))
    with pytest.raises(ValueError, match="Unsafe"):
        repeat.reference_paths(folder)
    assert victim.read_bytes() == b"keep"


def test_historical_preflight_is_read_only_and_requires_space(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source_run(source, keep_full=False)
    before = {str(path): path.read_bytes() for path in source.rglob("*") if path.is_file()}
    context = repeat.preflight(source, tmp_path / "output", {})
    assert context["source"] == source.resolve()
    assert before == {str(path): path.read_bytes() for path in source.rglob("*") if path.is_file()}
    with pytest.raises(ValueError, match="separate"):
        repeat.preflight(source, source / "output", {})
    monkeypatch.setattr(repeat.shutil, "disk_usage", lambda _: SimpleNamespace(free=0))
    with pytest.raises(OSError, match="free"):
        repeat.preflight(source, tmp_path / "output", {})


@pytest.mark.parametrize("getter_name", ["get_main_grads_for_grad_norm", "get_grads_for_grad_norm"])
@pytest.mark.parametrize("chained", [False, True])
def test_native_norm_calls_prepare_once_and_norm_twice_without_step(getter_name, chained):
    calls = []

    class Native:
        grad_scaler = None
        optimizer = SimpleNamespace(param_groups=[{"step": 0}], state={})

        def get_loss_scale(self):
            return torch.tensor(1.0)

        def prepare_grads(self):
            calls.append("prepare")
            self.gradient = torch.tensor([3.0, 4.0])
            return False

        def get_grad_norm(self):
            calls.append("norm")
            return self.gradient.norm()

        def step(self):
            pytest.fail("No clipping or optimizer update is allowed")

    setattr(Native, getter_name, lambda self: [self.gradient])
    native = Native()
    optimizer = native
    if chained:
        optimizer = SimpleNamespace(
            chained_optimizers=[native],
            get_loss_scale=native.get_loss_scale,
            prepare_grads=native.prepare_grads,
            get_grad_norm=native.get_grad_norm,
        )
    # Resolving API names must not read gradients or run any optimizer methods.
    assert native_gradient_getters(optimizer)[0].__self__ is native
    assert calls == []
    result = native_norms(optimizer, lambda: None)
    assert calls == ["prepare", "norm", "norm"]
    assert result["native_norms"] == [5.0, 5.0]
    assert result["native_norm_elements"] == 2
    assert len(result["native_gradient_list_functions"]) == 1
    native.grad_scaler = object()
    with pytest.raises(AssertionError, match="no gradient scaler"):
        native_norms(optimizer, lambda: None)


@pytest.mark.parametrize("missing", ["get_loss_scale", "prepare_grads", "get_grad_norm", "gradient_getter"])
def test_missing_native_norm_api_fails_before_preparation(missing):
    def must_not_run():
        pytest.fail("API validation must finish before any native computation")

    methods = dict.fromkeys(
        ("get_loss_scale", "prepare_grads", "get_grad_norm", "get_main_grads_for_grad_norm"), must_not_run
    )
    methods.pop("get_main_grads_for_grad_norm" if missing == "gradient_getter" else missing)
    with pytest.raises(RuntimeError, match="missing|neither"):
        native_norms(SimpleNamespace(**methods), lambda: None)

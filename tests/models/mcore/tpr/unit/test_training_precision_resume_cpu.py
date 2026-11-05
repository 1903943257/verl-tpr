# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Continue after diagnostic I/O failures without rerunning a completed Base."""

import pytest
import torch

from tests.models.mcore.tpr.parallel.training_precision_contract import input_digest, synthetic_tokens
from tests.models.mcore.tpr.parallel.training_precision_io import write_json
from tests.models.mcore.tpr.parallel.training_precision_resume import prepare_resume


def _baseline(root, monkeypatch):
    controls = dict(
        seed=261105,
        vanilla_mbridge=True,
        delayed_dkv_fp32="0",
        root_native_ring="0",
        leaf_batch_size="1",
        selected_logits_fp32="0",
    )
    for name, value in dict(
        TPR_TRAIN_SEED="261105",
        TPR_TRAIN_VANILLA_MBRIDGE="1",
        TPR_TRAIN_GRAD_ACCUM_FUSION="1",
        TPR_EXPERIMENT_DELAYED_DKV_FP32="0",
        TPR_ROOT_NATIVE_RING="0",
        TPR_LEAF_BATCH_SIZE="1",
        TPR_SELECTED_LOGITS_FP32="0",
    ).items():
        monkeypatch.setenv(name, value)
    fixture = dict(
        tokens=synthetic_tokens(151936, 261105),
        old_log_probs=torch.zeros(2, 1024),
        advantages=torch.ones(2, 1024),
        weights="same-initial-weights",
    )
    torch.save(fixture, str(root / "actor_fixture.pt"))
    grads = root / "base_cp1/grads"
    (grads / "rank0").mkdir(parents=True)
    torch.save(torch.arange(4).float(), str(grads / "rank0/p.pt"))
    write_json(
        grads / "index.json",
        dict(
            layout=[dict(name="p", shape=[4], elements=4)],
            parts=[[dict(index=0, start=0, end=4, file="rank0/p.pt")]],
        ),
    )
    baseline = dict(
        mode="base_cp1",
        cp=1,
        prefix_length=1024,
        suffix_length=1024,
        input_sha256=input_digest(fixture["tokens"], fixture["old_log_probs"], fixture["advantages"]),
        initial_weights_sha256=fixture["weights"],
        objective="unmodified_VERL_vanilla_PPO_token_mean",
        controls=controls,
        packages={"torch": "test"},
        precision=dict(
            main_grad_dtype="torch.float32",
            gradient_accumulation_fusion=True,
            grad_reduce_in_fp32=True,
            use_distributed_optimizer=True,
        ),
    )
    write_json(root / "base_cp1/result.json", baseline)
    failed = root / "base_cp2/grads/rank0"
    failed.mkdir(parents=True)
    (failed / "partial.pt").write_bytes(b"partial reference before failed load")
    (root / "base_cp2/run.log").write_text("mmap loader failed", encoding="utf-8")
    return fixture, baseline


def test_resume_reuses_base_cp1_and_preserves_failed_cp2_files(tmp_path, monkeypatch):
    _baseline(tmp_path, monkeypatch)
    original = (tmp_path / "base_cp1/result.json").read_bytes()
    completed, pending = prepare_resume(tmp_path)
    assert completed == ["base_cp1"] and pending == ["base_cp2", "tpr_cp1", "tpr_cp2"]
    assert (tmp_path / "base_cp1/result.json").read_bytes() == original
    assert (tmp_path / "base_cp1/grads/rank0/p.pt").is_file()
    assert not (tmp_path / "base_cp2").exists()
    (archive,) = tmp_path.glob("base_cp2.failed-*")
    assert (archive / "run.log").read_text() == "mmap loader failed"
    assert (archive / "grads/rank0/partial.pt").read_bytes() == b"partial reference before failed load"


@pytest.mark.parametrize("change", ["seed", "fixture", "missing_block", "fusion"])
def test_invalid_resume_rejected_before_archiving_anything(tmp_path, monkeypatch, change):
    fixture, _report = _baseline(tmp_path, monkeypatch)
    if change == "seed":
        monkeypatch.setenv("TPR_TRAIN_SEED", "8")
    elif change == "fixture":
        fixture["advantages"][0, 0] = 2
        torch.save(fixture, str(tmp_path / "actor_fixture.pt"))
    elif change == "fusion":
        monkeypatch.setenv("TPR_TRAIN_GRAD_ACCUM_FUSION", "0")
    else:
        (tmp_path / "base_cp1/grads/rank0/p.pt").unlink()
    with pytest.raises(AssertionError):
        prepare_resume(tmp_path)
    assert (tmp_path / "base_cp2/run.log").is_file()
    assert not list(tmp_path.glob("*.failed-*"))


def test_resume_fixture_load_also_uses_string_filename(tmp_path, monkeypatch):
    _baseline(tmp_path, monkeypatch)
    original_load = torch.load
    seen = []

    def npu_load(filename, **kwargs):
        assert isinstance(filename, str)
        seen.append(filename)
        return original_load(filename, **kwargs)

    monkeypatch.setattr(torch, "load", npu_load)
    prepare_resume(tmp_path)
    assert seen == [str(tmp_path / "actor_fixture.pt")]

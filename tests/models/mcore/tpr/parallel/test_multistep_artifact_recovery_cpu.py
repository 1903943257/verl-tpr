"""No-NPU preflight gates: safely replay failed real Actor PPO experiments."""
from pathlib import Path

import pytest

from .multistep_artifact_recovery import expected_snapshots, reconcile


def _snapshots(root: Path, mode: str, steps=(1, 2, 5)):
    for step in steps:
        for rank in range(8):
            (root / f"{mode}_step{step}_rank{rank}.pt").write_bytes(b"test-snapshot")


def _grads(root: Path, mode: str, step=1, tp=1):
    folder = root / f"{mode}_step{step}_tp{tp}_full_grads"
    folder.mkdir()
    (folder / "grad_0000.pt").write_bytes(b"test-gradient")
    (folder / "manifest.pt").write_bytes(b"test-manifest")
    return folder


def test_old_orphan_gradient_directory_is_archived_without_deletion(tmp_path):
    orphan = _grads(tmp_path, "tpr")
    old_bytes = (orphan / "grad_0000.pt").read_bytes()
    (tmp_path / "tpr_50step.log").write_text("failure trace")
    result = reconcile(tmp_path, mode="tpr", steps=5, has_manifest=False,
                       full_grad_steps={1, 5})
    assert result["state"] == "replay"
    assert not orphan.exists()
    archive = Path(result["archive"])
    assert (archive / orphan.name / "grad_0000.pt").read_bytes() == old_bytes
    assert (archive / "tpr_50step.log").read_text() == "failure trace"
    assert reconcile(tmp_path, mode="tpr", steps=5, has_manifest=False,
                     full_grad_steps={1, 5})["state"] == "run"


def test_completed_tpr_is_reused_while_failed_native_is_archived(tmp_path):
    _snapshots(tmp_path, "tpr")
    for step in (1, 5):
        for tp in (0, 1):
            _grads(tmp_path, "tpr", step=step, tp=tp)
    full_grad = _grads(tmp_path, "full")
    r = reconcile(tmp_path, mode="full", steps=5, has_manifest=False,
                  full_grad_steps={1, 5})
    assert r["state"] == "replay"
    assert not full_grad.exists()
    tpr = reconcile(tmp_path, mode="tpr", steps=5, has_manifest=False,
                    full_grad_steps={1, 5})
    assert tpr["state"] == "reuse"
    assert tpr["snapshots"] == 24
    assert (tmp_path / "tpr_step5_tp1_full_grads" / "manifest.pt").exists()


def test_partial_snapshot_set_never_accepted_as_completed_mode(tmp_path):
    (tmp_path / "tpr_step1_rank0.pt").write_text("unfinished")
    (tmp_path / "tpr_step1_tp0_full_grads").mkdir()
    outcome = reconcile(tmp_path, mode="tpr", steps=5, has_manifest=False,
                        full_grad_steps={1, 5})
    assert outcome["state"] == "replay"
    archived = Path(outcome["archive"])
    assert (archived / "tpr_step1_rank0.pt").is_file()
    assert (archived / "tpr_step1_tp0_full_grads").is_dir()


def test_done_missing_full_gradient_witness_fails_closed(tmp_path):
    _snapshots(tmp_path, "tpr")
    with pytest.raises(RuntimeError, match="lack requested full-grad"):
        reconcile(tmp_path, mode="tpr", steps=5, has_manifest=False,
                  full_grad_steps={1, 5})
    assert (tmp_path / "tpr_step1_rank0.pt").exists()


def test_manifest_requires_all_real_step_snapshots(tmp_path):
    assert expected_snapshots(5, has_manifest=True) == (1, 2, 3, 4, 5)
    assert expected_snapshots(5, has_manifest=False) == (1, 2, 5)
    _snapshots(tmp_path, "full", steps=(1, 2, 5))
    result = reconcile(tmp_path, mode="full", steps=5, has_manifest=True,
                       full_grad_steps=set())
    assert result["state"] == "replay"
    assert (Path(result["archive"]) / "full_step5_rank7.pt").exists()


def test_unrelated_files_and_other_mode_left_untouched(tmp_path):
    other = tmp_path / "notes.txt"
    other.write_text("keep")
    (tmp_path / "full_step1_rank0.pt").write_text("full-in-progress")
    _grads(tmp_path, "tpr")
    _ = reconcile(tmp_path, mode="tpr", steps=5, has_manifest=False,
                  full_grad_steps=set())
    assert other.read_text() == "keep"
    assert (tmp_path / "full_step1_rank0.pt").read_text() == "full-in-progress"


def test_completed_tpr_without_full_grad_requirement_is_reused(tmp_path):
    _snapshots(tmp_path, "tpr")
    result = reconcile(tmp_path, mode="tpr", steps=5, has_manifest=False,
                       full_grad_steps=set())
    assert result["state"] == "reuse"

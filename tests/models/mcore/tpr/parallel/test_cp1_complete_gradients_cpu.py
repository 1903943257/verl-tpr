"""CPU contract for streaming *complete* CP1 Qwen3-4B BF16 gradients."""
import pytest
import torch

from .compare_real_actor_cp1_short_full_tpr import (
    _complete_cp1_parameter_gradient_metrics,
)


def _write(root, name, grads):
    folder = root / name
    folder.mkdir()
    manifest = []
    for i, (param_name, value) in enumerate(grads.items()):
        filename = f"grad_{i:04d}.pt"
        torch.save(value, folder / filename)
        manifest.append({
            "name": param_name,
            "shape": tuple(value.shape),
            "filename": filename,
            "tp_sharded": False,
        })
    torch.save(manifest, folder / "manifest.pt")
    return str(folder)


def test_full_cp1_gradient_metrics_are_unsampled(tmp_path):
    a = {
        "layer.weight": torch.tensor([1., 2., 3., 4.], dtype=torch.bfloat16),
        "norm.weight": torch.tensor([5., 6.], dtype=torch.bfloat16),
    }
    b = {
        "layer.weight": torch.tensor([1., 2.5, 3., 4.], dtype=torch.bfloat16),
        "norm.weight": torch.tensor([5., 6.], dtype=torch.bfloat16),
    }
    left = {"full_grad_dir": _write(tmp_path, "full", a)}
    right = {"full_grad_dir": _write(tmp_path, "split", b)}
    result = _complete_cp1_parameter_gradient_metrics(left, right)
    x = torch.cat(list(a.values())).double()
    y = torch.cat(list(b.values())).double()
    assert result["complete_parameter_elements"] == 6
    assert result["relative_l2"] == pytest.approx(float(torch.linalg.vector_norm(y-x) / torch.linalg.vector_norm(x)))
    assert result["cosine"] == pytest.approx(float(torch.dot(x, y) / (torch.linalg.vector_norm(x)*torch.linalg.vector_norm(y))))
    assert result["top_absolute_gradient_errors"][0][0] == "layer.weight"


def test_diagnostic_requires_both_full_gradient_directories(tmp_path):
    with pytest.raises(AssertionError, match="Only one"):
        _complete_cp1_parameter_gradient_metrics(
            {"full_grad_dir": str(tmp_path)}, {"full_grad_dir": None},
        )


def test_unsaved_legacy_cp1_results_are_not_misrepresented():
    assert _complete_cp1_parameter_gradient_metrics(
        {"full_grad_dir": None}, {"full_grad_dir": None},
    ) is None

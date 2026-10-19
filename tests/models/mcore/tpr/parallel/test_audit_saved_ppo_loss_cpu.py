"""CPU formula control for historical PPO logprob-based scalar reconstruction."""
import torch
import pytest

from .audit_saved_ppo_loss_cpu import (
    _ppo_from_token_vectors,
    _ppo_token_gradient_diagnostics,
)


def test_zero_ratio_flat_ppo_uses_negative_advantage_token_mean():
    new = torch.tensor([-1.0, -2.0, -3.0])
    old = new.clone()
    advantage = torch.tensor([1.0, -2.0, 3.0])
    assert _ppo_from_token_vectors(new, old, advantage) == pytest.approx(-2.0/3)


def test_effective_clipping_positive_advantage():
    old = torch.tensor([-1.0, -1.0])
    new = old + torch.log(torch.tensor([1.5, 0.5]))
    advantage = torch.tensor([2.0, -2.0])
    # A>0 ratio1.5 clips to 1.2; A<0 ratio0.5 clips to 0.8.
    assert _ppo_from_token_vectors(new, old, advantage) == pytest.approx(
        (-2.4+1.6)/2, abs=1e-6
    )


def test_importance_weight_preserved():
    value = _ppo_from_token_vectors(
        torch.tensor([-1., -1.]), torch.tensor([-1., -1.]),
        torch.tensor([1., 2.]), torch.tensor([2., 0.5])
    )
    assert value == pytest.approx(-1.5)


def test_identical_saved_ppo_token_gradients_have_zero_drift():
    old = torch.tensor([-3.0, -1.0, -2.0])
    new = old.clone()
    adv = torch.tensor([1.0, -2.0, 0.5])
    report = _ppo_token_gradient_diagnostics(new, new, old, adv)
    assert report["logprob_rel_l2"] == 0
    assert report["effective_clip_flip"] == 0
    assert report["upstream_ppo_gradient_rel_l2"] == 0
    assert report["upstream_ppo_gradient_cosine"] == pytest.approx(1)


def test_saved_ppo_clip_flip_and_negative_advantage_dual_clip():
    old = torch.full((4,), -4.0)
    new_full = old.clone()
    # Positive advantage high-ratio; negative advantage low-ratio; negative
    # advantage >dual-clip threshold 3; negative advantage inside window.
    ratio = torch.tensor([1.5, 0.5, 3.5, 1.1])
    new_tpr = old + ratio.log()
    adv = torch.tensor([1.0, -1.0, -1.0, -1.0])
    report = _ppo_token_gradient_diagnostics(
        new_full, new_tpr, old, adv,
    )
    assert report["tokens"] == 4
    assert report["effective_clip_full"] == 0
    assert report["effective_clip_tpr"] == 3
    assert report["effective_clip_flip"] == 3
    assert report["effective_clip_flip_fraction"] == pytest.approx(.75)
    assert report["upstream_ppo_gradient_rel_l2"] > 0


def test_importance_weight_scales_upstream_gradient():
    old = torch.tensor([-1., -2.])
    new = old + torch.tensor([0.01, -0.01])
    adv = torch.tensor([1., 2.])
    report = _ppo_token_gradient_diagnostics(
        old, new, old, adv, torch.tensor([2., 0.])
    )
    assert report["effective_clip_flip"] == 0
    assert report["upstream_ppo_gradient_full_norm"] > 0

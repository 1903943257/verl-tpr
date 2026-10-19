"""CPU formula control for historical PPO logprob-based scalar reconstruction."""
import torch
import pytest

from .audit_saved_ppo_loss_cpu import _ppo_from_token_vectors


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

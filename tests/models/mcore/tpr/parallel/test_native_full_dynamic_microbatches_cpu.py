"""CPU contract: Native Full CP4 has dynamic PPO microbatch cardinality.

Two-row sibling replay and eight-Session original Actor replay share one native
Megatron schedule. No fake rows and no extra normalization by 8 are allowed.
"""
import pytest

from .native_full_megatron_schedule import _native_microbatch_count


@pytest.mark.parametrize("rows,supervised_tokens", [(1, 64), (2, 4616), (8, 24792)])
def test_native_original_row_count_is_dynamic(rows, supervised_tokens):
    assert _native_microbatch_count(rows, supervised_tokens, rows) == rows
    local_ppo_losses = [float(i + 1) / supervised_tokens for i in range(rows)]
    # Megatron multiplies 2-tuple loss by CP/num_microbatches. The callback
    # cancels num_microbatches exactly, even for a two-row fixed PPO smoke.
    cp = 4
    schedule_sum = sum(
        (value * rows) * cp / rows for value in local_ppo_losses
    )
    cp_average = schedule_sum / cp
    assert cp_average == pytest.approx(sum(local_ppo_losses))


@pytest.mark.parametrize("rows,tokens,metadata", [
    (0, 64, 0), (2, 0, 2), (2, 64, 8), (8, 64, 2), (2, 64, None)
])
def test_native_rejects_invalid_original_batch(rows, tokens, metadata):
    with pytest.raises(AssertionError):
        _native_microbatch_count(rows, tokens, metadata)

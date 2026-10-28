# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""CPU placement-contract tests; these do not enable distributed training."""

from itertools import combinations
import random
import sys
import types

import pytest
import torch

from verl.experimental.tpr.placement import (
    _tree_token_cost,
    plan_dta_dfs,
    plan_tpr_dta_dp,
    plan_verl_uid,
)


def _assert_exact_coverage(plan, n, k):
    assert len(plan.partitions) == k
    assert sorted(i for group in plan.partitions for i in group) == list(range(n))
    assert all(plan.partitions)


def test_dta_balances_shared_prefix_tree_cost_not_raw_token_sum():
    sequences = [
        [1, 2, 3, 4],
        [1, 2, 3, 5],
        [9, 8, 7],
        [9, 8, 6],
    ]
    plan = plan_dta_dfs(sequences, 2)
    _assert_exact_coverage(plan, 4, 2)
    assert plan.equal_rows_per_rank
    assert plan.global_tree_tokens == 9
    assert plan.max_tree_tokens == 5
    assert plan.duplicated_tree_tokens == 0


def test_dta_uneven_partitions_are_offline_only():
    # The 20-token sample forces DTA's optimal 1-vs-3 assignment.
    sequences = [list(range(20)), [40, 41], [50, 51], [60, 61]]
    offline = plan_dta_dfs(sequences, 2, enforce_equal_rows=False)
    _assert_exact_coverage(offline, 4, 2)
    assert sorted(map(len, offline.partitions)) == [1, 3]
    assert not offline.equal_rows_per_rank
    assert offline.max_tree_tokens == 20
    with pytest.raises(ValueError, match="unequal per-DP row counts"):
        plan_dta_dfs(sequences, 2)


def test_dta_duplicate_trajectories_remain_distinct_rows():
    sequences = [[1, 2, 3]] * 4
    plan = plan_dta_dfs(sequences, 2)
    _assert_exact_coverage(plan, 4, 2)
    assert tuple(map(len, plan.partitions)) == (2, 2)
    assert plan.equal_rows_per_rank
    assert plan.global_tree_tokens == 3
    assert plan.tree_tokens_by_rank == (3, 3)
    assert plan.duplicated_tree_tokens == 3



def test_dta_duplicate_ties_choose_equal_rows_for_three_replicas():
    sequences = [[11, 22, 33]] * 6
    plan = plan_dta_dfs(sequences, 3)
    _assert_exact_coverage(plan, 6, 3)
    assert tuple(map(len, plan.partitions)) == (2, 2, 2)
    assert plan.tree_tokens_by_rank == (3, 3, 3)
    assert plan.duplicated_tree_tokens == 6

def test_dta_minimax_matches_exhaustive_small_case():
    sequences = [
        [1, 2, 3, 4, 5],
        [1, 2, 3, 9],
        [1, 2, 4],
        [6, 7],
        [6, 8, 9],
        [9, 0],
    ]
    plan = plan_dta_dfs(sequences, 3, enforce_equal_rows=False)
    ordered = sorted(map(tuple, sequences))
    brute_force = min(
        max(_tree_token_cost(ordered[a:b]) for a, b in zip((0, *cuts), (*cuts, len(ordered))))
        for cuts in combinations(range(1, len(ordered)), 2)
    )
    assert plan.max_tree_tokens == brute_force


def test_dta_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="dp_size"):
        plan_dta_dfs([[1]], 0)
    with pytest.raises(ValueError, match=">="):
        plan_dta_dfs([[1]], 2)
    with pytest.raises(ValueError, match="empty"):
        plan_dta_dfs([[]], 1)
    with pytest.raises(TypeError, match="nonnegative"):
        plan_dta_dfs([[-1]], 1)


def test_native_uid_checks_contiguity_even_on_dp1():
    with pytest.raises(ValueError, match="contiguous"):
        plan_verl_uid([[1], [2], [3]], ["a", "b", "a"], 1)


def test_native_uid_dp1_preserves_every_row_without_importing_verl_balancer():
    plan = plan_verl_uid([[1, 2], [1, 3], [9]], ["a", "a", "b"], 1)
    assert plan.partitions == ((0, 1, 2),)
    assert plan.equal_rows_per_rank
    assert plan.policy == "verl_uid"


@pytest.fixture
def balancer(monkeypatch):
    # These are delegation tests, not tests of the upstream partitioner. Keep
    # its unrelated DataProto/Ray dependencies out of the isolated CPU suite.
    module = types.ModuleType("verl.utils.seqlen_balancing")
    module.get_group_balanced_partitions = None
    monkeypatch.setitem(sys.modules, module.__name__, module)
    from verl import utils
    monkeypatch.setattr(utils, "seqlen_balancing", module, raising=False)
    return module


def test_native_uid_delegates_to_verl_balancer(monkeypatch, balancer):

    captured = {}

    def fake_partition(*, seqlen_list, uid_list, k_partitions):
        captured.update(seqlen_list=seqlen_list, uid_list=uid_list, k=k_partitions)
        return [[0, 1], [2, 3]]

    monkeypatch.setattr(balancer, "get_group_balanced_partitions", fake_partition)
    plan = plan_verl_uid([[1, 2], [1, 3], [9, 8], [9, 7]], ["a", "a", "b", "b"], 2)
    assert plan.partitions == ((0, 1), (2, 3))
    assert captured == {
        "seqlen_list": [2, 2, 2, 2],
        "uid_list": ["a", "a", "b", "b"],
        "k": 2,
    }


def test_native_uid_rejects_unequal_rows_from_upstream(monkeypatch, balancer):
    monkeypatch.setattr(
        balancer, "get_group_balanced_partitions",
        lambda **kw: [[0], [1, 2, 3]],
    )
    with pytest.raises(ValueError, match="unequal per-DP row counts"):
        plan_verl_uid([[1], [2], [3], [4]], ["a", "a", "b", "b"], 2)



def test_tpr_dp_uid_scoped_cost_matches_turn_semantics():
    # Token-identical trajectories from different UIDs are NOT coalesced by
    # the Turn builder. There is also no COMPLETE shared turn between the two
    # distinct uidA rows: the actual forest has three 4-token paths.
    sequences = [[1, 2, 3, 4], [1, 2, 3, 5], [1, 2, 3, 4]]
    keys = ["uidA_rollout_0", "uidA_rollout_1", "uidB_rollout_0"]
    plan = plan_tpr_dta_dp(sequences, keys, dp_size=1)
    assert plan.policy == "tpr_dta"
    assert plan.global_tree_tokens == 12
    assert plan.tree_tokens_by_rank == (12,)
    assert plan.cost_model == "qwen_turn"
    assert _tree_token_cost(sequences) == 5


def test_tpr_dp_keeps_duplicate_logical_rows_and_equal_cardinality():
    sequences = [[1, 2, 3, 4]] * 4
    keys = [f"uidA_rollout_{i}" for i in range(4)]
    plan = plan_tpr_dta_dp(sequences, keys, dp_size=2)
    _assert_exact_coverage(plan, 4, 2)
    assert tuple(map(len, plan.partitions)) == (2, 2)
    assert plan.global_tree_tokens == 4
    assert plan.tree_tokens_by_rank == (4, 4)
    assert plan.duplicated_tree_tokens == 4


def test_tpr_dp_key_contract():
    with pytest.raises(ValueError, match="equal lengths"):
        plan_tpr_dta_dp([[1]], [], 1)
    with pytest.raises(ValueError, match="unique"):
        plan_tpr_dta_dp([[1], [1]], ["a_b_0", "a_b_0"], 1)
    with pytest.raises(ValueError, match="trajectory key"):
        plan_tpr_dta_dp([[1]], ["invalid"], 1)


def _execution_cost(sequences, keys, prompts=None, minimum=128):
    from verl.workers.engine.megatron.tpr.tree import build_tree_execution_plans
    boundaries = [1] * len(sequences) if prompts is None else [max(1, n) for n in prompts]
    batch = {"input_ids": [torch.tensor(s) for s in sequences],
             "response_mask": [torch.ones(len(s) - n, dtype=torch.bool)
                               for s, n in zip(sequences, boundaries, strict=True)]}
    if prompts is not None:
        batch["prompts"] = [torch.tensor(s[:n]) for s, n in zip(sequences, prompts, strict=True)]
    forest = build_tree_execution_plans(keys, batch, turn_min_segment_tokens=minimum)
    return sum(segment.length for tree in forest.trees for segment in tree.segment_plan.segments.values())


def test_partial_turn_is_not_invented_shared_work():
    seqs = [[11] * 100 + [21, 151645], [11] * 100 + [22, 151645]]
    keys = ["uid_tpr_0", "uid_tpr_1"]
    for dp in (1, 2):
        plan = plan_tpr_dta_dp(seqs, keys, dp)
        assert plan.global_tree_tokens == _execution_cost(seqs, keys) == 204
        assert plan.duplicated_tree_tokens == 0


@pytest.mark.parametrize("seed", range(24))
def test_turn_cost_matches_real_forest_with_prompts_uids_and_duplicates(seed):
    rng = random.Random(seed)
    seqs = [[1, 2, 151645, 3, 151645, 4 + i % 2, 151645] for i in range(6)]
    prompts = [rng.choice((2, 3, 5)) for _ in seqs]
    keys = [f"uid{i % 2}_tpr_{i}" for i in range(len(seqs))]
    minimum = rng.choice((1, 3, 128))
    plan = plan_tpr_dta_dp(seqs, keys, 2, enforce_equal_rows=False,
                           prompt_lengths=prompts, turn_min_segment_tokens=minimum)
    assert plan.global_tree_tokens == _execution_cost(seqs, keys, prompts, minimum)
    costs = tuple(_execution_cost([seqs[i] for i in part], [keys[i] for i in part],
                                  [prompts[i] for i in part], minimum) for part in plan.partitions)
    assert plan.tree_tokens_by_rank == costs
    assert plan.duplicated_tree_tokens == sum(costs) - plan.global_tree_tokens


def test_turn_minimax_matches_exhaustive_real_forest_intervals():
    from verl.workers.engine.megatron.tpr.tree import _turn_units
    seqs = [[1, 151645, 2, 151645], [1, 151645, 3, 151645],
            [1, 151645, 3, 4, 151645], [5, 151645], [5, 151645, 6, 151645], [7, 151645]]
    keys = [f"uid_tpr_{i}" for i in range(len(seqs))]
    order = sorted(range(len(seqs)), key=lambda i: tuple(u for u, _ in _turn_units(seqs[i], 0, 1)))
    expected = min(max(_execution_cost([seqs[i] for i in order[a:b]],
                                      [keys[i] for i in order[a:b]], minimum=1)
                       for a, b in zip((0, *cuts), (*cuts, len(order))))
                   for cuts in combinations(range(1, len(order)), 2))
    actual = plan_tpr_dta_dp(seqs, keys, 3, enforce_equal_rows=False, turn_min_segment_tokens=1)
    assert actual.max_tree_tokens == expected


@pytest.mark.parametrize("prompts,minimum", [([0], 1), ([-1, 0], 1), ([0, 4], 1), ([0, 0], 0),
                                            ([0, 0], True)])
def test_turn_cost_rejects_wrong_boundary_configuration(prompts, minimum):
    with pytest.raises(ValueError):
        plan_tpr_dta_dp([[1, 2], [1, 3]], ["u_tpr_0", "u_tpr_1"], 1,
                        prompt_lengths=prompts, turn_min_segment_tokens=minimum)

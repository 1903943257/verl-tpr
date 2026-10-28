# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Prefix-aware DP placement *planning*, not distributed data dispatch.

Only **DP placement** follows DTA's DFS-contiguous minimax idea:
https://github.com/areal-project/AReaL/blob/feat/dta/areal/experimental/dta/dp.py
Tree construction and training use the Turn SegmentPlan and SegmentExecutor.
No DTA runtime or trie is installed.

This module intentionally does not touch Megatron process groups, split a
TensorDict, or override VERL's gradient/optimizer synchronization.  In
particular, VERL's current trainer / worker contract assumes an equal number
of examples on every DP replica.  DTA is an offline planning alternative until
a compatible variable-cardinality dispatch + mini-batch contract exists.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable
from typing import Literal, Sequence

Policy = Literal["verl_uid", "dta_dfs", "tpr_dta"]


@dataclass(frozen=True)
class DPPlacementPlan:
    """Original example indices assigned to each *logical DP replica*."""

    partitions: tuple[tuple[int, ...], ...]
    tree_tokens_by_rank: tuple[int, ...]
    global_tree_tokens: int
    duplicated_tree_tokens: int
    equal_rows_per_rank: bool
    policy: Policy
    cost_model: str = "exact_tokens"

    @property
    def max_tree_tokens(self) -> int:
        return max(self.tree_tokens_by_rank, default=0)


def _sequences(token_sequences: Sequence[Sequence[int]]) -> tuple[tuple[int, ...], ...]:
    result = []
    for row, sequence in enumerate(token_sequences):
        tokens = tuple(sequence)
        if not tokens:
            raise ValueError(f"sequence {row} is empty; pass unpadded real tokens")
        # Convert ordinary CPU tensors with .tolist() at the caller boundary.
        if any(not isinstance(token, int) or isinstance(token, bool) or token < 0 for token in tokens):
            raise TypeError(f"sequence {row} must contain nonnegative Python int token IDs")
        result.append(tokens)
    return tuple(result)


def _lcp(left: Sequence, right: Sequence) -> int:
    length = 0
    for a, b in zip(left, right):
        if a != b:
            break
        length += 1
    return length


def _tree_token_cost(sequences: Sequence[tuple[int, ...]]) -> int:
    if not sequences:
        return 0
    ordered = sorted(sequences)
    cost = len(ordered[0])
    for previous, current in zip(ordered, ordered[1:]):
        cost += len(current) - _lcp(previous, current)
    return cost



def _uid_scoped_tree_cost(
    sequences: Sequence[tuple[int, ...]],
    uids: Sequence[str],
) -> int:
    """UID-scoped exact-token reference, NOT the Turn execution cost."""
    if len(sequences) != len(uids):
        raise ValueError("number of UIDs must match number of trajectories")
    groups: dict[str, list[tuple[int, ...]]] = {}
    for uid, sequence in zip(uids, sequences, strict=True):
        groups.setdefault(uid, []).append(sequence)
    return sum(_tree_token_cost(group) for group in groups.values())


def _check_partitions(partitions: Sequence[Sequence[int]], n: int, dp_size: int) -> None:
    if len(partitions) != dp_size:
        raise ValueError(f"expected {dp_size} DP partitions, got {len(partitions)}")
    flat = [index for part in partitions for index in part]
    if any(type(index) is not int for index in flat):
        raise ValueError("DP partitions must contain integer row indices")
    if any(not part for part in partitions):
        raise ValueError("each DP replica must receive at least one trajectory")
    if sorted(flat) != list(range(n)):
        raise ValueError("DP partitions must cover each input row exactly once")


def _result(
    sequences: tuple[tuple[int, ...], ...],
    partitions: Sequence[Sequence[int]],
    policy: Policy,
    dp_size: int,
    enforce_equal_rows: bool,
    uid_list: Sequence[str] | None = None,
    *,
    tree_cost: Callable[[Sequence[int]], int] | None = None,
    cost_model: str = "exact_tokens",
) -> DPPlacementPlan:
    _check_partitions(partitions, len(sequences), dp_size)
    equal = len({len(part) for part in partitions}) == 1
    if enforce_equal_rows and not equal:
        raise ValueError(
            "DTA produced unequal per-DP row counts; VERL's current batch "
            "dispatcher and TrainingWorker mini-batch contract require equal "
            "counts. Keep this plan offline; do not pass it to batch.reorder()."
        )
    def cost_of(indices: Sequence[int]) -> int:
        if tree_cost is not None:
            return tree_cost(indices)
        subset = [sequences[i] for i in indices]
        return (
            _tree_token_cost(subset)
            if uid_list is None
            else _uid_scoped_tree_cost(subset, [uid_list[i] for i in indices])
        )

    costs = tuple(cost_of(part) for part in partitions)
    global_cost = cost_of(range(len(sequences)))
    duplication = sum(costs) - global_cost
    if duplication < 0:
        raise AssertionError("DP partitions cannot require fewer unique tree tokens than the global forest")
    return DPPlacementPlan(
        partitions=tuple(tuple(part) for part in partitions),
        tree_tokens_by_rank=costs,
        global_tree_tokens=global_cost,
        duplicated_tree_tokens=duplication,
        equal_rows_per_rank=equal,
        policy=policy,
        cost_model=cost_model,
    )


def plan_dta_dfs(
    token_sequences: Sequence[Sequence[int]],
    dp_size: int,
    *,
    enforce_equal_rows: bool = True,
    uid_list: Sequence[str] | None = None,
) -> DPPlacementPlan:
    """DTA DFS-contiguous minimax partition using unique trie-token cost.

    Duplicate trajectories remain distinct *logical rows*. With uid_list,
    sorting and tree-token costs follow the **existing TPR UID boundary**;
    without it, this is the flat lexical-token DTA baseline. This function
    only plans sample placement; it never builds another execution trie.
    No KV or dKV crosses DP ranks. Set enforce_equal_rows=False only for
    offline inspection of variable-cardinality assignments.
    """
    if type(dp_size) is not int or dp_size <= 0:
        raise ValueError("dp_size must be a positive integer")
    sequences = _sequences(token_sequences)
    if len(sequences) < dp_size:
        raise ValueError("number of trajectories must be >= dp_size")
    if uid_list is not None and len(uid_list) != len(sequences):
        raise ValueError("uid_list and trajectories must have the same row count")

    # DTA sorts tokens by lexical DFS.  TPR's real execution tree first groups
    # by UID; do the same here so planning cannot invent cross-UID KV sharing.
    order = sorted(
        range(len(sequences)),
        key=(
            (lambda i: (sequences[i], i)) if uid_list is None
            else (lambda i: (uid_list[i], sequences[i], i))
        ),
    )
    overlap = [0] + [
        _lcp(sequences[a], sequences[b])
        if uid_list is None or uid_list[a] == uid_list[b] else 0
        for a, b in zip(order, order[1:])
    ]
    partitions = _dfs_partitions(order, sequences, overlap, dp_size, enforce_equal_rows)
    return _result(sequences, partitions, "dta_dfs" if uid_list is None else "tpr_dta",
                   dp_size, enforce_equal_rows, uid_list=uid_list,
                   cost_model="exact_tokens" if uid_list is None else "uid_exact_tokens")


def _dfs_partitions(order, sequences, overlap, dp_size, enforce_equal_rows):
    """Minimax over lexical DFS intervals; overlap is in real token units."""
    lengths = [len(sequences[i]) for i in order]
    increment = [0] + [lengths[i] - overlap[i] for i in range(1, len(order))]
    cumulative = [0]
    for value in increment:
        cumulative.append(cumulative[-1] + value)

    def cost(start: int, stop: int) -> int:
        """Cost of [start, stop) in sorted DFS order."""
        return lengths[start] + cumulative[stop] - cumulative[start + 1]

    def divide(max_cost: int) -> list[tuple[int, int]]:
        groups: list[tuple[int, int]] = []
        start = 0
        n = len(order)
        while start < n:
            left, right = start + 1, n
            while left < right:
                mid = (left + right + 1) // 2
                if cost(start, mid) <= max_cost:
                    left = mid
                else:
                    right = mid - 1
            groups.append((start, left))
            start = left
        return groups

    low = max(lengths)
    high = cost(0, len(order))
    while low < high:
        mid = (low + high) // 2
        if len(divide(mid)) <= dp_size:
            high = mid
        else:
            low = mid + 1
    groups = divide(low)
    # DTA's greedy feasibility scan deliberately fills each rank as far as it
    # can.  When many rows have identical token sequences, adding another row
    # costs zero tree tokens: a valid minimax solution may then contain 3/1
    # rows even though a 2/2 split has exactly the SAME optimal tree cost.
    #
    # For VERL's equal-cardinality dispatch, prefer equally sized contiguous
    # intervals *only if* they preserve the DTA minimax objective.  Do not
    # silently sacrifice load balance to force equal rows: when that is
    # impossible, keep the original unequal solution and let _result reject
    # it until variable-cardinality worker dispatch is supported.
    if enforce_equal_rows and len(order) % dp_size == 0:
        rows_per_rank = len(order) // dp_size
        equal_groups = [
            (rank * rows_per_rank, (rank + 1) * rows_per_rank)
            for rank in range(dp_size)
        ]
        if all(cost(start, end) <= low for start, end in equal_groups):
            groups = equal_groups

    # Monotone partition cost means splitting an interval cannot increase its
    # tree-token cost.  Split until all DP ranks have a non-empty interval.
    while len(groups) < dp_size:
        i = next((j for j in range(len(groups) - 1, -1, -1)
                  if groups[j][1] - groups[j][0] > 1), None)
        if i is None:
            raise AssertionError("could not construct non-empty DP partitions")
        start, end = groups[i]
        groups[i:i + 1] = [(start, end - 1), (end - 1, end)]
    return [[order[i] for i in range(start, end)] for start, end in groups]




def plan_tpr_dta_dp(
    token_sequences: Sequence[Sequence[int]],
    trajectory_keys: Sequence[str],
    dp_size: int,
    *,
    enforce_equal_rows: bool = True,
    prompt_lengths: Sequence[int] | None = None,
    turn_min_segment_tokens: int = 128,
) -> DPPlacementPlan:
    """DP-only minimax using the SAME Turn units as the executable forest.

    Prompt endpoints and the Turn minimum affect sharing and must match the
    training batch. Planning never changes the captured training data.
    Costs count physical topology tokens, before any zero-objective pruning.
    The flat exact-token DTA reference remains separately available above.
    """
    from verl.workers.engine.megatron.tpr.tree import _parse_key, _turn_units

    if type(dp_size) is not int or dp_size <= 0:
        raise ValueError("dp_size must be a positive integer")
    sequences = _sequences(token_sequences)
    if len(sequences) < dp_size:
        raise ValueError("number of trajectories must be >= dp_size")
    if len(trajectory_keys) != len(token_sequences):
        raise ValueError("trajectory_keys and token_sequences must have equal lengths")
    if len(set(trajectory_keys)) != len(trajectory_keys):
        raise ValueError("trajectory_keys must be unique")
    uids = tuple(_parse_key(key)[0] for key in trajectory_keys)
    if type(turn_min_segment_tokens) is not int or turn_min_segment_tokens < 1:
        raise ValueError("turn_min_segment_tokens must be a positive integer")
    prompts = [0] * len(sequences) if prompt_lengths is None else list(prompt_lengths)
    if len(prompts) != len(sequences) or any(
        type(length) is not int or not 0 <= length <= len(sequence)
        for length, sequence in zip(prompts, sequences)
    ):
        raise ValueError("prompt_lengths must contain one valid token length per row")
    units = [tuple(unit for unit, _ in _turn_units(list(tokens), prompt, turn_min_segment_tokens))
             for tokens, prompt in zip(sequences, prompts, strict=True)]

    def shared(a, b):
        if uids[a] != uids[b]:
            return 0
        return sum(len(unit) for unit in units[a][:_lcp(units[a], units[b])])

    def cost(indices):
        ordered = sorted(indices, key=lambda i: (uids[i], units[i], i))
        return (len(sequences[ordered[0]]) + sum(
            len(sequences[b]) - shared(a, b) for a, b in zip(ordered, ordered[1:])
        )) if ordered else 0

    order = sorted(range(len(sequences)), key=lambda i: (uids[i], units[i], i))
    overlap = [0] + [shared(a, b) for a, b in zip(order, order[1:])]
    partitions = _dfs_partitions(order, sequences, overlap, dp_size, enforce_equal_rows)
    return _result(sequences, partitions, "tpr_dta", dp_size, enforce_equal_rows,
                   tree_cost=cost, cost_model="qwen_turn")


def plan_verl_uid(
    token_sequences: Sequence[Sequence[int]],
    uid_list: Sequence[str],
    dp_size: int,
) -> DPPlacementPlan:
    """Delegate initial equal-cardinality UID placement to upstream VERL.

    This is the *safe* integration baseline.  It does not use DTA's cost model;
    it preserves existing VERL Karmarkar-Karp group balancing instead.
    """
    if type(dp_size) is not int or dp_size <= 0:
        raise ValueError("dp_size must be a positive integer")
    sequences = _sequences(token_sequences)
    if len(uid_list) != len(sequences):
        raise ValueError("uid_list and trajectories must have the same row count")
    if len(sequences) < dp_size:
        raise ValueError("number of trajectories must be >= dp_size")

    # Upstream get_group_balanced_partitions assumes each UID is one contiguous
    # group.  Do not silently turn repeated non-contiguous UIDs into new groups.
    seen = set()
    last_uid = object()
    for uid in uid_list:
        if uid != last_uid:
            if uid in seen:
                raise ValueError("VERL UID balancing requires contiguous rows for each UID")
            seen.add(uid)
            last_uid = uid

    if dp_size == 1:
        partitions = [list(range(len(sequences)))]
    else:
        from verl.utils.seqlen_balancing import get_group_balanced_partitions

        partitions = get_group_balanced_partitions(
            seqlen_list=[len(seq) for seq in sequences],
            uid_list=list(uid_list),
            k_partitions=dp_size,
        )
    return _result(sequences, partitions, "verl_uid", dp_size, True,
                   uid_list=uid_list, cost_model="uid_exact_tokens")

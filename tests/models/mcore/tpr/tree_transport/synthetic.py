"""Deterministic Session Trees plus independently authored Full-row oracles."""
from __future__ import annotations

import torch
from torch import Tensor

from .schema import TREE_SCHEMA_VERSION, validate_session_tree

IM_START = 151644
IM_END = 151645


def _message(role_id: int, *body: int) -> Tensor:
    return torch.tensor([IM_START, role_id, *body, IM_END], dtype=torch.long)


def _logprobs(tokens: Tensor, offset: int) -> list[float]:
    return [-(offset + index + 1) / 100.0 for index in range(tokens.numel())]


def make_synthetic_session_tree(*, uid: str, session_id: str, user_token: int, token_offset: int):
    """One Session with a root Chain and a two-level branch plus a sibling Chain.

    Chain IDs are unique across active Leaves, as in the Gateway leaves map.
    Node 1 and Node 3 are one Chain; Node 2 is a sibling Chain.
    """
    system = _message(1, 700, 701, 702)
    user = _message(2, user_token, user_token + 1)
    root_prompt = torch.cat((system, user))
    root_response = _message(3, token_offset + 10, token_offset + 11)
    tool_a = _message(4, token_offset + 20)
    response_a = _message(3, token_offset + 30, token_offset + 31)
    tool_a2 = _message(4, token_offset + 35)
    response_a2 = _message(3, token_offset + 36, token_offset + 37)
    tool_b = _message(4, token_offset + 40)
    response_b = _message(3, token_offset + 50, token_offset + 51)

    nodes = [
        {"node_id": 0, "parent_id": None, "chain_id": 0, "prompt_ids": root_prompt,
         "response_ids": root_response, "response_logprobs": _logprobs(root_response, token_offset),
         "removed_prefix_token_count": 0},
        {"node_id": 1, "parent_id": 0, "chain_id": 1, "prompt_ids": tool_a,
         "response_ids": response_a, "response_logprobs": _logprobs(response_a, token_offset + 100),
         "removed_prefix_token_count": 0},
        {"node_id": 2, "parent_id": 0, "chain_id": 2, "prompt_ids": tool_b,
         "response_ids": response_b, "response_logprobs": _logprobs(response_b, token_offset + 200),
         "removed_prefix_token_count": 0},
        {"node_id": 3, "parent_id": 1, "chain_id": 1, "prompt_ids": tool_a2,
         "response_ids": response_a2, "response_logprobs": _logprobs(response_a2, token_offset + 300),
         "removed_prefix_token_count": 0},
    ]

    root_mask = torch.ones(root_response.numel(), dtype=torch.bool)
    nested_response_mask = torch.ones(response_a.numel(), dtype=torch.bool)
    nested_response_mask[0] = False  # synthetic masked/error token in this Leaf only
    nested_mask = torch.cat((nested_response_mask,
                             torch.zeros(tool_a2.numel(), dtype=torch.bool),
                             torch.ones(response_a2.numel(), dtype=torch.bool)))
    sibling_mask = torch.ones(response_b.numel(), dtype=torch.bool)
    leaves = [
        {"leaf_id": f"{session_id}_root", "leaf_index": 0, "kind": "trajectory", "chain_id": 0,
         "trajectory": {"prompt_ids": root_prompt, "response_ids": root_response,
                        "response_logprobs": _logprobs(root_response, token_offset)},
         "response_mask": root_mask, "loss_mask": root_mask.clone(),
         "reward_score": 0.0, "finished": True},
        {"leaf_id": f"{session_id}_nested", "leaf_index": 1, "kind": "node_path",
         "leaf_node_id": 3, "chain_id": 1, "response_mask": nested_mask,
         "loss_mask": nested_mask.clone(), "reward_score": 1.0, "finished": True},
        {"leaf_id": f"{session_id}_sibling", "leaf_index": 2, "kind": "node_path",
         "leaf_node_id": 2, "chain_id": 2, "response_mask": sibling_mask,
         "loss_mask": sibling_mask.clone(), "reward_score": 0.5, "finished": True},
    ]
    tree = {"schema_version": TREE_SCHEMA_VERSION, "tree_id": f"tree_{session_id}", "uid": uid,
            "session_id": session_id, "root_node_id": 0, "nodes": nodes, "leaves": leaves}
    validate_session_tree(tree)

    # Ground truth is built directly from source chunks, not by calling the
    # Tree reconstruction code under test.
    expected = [
        {"key": f"{uid}_{session_id}_0", "uid": uid, "session_id": session_id,
         "leaf_id": leaves[0]["leaf_id"], "prompts": root_prompt.clone(), "responses": root_response.clone(),
         "input_ids": torch.cat((root_prompt, root_response)), "response_mask": root_mask.clone(),
         "loss_mask": root_mask.clone(), "rollout_log_probs": torch.tensor(_logprobs(root_response, token_offset)),
         "reward_score": 0.0},
        {"key": f"{uid}_{session_id}_1", "uid": uid, "session_id": session_id,
         "leaf_id": leaves[1]["leaf_id"],
         "prompts": torch.cat((root_prompt, root_response, tool_a)),
         "responses": torch.cat((response_a, tool_a2, response_a2)),
         "input_ids": torch.cat((root_prompt, root_response, tool_a, response_a, tool_a2, response_a2)),
         "response_mask": nested_mask.clone(), "loss_mask": nested_mask.clone(),
         "rollout_log_probs": torch.tensor(_logprobs(response_a, token_offset + 100)
                                               + [0.0] * tool_a2.numel()
                                               + _logprobs(response_a2, token_offset + 300)),
         "reward_score": 1.0},
        {"key": f"{uid}_{session_id}_2", "uid": uid, "session_id": session_id,
         "leaf_id": leaves[2]["leaf_id"],
         "prompts": torch.cat((root_prompt, root_response, tool_b)),
         "responses": response_b.clone(),
         "input_ids": torch.cat((root_prompt, root_response, tool_b, response_b)),
         "response_mask": sibling_mask.clone(), "loss_mask": sibling_mask.clone(),
         "rollout_log_probs": torch.tensor(_logprobs(response_b, token_offset + 200)),
         "reward_score": 0.5},
    ]
    return tree, expected


def make_synthetic_batch():
    """Two independent Session Trees with the same UID/System and different Users."""
    trees, expected = [], []
    for index in range(2):
        tree, rows = make_synthetic_session_tree(
            uid="uid_shared", session_id=f"s{index}", user_token=800 + index * 10,
            token_offset=1000 + index * 1000,
        )
        trees.append(tree)
        expected.extend(rows)
    return trees, expected

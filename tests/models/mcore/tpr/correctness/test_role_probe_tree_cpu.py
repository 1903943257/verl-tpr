"""CPU-only invariants for experimental role_probe tree building.

The fake tokenizer changes special IDs at will: the builder must learn
tokenized role headers from the tokenizer rather than from Qwen constants.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch


_ROOT = Path(__file__).resolve().parents[5]
_TREE_FILE = _ROOT / "verl/workers/engine/megatron/tpr/tree.py"
_SPEC = importlib.util.spec_from_file_location("tpr_role_probe_tree_under_test", _TREE_FILE)
tree = importlib.util.module_from_spec(_SPEC)
import sys
sys.modules[_SPEC.name] = tree
_SPEC.loader.exec_module(tree)


class FakeTokenizer:
    """Chat template with non-Qwen special IDs and explicit role headers."""

    def encode(self, content, add_special_tokens=False):
        return [ord(c) for c in content]

    def decode(self, tokens, skip_special_tokens=False):
        return "".join(chr(int(t)) for t in tokens if int(t) < 256)

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False):
        result = []
        for message in messages:
            result.extend([6001, *self.encode(message["role"]), 6003])
            result.extend(self.encode(message["content"]))
            result.append(6002)
        return result


def _tokens(messages):
    return torch.tensor(FakeTokenizer().apply_chat_template(messages), dtype=torch.long)


def _make_batch():
    prompt = [
        {"role": "system", "content": "You are a coding agent."},
        {"role": "user", "content": "Fix the bug; the word assistant is ordinary text."},
    ]
    shared = [
        {"role": "assistant", "content": "I will inspect the code carefully."},
        {"role": "tool", "content": "The answer mentions system, user and assistant."},
    ]
    alternatives = ["Apply patch A to the module.", "Apply patch B to the module."]
    sequences, prompts, responses, masks = [], [], [], []
    for text in alternatives:
        complete = prompt + shared + [{"role": "assistant", "content": text}]
        seq = _tokens(complete)
        p = _tokens(prompt)
        sequences.append(seq)
        prompts.append(p)
        responses.append(seq[len(p):])
        mask = []
        for msg in complete[len(prompt):]:
            mask.extend([int(msg["role"] == "assistant")] * len(_tokens([msg])))
        masks.append(torch.tensor(mask, dtype=torch.long))
    return {
        "input_ids": sequences,
        "prompts": prompts,
        "responses": responses,
        "response_mask": masks,
        "loss_mask": masks,
    }, ["uid_0_0", "uid_0_1"], prompt, shared


def _audit_exact_replay(forest, batch):
    seen = set()
    for item in forest.trees:
        plan = item.segment_plan
        for row in item.member_rows:
            source = batch["input_ids"][row]
            paths = (
                plan.path_to(s.segment_id)
                for s in plan.segments.values()
                if s.position_end == source.numel()
            )
            assert any(torch.equal(torch.cat([n.token_ids for n in path]), source) for path in paths)
        for ref in item.objective_refs:
            row, offset = ref.sample_row, ref.response_offset
            key = (row, offset)
            assert key not in seen, f"duplicated logical PPO token {key}"
            seen.add(key)
            seq = batch["input_ids"][row]
            response_start = len(batch["prompts"][row])
            target_pos = response_start + offset
            segment = plan.get(ref.segment_id)
            query_pos = segment.position_start + ref.query_offset
            assert query_pos + 1 == target_pos
            assert seq[query_pos].item() == segment.token_ids[ref.query_offset].item()
            assert seq[target_pos].item() == ref.target_token_id
            assert batch["response_mask"][row][offset].item() == 1
    expected = {
        (row, off) for row, mask in enumerate(batch["response_mask"])
        for off, value in enumerate(mask.tolist()) if value
    }
    assert seen == expected
    assert forest.logical_loss_tokens == len(expected)


def test_role_probe_recovers_headers_and_matches_two_builders(monkeypatch):
    probe = tree._probe_role_template(FakeTokenizer())
    batch, keys, prompt, shared = _make_batch()
    monkeypatch.setenv("TPR_ROLE_TOKENIZER_PATH", "/fake-model")
    monkeypatch.setattr(tree, "_role_probe_from_path", lambda path: probe)
    anchors = tree._role_anchors(batch["input_ids"][0], probe)
    expected = [
        (0, "system"),
        (len(_tokens(prompt[:1])), "user"),
        (len(_tokens(prompt)), "assistant"),
        (len(_tokens(prompt + shared[:1])), "tool"),
        (len(_tokens(prompt + shared)), "assistant"),
    ]
    assert anchors == tuple(expected)
    first = tree.build_tree_execution_plans(
        keys, batch, tree_policy="role_probe", tree_builder="trie",
        turn_min_segment_tokens=1, require_loss_mask_alignment=True
    )
    sorted_lcp = tree.build_tree_execution_plans(
        keys, batch, tree_policy="role_probe", tree_builder="lcp_sort",
        turn_min_segment_tokens=1, require_loss_mask_alignment=True
    )
    assert tree.digest_forest_plan(first) == tree.digest_forest_plan(sorted_lcp)
    _audit_exact_replay(first, batch)
    _audit_exact_replay(sorted_lcp, batch)
    # Baseline with no matching Qwen special ID can only share the initial prompt.
    qwen_baseline = tree.build_tree_execution_plans(
        keys, batch, tree_policy="qwen_turn", tree_builder="trie",
        turn_min_segment_tokens=1
    )
    reused_unique = sum(s.length for t in first.trees for s in t.segment_plan.segments.values())
    baseline_unique = sum(s.length for t in qwen_baseline.trees for s in t.segment_plan.segments.values())
    assert reused_unique < baseline_unique


def test_role_probe_rejects_missing_model_path(monkeypatch):
    monkeypatch.delenv("TPR_ROLE_TOKENIZER_PATH", raising=False)
    batch, keys, _, _ = _make_batch()
    with pytest.raises(ValueError, match="TPR_ROLE_TOKENIZER_PATH"):
        tree.build_tree_execution_plans(keys, batch, tree_policy="role_probe")


def test_role_units_preserve_every_token_with_minimum_length():
    batch, _, _, _ = _make_batch()
    probe = tree._probe_role_template(FakeTokenizer())
    source = batch["input_ids"][0]
    prompt_length = len(batch["prompts"][0])
    many = tree._role_units_bytes(source, prompt_length, 1, probe)
    few = tree._role_units_bytes(source, prompt_length, 128, probe)
    assert len(few) < len(many)
    for units in (many, few):
        assert b"".join(signature for signature, _ in units) == source.numpy().tobytes()
        assert units[-1][1] == len(source)

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


class ContextSensitiveProbeTokenizer(FakeTokenizer):
    """The content marker has different token IDs alone vs. in chat context."""

    def encode(self, content, add_special_tokens=False):
        if content == "TPRRoleProbeContentQ7X9":
            return [987654, 987655]
        return [ord(c) for c in content]

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False):
        rendered = "".join(
            chr(6001) + message["role"] + chr(6003)
            + message["content"] + chr(6002)
            for message in messages
        )
        if not tokenize:
            return rendered
        # Real full-template tokenization need not equal standalone marker ids.
        return [ord(c) for c in rendered]


def test_role_probe_recovers_context_sensitive_marker_alignment():
    tok = ContextSensitiveProbeTokenizer()
    probe = tree._probe_role_template(tok)
    sample = [
        {"role": "system", "content": "A"},
        {"role": "user", "content": "B"},
        {"role": "assistant", "content": "C"},
    ]
    encoded = torch.tensor(tok.apply_chat_template(sample), dtype=torch.long)
    anchors = tree._role_anchors(encoded, probe)
    expected = (
        (0, "system"),
        (len(tok.apply_chat_template(sample[:1])), "user"),
        (len(tok.apply_chat_template(sample[:2])), "assistant"),
    )
    assert anchors == expected


class WrappedChatTemplateTokenizer(FakeTokenizer):
    """Valid single-sequence token outputs that are not a flat Python list."""

    def __init__(self, return_mode):
        self.return_mode = return_mode

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False):
        ids = super().apply_chat_template(messages, tokenize, add_generation_prompt)
        if self.return_mode == "batch_encoding":
            return {"input_ids": [ids], "attention_mask": [[1] * len(ids)]}
        if self.return_mode == "tuple":
            return tuple(ids)
        if self.return_mode == "numpy":
            import numpy as np
            return np.asarray([ids], dtype=np.int64)
        if self.return_mode == "tensor":
            return torch.tensor([ids], dtype=torch.long)
        if self.return_mode == "singleton_batch":
            return [ids]
        raise AssertionError("unknown return mode")


@pytest.mark.parametrize(
    "return_mode",
    ["batch_encoding", "tuple", "numpy", "tensor", "singleton_batch"],
)
def test_role_probe_accepts_single_sequence_tokenizer_wrappers(return_mode):
    tok = WrappedChatTemplateTokenizer(return_mode)
    probe = tree._probe_role_template(tok)
    messages = [
        {"role": "system", "content": "setup"},
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "answer"},
    ]
    ids = FakeTokenizer().apply_chat_template(messages)
    anchors = tree._role_anchors(torch.tensor(ids, dtype=torch.long), probe)
    assert anchors == (
        (0, "system"),
        (len(FakeTokenizer().apply_chat_template(messages[:1])), "user"),
        (len(FakeTokenizer().apply_chat_template(messages[:2])), "assistant"),
    )


def test_role_probe_rejects_multi_sequence_tokenizer_result():
    class MultiSequenceTokenizer(FakeTokenizer):
        def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False):
            ids = super().apply_chat_template(messages, tokenize, add_generation_prompt)
            return {"input_ids": [ids, ids]}

    with pytest.raises(ValueError, match="multi-row token IDs"):
        tree._probe_role_template(MultiSequenceTokenizer())


def test_role_probe_real_qwen_tokenizer_when_configured():
    """Optional integration test: exercise the actual local chat template."""
    import os
    from transformers import AutoTokenizer

    tokenizer_path = os.environ.get("TPR_ROLE_TOKENIZER_PATH")
    if not tokenizer_path:
        pytest.skip("set TPR_ROLE_TOKENIZER_PATH for real Qwen chat template validation")
    tok = AutoTokenizer.from_pretrained(
        tokenizer_path, local_files_only=True, trust_remote_code=False
    )
    probe = tree._probe_role_template(tok)
    assert probe.headers and probe.endings
    messages = [
        {"role": "user", "content": "TPR_real_chat_role_probe_user"},
        {"role": "assistant", "content": "TPR_real_chat_role_probe_assistant"},
    ]
    ids = tree._coerce_probe_token_ids(
        tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
    )
    anchors = tree._role_anchors(torch.tensor(ids, dtype=torch.long), probe)
    assert (0, "user") in anchors
    assert any(role == "assistant" for _, role in anchors)


class ToolWrappedAsUserTokenizer(FakeTokenizer):
    """Model a chat template where tool has an overlapping user header."""

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False):
        ids = []
        for message in messages:
            role = message["role"]
            if role == "tool":
                ids.extend([6001, *self.encode("user"), 6003])
                ids.extend(self.encode("<tool_response>\n"))
                ids.extend(self.encode(message["content"]))
                ids.extend(self.encode("\n</tool_response>"))
            else:
                ids.extend([6001, *self.encode(role), 6003])
                ids.extend(self.encode(message["content"]))
            ids.append(6002)
        return ids


def test_role_probe_prefers_complete_tool_header_over_user_prefix():
    tok = ToolWrappedAsUserTokenizer()
    probe = tree._probe_role_template(tok)
    messages = [
        {"role": "user", "content": "Run a test"},
        {"role": "tool", "content": "3 passed"},
        {"role": "assistant", "content": "Finished"},
    ]
    ids = tok.apply_chat_template(messages)
    anchors = tree._role_anchors(torch.tensor(ids, dtype=torch.long), probe)
    assert anchors == (
        (0, "user"),
        (len(tok.apply_chat_template(messages[:1])), "tool"),
        (len(tok.apply_chat_template(messages[:2])), "assistant"),
    )


def test_role_probe_fails_on_equal_length_cross_role_ambiguity():
    header = torch.tensor([6001, 42, 6003], dtype=torch.long).numpy().tobytes()
    probe = tree._RoleProbe(
        headers=(("user", header, True), ("tool", header, True)),
        endings=(torch.tensor([6002], dtype=torch.long).numpy().tobytes(),),
    )
    with pytest.raises(ValueError, match="conflicting longest Role candidates at 0"):
        tree._role_anchors(torch.tensor([6001, 42, 6003, 99], dtype=torch.long), probe)


def test_role_probe_real_qwen_tool_header_when_configured():
    """Verify actual ChatML tool wrapper in Qwen's installed template."""
    import os
    from transformers import AutoTokenizer

    path = os.environ.get("TPR_ROLE_TOKENIZER_PATH")
    if not path:
        pytest.skip("set TPR_ROLE_TOKENIZER_PATH for real-tokenizer validation")
    tok = AutoTokenizer.from_pretrained(
        path, local_files_only=True, trust_remote_code=False
    )
    messages = [
        {"role": "user", "content": "Call a tool"},
        {"role": "tool", "content": "Tool output"},
        {"role": "assistant", "content": "Tool processed"},
    ]
    rendered = tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    if "<|im_start|>user\n<tool_response>" not in rendered:
        pytest.skip("tokenizer does not wrap tool replies as ChatML user messages")
    probe = tree._probe_role_template(tok)
    ids = tree._coerce_probe_token_ids(
        tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
    )
    anchors = tree._role_anchors(torch.tensor(ids, dtype=torch.long), probe)
    tool_start = len(tree._coerce_probe_token_ids(
        tok.apply_chat_template(messages[:1], tokenize=True, add_generation_prompt=False)
    ))
    assert (tool_start, "tool") in anchors
    assert (0, "user") in anchors
    assert len(anchors) >= 3


class FinalAssistantThinkingTokenizer(FakeTokenizer):
    """Final assistant has a thinking prelude; non-final assistant does not."""

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False):
        ids = []
        for i, msg in enumerate(messages):
            ids.extend([6001, *self.encode(msg["role"]), 6003])
            if msg["role"] == "assistant" and i == len(messages) - 1:
                ids.extend(self.encode("<think>\n\n</think>\n\n"))
            ids.extend(self.encode(msg["content"]))
            ids.append(6002)
        return ids


def test_role_probe_recovers_nonterminal_assistant_header():
    tok = FinalAssistantThinkingTokenizer()
    probe = tree._probe_role_template(tok)
    msgs = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "middle"},
        {"role": "user", "content": "second"},
        {"role": "assistant", "content": "last"},
    ]
    ids = tok.apply_chat_template(msgs)
    anchors = tree._role_anchors(torch.tensor(ids, dtype=torch.long), probe)
    expected = tuple(
        (len(tok.apply_chat_template(msgs[:i])) if i else 0, msg["role"])
        for i, msg in enumerate(msgs)
    )
    # For this context-sensitive fake, rendering a truncated prefix changes
    # the last assistant. Read its offset from the complete token stream instead.
    starts = [i for i, token in enumerate(ids) if token == 6001]
    assert anchors == tuple((pos, msg["role"]) for pos, msg in zip(starts, msgs))
    assert len(starts) == 4


def test_real_qwen_nonterminal_assistant_role_recall_when_configured():
    import os
    from transformers import AutoTokenizer

    path = os.environ.get("TPR_ROLE_TOKENIZER_PATH")
    if not path:
        pytest.skip("set TPR_ROLE_TOKENIZER_PATH for real-tokenizer validation")
    tok = AutoTokenizer.from_pretrained(
        path, local_files_only=True, trust_remote_code=False
    )
    msgs = [
        {"role": "user", "content": "Run a test"},
        {"role": "assistant", "content": "Calling tool now"},
        {"role": "tool", "content": "3 tests passed"},
        {"role": "assistant", "content": "Finished processing results"},
        {"role": "user", "content": "Thanks"},
    ]
    ids = tree._coerce_probe_token_ids(tok.apply_chat_template(
        msgs, tokenize=True, add_generation_prompt=False
    ))
    marker = tok.convert_tokens_to_ids("<|im_start|>")
    assert isinstance(marker, int) and marker >= 0
    expected = [i for i, token in enumerate(ids) if token == marker]
    probe = tree._probe_role_template(tok)
    anchors = tree._role_anchors(torch.tensor(ids, dtype=torch.long), probe)
    assert [i for i, _ in anchors] == expected
    assert [role for _, role in anchors] == [
        "user", "assistant", "tool", "assistant", "user"
    ]


def test_audit_extracts_exact_system_prompt_content():
    import importlib.util
    from pathlib import Path

    audit_path = (
        Path(__file__).resolve().parents[1]
        / "parallel"
        / "audit_role_probe_from_tq.py"
    )
    audit_spec = importlib.util.spec_from_file_location("tpr_system_span_audit_under_test", audit_path)
    audit = importlib.util.module_from_spec(audit_spec)
    audit_spec.loader.exec_module(audit)

    batch, _, _, _ = _make_batch()
    summary = audit._system_prompt_summary(batch, tree._probe_role_template(FakeTokenizer()), 6002)
    assert summary["rows"] == 2
    assert summary["system_messages"] == 2
    assert summary["complete_system_spans"] is True
    assert summary["identical_system_content_across_rows"] is True
    assert all(span["starts_at_zero"] for span in summary["per_row"])
    assert all(span["ends_with_im_end"] for span in summary["per_row"])
    assert all(span["within_prompt"] for span in summary["per_row"])
    assert all(n > 0 for n in summary["system_content_lengths"])


class NewlineTerminatedChatTokenizer(FakeTokenizer):
    """Chat messages end with [im_end, newline] before the next header."""

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False):
        ids = super().apply_chat_template(messages, tokenize, add_generation_prompt)
        output = []
        for token in ids:
            output.append(token)
            if token == 6002:
                output.append(ord("\n"))
        return output


def _load_system_prompt_audit():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "parallel" / "audit_role_probe_from_tq.py"
    spec = importlib.util.spec_from_file_location("tpr_system_end_audit_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_system_audit_accepts_template_im_end_plus_newline():
    tok = NewlineTerminatedChatTokenizer()
    prompt = [
        {"role": "system", "content": "shared system instructions"},
        {"role": "user", "content": "task"},
    ]
    sequences = [
        tok.apply_chat_template(prompt + [{"role": "assistant", "content": "answer A"}]),
        tok.apply_chat_template(prompt + [{"role": "assistant", "content": "answer B"}]),
    ]
    batch = {
        "input_ids": [torch.tensor(seq, dtype=torch.long) for seq in sequences],
        "prompts": [
            torch.tensor(tok.apply_chat_template(prompt), dtype=torch.long) for _ in sequences
        ],
    }
    result = _load_system_prompt_audit()._system_prompt_summary(
        batch, tree._probe_role_template(tok), 6002
    )
    assert result["rows_with_exactly_one_system"] == 2
    assert result["complete_system_spans"] is True
    assert result["identical_system_content_across_rows"] is True
    assert all(span["template_ending_token_ids"] == [6002, 10] for span in result["per_row"])
    assert all(span["content_tokens"] == len("shared system instructions") for span in result["per_row"])


def test_system_audit_rejects_unknown_trailing_token():
    tok = NewlineTerminatedChatTokenizer()
    system_only = [{"role": "system", "content": "some system prompt"}]
    encoded = tok.apply_chat_template(system_only)
    encoded[-1] = ord("!")  # im_end is present, but the actual suffix is invalid
    data = torch.tensor(encoded, dtype=torch.long)
    summary = _load_system_prompt_audit()._system_prompt_summary(
        {"input_ids": [data], "prompts": [data]},
        tree._probe_role_template(tok), 6002
    )
    assert summary["complete_system_spans"] is False
    assert summary["per_row"][0]["template_ending_token_ids"] is None
    assert summary["per_row"][0]["tail_token_ids_if_unmatched"][-2:] == [6002, ord("!")]


def test_real_qwen_system_audit_when_configured():
    import os
    from transformers import AutoTokenizer

    model = os.environ.get("TPR_ROLE_TOKENIZER_PATH")
    if not model:
        pytest.skip("set TPR_ROLE_TOKENIZER_PATH for real-tokenizer validation")
    tok = AutoTokenizer.from_pretrained(
        model, local_files_only=True, trust_remote_code=False
    )
    msgs = [
        {"role": "system", "content": "Code review instructions."},
        {"role": "user", "content": "Review this change."},
        {"role": "assistant", "content": "Looks good."},
    ]
    ids = tree._coerce_probe_token_ids(
        tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=False)
    )
    prompt = tree._coerce_probe_token_ids(
        tok.apply_chat_template(msgs[:2], tokenize=True, add_generation_prompt=False)
    )
    data = {
        "input_ids": [torch.tensor(ids, dtype=torch.long)],
        "prompts": [torch.tensor(prompt, dtype=torch.long)],
    }
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    summary = _load_system_prompt_audit()._system_prompt_summary(
        data, tree._probe_role_template(tok), im_end
    )
    assert summary["complete_system_spans"] is True
    assert summary["system_messages"] == 1
    assert summary["per_row"][0]["template_ending_token_ids"][0] == im_end
    assert summary["per_row"][0]["content_tokens"] > 0


class TerminalEosChatTokenizer(FakeTokenizer):
    """A template that appends eos_token_id only to the final rendered chat."""

    eos_token_id = 6004

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False):
        ids = super().apply_chat_template(messages, tokenize, add_generation_prompt)
        if not add_generation_prompt:
            ids.append(self.eos_token_id)
        return ids


def test_role_probe_handles_terminal_eos_added_to_prefix_rendering():
    tok = TerminalEosChatTokenizer()
    probe = tree._probe_role_template(tok)
    messages = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "middle"},
        {"role": "user", "content": "last"},
    ]
    ids = tok.apply_chat_template(messages)
    anchors = tree._role_anchors(torch.tensor(ids, dtype=torch.long), probe)
    expected = (
        (0, "user"),
        (len(tok.apply_chat_template(messages[:1])) - 1, "assistant"),
        (len(tok.apply_chat_template(messages[:2])) - 1, "user"),
    )
    assert anchors == expected


def test_real_llama3_tokenizer_role_probe_and_system_audit_when_configured():
    import os
    from transformers import AutoTokenizer

    path = os.environ.get("TPR_LLAMA3_TOKENIZER_PATH")
    if not path:
        pytest.skip("set TPR_LLAMA3_TOKENIZER_PATH for Llama 3 tokenizer validation")
    tok = AutoTokenizer.from_pretrained(
        path, local_files_only=True, trust_remote_code=False
    )

    def special_id(token):
        value = tok.convert_tokens_to_ids(token)
        assert isinstance(value, int)
        assert tok.convert_ids_to_tokens(value) == token
        return value

    start_header_id = special_id("<|start_header_id|>")
    eot_id = special_id("<|eot_id|>")
    assert tok.bos_token_id is not None

    shared_system = "You are a code review assistant. Follow the repository conventions."
    message_sets = [
        [
            {"role": "system", "content": shared_system},
            {"role": "user", "content": "Review change A."},
            {"role": "assistant", "content": "I will inspect change A."},
            {"role": "ipython", "content": "tests passed"},
            {"role": "assistant", "content": "Change A looks good."},
        ],
        [
            {"role": "system", "content": shared_system},
            {"role": "user", "content": "Review change B."},
            {"role": "assistant", "content": "I will inspect change B."},
            {"role": "ipython", "content": "one test failed"},
            {"role": "assistant", "content": "Change B needs a fix."},
        ],
    ]
    expected_roles = ["system", "user", "assistant", "ipython", "assistant"]
    probe = tree._probe_role_template(tok)
    input_rows = []
    prompt_rows = []

    for messages in message_sets:
        ids = tree._coerce_probe_token_ids(tok.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=False
        ))
        prompt_ids = tree._coerce_probe_token_ids(tok.apply_chat_template(
            messages[:2], tokenize=True, add_generation_prompt=False
        ))
        assert ids[0] == tok.bos_token_id

        header_positions = [i for i, token_id in enumerate(ids) if token_id == start_header_id]
        assert len(header_positions) == len(messages)
        expected = tuple(
            [(0, expected_roles[0])]
            + list(zip(header_positions[1:], expected_roles[1:]))
        )
        actual = tree._role_anchors(torch.tensor(ids, dtype=torch.long), probe)
        assert actual == expected

        input_rows.append(torch.tensor(ids, dtype=torch.long))
        prompt_rows.append(torch.tensor(prompt_ids, dtype=torch.long))

    summary = _load_system_prompt_audit()._system_prompt_summary(
        {"input_ids": input_rows, "prompts": prompt_rows},
        probe,
        eot_id,
    )
    assert summary["rows"] == 2
    assert summary["system_messages"] == 2
    assert summary["complete_system_spans"] is True
    assert summary["identical_system_content_across_rows"] is True
    assert summary["system_content_lengths"][0] > 0
    assert summary["system_content_lengths"][0] == summary["system_content_lengths"][1]
    assert all(span["within_prompt"] for span in summary["per_row"])

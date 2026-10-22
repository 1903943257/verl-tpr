"""CPU contracts for immutable short Actor captures in real Engine replay."""
from __future__ import annotations

import json

import pytest
import torch

pytest.importorskip("tensordict")
from tensordict import TensorDict

from verl.utils import tensordict_utils as tu
from ..parallel.prepare_verl_engine_short_manifest import prepare_short_manifest


def _eight_row_actor(path, *, same_uid: bool):
    # A controlled common prompt + eight actual divergent responses.
    # 10K shortening must preserve original labels at retained indices.
    prompt = torch.arange(1024, dtype=torch.int64) + 100
    prompt[-1] = 151645  # ChatML end turn
    responses = [
        torch.cat((
            torch.tensor([200000 + i], dtype=torch.int64),
            torch.arange(10500, dtype=torch.int64) + 600,
        ))
        for i in range(8)
    ]
    input_ids = [torch.cat((prompt, response)) for response in responses]
    fields = {
        "input_ids": input_ids,
        "prompts": [prompt.clone() for _ in range(8)],
        "responses": responses,
        "response_mask": [torch.ones(len(r), dtype=torch.bool) for r in responses],
        "loss_mask": [torch.ones(len(r), dtype=torch.int64) for r in responses],
        "old_log_probs": [torch.arange(len(r), dtype=torch.float32) / 100 for r in responses],
        "advantages": [torch.arange(len(r), dtype=torch.float32) / 200 for r in responses],
    }
    nested = {
        key: tu.nested_tensor_from_tensor_list(rows, ragged_idx=1)
        for key, rows in fields.items()
    }
    nested["temperature"] = torch.ones(8, dtype=torch.float32)
    data = TensorDict(nested, batch_size=[8])
    keys = [f"{'uid1' if same_uid else f'uid{i}'}_tpr_{i}" for i in range(8)]
    torch.save({
        "capture_stage": "actor_update_mini_batch",
        "capture_global_step": 1,
        "keys": keys,
        "tensordict": data,
    }, path)


def test_short_replay_keeps_labels_and_actual_branching(tmp_path):
    source = tmp_path / "original.pt"
    _eight_row_actor(source, same_uid=True)
    before = torch.load(source, map_location="cpu", weights_only=False)
    out = tmp_path / "replay"
    manifest = prepare_short_manifest(source, out, steps=1, cap=10240)
    parsed = json.loads(manifest.read_text())
    item = parsed["items"][0]
    assert item["actual_tree_count"] == 1
    assert item["actual_segment_count"] > item["actual_tree_count"]
    assert item["actual_prefix_reuse_ratio"] > 1
    assert item["max_sequence_length"] == 10240
    assert item["conditioning_changed"] is False
    assert parsed["controlled_short_replay"] is True
    derived = torch.load(item["capture"], map_location="cpu", weights_only=False)
    for row in range(8):
        assert torch.equal(
            derived["tensordict"]["old_log_probs"][row],
            before["tensordict"]["old_log_probs"][row][:9216],
        )
        assert torch.equal(
            derived["tensordict"]["advantages"][row],
            before["tensordict"]["advantages"][row][:9216],
        )
    assert len(before["tensordict"]["input_ids"][0]) > 10240


def test_short_replay_rejects_unshared_roots(tmp_path):
    source = tmp_path / "independent.pt"
    _eight_row_actor(source, same_uid=False)
    with pytest.raises(ValueError, match="no genuine shared prefix"):
        prepare_short_manifest(source, tmp_path / "replay", steps=1, cap=10240)


def test_short_replay_accepts_existing_derived_pt(tmp_path):
    source = tmp_path / "original.pt"
    _eight_row_actor(source, same_uid=True)
    first = tmp_path / "first"
    original_manifest = prepare_short_manifest(source, first, steps=1, cap=10240)
    derived_path = json.loads(original_manifest.read_text())["items"][0]["capture"]
    second = tmp_path / "second"
    result = prepare_short_manifest(
        type(source)(derived_path), second, steps=1, cap=0,
    )
    manifest = json.loads(result.read_text())
    assert manifest["items"][0]["capture"] == derived_path
    assert manifest["items"][0]["actual_prefix_reuse_ratio"] > 1

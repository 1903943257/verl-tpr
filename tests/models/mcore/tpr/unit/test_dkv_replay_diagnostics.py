import json

import torch

from verl.workers.engine.megatron.tpr.dkv_diagnostics import observe_dkv_replay_root
from verl.workers.engine.megatron.tpr.state import KVPrefixState


def _read_records(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_accumulation_diagnostic_distinguishes_bf16_input_and_fp32_buffer(tmp_path, monkeypatch):
    output = tmp_path / "dkv.jsonl"
    monkeypatch.setenv("TPR_DKV_REPLAY_DIAG_PATH", str(output))
    monkeypatch.setenv("TPR_EXPERIMENT_DELAYED_DKV_FP32", "1")

    key = torch.ones((4, 1, 2, 4), dtype=torch.bfloat16)
    value = torch.full_like(key, 2)
    state = KVPrefixState(0, 4, {1: (key, value)})

    for factor in (1.0, 1.0 / 256):
        anchors = state.make_anchors()
        anchor_key, anchor_value = anchors.key_values[1]
        (factor * (anchor_key.sum() + anchor_value.sum())).backward()
        state.accumulate_anchor_gradients(anchors, consume=True)

    records = _read_records(output)
    assert len(records) == 4
    for record in records:
        assert record["event"] == "dkv_accumulate"
        assert record["segment_id"] == "0"
        assert record["layer"] == 1
        assert record["incoming"]["dtype"] == "torch.bfloat16"
        assert record["buffer"]["dtype"] == "torch.float32"
    assert {record["accumulation_count"] for record in records} == {1, 2}


def test_replay_root_hook_records_actual_autograd_gradient_dtype(tmp_path, monkeypatch):
    output = tmp_path / "replay.jsonl"
    monkeypatch.setenv("TPR_DKV_REPLAY_DIAG_PATH", str(output))

    root = torch.ones(8, dtype=torch.bfloat16, requires_grad=True)
    seed = torch.full((8,), 1.0001, dtype=torch.float32)
    observe_dkv_replay_root(
        root,
        seed,
        segment_id=7,
        layer=3,
        kind="key",
    )
    torch.autograd.backward((root,), (seed,))

    records = _read_records(output)
    assert [record["event"] for record in records] == [
        "dkv_replay_input",
        "dkv_replay_root_hook",
    ]
    replay_input, replay_hook = records
    assert replay_input["root"]["dtype"] == "torch.bfloat16"
    assert replay_input["seed_gradient"]["dtype"] == "torch.float32"
    assert replay_hook["actual_gradient"]["dtype"] in {
        "torch.bfloat16",
        "torch.float16",
        "torch.float32",
    }
    assert replay_hook["seed_to_hook"]["shape_match"] is True


def test_dkv_diagnostics_are_disabled_without_output_path(tmp_path, monkeypatch):
    output = tmp_path / "should_not_exist.jsonl"
    monkeypatch.delenv("TPR_DKV_REPLAY_DIAG_PATH", raising=False)

    root = torch.ones(4, dtype=torch.bfloat16, requires_grad=True)
    seed = torch.ones(4, dtype=torch.float32)
    observe_dkv_replay_root(root, seed, segment_id=1, layer=1, kind="value")

    assert not output.exists()

"""CPU-only capture/control gate for Qwen3-4B N8 CP1 8:1 performance."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from tensordict import TensorDict

from verl.utils import tensordict_utils as tu
from .prepare_tpr_cp1_n8_perf_capture import derive_capture, PREFIX, SUFFIX, BRANCHES
from .compare_tpr_cp1_n8_perf import compare


def _nested(parts):
    return tu.nested_tensor_from_tensor_list(parts, ragged_idx=1)


def _original_source(root: Path, *, divergent_prompt: bool = False,
                     prefix: int = PREFIX, suffix: int = SUFFIX):
    prompt = torch.arange(prefix + 32, dtype=torch.long) + 100
    prompts = [prompt.clone() for _ in range(BRANCHES)]
    if divergent_prompt:
        prompts[7][-1] = 55555
    responses = [
        torch.arange(suffix + 20, dtype=torch.long) + 25000 + i * 2000
        for i in range(BRANCHES)
    ]
    masks = [torch.ones(suffix + 20, dtype=torch.bool) for _ in range(BRANCHES)]
    data = TensorDict({
        "input_ids": _nested([torch.cat((p, r)) for p, r in zip(prompts, responses)]),
        "prompts": _nested(prompts),
        "responses": _nested(responses),
        "response_mask": _nested(masks),
        "loss_mask": _nested([x.clone() for x in masks]),
        "old_log_probs": _nested([
            torch.arange(suffix+20, dtype=torch.float32) * (-0.01)
            for _ in range(BRANCHES)
        ]),
        "advantages": _nested([
            torch.arange(suffix+20, dtype=torch.float32) * 0.01
            for _ in range(BRANCHES)
        ]),
        "temperature": torch.full((BRANCHES,), 0.6),
    }, batch_size=[BRANCHES])
    source = root/"source.pt"
    torch.save({
        "capture_stage":"actor_update_mini_batch",
        "tensordict":data,
        "keys":tuple(f"session_{i}_tpr_0" for i in range(BRANCHES)),
    }, source)
    return source, data


def test_n8_capture_preserves_real_ppo_labels_and_actual_turn_root(tmp_path: Path):
    source, data = _original_source(tmp_path)
    target = tmp_path/"p8192_s1024.pt"
    audit = derive_capture(source, target)
    assert audit["n"] == 8
    assert audit["prefix"] == 8192 and audit["suffix"] == 1024
    assert audit["forest_trees"] == 1
    assert audit["forest_segments"] == 9
    assert audit["ppo_tokens"] == 8192
    assert audit["changed_conditioning"] is True
    output = torch.load(target, map_location="cpu", weights_only=False)
    assert len(set(output["keys"])) == 8
    derived = output["tensordict"]
    for i in range(BRANCHES):
        assert len(derived["prompts"][i]) == PREFIX
        assert len(derived["responses"][i]) == SUFFIX
        assert len(derived["input_ids"][i]) == PREFIX + SUFFIX
        for k in ("responses", "response_mask", "loss_mask",
                  "old_log_probs", "advantages"):
            torch.testing.assert_close(derived[k][i], data[k][i][:SUFFIX])
    with pytest.raises(FileExistsError):
        derive_capture(source, target)


def test_n8_qwen_turn_root_rounding_at_short_prompt_tail(tmp_path: Path):
    # The identical prompt ends 64 tokens after the preceding ChatML turn.
    # With production turn_min_segment_tokens=128, this short final piece
    # is merged with a response-dependent turn: true reusable Root=P-64.
    source, batch = _original_source(tmp_path)
    from verl.workers.engine.megatron.tpr.tree import QWEN3_IM_END_TOKEN_ID
    original = torch.load(source, map_location="cpu", weights_only=False)
    prompt = batch["prompts"][0].clone()
    prompt[-65] = QWEN3_IM_END_TOKEN_ID
    data = batch.clone(recurse=False)
    prompts = [prompt.clone() for _ in range(BRANCHES)]
    responses = [batch["responses"][i] for i in range(BRANCHES)]
    data.set("prompts", _nested(prompts))
    data.set("input_ids", _nested([
        torch.cat((prompts[i], responses[i])) for i in range(BRANCHES)
    ]))
    original["tensordict"] = data
    torch.save(original, source)
    audit = derive_capture(source, tmp_path/"rounded.pt")
    assert audit["actual_shared_root_tokens"] == PREFIX - 64
    assert audit["nominal_prefix_minus_actual_root"] == 64
    assert audit["effective_root_to_suffix_ratio"] == (PREFIX - 64) / SUFFIX
    assert audit["forest_trees"] == 1
    assert audit["forest_segments"] == 9


def test_n8_capture_rejects_missing_common_prefix(tmp_path: Path):
    source, _ = _original_source(tmp_path, divergent_prompt=True)
    with pytest.raises(ValueError, match="identical"):
        derive_capture(source, tmp_path/"invalid.pt")


@pytest.mark.parametrize("p,s", [(13824, 13824), (24576, 3072)])
def test_cp4_27k_capture_exact_real_label_contract(tmp_path: Path, p: int, s: int):
    source, original = _original_source(tmp_path, prefix=p, suffix=s)
    target = tmp_path/"cp4_27k.pt"
    audit = derive_capture(source, target, prefix=p, suffix=s)
    assert (audit["prefix"], audit["suffix"], audit["length"], audit["n"]) == (
        p, s, 27648, 8
    )
    assert audit["forest_trees"] == 1
    assert audit["forest_segments"] >= 9
    assert audit["ppo_tokens"] == 8*s
    data = torch.load(target, map_location="cpu", weights_only=False)["tensordict"]
    for i in range(BRANCHES):
        assert data["input_ids"][i].numel() == 27648
        assert data["prompts"][i].numel() == p
        assert data["responses"][i].numel() == s
        for key in ("responses", "response_mask", "loss_mask",
                    "old_log_probs", "advantages"):
            assert torch.equal(data[key][i], original[key][i][:s])


def test_cp4_27k_rejects_short_response_without_padding(tmp_path: Path):
    source, original = _original_source(tmp_path, prefix=13824, suffix=10000)
    with pytest.raises(ValueError, match="response_shortfall"):
        derive_capture(source, tmp_path/"invalid.pt",
                       prefix=13824, suffix=13824)


def test_cpu_comparison_has_strict_same_input_and_baseline_label(tmp_path: Path):
    base = {
        "model":"Qwen3-4B-128K", "checkpoint":"qwen",
        "topology":{"tp":2,"cp":1,"dp":1,"world":2},
        "branches":8, "prefix":8192, "suffix":1024, "length":9216,
        "source_capture":"source.pt", "source_input_sha256":"a"*64,
        "ppo_tokens":8192, "condition_changed":True,
        "synthetic_shared_uid":True, "recompute":False, "optimizer_step":False,
        "warmups":1, "repeats":3,
        "forest_trees":8, "forest_segments":8,
        "actual_shared_root_tokens":8192, "unique_tree_tokens":16384,
        "baseline_kind":"CONTROLLED_WHOLE_SQUARE_CANN_NO_TPR_CONTEXT",
        "median_fwd_bwd_s":4.,
        "worst_peak_allocated_gib":30.,
        "worst_peak_reserved_gib":34.,
        "runs":[
            {"warmup": i==0, "critical_fwd_bwd_finalize_s":4.,
             "peak_allocated_gib":30.}
            for i in range(4)
        ],
    }
    full = base
    tpr = {**base, "mode":"tpr", "forest_trees":1, "forest_segments":9,
           "baseline_kind":"PRODUCTION_TPR_SEGMENT_EXECUTOR_CP1",
           "median_fwd_bwd_s":2.,
           "worst_peak_allocated_gib":20.}
    (tmp_path/"full.json").write_text(json.dumps(full))
    (tmp_path/"tpr.json").write_text(json.dumps(tpr))
    report = compare(tmp_path)
    assert report["speedup_full_over_tpr"] == 2
    assert report["memory_saving_fraction"] == pytest.approx(1/3)
    with pytest.raises(FileExistsError):
        compare(tmp_path)
    (tmp_path/"comparison.json").unlink()
    tpr["source_input_sha256"] = "b"*64
    (tmp_path/"tpr.json").write_text(json.dumps(tpr))
    with pytest.raises(AssertionError, match="mismatch"):
        compare(tmp_path)

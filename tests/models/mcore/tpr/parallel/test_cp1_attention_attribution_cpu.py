"""CPU-only contract for automated CP1 attention numerical attribution."""
from __future__ import annotations

import json

import pytest
import torch

from .cp1_attention_attribution import _difference
from .analyze_cp1_attention_attribution import _intersection, analyze


def _sample(value: float):
    return {"positions": [0, 1, 3], "values": torch.tensor([[1., 2.], [3., 4.], [value, 6.]])}


def _record(kind):
    return {f"r0|{kind}|L1|{stage}": _sample(5.0 if stage == "qkv_in" else 4.5)
            for stage in ("attn_in", "qkv_in", "qkv_out", "post_rope_k", "raw_v")}


def test_same_shape_metrics_and_aligned_absolute_positions():
    x = _sample(7.0)
    y = _sample(7.0)
    assert _difference(x["values"], y["values"])["exact"]
    assert _intersection(x, y)["changed"] == 0
    z = {"positions": [3, 0], "values": x["values"][[2, 0]].clone()}
    assert _intersection(x, z)["exact"]
    with pytest.raises(AssertionError, match="shape"):
        _difference(torch.zeros(2), torch.zeros(3))


def test_autoconclusion_identifies_real_qkv_m_effect(tmp_path):
    modes = {}
    for mode, kind in (("full", "full"), ("split_no_share", "push"),
                       ("shared", "push")):
        records = _record(kind)
        if mode != "full":
            for stage in ("attn_in", "qkv_in", "qkv_out", "post_rope_k", "raw_v"):
                records[f"r0|pop|L1|{stage}"] = _sample(
                    5.0 if stage == "qkv_in" else 4.5
                )
            records.update(_record("leaf"))
        if mode == "split_no_share":
            records["r0|push|L1|qkv_out"] = _sample(4.0)
        modes[mode] = {
            "mode": mode, "no_fixed_tile": True, "row_lengths": [8, 8],
            "root_length": 4, "records": records,
            "same_qkv_attention": {},
            "same_input_gemm_replay": {
                "input_equal": True,
                "qkv_same_input_m_replay": {"changed": 1},
                "full_qkv_replay_vs_actual": {"exact": True},
                "split_root_qkv_replay_vs_actual": {"exact": True},
            } if mode == "split_no_share" else {},
            "kv_relay_roundoff": {},
        }
        (tmp_path / mode).mkdir()
        torch.save(modes[mode], tmp_path / mode / "trace.pt")
    (tmp_path / "comparison_split_no_share.json").write_text(json.dumps({
        "ppo_tokens": 5,
        "comparisons": {
            k: {
                "logprob": {"relative_l2": 0.0},
                "loss_abs_gap": 0.0,
                "canonical_sampled_grad": {"relative_l2": 0.0, "cosine": 1.0},
                "complete_norm_grad": {"relative_l2": 0.0},
            }
            for k in ("full_vs_one_node", "one_node_vs_split_no_share",
                      "split_no_share_vs_shared", "full_vs_shared")
        },
    }))
    result = analyze(tmp_path)
    assert result["qkv_m_shape_effect_confirmed"] is True
    assert result["first_root_stage_difference"]["stage"] == "qkv_out"
    assert result["fixed_tile_enabled"] is False
    assert any("CONFIRMED" in line for line in result["conclusions"])

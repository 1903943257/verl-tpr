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
from .prepare_tpr_n8_synthetic_windows import (prepare_window_source, _choose_later_supervised_window)


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


@pytest.mark.parametrize("p,s", [(4608, 4608), (16384, 2048),
                                 (9216, 9216), (6016, 6016),
                                 (12416, 6016), (13824, 13824),
                                 (24576, 3072), (18432, 18432),
                                 (32768, 4096), (17440, 17440)])
def test_extra_n8_shapes_preserve_real_ppo_labels(tmp_path: Path, p: int, s: int):
    source, original = _original_source(tmp_path, prefix=p, suffix=s)
    target = tmp_path/f"p{p}_s{s}.pt"
    audit = derive_capture(source, target, prefix=p, suffix=s)
    assert (audit["prefix"], audit["suffix"], audit["length"], audit["n"]) == (
        p, s, p+s, 8
    )
    assert audit["forest_trees"] == 1
    assert audit["forest_segments"] >= 9
    assert audit["ppo_tokens"] == 8*s
    data = torch.load(target, map_location="cpu", weights_only=False)["tensordict"]
    for i in range(BRANCHES):
        assert data["input_ids"][i].numel() == p+s
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


@pytest.mark.parametrize("p,s", [(9216, 9216), (17440, 17440)])
def test_n8_synthetic_window_source_keeps_aligned_original_ppo_labels(tmp_path: Path, p: int, s: int):
    # The source has long enough responses; the two extra branches are
    # shifted windows of rows 0 and 3, not exact copied response duplicates.
    source, original = _original_source(tmp_path, prefix=p, suffix=2*s)
    # Mimic the production actor capture containing critic-only nested
    # returns/values: once P == S these become alignment-ambiguous.
    recorded = torch.load(source, map_location="cpu", weights_only=False)
    recorded["tensordict"].set("returns", _nested([
        torch.arange(original["responses"][i].numel(), dtype=torch.float32)
        for i in range(BRANCHES)
    ]))
    recorded["tensordict"].set("values", _nested([
        torch.ones(original["responses"][i].numel(), dtype=torch.float32)
        for i in range(BRANCHES)
    ]))
    torch.save(recorded, source)
    stage = tmp_path / "window_source.pt"
    provenance = prepare_window_source(source, stage, prefix=p, suffix=s)
    assert provenance["synthetic_response_windows"] is True
    assert provenance["on_policy_valid"] is False
    assert [entry["donor_source_row"] for entry in provenance["source_rows"]] == [
        0, 1, 3, 4, 5, 6, 0, 3,
    ]
    assert [entry["response_start"] for entry in provenance["source_rows"]] == [
        0, 0, 0, 0, 0, 0, s, s,
    ]
    staged = torch.load(stage, map_location="cpu", weights_only=False)
    batch = staged["tensordict"]
    assert "returns" in batch and "values" in batch
    for i, entry in enumerate(provenance["source_rows"]):
        donor, offset = entry["donor_source_row"], entry["response_start"]
        assert batch["input_ids"][i].numel() == p+s
        assert torch.equal(batch["prompts"][i], original["prompts"][donor][-p:])
        for key in ("responses", "response_mask", "loss_mask",
                    "old_log_probs", "advantages"):
            assert torch.equal(batch[key][i],
                               original[key][donor][offset:offset+s])
    with pytest.raises(FileExistsError):
        prepare_window_source(source, stage, prefix=p, suffix=s)

    derived = tmp_path / "derived.pt"
    audit = derive_capture(stage, derived, prefix=p, suffix=s)
    assert audit["synthetic_response_windows"] is True
    assert audit["original_actor_capture"] == str(source.resolve())
    assert set(("returns", "values")).issubset(audit["dropped_actor_unused_fields"])
    derived_batch = torch.load(derived, map_location="cpu", weights_only=False)["tensordict"]
    assert "returns" not in derived_batch and "values" not in derived_batch
    assert all(key in derived_batch for key in (
        "input_ids", "prompts", "responses", "response_mask",
        "loss_mask", "old_log_probs", "advantages", "temperature"
    ))
    assert audit["forest_trees"] == 1 and audit["forest_segments"] >= 9
    assert audit["prefix"] == p and audit["suffix"] == s


@pytest.mark.parametrize("suffix", [9216, 17440])
def test_synthetic_later_window_falls_back_to_supervised_span(suffix: int):
    mask = torch.ones(2*suffix + 20, dtype=torch.bool)
    mask[suffix:2*suffix] = False
    start, count = _choose_later_supervised_window(mask, suffix)
    assert start > suffix
    assert start + suffix <= len(mask)
    assert count == int(mask[start:start+suffix].sum()) > 0
    with pytest.raises(ValueError, match="No later nonoverlapping window"):
        _choose_later_supervised_window(torch.cat((
            torch.ones(suffix, dtype=torch.bool),
            torch.zeros(suffix + 20, dtype=torch.bool),
        )), suffix)


def test_cp2_synthetic_window_adapts_to_sparse_supervision(tmp_path: Path):
    p = s = 9216
    source, original = _original_source(tmp_path, prefix=p, suffix=2*s)
    raw = torch.load(source, map_location="cpu", weights_only=False)
    masks = [raw["tensordict"]["response_mask"][i].clone()
             for i in range(BRANCHES)]
    for donor in (0, 3):
        masks[donor][s:2*s] = False
    raw["tensordict"].set("response_mask", _nested(masks))
    torch.save(raw, source)
    stage = tmp_path / "sparse_window_source.pt"
    provenance = prepare_window_source(source, stage, prefix=p, suffix=s)
    shifted = provenance["source_rows"][6:]
    assert [item["donor_source_row"] for item in shifted] == [0, 3]
    assert all(item["response_start"] > s for item in shifted)
    assert all(item["supervised_tokens"] > 0 for item in shifted)
    assert all(item["window_selection"] == "adaptive_later_supervised"
               for item in shifted)
    prepared = torch.load(stage, map_location="cpu", weights_only=False)["tensordict"]
    for i, window in enumerate(provenance["source_rows"]):
        donor, start = window["donor_source_row"], window["response_start"]
        for name in ("responses", "response_mask", "loss_mask",
                     "old_log_probs", "advantages"):
            assert torch.equal(
                prepared[name][i],
                raw["tensordict"][name][donor][start:start+s],
            )
    audit = derive_capture(stage, tmp_path / "adapted.pt", prefix=p, suffix=s)
    assert audit["synthetic_response_windows"] is True
    assert audit["forest_segments"] >= 9


@pytest.mark.parametrize("p,s", [(8192, 1024), (4608, 4608)])
def test_cpu_comparison_has_strict_same_input_and_baseline_label(tmp_path: Path, p: int, s: int):
    base = {
        "model":"Qwen3-4B-128K", "checkpoint":"qwen",
        "topology":{"tp":2,"cp":1,"dp":1,"world":2},
        "branches":8, "prefix":p, "suffix":s, "length":p+s,
        "source_capture":"source.pt", "source_input_sha256":"a"*64,
        "ppo_tokens":8*s, "condition_changed":True,
        "synthetic_shared_uid":True, "recompute":False, "optimizer_step":False,
        "warmups":1, "repeats":3,
        "forest_trees":8, "forest_segments":8,
        "actual_shared_root_tokens":p, "unique_tree_tokens":p+8*s,
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
    report = compare(tmp_path, prefix=p, suffix=s)
    assert report["speedup_full_over_tpr"] == 2
    assert report["memory_saving_fraction"] == pytest.approx(1/3)
    assert report["actual_shared_root_tokens"] == p
    comparison_path = tmp_path / "comparison.json"
    assert comparison_path.is_file()
    assert json.loads(comparison_path.read_text()) == report
    with pytest.raises(FileExistsError):
        compare(tmp_path, prefix=p, suffix=s)
    (tmp_path/"comparison.json").unlink()
    tpr["source_input_sha256"] = "b"*64
    (tmp_path/"tpr.json").write_text(json.dumps(tpr))
    with pytest.raises(AssertionError, match="mismatch"):
        compare(tmp_path, prefix=p, suffix=s)


def test_phase_trace_classifies_only_npu_kernel_and_hccl_lanes(tmp_path: Path):
    from .tpr_phase_diagnostics import trace_kernel_summary, segment_event_summary
    trace = tmp_path / "trace.json"
    trace.write_text(json.dumps({"traceEvents": [
        {"ph": "X", "cat": "NPU Kernel", "name": "FlashAttentionScore", "dur": 2800},
        {"ph": "X", "cat": "HCCL Communication", "name": "HCCL_AllReduce", "dur": 1200},
        # Current CANN Chrome export sometimes omits `cat` for device events.
        {"ph": "X", "name": "FlashAttentionScoreGrad", "dur": 3100,
         "args": {"Task Type": "KERNEL_AICORE", "Physic Stream Id": 46, "Task Id": 3}},
        {"ph": "X", "name": "HcclAllReduce", "dur": 600,
         "args": {"Task Type": "KERNEL_AIVEC", "Physic Stream Id": 47, "Task Id": 4}},
        {"ph": "X", "cat": "CPU", "name": "HCCL_AllReduce", "dur": 700000},
        {"ph": "X", "cat": "CPU", "name": "aten::matmul", "dur": 8000},
        {"ph": "X", "cat": "CPU", "name": "FlashAttentionScore", "dur": 8000},
        {"ph": "X", "cat": "HCCL API", "name": "HcclAllReduce", "dur": 100000},
    ]}))
    metrics = trace_kernel_summary(trace)
    assert metrics["attention_kernel_sum_ms"] == pytest.approx(5.9)
    assert metrics["hccl_kernel_sum_ms"] == pytest.approx(1.8)
    assert metrics["attention_events"] == 2 and metrics["hccl_events"] == 2
    assert trace_kernel_summary(None)["attention_kernel_sum_ms"] is None
    segment = segment_event_summary([
        {"kind": "push", "npu_event_ms": 1.5},
        {"kind": "visit_leaf", "npu_event_ms": 2.0},
        {"kind": "visit_leaf", "npu_event_ms": 3.0},
        {"kind": "pop", "npu_event_ms": 0.7},
    ])
    assert segment["visit_leaf"] == {"npu_event_sum_ms": 5.0, "segments": 2}



def test_phase_trace_invalid_json_keeps_npu_kernel_timings_null(tmp_path: Path):
    from .tpr_phase_diagnostics import trace_kernel_summary
    trace = tmp_path / "incomplete_trace.json"
    # Mirrors a truncated CANN Chrome trace: events exported, closing ']' absent.
    invalid = '[{"ph": "X", "cat": "NPU Kernel", "name": "FlashAttentionScore", "dur": 2000}'
    trace.write_text(invalid)
    result = trace_kernel_summary(trace)
    assert result["attention_kernel_sum_ms"] is None
    assert result["hccl_kernel_sum_ms"] is None
    assert result["reason"] == "trace_invalid_json"
    assert "JSONDecodeError" in result["trace_error"]
    assert trace.read_text() == invalid  # original trace left untouched

    trace.write_text('{"unexpected": []}')
    result = trace_kernel_summary(trace)
    assert result["reason"] == "trace_events_missing"
    assert result["attention_kernel_sum_ms"] is None


def test_phase_comparison_requires_matching_unprofiled_sha(tmp_path: Path):
    from .compare_tpr_cp1_cp4_phase_diagnostics import collect_diagnostic
    base = {
        "mode": "tpr", "checkpoint": "/model/qwen",
        "topology": {"tp": 2, "cp": 4, "dp": 1, "world": 8},
        "input_sha256": "a" * 64, "median_fwd_bwd_s": 5.,
    }
    current = dict(base, phase_diagnostic=True, median_fwd_bwd_s=6.,
                   runs=[{"warmup": False, "phase_diagnostics": [
                       {"rank": 0, "stages": {
                           "forward": {"npu_event_sum_ms": 2500., "events": 9},
                           "backward": {"npu_event_sum_ms": 1800., "events": 9},
                       }, "kernels": {
                           "attention_kernel_sum_ms": 800.,
                           "hccl_kernel_sum_ms": 200.,
                       }},
                   ], "profiles": [{"rank": 0, "stage_ms": {
                       "tree_build_cpu_ms": 20., "tp_plan_collective_ms": 4.,
                       "executor_setup_cpu_ms": 6.,
                   }, "events": [{"kind": "push", "npu_event_ms": 100.}]}]}])
    ref, phase = tmp_path / "ref.json", tmp_path / "phase.json"
    ref.write_text(json.dumps(base))
    phase.write_text(json.dumps(current))
    result = collect_diagnostic(phase, ref, topology="TP2CP4")
    assert result["stages_ms"]["forward"] == 2500.
    assert result["stages_ms"]["attention"] == 800.
    assert result["stages_ms"]["hccl"] == 200.
    assert result["stages_ms"]["setup"] == 30.
    assert result["stages_ms"]["profiled"] == 6000.
    assert result["stages_ms"]["unprofiled"] == 5000.
    assert result["stages_ms"]["ppo_loss"] is None
    current["input_sha256"] = "b" * 64
    phase.write_text(json.dumps(current))
    with pytest.raises(AssertionError, match="inputs differ"):
        collect_diagnostic(phase, ref, topology="TP2CP4")

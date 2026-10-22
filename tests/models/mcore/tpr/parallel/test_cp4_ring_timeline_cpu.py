"""CPU-only regression for Ascend CP4 Ring timestamp attribution."""
from __future__ import annotations

import csv
import pytest

from tests.models.mcore.tpr.parallel.compare_cp4_ring_timeline import (
    classify, concurrency, same_stream_conflicts, summarize, union,
    Event,
)


def test_classify_non_hccl_compute():
    assert classify("aclnnMatMul") == "GEMM"
    assert classify("npu_fusion_attention") == "Attention"
    assert classify("hcom_allGather__030") == "HCCL"
    assert classify("relu") == "Other NPU"


def test_interval_union_not_sum_and_stream_conflict():
    records = [
        Event("hcom_send", "HCCL", 1000, 1000, "7", "unit"),
        Event("hcom_send", "HCCL", 1250, 750, "7", "unit"),
        Event("hcom_recv", "HCCL", 1250, 750, "8", "unit"),
    ]
    assert union((e.ts, e.end) for e in records) == [[1000, 2000]]
    assert concurrency(records) == 3
    overlaps = same_stream_conflicts(records)
    assert len(overlaps) == 1
    assert overlaps[0]["stream"] == "7"


def test_csv_timeline_gemm_attention_hccl_and_gap(tmp_path):
    p = tmp_path / "ASCEND_PROFILER_OUTPUT"
    p.mkdir()
    with (p / "kernel_details.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(("Name", "Type", "Start Time", "Duration(us)", "Stream ID"))
        writer.writerows((
            ("aclnnMatMul", "Compute", 1000, 2000, 1),
            ("npu_fusion_attention", "Compute", 3000, 1000, 1),
            ("hcom_send", "HCCL", 4000, 1000, 3),
            ("hcom_send", "HCCL", 4250, 750, 3),
            ("hcom_allGather", "HCCL", 6000, 500, 3),
        ))
    report = summarize(p, "us", 0, None, 0.1, no_trace=True)
    assert report["by_category"]["GEMM"]["count"] == 1
    assert report["by_category"]["Attention"]["count"] == 1
    assert report["by_category"]["HCCL"]["count"] == 3
    assert report["hccl_union_ms"] == pytest.approx(1.5)
    assert report["npu_gap_total_ms"] == pytest.approx(1.0)
    assert report["max_concurrent_hccl"] == 2
    assert report["same_stream_overlap_examples"]
    assert any("Same-stream NPU tasks overlap" in x for x in report["warnings"])


def test_missing_kernel_data_is_explicit_error(tmp_path):
    with pytest.raises(ValueError, match="No device time axis"):
        summarize(tmp_path, "us", 0, None, 0.1, no_trace=True)

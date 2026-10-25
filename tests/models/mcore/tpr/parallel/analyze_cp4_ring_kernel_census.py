#!/usr/bin/env python3
"""Offline same-rank kernel census for CP4 Full vs TPR (stdlib only).

No NPU, no trace_view.json load, and no ungrounded Prefix/Current attribution.
Device kernel durations overlap with HCCL and must NOT be summed as wall time.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import re
import statistics


def normalized(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def value(row, *names):
    keys = {normalized(k): v for k, v in row.items() if k is not None}
    for name in names:
        found = keys.get(normalized(name))
        if found is not None and str(found).strip():
            return found
    return None


def locate_csv(root: Path) -> Path:
    if root.is_file() and root.name == "kernel_details.csv":
        return root
    if not root.exists():
        raise FileNotFoundError(root)
    matches = list(root.rglob("kernel_details.csv"))
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one rank's kernel_details.csv under {root}, "
            f"found {len(matches)}: {list(map(str, matches[:5]))}"
        )
    return matches[0]


def classify(name: str) -> str:
    s = name.lower()
    if re.search(r"hcom|hccl|allreduce|allgather|reducescatter|alltoall|\bsend\b|\breceive\b", s):
        return "HCCL"
    if "flashattention" in s or "flash_attention" in s or "attentionscore" in s:
        if "grad" in s or "backward" in s:
            return "FA_BWD"
        if "score" in s or "forward" in s:
            return "FA_FWD"
        return "Attention_OTHER"
    if re.search(r"matmul|groupedmat|\bgemm\b|batchmat|\bbmm\b", s):
        return "GEMM"
    return "Other"


def family(name: str) -> str:
    return name.split("_", 1)[0].strip() or "UNKNOWN"


def summary(values: list[float]) -> dict:
    if not values:
        return {"count": 0, "sum_ms": 0.0, "mean_us": 0.0,
                "p50_us": 0.0, "p90_us": 0.0, "under_100us_pct": 0.0}
    ordered = sorted(values)
    n = len(ordered)
    return {
        "count": n,
        "sum_ms": sum(values) / 1000,
        "mean_us": statistics.mean(values),
        "p50_us": ordered[(n - 1) // 2],
        "p90_us": ordered[int((n - 1) * .9)],
        "under_100us_pct": 100 * sum(x < 100 for x in values) / n,
    }


def collect(path: Path) -> dict:
    groups = defaultdict(list)
    families = defaultdict(list)
    shape_buckets = defaultdict(list)
    shape_rows = attention_rows = 0
    with path.open(encoding="utf-8-sig", newline="") as fp:
        reader = csv.DictReader(fp)
        headers = reader.fieldnames or []
        shape_columns = [k for k in headers if "shape" in normalized(k)]
        for line_number, row in enumerate(reader, 2):
            name = value(row, "Name", "op_name", "Operator Name")
            duration = value(row, "Duration(us)", "Duration (us)", "Duration",
                             "task_duration(us)", "task_duration")
            if duration is None:
                raise ValueError(f"{path}:{line_number}: missing duration; headers={headers}")
            try:
                dur = float(str(duration).replace(",", ""))
            except ValueError as exc:
                raise ValueError(f"{path}:{line_number}: invalid duration={duration!r}") from exc
            if not math.isfinite(dur) or dur < 0:
                raise ValueError(f"{path}:{line_number}: invalid duration={duration!r}")
            kind = classify(str(name or ""))
            fam = family(str(name or "UNKNOWN"))
            groups[kind].append(dur)
            families[(kind, fam)].append(dur)
            if kind in ("FA_FWD", "FA_BWD", "Attention_OTHER"):
                attention_rows += 1
                shape = " | ".join(
                    f"{column}={row[column]}" for column in shape_columns
                    if row.get(column) and str(row[column]).strip()
                )
                if shape:
                    shape_rows += 1
                    shape_buckets[(kind, fam, shape)].append(dur)
    return {
        "path": str(path), "headers": headers, "shape_headers": shape_columns,
        "attention_rows": attention_rows,
        "attention_rows_with_shapes": shape_rows,
        "groups": {kind: summary(groups[kind]) for kind in
                   ("FA_FWD", "FA_BWD", "Attention_OTHER", "GEMM", "HCCL", "Other")},
        "families": sorted(
            ({"kind": k, "family": f, **summary(v)} for (k, f), v in families.items()),
            key=lambda a: -a["sum_ms"]
        ),
        "attention_shapes": sorted(
            ({"kind": k, "family": f, "raw_input_shapes": shape, **summary(v)}
             for (k, f, shape), v in shape_buckets.items()),
            key=lambda a: -a["sum_ms"]
        ),
    }


def print_table(items, columns):
    widths = {name: max([len(name)] + [len(str(item.get(name, ""))) for item in items])
              for name in columns}
    print("  ".join(name.ljust(widths[name]) for name in columns))
    for item in items:
        print("  ".join(str(item.get(name, "")).ljust(widths[name]) for name in columns))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", type=Path, required=True)
    parser.add_argument("--tpr", type=Path, required=True)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--top", type=int, default=16)
    args = parser.parse_args()
    full, tpr = (collect(locate_csv(root)) for root in (args.full, args.tpr))
    report = {"full": full, "tpr": tpr}
    print("CP4 RING KERNEL CENSUS — DEVICE KERNEL DURATION, NOT ADDITIVE WALL TIME")
    print("FULL:", full["path"])
    print("TPR: ", tpr["path"])
    rows = []
    for kind in ("FA_FWD", "FA_BWD", "Attention_OTHER", "GEMM", "HCCL", "Other"):
        a, b = full["groups"][kind], tpr["groups"][kind]
        rows.append({
            "group": kind, "full_n": a["count"], "tpr_n": b["count"],
            "full_ms": round(a["sum_ms"], 2), "tpr_ms": round(b["sum_ms"], 2),
            "saved_ms": round(a["sum_ms"] - b["sum_ms"], 2),
            "full_p50us": round(a["p50_us"], 1), "tpr_p50us": round(b["p50_us"], 1),
            "full_lt100pct": round(a["under_100us_pct"], 1),
            "tpr_lt100pct": round(b["under_100us_pct"], 1),
        })
    print("\n== GROUPS: CALL COUNT / DURATION / MEDIAN / SHORT-KERNEL FRACTION ==")
    print_table(rows, ("group", "full_n", "tpr_n", "full_ms", "tpr_ms", "saved_ms",
                       "full_p50us", "tpr_p50us", "full_lt100pct", "tpr_lt100pct"))
    for label, sample in (("FULL", full), ("TPR", tpr)):
        print(f"\n== {label}: TOP OP FAMILIES (SUM OF KERNEL DURATIONS) ==")
        entries = [
            {"kind": x["kind"], "family": x["family"], "n": x["count"],
             "ms": round(x["sum_ms"], 2), "p50us": round(x["p50_us"], 1)}
            for x in sample["families"][:args.top]
        ]
        print_table(entries, ("kind", "family", "n", "ms", "p50us"))
        print(f"\n== {label}: MERGE-LIKE VECTOR OPS (NOT PROVEN RING-OWNED) ==")
        candidates = ("maximum", "sub", "exp", "mul", "add", "div", "cast", "transpose")
        entries = [
            {"family": x["family"], "n": x["count"], "ms": round(x["sum_ms"], 2)}
            for x in sample["families"] if x["kind"] == "Other"
            and any(c in x["family"].lower() for c in candidates)
        ]
        print_table(entries[:args.top], ("family", "n", "ms"))
        present = sample["attention_rows_with_shapes"]
        total = sample["attention_rows"]
        print(f"\n{label} attention rows with shape metadata: {present}/{total}; "
              f"shape columns={sample['shape_headers']!r}")
        if not present:
            print("NO Q/KV SHAPES in kernel_details.csv. Prefix/Current/Root "
                  "and input shapes CANNOT be inferred from op names.")
        else:
            entries = [
                {"kind": x["kind"], "n": x["count"], "ms": round(x["sum_ms"], 2),
                 "raw_shapes": x["raw_input_shapes"][:180]}
                for x in sample["attention_shapes"][:args.top]
            ]
            print("Raw input-shape groups (not validated Q/KV dimensions):")
            print_table(entries, ("kind", "n", "ms", "raw_shapes"))
    print("\nWARNING: merge-like op names cannot be identified as Ring-owned without operator correlation.")
    print("WARNING: kernel sums are not additive wall times; HCCL can overlap compute.")
    print("WARNING: compare only matched rank, model, capture, topology and code version.")
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                                 encoding="utf-8")
        print("json_out=", args.json_out)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Summarize opt-in dKV accumulation and Replay-root JSONL diagnostics."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path


def _read(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _summarize_arm(arm: str, path: Path) -> dict:
    records = _read(path)
    accumulation = [item for item in records if item["event"] == "dkv_accumulate"]
    replay_inputs = [item for item in records if item["event"] == "dkv_replay_input"]
    replay_hooks = [item for item in records if item["event"] == "dkv_replay_root_hook"]

    groups: dict[tuple[str, int, str], list[dict]] = defaultdict(list)
    for item in accumulation:
        groups[(item["segment_id"], item["layer"], item["kind"])].append(item)

    accumulation_groups = []
    for (segment_id, layer, kind), items in sorted(
        groups.items(), key=lambda pair: (pair[0][0], pair[0][1], pair[0][2])
    ):
        last = max(items, key=lambda item: item["accumulation_count"])
        accumulation_groups.append({
            "segment_id": segment_id,
            "layer": layer,
            "kind": kind,
            "contributions": len(items),
            "max_accumulation_count": last["accumulation_count"],
            "incoming_dtypes": sorted({item["incoming"]["dtype"] for item in items}),
            "buffer_dtypes": sorted({item["buffer"]["dtype"] for item in items}),
            "last_buffer_norm": last["buffer"]["norm"],
            "last_buffer_max_abs": last["buffer"]["max_abs"],
        })

    hook_errors = [
        item["seed_to_hook"]["relative_l2"]
        for item in replay_hooks
        if item["seed_to_hook"].get("shape_match") is True
    ]
    hook_max_abs = [
        item["seed_to_hook"]["max_abs"]
        for item in replay_hooks
        if item["seed_to_hook"].get("shape_match") is True
    ]
    dtype_mismatches = [
        {
            "segment": item["replay_segment_id"],
            "layer": item["layer"],
            "kind": item["kind"],
            "seed_dtype": item["seed_gradient"]["dtype"],
            "actual_dtype": item["actual_gradient"]["dtype"],
            "relative_l2": item["seed_to_hook"].get("relative_l2"),
            "max_abs": item["seed_to_hook"].get("max_abs"),
        }
        for item in replay_hooks
        if item["seed_gradient"]["dtype"] != item["actual_gradient"]["dtype"]
    ]

    return {
        "arm": arm,
        "records": len(records),
        "accumulation_tensor_records": len(accumulation),
        "accumulation_groups": accumulation_groups,
        "incoming_dtype_counts": dict(Counter(item["incoming"]["dtype"] for item in accumulation)),
        "buffer_dtype_counts": dict(Counter(item["buffer"]["dtype"] for item in accumulation)),
        "replay_input_records": len(replay_inputs),
        "replay_root_dtype_counts": dict(Counter(item["root"]["dtype"] for item in replay_inputs)),
        "replay_seed_dtype_counts": dict(Counter(item["seed_gradient"]["dtype"] for item in replay_inputs)),
        "replay_hook_records": len(replay_hooks),
        "replay_actual_gradient_dtype_counts": dict(Counter(
            item["actual_gradient"]["dtype"] for item in replay_hooks
        )),
        "replay_hook_relative_l2_median": statistics.median(hook_errors) if hook_errors else None,
        "replay_hook_relative_l2_max": max(hook_errors) if hook_errors else None,
        "replay_hook_max_abs_max": max(hook_max_abs) if hook_max_abs else None,
        "replay_seed_actual_dtype_mismatches": dtype_mismatches,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()

    summaries = [
        _summarize_arm("tpr_old", args.root / "tpr_old" / "dkv.jsonl"),
        _summarize_arm("tpr_fp32", args.root / "tpr_fp32" / "dkv.jsonl"),
    ]
    output = args.root / "summary.json"
    output.write_text(json.dumps(summaries, indent=2, sort_keys=True))

    for summary in summaries:
        print(f"========== {summary['arm']} ==========")
        print(
            "accumulation:",
            f"tensor_records={summary['accumulation_tensor_records']}",
            f"incoming_dtypes={summary['incoming_dtype_counts']}",
            f"buffer_dtypes={summary['buffer_dtype_counts']}",
        )
        print(
            "replay:",
            f"inputs={summary['replay_input_records']}",
            f"root_dtypes={summary['replay_root_dtype_counts']}",
            f"seed_dtypes={summary['replay_seed_dtype_counts']}",
            f"hooks={summary['replay_hook_records']}",
            f"actual_dtypes={summary['replay_actual_gradient_dtype_counts']}",
        )
        print(
            "seed_to_hook:",
            f"median_rel_l2={summary['replay_hook_relative_l2_median']}",
            f"max_rel_l2={summary['replay_hook_relative_l2_max']}",
            f"max_abs={summary['replay_hook_max_abs_max']}",
        )
        for item in summary["replay_seed_actual_dtype_mismatches"][:10]:
            print("dtype_mismatch:", json.dumps(item, sort_keys=True))
        for item in summary["accumulation_groups"]:
            print(
                "accum",
                f"segment={item['segment_id']}",
                f"layer={item['layer']}",
                f"kind={item['kind']}",
                f"contributions={item['contributions']}",
                f"max_count={item['max_accumulation_count']}",
                f"incoming={item['incoming_dtypes']}",
                f"buffer={item['buffer_dtypes']}",
                f"last_norm={item['last_buffer_norm']:.8g}",
            )

    print(f"TPR_DKV_REPLAY_DIAG_SUMMARY saved={output}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""CPU-only SWE-reBench short-fix screening for UniAgent / Claude Code / TPR.

The gold patch is used ONLY for offline instance selection; the saved subset
preserves the original UniAgent rows and never inserts the gold patch into a
model-facing prompt. This is a proxy for short agent trajectories, NOT a
measurement of response length or prefix reuse.

Supported metadata layouts include both:
  extra_info.tools_kwargs.task.metadata
  extra_info.tools_kwargs.reward.metadata

Usage:
  python tests/models/mcore/tpr/parallel/filter_swe_rebench_shortfix.py \
      /path/to/swe_rebench_filtered.parquet --out /path/to/shortfix --limit 50
Requires pandas and pyarrow.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

DIFF_FILE = re.compile(r"^diff --git a/(.+?) b/(.+)$", re.MULTILINE)
DIFF_HUNK = re.compile(r"^@@\s", re.MULTILINE)
NON_SOURCE = re.compile(
    r"(^|/)(tests?|docs?|examples?|vendor|node_modules|fixtures|migrations)/"
    r"|(^|/)(README|CHANGELOG|LICENSE|NOTICE|setup\.cfg|pyproject\.toml|"
    r"package-lock\.json|poetry\.lock|uv\.lock|Cargo\.lock)$"
    r"|\.(md|rst|txt|lock|json|yaml|yml|toml|csv|ipynb|png|jpg|svg)$",
    re.IGNORECASE,
)
TIERS = ("strict", "relaxed", "reject")


def decode_object(value):
    if isinstance(value, dict):
        return value
    if hasattr(value, "as_py"):
        return decode_object(value.as_py())
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except ValueError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def decode_array(value):
    if value is None:
        return []
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            return []
    if hasattr(value, "as_py"):
        value = value.as_py()
    if hasattr(value, "tolist"):
        value = value.tolist()
    return list(value) if isinstance(value, (list, tuple)) else []


def extract_metadata(row):
    """Read metadata from both UniAgent task and reward configuration layouts."""
    root = decode_object(row.get("extra_info"))
    tools_kwargs = decode_object(root.get("tools_kwargs"))

    task = decode_object(tools_kwargs.get("task"))
    task_meta = decode_object(task.get("metadata"))
    if task_meta:
        return task_meta

    reward = decode_object(tools_kwargs.get("reward"))
    reward_meta = decode_object(reward.get("metadata"))
    if reward_meta:
        return reward_meta

    # Also accept raw datasets where task fields are top-level columns.
    return row.to_dict()


def patch_metrics(patch):
    paths = [target for _, target in DIFF_FILE.findall(patch)]
    changed = sum(
        (line.startswith("+") and not line.startswith("+++"))
        or (line.startswith("-") and not line.startswith("---"))
        for line in patch.splitlines()
    )
    source = [path for path in paths if not NON_SOURCE.search(path)]
    return paths, source, changed, len(DIFF_HUNK.findall(patch))


def describe(row):
    metadata = extract_metadata(row)
    patch = metadata.get("patch") or ""
    if not isinstance(patch, str):
        patch = ""
    files, source, changed, hunks = patch_metrics(patch)
    issue = metadata.get("problem_statement") or ""
    if not isinstance(issue, str):
        issue = ""
    return {
        "instance_id": str(metadata.get("instance_id") or ""),
        "repo": str(metadata.get("repo") or ""),
        "gold_files": len(files),
        "source_files": len(source),
        "gold_changed_lines": changed,
        "gold_hunks": hunks,
        "fail_to_pass": len(decode_array(metadata.get("FAIL_TO_PASS"))),
        "issue_chars": len(issue),
        "gold_source_paths": ";".join(source),
        "issue_preview": " ".join(issue.split())[:140],
    }


def tier(row):
    if not row["instance_id"] or not row["gold_files"] or not row["source_files"]:
        return "reject"
    if not 1 <= row["fail_to_pass"] <= 5:
        return "reject"
    if (
        row["source_files"] == 1
        and row["gold_files"] == 1
        and row["gold_changed_lines"] <= 30
        and row["gold_hunks"] <= 3
        and row["fail_to_pass"] <= 3
        and row["issue_chars"] <= 3000
    ):
        return "strict"
    if (
        row["source_files"] <= 2
        and row["gold_files"] <= 2
        and row["gold_changed_lines"] <= 60
        and row["gold_hunks"] <= 4
        and row["issue_chars"] <= 4500
    ):
        return "relaxed"
    return "reject"


def screen(df, limit=50):
    """Return ranked audit, candidate rows, and tier counts; preserve row schema."""
    if limit < 1:
        raise ValueError("limit must be >= 1")
    columns = [
        "instance_id", "repo", "gold_files", "source_files",
        "gold_changed_lines", "gold_hunks", "fail_to_pass",
        "issue_chars", "gold_source_paths", "issue_preview",
    ]
    rank = pd.DataFrame([describe(row) for _, row in df.iterrows()], columns=columns)
    if rank.empty:
        rank["tier"] = pd.Series(dtype=str)
        return rank, df.iloc[0:0].copy(), {t: 0 for t in TIERS}
    rank["tier"] = rank.apply(tier, axis=1)
    rank["_row_number"] = range(len(rank))
    rank["_tier_priority"] = rank["tier"].map(dict(zip(TIERS, range(len(TIERS)))))
    rank = rank.sort_values(
        ["_tier_priority", "gold_changed_lines", "fail_to_pass", "gold_hunks", "issue_chars"],
        kind="stable",
    )
    counts = {t: int((rank["tier"] == t).sum()) for t in TIERS}
    shortlist = rank[rank["tier"] != "reject"].head(limit)
    selected = df.iloc[shortlist["_row_number"].tolist()].reset_index(drop=True)
    return shortlist.drop(columns=["_row_number", "_tier_priority"]), selected, counts


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_parquet", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=50)
    args = parser.parse_args(argv)
    if args.limit < 1:
        parser.error("--limit must be >= 1")
    df = pd.read_parquet(args.input_parquet)
    ranked, selected, counts = screen(df, args.limit)
    args.out.mkdir(parents=True, exist_ok=True)
    ranked.to_csv(args.out / "shortfix_ranked.csv", index=False)
    # Always write a valid Parquet, even if nothing passes the filter.
    selected.to_parquet(args.out / "shortfix_candidates.parquet", index=False)
    (args.out / "shortfix_counts.json").write_text(
        json.dumps({"input": len(df), "selected": len(selected), "tiers": counts},
                   indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Input={len(df)}, selected={len(selected)}, tiers={counts}")
    print(f"Outputs: {args.out}/shortfix_ranked.csv, shortfix_candidates.parquet, shortfix_counts.json")
    print("Gold patches used only for screening; verify suffix length and reward in real Claude Code rollouts.")


if __name__ == "__main__":
    main()

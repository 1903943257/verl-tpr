#!/usr/bin/env python3
"""Create an isolated prompt-hint variant of an existing normalized SWE-ReBench Parquet.

The source dataset is never overwritten. Only the selected task's final user
message is extended; task metadata, reward configuration, and sandbox image
remain unchanged. This is intended for controlled rollout experiments, not
for reporting an unassisted SWE-agent success rate.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

TARGET_INSTANCE = "ASPP__pelita-875"
HINTS = {
    "h1": (
        "Focus on pelita/utils.py, especially run_background_game() and the "
        "branch handling a non-None layout. Trace how a custom layout string "
        "is converted to the dictionary passed to run_game(). Add a focused "
        "regression test for a supplied multiline layout string, while "
        "preserving the existing behavior when layout is None."
    ),
    "h2": (
        "In pelita/utils.py::run_background_game(), the non-None layout branch "
        "should parse the supplied layout string with parse_layout(layout) and "
        "pass the resulting layout dictionary to run_game(). Add a regression "
        "test in test/test_utils.py for a custom layout string and verify the "
        "existing default-layout behavior remains unchanged."
    ),
    "h3": (
        "Implementation hint: in pelita/utils.py::run_background_game(), the "
        "custom-layout branch should use layout_dict = parse_layout(layout) "
        "before calling run_game(). Apply the minimal code change, run the "
        "focused regression test test/test_utils.py::test_run_background_game_with_layout, "
        "and verify the default layout path still works. This is an "
        "answer-assisted/oracle-style condition, not an unassisted benchmark."
    ),
}
HINT_MARKER = "[TPR_TASK_HINT:"


def decode_object(value: Any) -> dict:
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


def get_instance_id(row: pd.Series) -> str:
    extra = decode_object(row.get("extra_info"))
    tools_kwargs = decode_object(extra.get("tools_kwargs"))
    task = decode_object(tools_kwargs.get("task"))
    metadata = decode_object(task.get("metadata"))
    return str(metadata.get("instance_id") or "")


def append_to_prompt(prompt: Any, hint: str, level: str) -> Any:
    marker = f"{HINT_MARKER}{level.upper()}]"
    suffix = f"\n\n{marker}\n{hint}"
    if isinstance(prompt, str):
        if HINT_MARKER in prompt:
            raise ValueError("prompt already contains a TPR_TASK_HINT marker")
        return prompt + suffix
    if isinstance(prompt, dict):
        if prompt.get("role") != "user" or not isinstance(prompt.get("content"), str):
            raise ValueError("dict prompt must be a user message with string content")
        updated = dict(prompt)
        if HINT_MARKER in updated["content"]:
            raise ValueError("prompt already contains a TPR_TASK_HINT marker")
        updated["content"] += suffix
        return updated
    if isinstance(prompt, np.ndarray):
        # pandas/pyarrow may deserialize a list-of-messages column as ndarray.
        # Normalize only the selected row to a regular message list before editing.
        return append_to_prompt(prompt.tolist(), hint, level)
    if isinstance(prompt, list):
        updated = [dict(message) if isinstance(message, dict) else message for message in prompt]
        user_indices = [
            i for i, message in enumerate(updated)
            if isinstance(message, dict) and message.get("role") == "user"
        ]
        if not user_indices:
            raise ValueError("prompt list contains no user message")
        index = user_indices[-1]
        message = updated[index]
        if not isinstance(message.get("content"), str):
            raise ValueError("last user message content is not a string")
        if HINT_MARKER in message["content"]:
            raise ValueError("prompt already contains a TPR_TASK_HINT marker")
        message["content"] += suffix
        return updated
    raise ValueError(f"unsupported prompt representation: {type(prompt).__name__}")


def build_variant(df: pd.DataFrame, instance_id: str, level: str) -> pd.DataFrame:
    if "prompt" not in df.columns or "extra_info" not in df.columns:
        raise ValueError("input Parquet must contain prompt and extra_info columns")
    ids = df.apply(get_instance_id, axis=1)
    matches = ids[ids == instance_id].index.tolist()
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one row for {instance_id!r}, found {len(matches)}"
        )
    out = df.copy(deep=True)
    idx = matches[0]
    out.at[idx, "prompt"] = append_to_prompt(out.at[idx, "prompt"], HINTS[level], level)
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Existing normalized task Parquet")
    parser.add_argument("--output", type=Path, required=True, help="New output path; source is never overwritten")
    parser.add_argument("--instance-id", default=TARGET_INSTANCE)
    parser.add_argument("--level", choices=tuple(HINTS), default="h2")
    parser.add_argument("--only-target", action="store_true", help="Write only the selected task row (recommended for N=1 screening)")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing the output file only")
    args = parser.parse_args(argv)

    source = args.input.resolve()
    destination = args.output.resolve()
    if source == destination:
        parser.error("--output must differ from --input; the source dataset is immutable")
    if not source.is_file():
        parser.error(f"input file does not exist: {source}")
    if destination.exists() and not args.overwrite:
        parser.error(f"output already exists: {destination}; choose a new path or pass --overwrite")

    df = pd.read_parquet(source)
    out = build_variant(df, args.instance_id, args.level)
    if args.only_target:
        out = out.loc[out.apply(get_instance_id, axis=1) == args.instance_id].reset_index(drop=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(destination, index=False)
    print(json.dumps({
        "source": str(source),
        "output": str(destination),
        "rows": len(out),
        "instance_id": args.instance_id,
        "hint_level": args.level,
        "hinted_rows": 1,
        "only_target_row": args.only_target,
        "original_prompt_unchanged_for_other_rows": True,
        "source_overwritten": False,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

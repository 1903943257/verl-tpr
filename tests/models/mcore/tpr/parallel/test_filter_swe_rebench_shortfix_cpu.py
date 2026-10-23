"""CPU-only regression for offline SWE-reBench candidate selection."""
from __future__ import annotations

import json

import pandas as pd
import pytest

from .filter_swe_rebench_shortfix import (
    decode_array,
    describe,
    patch_metrics,
    screen,
    tier,
)


def _patch(path: str, added: int = 1) -> str:
    body = "\n".join(f"+new_{index}" for index in range(added))
    return (
        f"diff --git a/{path} b/{path}\n"
        f"--- a/{path}\n+++ b/{path}\n"
        f"@@ -1,1 +1,{added} @@\n"
        "-old\n" + body + "\n"
    )


def _row(instance_id: str, patch: str, *, f2p=1, issue_chars=100,
         serialized=False):
    metadata = {
        "instance_id": instance_id,
        "repo": "demo/pkg",
        "base_commit": "abc123",
        "patch": patch,
        "test_patch": "test-unchanged",
        "problem_statement": "x" * issue_chars,
        "FAIL_TO_PASS": json.dumps([f"test_{i}" for i in range(f2p)]),
        "PASS_TO_PASS": "[]",
    }
    extra = {"tools_kwargs": {"task": {
        "name": "swe_rebench",
        "sandbox": {"image": "swerebench/sample"},
        "metadata": metadata,
    }}}
    return {
        "prompt": [{"role": "user", "content": "a problem"}],
        "data_source": "nebius/SWE-rebench",
        "extra_info": json.dumps(extra) if serialized else extra,
    }


def test_patch_metrics_excludes_headers_and_test_sources():
    patch = _patch("pkg/code.py", 2) + _patch("tests/test_code.py", 1)
    paths, sources, changed, hunks = patch_metrics(patch)
    assert paths == ["pkg/code.py", "tests/test_code.py"]
    assert sources == ["pkg/code.py"]
    assert changed == 5
    assert hunks == 2


def test_serialized_uniagent_metadata_is_read_without_mutation():
    item = _row("case_1", _patch("src/code.py"), serialized=True)
    got = describe(pd.Series(item))
    assert (got["instance_id"], got["source_files"], got["fail_to_pass"]) == ("case_1", 1, 1)
    assert tier(got) == "strict"
    assert decode_array("not-json") == []
    assert item["extra_info"]  # Original row was not mutated.


def test_screen_preserves_original_task_rows_and_prioritizes_strict():
    large = _row("relaxed", _patch("src/code.py", 35))
    small = _row("strict", _patch("src/other.py", 2))
    invalid = _row("reject", _patch("docs/README.md", 1))
    original = pd.DataFrame([large, invalid, small])
    shortlist, selected, counts = screen(original, limit=10)
    assert counts == {"strict": 1, "relaxed": 1, "reject": 1}
    assert shortlist["instance_id"].tolist() == ["strict", "relaxed"]
    assert selected.to_dict("records") == [small, large]
    assert "patch" not in selected.columns  # Only inside original metadata, never a prompt column.
    assert original.iloc[0]["extra_info"] == large["extra_info"]


def test_screen_empty_or_bad_limit():
    original = pd.DataFrame(columns=["prompt", "extra_info"])
    ranked, subset, counts = screen(original)
    assert ranked.empty and subset.empty
    assert counts == {"strict": 0, "relaxed": 0, "reject": 0}
    with pytest.raises(ValueError, match="limit"):
        screen(original, limit=0)

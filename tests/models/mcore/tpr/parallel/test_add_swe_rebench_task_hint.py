"""CPU-only tests for isolated SWE-ReBench task prompt hint variants."""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from .add_swe_rebench_task_hint import HINTS, append_to_prompt, build_variant


def _row(instance_id: str, prompt_text: str = "Original issue") -> dict:
    return {
        "prompt": [{"role": "user", "content": prompt_text}],
        "data_source": "nebius/SWE-rebench",
        "extra_info": {
            "tools_kwargs": {
                "task": {
                    "name": "swe_rebench",
                    "sandbox": {"image": "swerebench/example"},
                    "metadata": {"instance_id": instance_id},
                }
            }
        },
    }


@pytest.mark.parametrize("level", ["h1", "h2", "h3"])
def test_only_target_task_prompt_is_extended(level):
    original = pd.DataFrame([
        _row("ASPP__pelita-875"),
        _row("Algebra8__pyopenapi3-83", "Keep unchanged"),
    ])
    original_snapshot = copy.deepcopy(original.to_dict("records"))

    result = build_variant(original, "ASPP__pelita-875", level)

    target_content = result.iloc[0]["prompt"][-1]["content"]
    other_content = result.iloc[1]["prompt"][-1]["content"]
    assert f"[TPR_TASK_HINT:{level.upper()}]" in target_content
    assert HINTS[level] in target_content
    assert other_content == "Keep unchanged"
    assert original.to_dict("records") == original_snapshot
    assert result.iloc[0]["extra_info"] == original.iloc[0]["extra_info"]


def test_exactly_one_target_row_is_required():
    with pytest.raises(ValueError, match="found 0"):
        build_variant(pd.DataFrame([_row("other")]), "ASPP__pelita-875", "h2")
    with pytest.raises(ValueError, match="found 2"):
        build_variant(
            pd.DataFrame([_row("ASPP__pelita-875"), _row("ASPP__pelita-875")]),
            "ASPP__pelita-875",
            "h2",
        )


def test_prompt_hint_cannot_be_appended_twice():
    prompt = [{"role": "user", "content": "Issue\n\n[TPR_TASK_HINT:H2]\nexisting"}]
    with pytest.raises(ValueError, match="already contains"):
        append_to_prompt(prompt, HINTS["h2"], "h2")


def test_zero_dimensional_ndarray_prompt_from_pandas_is_supported():
    original = pd.DataFrame([_row("ASPP__pelita-875")])
    # Assigning a length-one ndarray through .at can be scalarized by pandas
    # into a zero-dimensional ndarray containing the single message dict.
    original.at[0, "prompt"] = np.array(
        [{"role": "user", "content": "Original issue"}], dtype=object
    )
    assert original.at[0, "prompt"].shape == ()

    result = build_variant(original, "ASPP__pelita-875", "h2")

    updated_prompt = result.iloc[0]["prompt"]
    assert isinstance(updated_prompt, dict)
    assert updated_prompt["role"] == "user"
    assert "[TPR_TASK_HINT:H2]" in updated_prompt["content"]
    assert updated_prompt["content"].startswith("Original issue")
    assert isinstance(original.iloc[0]["prompt"], np.ndarray)
    assert "TPR_TASK_HINT" not in original.iloc[0]["prompt"].item()["content"]


def test_one_dimensional_ndarray_prompt_is_preserved_as_message_list():
    original = pd.DataFrame([_row("ASPP__pelita-875")])
    prompt_array = np.array(
        [{"role": "user", "content": "Original issue"}], dtype=object
    )
    # Construct the object column without pandas scalarizing the ndarray.
    original["prompt"] = pd.Series([prompt_array], dtype=object)
    assert original.at[0, "prompt"].shape == (1,)

    result = build_variant(original, "ASPP__pelita-875", "h2")

    updated_prompt = result.iloc[0]["prompt"]
    assert isinstance(updated_prompt, list)
    assert len(updated_prompt) == 1
    assert "[TPR_TASK_HINT:H2]" in updated_prompt[-1]["content"]
    assert updated_prompt[-1]["content"].startswith("Original issue")
    assert isinstance(original.iloc[0]["prompt"], np.ndarray)
    assert "TPR_TASK_HINT" not in original.iloc[0]["prompt"][0]["content"]

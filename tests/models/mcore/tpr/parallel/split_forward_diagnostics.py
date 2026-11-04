# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Forward statistics only; activation snapshots never leave CPU memory."""

import json

from .split_fp32_attribution import _compare_traces


def write_forward_comparison(reference, candidate, output, *, atol=1e-5, rtol=1e-4):
    records = _compare_traces(reference, candidate, atol=atol, rtol=rtol)
    if any("gradient" in item for item in records):
        raise AssertionError("Forward diagnostic unexpectedly retained a backward trace")
    output.write_text(json.dumps(records, indent=2), encoding="utf-8")

    def first(predicate):
        return next(
            (
                {key: item[key] for key in ("stage", "start", "end", "grad_enabled", "forward")}
                for item in records
                if predicate(item["forward"])
            ),
            None,
        )

    return {
        "file": output.name,
        "records": len(records),
        "first_nonzero": first(lambda metric: metric["max_abs"] > 0),
        "first_above_tolerance": first(lambda metric: not metric["within_tolerance"]),
        "atol": atol,
        "rtol": rtol,
        "scope": "first recorded stage in Full forward order; not proof of the originating operator",
    }

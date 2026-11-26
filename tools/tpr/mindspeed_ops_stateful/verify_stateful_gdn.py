#!/usr/bin/env python3
"""Verify the MindSpeed-Ops Stateful GDN migration."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

GDR_ERROR = (
    'The chunk_gated_delta_rule_fwd_h does not support initial_state.'
)

FILES = (
    Path("mindspeed_ops/api/triton/chunk_gated_delta_rule.py"),
    Path("mindspeed_ops/arch32/triton/gdn/chunk_gated_delta_rule_fwd_h.py"),
    Path("mindspeed_ops/arch32/triton/convolution.py"),
)


def fail(message: str) -> None:
    raise RuntimeError(message)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("ops_root", type=Path)
    args = parser.parse_args()
    root = args.ops_root.resolve()

    for relative in FILES:
        if not (root / relative).is_file():
            fail(f"missing file: {relative}")

    public, fwd_h, conv = [(root / p).read_text(encoding="utf-8") for p in FILES]

    active_gate = re.compile(
        r'(?m)^[ \t]*if initial_state is not None:\n'
        r'[ \t]+raise ValueError\(' + re.escape(repr(GDR_ERROR)) + r'\)'
    )
    for relative, text in zip(FILES[:2], (public, fwd_h)):
        if active_gate.search(text):
            fail(f"active Stateful GDR gate remains in {relative}")

    conv_checks = {
        "single-tile dh0 ownership": "if i_t == 0:",
        "global time indexing": "dy + bos * D + i_t2 * D + o_d",
        "short-sequence mask": "m_t2 = m_d & (i_t2 < T_len)",
        "extra dw reduction slices": "dw_extra = initial_state.shape[0] if initial_state is not None else 0",
        "extra dw store": "dw + (NUM_CHKS + i_n) * W * D",
        "unsharded fp32 dh0 allocation": "dh0 = initial_state.new_zeros(*initial_state.shape, dtype=torch.float32)",
        "no time-tile dh0 reduction": "dh0 = dh0.to(initial_state)",
    }
    for label, marker in conv_checks.items():
        if marker not in conv:
            fail(f"CausalConv check failed: missing {label} ({marker})")

    forbidden = (
        "i_t * BT + i_t2",
        "dh0 + i_t * B * D * W",
        "dh0 = initial_state.new_zeros(min(eff_NT, triton.cdiv(W, BT))",
        "dh0 = dh0.sum(0, dtype=torch.float32).to(initial_state)",
    )
    for marker in forbidden:
        if marker in conv:
            fail(f"legacy unsafe CausalConv pattern remains: {marker}")

    for relative in FILES:
        result = subprocess.run(
            [sys.executable, "-m", "py_compile", str(root / relative)],
            check=False,
        )
        if result.returncode:
            fail(f"Python syntax check failed: {relative}")

    try:
        sha = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        sha = "unknown"

    print("STATEFUL GDN PATCH VERIFY PASS")
    print(f"MindSpeed-Ops HEAD: {sha}")
    print("GDR public gate: open")
    print("GDR Arch32 fwd-h gate: open")
    for label in conv_checks:
        print(f"CausalConv: {label}: OK")
    print("Python syntax: OK")
    print("NOTE: static verification only; NPU numerical tests are still required.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(f"VERIFY FAILED: {exc}", file=sys.stderr)
        raise SystemExit(2)

#!/usr/bin/env python3
"""Apply the version-adaptable MindSpeed-Ops Stateful GDN migration."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

GDR_ERROR = (
    'The chunk_gated_delta_rule_fwd_h does not support initial_state.'
)

GDR_FILES = (
    Path("mindspeed_ops/api/triton/chunk_gated_delta_rule.py"),
    Path("mindspeed_ops/arch32/triton/gdn/chunk_gated_delta_rule_fwd_h.py"),
)

CONV_FILE = Path("mindspeed_ops/arch32/triton/convolution.py")


class PatchError(RuntimeError):
    pass


def _active_gate_pattern() -> re.Pattern[str]:
    return re.compile(
        r'(?m)^(?P<indent>[ \t]*)if initial_state is not None:\n'
        r'(?P=indent)[ \t]+raise ValueError\('
        + r'["\']' + re.escape(GDR_ERROR) + r'["\']'
        + r'\)\n'
    )


def patch_gdr_gate(path: Path, text: str) -> tuple[str, str]:
    pattern = _active_gate_pattern()
    matches = list(pattern.finditer(text))
    if len(matches) == 1:
        match = matches[0]
        indent = match.group("indent")
        replacement = (
            indent
            + "# TPR Stateful GDN: non-empty initial_state is supported by the "
              "fwd-h/bwd-dhu path.\n"
        )
        return text[:match.start()] + replacement + text[match.end():], "patched"

    if len(matches) > 1:
        raise PatchError(f"{path}: found {len(matches)} active initial_state gates")

    if GDR_ERROR not in text or re.search(
        r'(?m)^[ \t]*raise ValueError\(' + re.escape(repr(GDR_ERROR)) + r'\)',
        text,
    ) is None:
        return text, "already-open"

    raise PatchError(
        f"{path}: found the historical gate in an unsupported layout; "
        "inspect this file instead of applying a fuzzy edit"
    )


def _remove_weight_initial_state_block(lines: list[str], path: Path) -> list[str]:
    starts = [
        i for i in range(len(lines) - 1)
        if lines[i].strip() == "if USE_INITIAL_STATE:"
        and lines[i + 1].strip() == "mask_head_rows = o_t < i_w"
    ]
    if len(starts) != 1:
        raise PatchError(
            f"{path}: expected exactly one legacy per-tile initial_state/dw "
            f"correction block, found {len(starts)}"
        )

    start = starts[0]
    end_marker = "b_dw += tl.sum(b_dy_head * b_xc, 0)"
    ends = [
        i for i in range(start + 1, len(lines))
        if lines[i].strip() == end_marker
    ]
    if len(ends) != 1:
        raise PatchError(
            f"{path}: could not uniquely locate the end of the legacy "
            "initial_state/dw correction block"
        )
    return lines[:start] + lines[ends[0] + 1:]


def _replace_dh0_kernel_block(lines: list[str], path: Path) -> list[str]:
    starts = [
        i for i in range(len(lines) - 1)
        if lines[i].strip() == "if USE_INITIAL_STATE:"
        and lines[i + 1].strip() == "for i_w in tl.static_range(1, W):"
    ]
    if len(starts) != 1:
        raise PatchError(
            f"{path}: expected one legacy dh0 block, found {len(starts)}"
        )

    start = starts[0]
    ends = [
        i for i in range(start + 1, len(lines))
        if lines[i].rstrip("\r\n") == "        if HAS_BIAS:"
    ]
    if len(ends) != 1:
        raise PatchError(
            f"{path}: could not uniquely locate the end of the legacy dh0 block"
        )
    end = ends[0]

    replacement = '''            if USE_INITIAL_STATE:
                if i_t == 0:
                    # initial_state affects only the first W-1 outputs. Its
                    # dh0 and weight-gradient contributions are global-time
                    # quantities, independent of the current time tile.
                    for i_w in tl.static_range(0, W):
                        b_dh0_s = tl.zeros((BD,), dtype=tl.float32)
                        b_dw_h = tl.zeros((BD,), dtype=tl.float32)
                        for i_t2 in tl.static_range(0, W - 1):
                            if i_t2 < i_w:
                                # Short sequences may have fewer than W-1 rows.
                                m_t2 = m_d & (i_t2 < T_len)
                                dy0_row = tl.load(
                                    dy + bos * D + i_t2 * D + o_d,
                                    mask=m_t2,
                                    other=0.0,
                                ).to(tl.float32)
                                if ACTIVATION == "swish" or ACTIVATION == "silu":
                                    y0_row = tl.load(
                                        y + bos * D + i_t2 * D + o_d,
                                        mask=m_t2,
                                        other=0.0,
                                    ).to(tl.float32)
                                    y0_s = tl.sigmoid(y0_row)
                                    dy0_row = dy0_row * y0_s * (
                                        1 + y0_row * (1 - y0_s)
                                    )
                                if HAS_WEIGHT:
                                    w_row = tl.extract_slice(
                                        b_w,
                                        [i_w - 1 - i_t2, 0],
                                        [1, BD],
                                        [1, 1],
                                    )
                                    b_dh0_s += tl.sum(
                                        dy0_row[None, :] * w_row, 0
                                    ).to(tl.float32)
                                    # W - (i_w - i_t2) is in [1, W-1].
                                    b_xc = tl.load(
                                        initial_state
                                        + i_n * D * W
                                        + o_d * W
                                        + (W - i_w + i_t2),
                                        mask=m_d,
                                        other=0.0,
                                    ).to(tl.float32)
                                    b_dw_h += dy0_row * b_xc
                                else:
                                    b_dh0_s += dy0_row

                        if i_w > 0:
                            tl.store(
                                dh0 + i_n * D * W + o_d * W + i_w,
                                b_dh0_s.to(
                                    dh0.dtype.element_ty,
                                    fp_downcast_rounding="rtne",
                                ),
                                mask=m_d,
                            )
                        if HAS_WEIGHT:
                            # One extra dw reduction slice per sequence.
                            tl.store(
                                dw + (NUM_CHKS + i_n) * W * D
                                + (W - i_w - 1) * D + o_d,
                                b_dw_h.to(dw.dtype.element_ty),
                                mask=m_d,
                            )

'''
    return lines[:start] + replacement.splitlines(keepends=True) + lines[end:]


def patch_convolution(path: Path, text: str) -> tuple[str, str]:
    required = (
        "dw_extra = initial_state.shape[0] if initial_state is not None else 0",
        "dh0 = initial_state.new_zeros(*initial_state.shape, dtype=torch.float32)",
        "if i_t == 0:",
        "dw + (NUM_CHKS + i_n) * W * D",
        "dh0 = dh0.to(initial_state)",
    )
    if all(marker in text for marker in required):
        return text, "already-patched"

    if any(marker in text for marker in required):
        raise PatchError(
            f"{path}: partial PR169 migration detected; inspect the file before retrying"
        )

    if "i_t * BT + i_t2" not in text:
        raise PatchError(
            f"{path}: legacy dh0 time-index expression not found; "
            "this source revision differs from the inspected 26.2.0 layout"
        )
    if "dh0 + i_t * B * D * W" not in text:
        raise PatchError(
            f"{path}: legacy time-tile dh0 store not found; refusing an unsafe patch"
        )

    lines = text.splitlines(keepends=True)
    lines = _remove_weight_initial_state_block(lines, path)
    lines = _replace_dh0_kernel_block(lines, path)
    patched = "".join(lines)

    old_dw = (
        "    dw = weight.new_empty(B * NT, W, D, dtype=torch.float) "
        "if weight is not None else None\n"
    )
    new_dw = (
        "    # PR169: reserve one extra reduction slice per initial-state sequence.\n"
        "    dw_extra = initial_state.shape[0] if initial_state is not None else 0\n"
        "    dw = weight.new_empty(B * NT + dw_extra, W, D, dtype=torch.float) "
        "if weight is not None else None\n"
    )
    if patched.count(old_dw) != 1:
        raise PatchError(
            f"{path}: expected one legacy dw allocation, found {patched.count(old_dw)}"
        )
    patched = patched.replace(old_dw, new_dw, 1)

    old_dh0 = '''    if initial_state is not None:
        if cu_seqlens is not None:
            eff_NT = len(chunk_indices)
        else:
            eff_NT = triton.cdiv(T, BT)

        dh0 = initial_state.new_zeros(min(eff_NT, triton.cdiv(W, BT)), *initial_state.shape)
    else:
        dh0 = None
'''
    new_dh0 = '''    if initial_state is not None:
        # PR169: only time tile zero writes dh0, so no time-tile dimension is needed.
        dh0 = initial_state.new_zeros(*initial_state.shape, dtype=torch.float32)
    else:
        dh0 = None
'''
    if patched.count(old_dh0) != 1:
        raise PatchError(
            f"{path}: expected one legacy dh0 allocation block, found {patched.count(old_dh0)}"
        )
    patched = patched.replace(old_dh0, new_dh0, 1)

    old_reduce = "        dh0 = dh0.sum(0, dtype=torch.float32).to(initial_state)\n"
    new_reduce = "        dh0 = dh0.to(initial_state)\n"
    if patched.count(old_reduce) != 1:
        raise PatchError(
            f"{path}: expected one legacy dh0 reduction, found {patched.count(old_reduce)}"
        )
    patched = patched.replace(old_reduce, new_reduce, 1)

    old_o_t = "            o_t = i_t * BT + tl.arange(0, BT)\n"
    if patched.count(old_o_t) == 1:
        patched = patched.replace(old_o_t, "", 1)

    if "i_t * BT + i_t2" in patched or "dh0 + i_t * B * D * W" in patched:
        raise PatchError(f"{path}: legacy dh0 indexing/store remains after patch")
    return patched, "patched"


def _is_git_dirty(root: Path, relative: Path) -> bool:
    for args in (
        ["diff", "--quiet", "--"],
        ["diff", "--cached", "--quiet", "--"],
    ):
        result = subprocess.run(
            ["git", "-C", str(root), *args, str(relative)],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if result.returncode == 1:
            return True
    return False


def _backup(path: Path) -> None:
    backup = path.with_name(path.name + ".bak.tpr-stateful")
    index = 1
    while backup.exists():
        backup = path.with_name(path.name + f".bak.tpr-stateful.{index}")
        index += 1
    backup.write_bytes(path.read_bytes())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("ops_root", type=Path, help="MindSpeed-Ops checkout root")
    parser.add_argument(
        "--check",
        action="store_true",
        help="validate source layout and report planned changes without writing files",
    )
    args = parser.parse_args()

    root = args.ops_root.resolve()
    if not (root / ".git").exists() and not (root / ".git").is_file():
        raise PatchError(f"{root} is not a Git checkout")
    if not (root / "mindspeed_ops").is_dir():
        raise PatchError(f"{root} does not look like a MindSpeed-Ops repository")

    targets = [*GDR_FILES, CONV_FILE]
    missing = [str(p) for p in targets if not (root / p).is_file()]
    if missing:
        raise PatchError("missing expected source files: " + ", ".join(missing))

    dirty = [str(p) for p in targets if _is_git_dirty(root, p)]
    if dirty:
        raise PatchError(
            "refusing to overwrite locally modified target files: "
            + ", ".join(dirty)
            + ". Commit/stash them or review the diff first."
        )

    results: list[tuple[Path, str, str]] = []
    for relative in GDR_FILES:
        path = root / relative
        original = path.read_text(encoding="utf-8")
        updated, status = patch_gdr_gate(relative, original)
        results.append((path, status, updated))

    conv_path = root / CONV_FILE
    conv_updated, conv_status = patch_convolution(
        CONV_FILE, conv_path.read_text(encoding="utf-8")
    )
    results.append((conv_path, conv_status, conv_updated))

    print(f"MindSpeed-Ops root: {root}")
    try:
        sha = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip()
        print(f"Git HEAD: {sha}")
    except Exception:
        pass

    for path, status, updated in results:
        changed = updated != path.read_text(encoding="utf-8")
        label = "WOULD PATCH" if args.check and changed else status.upper()
        print(f"[{label}] {path.relative_to(root)}")

    if args.check:
        print("CHECK PASS: source layout matches the migration patch; no files changed.")
        return 0

    for path, status, updated in results:
        if updated == path.read_text(encoding="utf-8"):
            continue
        _backup(path)
        path.write_text(updated, encoding="utf-8")
        print(f"[WROTE] {path.relative_to(root)}")

    print("APPLY COMPLETE. Run verify.sh and the NPU correctness tests.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PatchError as exc:
        print(f"PATCH ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)

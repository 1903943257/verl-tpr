#!/usr/bin/env python3
"""Discover TPR gates, validate their references and run *explicit* suites.

Examples (from repository root):
  python tests/models/mcore/tpr/run_suite.py list
  python tests/models/mcore/tpr/run_suite.py inventory
  python tests/models/mcore/tpr/run_suite.py check
  python tests/models/mcore/tpr/run_suite.py show npu-recompute-primitive
  python tests/models/mcore/tpr/run_suite.py run cpu-core
  python tests/models/mcore/tpr/run_suite.py run npu-ring --ack-npu

No heavy test runs merely because a suite was listed or checked.
"""
from __future__ import annotations

import argparse
import ast
import os
import re
from pathlib import Path
import shlex
import subprocess
import sys

from suite_catalog import GATES, SUITES, SUITE_DESCRIPTIONS, Gate

ROOT = Path(__file__).resolve().parents[4]
TEST_ROOT = Path(__file__).resolve().parent


def inventory() -> dict[str, int]:
    """Counts per existing physical folder; not a pass/fail inventory."""
    result: dict[str, int] = {}
    for path in TEST_ROOT.rglob("test_*.py"):
        folder = path.relative_to(TEST_ROOT).parts[0]
        result[folder] = result.get(folder, 0) + 1
    return dict(sorted(result.items()))



def _check_relative_imports() -> list[str]:
    """Check local Python module paths without importing NPU/Megatron.

    This catches historical test dependencies accidentally stranded by file
    deletion or experimental/ relocation; it does not validate dynamic
    importlib calls or shell script paths.
    """
    errors: list[str] = []
    for path in sorted(TEST_ROOT.rglob("*.py")):
        relative = path.relative_to(TEST_ROOT)
        try:
            syntax = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (SyntaxError, UnicodeDecodeError) as exc:
            errors.append(f"{relative}: unparseable module: {exc}")
            continue
        for node in ast.walk(syntax):
            if not isinstance(node, ast.ImportFrom) or not node.level:
                continue
            target = path.parent
            for _ in range(node.level - 1):
                target = target.parent
            if node.module:
                target = target.joinpath(*node.module.split("."))
            if not target.is_relative_to(TEST_ROOT):
                continue
            exists = target.with_suffix(".py").is_file() or (target / "__init__.py").is_file()
            if not exists:
                errors.append(
                    f"{relative}:{node.lineno}: missing relative module "
                    f"{'.' * node.level}{node.module or ''}"
                )
                continue
            # 'from . import test_fixture_module' imports a sibling module;
            # check it even though ImportFrom.module is None.
            if node.module is None:
                for alias in node.names:
                    if alias.name.startswith("test_") and not (target / (alias.name + ".py")).is_file():
                        errors.append(
                            f"{relative}:{node.lineno}: missing sibling test module {alias.name}"
                        )
    return errors



def _check_shell_references() -> list[str]:
    """Catch shell launchers referring to Python/tests removed by pruning."""
    errors: list[str] = []
    pattern = r"tests/models/mcore/tpr/[A-Za-z0-9_./-]+\.(?:py|sh)"
    for path in sorted(TEST_ROOT.rglob("*.sh")):
        source = path.relative_to(TEST_ROOT)
        for reference in set(re.findall(pattern, path.read_text(encoding="utf-8"))):
            if not (ROOT / reference).is_file():
                errors.append(f"{source}: missing shell target {reference}")
    return errors


def check() -> list[str]:
    errors: list[str] = []
    if set(SUITES) != set(SUITE_DESCRIPTIONS):
        errors.append("suite descriptions and suite names differ")
    for name, gate in GATES.items():
        if gate.kind not in {"pytest", "torchrun", "isolated"}:
            errors.append(f"{name}: invalid kind {gate.kind}")
        if gate.ranks < 1:
            errors.append(f"{name}: ranks must be positive")
        if not gate.path.startswith("tests/models/mcore/tpr/"):
            errors.append(f"{name}: path escapes TPR test folder")
        if not (ROOT / gate.path).is_file():
            errors.append(f"{name}: missing {gate.path}")
        if gate.kind != "torchrun" and gate.ranks != 1:
            errors.append(f"{name}: non-torchrun must have 1 rank")
        keys = [key for key, _ in gate.opt_in]
        if len(keys) != len(set(keys)):
            errors.append(f"{name}: duplicate opt-in environment variable")
    for suite, identifiers in SUITES.items():
        if not identifiers:
            errors.append(f"{suite}: empty suite")
        if len(identifiers) != len(set(identifiers)):
            errors.append(f"{suite}: duplicated gate")
        for identifier in identifiers:
            if identifier not in GATES:
                errors.append(f"{suite}: unknown gate {identifier}")
    errors.extend(_check_relative_imports())
    errors.extend(_check_shell_references())
    return errors


def command(gate: Gate) -> list[str]:
    if gate.kind == "isolated":
        return [sys.executable, gate.path]
    pytest_args = ["-q", "-s", "-ra", gate.path]
    bootstrap = "tests/models/mcore/tpr/_mindspeed_first_pytest.py"
    if gate.kind == "pytest":
        if gate.bootstrap:
            return [sys.executable, bootstrap, *pytest_args]
        return [sys.executable, "-m", "pytest", *pytest_args]
    return [
        "torchrun", f"--nproc_per_node={gate.ranks}",
        "--master_addr=localhost",
        f"--master_port={os.environ.get('TPR_TEST_MASTER_PORT', '29541')}",
        bootstrap, *pytest_args,
    ]


def describe(suite: str) -> None:
    print(f"{suite}: {SUITE_DESCRIPTIONS[suite]}")
    for identifier in SUITES[suite]:
        gate = GATES[identifier]
        env = " ".join(f"{k}={v}" for k, v in gate.opt_in)
        cmd = shlex.join(command(gate))
        print(f"  {identifier} [{gate.kind}, ranks={gate.ranks}]")
        print(f"    {env + ' ' if env else ''}{cmd}")
        if gate.required_env:
            print("    requires environment: " + ", ".join(gate.required_env))
        if gate.expected_failure:
            print("    EXPECTED FAILURE: diagnostic only, not acceptance")
        if gate.note:
            print(f"    {gate.note}")


def run(suite: str, *, ack_npu: bool, ack_negative: bool) -> int:
    gates = [(name, GATES[name]) for name in SUITES[suite]]
    if any(g.kind == "torchrun" for _, g in gates) and not ack_npu:
        print("NPU/distributed execution requires --ack-npu.", file=sys.stderr)
        return 2
    if any(g.expected_failure for _, g in gates):
        print("Negative controls are intentionally failing diagnostics. "
              "Use 'show' and run the underlying pytest manually; "
              "never report them as a green suite.", file=sys.stderr)
        return 2
    for name, gate in gates:
        missing = [var for var in gate.required_env if not os.environ.get(var)]
        if missing:
            print(f"{name}: missing required environment: {', '.join(missing)}",
                  file=sys.stderr)
            return 2
    for name, gate in gates:
        env = os.environ.copy()
        for key, value in gate.opt_in:
            env[key] = value
        print(f"=== {name}: {shlex.join(command(gate))} ===", flush=True)
        # Stream output while ensuring an all-skipped pytest never looks green.
        passed = 0
        with subprocess.Popen(
            command(gate), cwd=ROOT, env=env, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1,
        ) as proc:
            assert proc.stdout is not None
            for line in proc.stdout:
                print(line, end="", flush=True)
                match = re.search(r"(?<!\w)(\d+) passed\b", line)
                if match:
                    passed += int(match.group(1))
            rc = proc.wait()
        if rc:
            print(f"{name}: exit={rc}", file=sys.stderr)
            return rc
        if passed < gate.ranks:
            print(f"{name}: no passing test summary from every rank "
                  f"(saw {passed}, require >= {gate.ranks}); skips are not PASS.",
                  file=sys.stderr)
            return 1
    print("PASS: every selected gate reported passing pytest cases. "
          "Device gates still require numerical witness review.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("list", help="List supported suites and purpose")
    sub.add_parser("inventory", help="Count discovered pytest files by physical folder")
    sub.add_parser("check", help="Validate registry and local Python import paths (no imports)")
    show = sub.add_parser("show", help="Print exact commands for a suite")
    show.add_argument("suite", choices=sorted(SUITES))
    execute = sub.add_parser("run", help="Execute a curated suite, not every test")
    execute.add_argument("suite", choices=sorted(SUITES))
    execute.add_argument("--ack-npu", action="store_true")
    execute.add_argument("--ack-negative", action="store_true")
    args = parser.parse_args(argv)

    if args.action == "list":
        for name in sorted(SUITES):
            print(f"{name:25s} {SUITE_DESCRIPTIONS[name]}")
        return 0
    if args.action == "inventory":
        counts = inventory()
        for name, number in counts.items():
            print(f"{name:18s} {number:4d}")
        print(f"total pytest files: {sum(counts.values())}; core: {sum(counts.values()) - counts.get('experimental', 0)}; experimental: {counts.get('experimental', 0)}; curated gates: {len(GATES)}")
        return 0
    if args.action == "check":
        errors = check()
        if errors:
            for issue in errors:
                print(f"ERROR: {issue}", file=sys.stderr)
            return 1
        print(f"PASS: {len(SUITES)} suites / {len(GATES)} gates; registry and local import paths valid.")
        return 0
    if args.action == "show":
        describe(args.suite)
        return 0
    return run(args.suite, ack_npu=args.ack_npu, ack_negative=args.ack_negative)


if __name__ == "__main__":
    raise SystemExit(main())

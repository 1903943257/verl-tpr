#!/usr/bin/env python3
"""Summarize *existing* UniAgent session timing logs; never starts Ray/vLLM.

Usage:
    python examples/tpr/analyze_rollout_stages.py /workspace/tpr_experiments/e2e/RUN_NAME

Intervals are approximate and use second-precision log timestamps:
- prep: run_task start -> Claude launch (sandbox start, git clean, setup)
- agent: Claude launch -> Claude finish (both LLM requests AND tool calls)
- reward: eval starts -> eval finishes (SWE-ReBench evaluation)
- post: Claude finish -> task done (includes reward; NOT additive to it)
"""
import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from statistics import median

TS = re.compile(r"(20\d\d-\d\d-\d\d \d\d:\d\d:\d\d)")
RULES = {
    "task_start": re.compile(r"\brun_task start:"),
    "claude_start": re.compile(r"\bclaude_code: launch"),
    "claude_end": re.compile(r"\bclaude_code: claude (?:finished|exited)"),
    "reward_start": re.compile(r"\brunning eval for"),
    "reward_end": re.compile(r"\beval finished in"),
    "task_end": re.compile(r"\btask done: resolved="),
}

def scan_log(path):
    if not path.is_file():
        return []
    rows = []
    for line in path.open(errors="replace"):
        match = TS.search(line)
        if match:
            rows.append((
                datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S"),
                line,
            ))
    return rows

def seconds(times, before, after):
    if before not in times or after not in times:
        return None
    return max(0., (times[after] - times[before]).total_seconds())

def fmt(value):
    return "-" if value is None else f"{value:.0f}"

def pct(numbers, quantile):
    ordered = sorted(numbers)
    index = (len(ordered) - 1) * quantile
    lower = int(index)
    return ordered[lower] + (
        ordered[min(lower + 1, len(ordered) - 1)] - ordered[lower]
    ) * (index - lower)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="E2E run root or agent/step_1")
    args = parser.parse_args()
    root = args.run_dir
    paths = sorted(root.glob("agent/step_*/session-*"))
    if not paths:
        paths = sorted(root.glob("step_*/session-*"))
    if not paths:
        paths = sorted(root.glob("session-*"))
    if not paths:
        parser.error(f"No session folders found under {root}")
    output = []
    for folder in paths:
        logs = scan_log(folder / "framework.log") + scan_log(folder / "task.log")
        if not logs:
            continue
        logs.sort(key=lambda row: row[0])
        times = {}
        for timestamp, line in logs:
            for key, pattern in RULES.items():
                if key not in times and pattern.search(line):
                    times[key] = timestamp
        trajectories = []
        if (folder / "trajectory.json").is_file():
            try:
                trajectories = json.loads(
                    (folder / "trajectory.json").read_text()
                ).get("trajectories", [])
            except (OSError, ValueError):
                pass
        ident = re.search(r"session-sample-(\d+)-rollout-(\d+)", folder.name)
        output.append({
            "name": f"s{ident.group(1)}r{ident.group(2)}" if ident else folder.name[:16],
            "elapsed": (logs[-1][0] - logs[0][0]).total_seconds(),
            "prep": seconds(times, "task_start", "claude_start"),
            "agent": seconds(times, "claude_start", "claude_end"),
            "reward": seconds(times, "reward_start", "reward_end"),
            "post": seconds(times, "claude_end", "task_end"),
            "tracks": len(trajectories),
            "turns": sum(int(t.get("num_turns") or 0) for t in trajectories),
            "tokens": sum(int(t.get("model_token_count") or 0) for t in trajectories),
            "rewards": ",".join(str(t.get("reward_score")) for t in trajectories),
        })
    print(f"Sessions with logs: {len(output)}/{len(paths)}")
    print("Approximate stage seconds; 'post' includes 'reward', so do not sum both.")
    print("stage                 count    p50    p90    max")
    for key in ("elapsed", "prep", "agent", "reward", "post"):
        vals = [entry[key] for entry in output if entry[key] is not None]
        if vals:
            print(f"{key:20s}{len(vals):5d} {fmt(median(vals)):>6s} "
                  f"{fmt(pct(vals, .9)):>6s} {fmt(max(vals)):>6s}")
        else:
            print(f"{key:20s}    0      -      -      -")
    print("\nLongest sessions:")
    print("session      total  prep agent reward tracks turns model_tokens rewards")
    for entry in sorted(output, key=lambda item: item["elapsed"], reverse=True)[:12]:
        print(
            f"{entry['name']:<11s}{fmt(entry['elapsed']):>6s} "
            f"{fmt(entry['prep']):>5s} {fmt(entry['agent']):>5s} "
            f"{fmt(entry['reward']):>6s} {entry['tracks']:>6d} "
            f"{entry['turns']:>5d} {entry['tokens']:>12d} {entry['rewards']}"
        )
    print(f"\nMulti-trajectory sessions: {sum(item['tracks'] > 1 for item in output)}")
    print("Agent time includes tool execution; use Gateway/vLLM traces to isolate pure model time.")

if __name__ == "__main__":
    main()

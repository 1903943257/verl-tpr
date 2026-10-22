#!/usr/bin/env python3
"""Offline, stdlib-only Ascend NPU Full/TPR trace comparison.

Input is ONE RANK's ASCEND_PROFILER_OUTPUT folder for each mode.
Use the same step and execution window for Full/TPR. This script NEVER claims
cross-rank timestamp alignment or host causality without supporting markers.
"""
from __future__ import annotations

import argparse
import collections
import csv
from dataclasses import dataclass
from datetime import datetime
import gzip
import json
import math
from pathlib import Path
import re
import statistics


def norm(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def field(row, *names):
    d = {norm(k): v for k, v in row.items()}
    for name in names:
        if norm(name) in d and d[norm(name)] not in (None, ""):
            return d[norm(name)]
    return None


def number(v):
    if v is None:
        return None
    try:
        return float(str(v).replace(",", "").strip())
    except ValueError:
        return None


def start_us(value, unit):
    n = number(value)
    if n is not None:
        return n * {"ns": .001, "us": 1., "ms": 1000., "s": 1e6}[unit]
    try:
        return datetime.fromisoformat(str(value).strip().replace("Z", "+00:00")).timestamp() * 1e6
    except ValueError:
        raise ValueError(f"Unknown Start Time format {value!r}. Set --csv-start-unit or check CSV.")


@dataclass(frozen=True)
class Event:
    name: str
    category: str
    ts: float   # microseconds
    dur: float  # microseconds
    stream: str
    origin: str

    @property
    def end(self):
        return self.ts + self.dur


def classify(name, typ=""):
    t = (name + " " + typ).lower()
    if re.search(r"hcom|hccl|allreduce|allgather|reducescatter|alltoall|\breceive\b|\bsend\b|hcpre|hcpost", t):
        return "HCCL"
    if re.search(r"fusion_attention|flash_attention|flashattention|attentionscore|attention_score|promptflash|increflash|sparseflash", t):
        return "Attention"
    if re.search(r"matmul|batchmat|groupedmat|\bgemm\b|\bbmm\b|\bmm\b|mat_mul|mat_mul_v2|batch_mat", t):
        return "GEMM"
    return "Other NPU"


def union(intervals):
    x = sorted((s, e) for s, e in intervals if e > s)
    if not x:
        return []
    merged = [[x[0][0], x[0][1]]]
    for s, e in x[1:]:
        if s > merged[-1][1]:
            merged.append([s, e])
        else:
            merged[-1][1] = max(merged[-1][1], e)
    return merged


def length_ms(intervals):
    return sum(e - s for s, e in union(intervals)) / 1000


def stats(evs):
    if not evs:
        return {"count": 0, "sum_ms": 0, "union_ms": 0, "mean_us": 0}
    return {"count": len(evs), "sum_ms": sum(e.dur for e in evs) / 1000,
            "union_ms": length_ms((e.ts, e.end) for e in evs),
            "mean_us": statistics.mean(e.dur for e in evs)}


def concurrency(evs):
    edges = []
    for e in evs:
        if e.dur > 0:
            edges.extend(((e.ts, 1), (e.end, -1)))
    edges.sort(key=lambda x: (x[0], x[1]))  # finish before start
    curr = best = 0
    for _, sign in edges:
        curr += sign
        best = max(best, curr)
    return best


def same_stream_conflicts(evs):
    groups = collections.defaultdict(list)
    for e in evs:
        groups[e.stream].append(e)
    examples = []
    for stream, rows in groups.items():
        rows.sort(key=lambda x: (x.ts, x.end))
        far = None
        for e in rows:
            if far is not None and e.ts < far.end - 1.0:  # >=1us overlap
                if len(examples) < 8:
                    examples.append({"stream": stream, "op1": far.name, "op2": e.name,
                                     "overlap_ms_at_least": round((min(far.end, e.end) - e.ts) / 1000, 5)})
            if far is None or e.end > far.end:
                far = e
    return examples


def findfile(root, filename):
    if root.is_file() and root.name == filename:
        return root
    matches = list(root.glob(filename)) or list(root.glob(f"ASCEND_PROFILER_OUTPUT/{filename}"))
    if not matches:
        matches = list(root.rglob(filename))
    if len(matches) > 1:
        raise RuntimeError(f"Multiple {filename} under {root}: pass a single rank's ASCEND_PROFILER_OUTPUT directory")
    return matches[0] if matches else None


def read_kernels(root, unit):
    path = findfile(root, "kernel_details.csv")
    if path is None:
        return [], None
    events = []
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        for idx, row in enumerate(reader, 2):
            name = field(row, "Name", "op_name", "Operator Name")
            typ = field(row, "Type", "op_type", "task_type") or ""
            start = field(row, "Start Time", "start_time", "Start Time(us)", "start")
            duration = field(row, "Duration(us)", "Duration", "task_duration", "task_duration(us)", "Duration (us)")
            if start is None or duration is None:
                raise ValueError(f"{path}:{idx}: no Start Time or Duration. Columns={cols}. Cannot build time axis.")
            dur = number(duration)
            if dur is None or not math.isfinite(dur) or dur < 0:
                raise ValueError(f"Bad duration {duration!r} in {path}:{idx}")
            stream = str(field(row, "Stream ID", "stream_id", "Stream") or "unknown")
            events.append(Event(str(name or typ), classify(str(name or ""), str(typ)),
                                start_us(start, unit), dur, stream, "kernel_details.csv"))
    return events, str(path)


def read_trace(root):
    path = findfile(root, "trace_view.json")
    if path is None:
        path = findfile(root, "trace_view.json.gz")
    if path is None:
        return [], [], None
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        obj = json.load(f)
    rows = obj.get("traceEvents", []) if isinstance(obj, dict) else obj
    tracks = {}
    for e in rows:
        if e.get("ph") == "M" and e.get("name") in {"process_name", "thread_name"}:
            key = (str(e.get("pid")), str(e.get("tid")))
            value = str((e.get("args") or {}).get("name", ""))
            if e["name"] == "process_name":
                tracks[(key[0], "*")] = value
            else:
                tracks[key] = value
    host, device = [], []
    for e in rows:
        if e.get("ph") != "X":
            continue
        ts, dur = number(e.get("ts")), number(e.get("dur"))
        if ts is None or dur is None or dur < 0:
            continue
        pid, tid = str(e.get("pid")), str(e.get("tid"))
        lane = (tracks.get((pid, "*"), "") + " " + tracks.get((pid, tid), "")).lower()
        cat = str(e.get("cat", "")).lower()
        name = str(e.get("name", ""))
        args = e.get("args") or {}
        if not isinstance(args, dict):
            args = {}
        stream = str(args.get("Stream ID", args.get("stream_id", tid)))
        ev = Event(name, classify(name, cat), ts, dur, stream, "trace_view.json")
        is_npu = (bool(re.search(r"npu|device|stream|aicore|ai core|task scheduler|communication|hccl", lane))
                  or bool(re.search(r"kernel|npu|device", cat)))
        is_host = bool(re.search(r"cpu|python|host|pytorch|cann|runtime", lane)) or not is_npu
        if is_npu and not bool(re.search(r"cpu|python|host", lane)):
            device.append(ev)
        elif is_host:
            host.append(ev)
    return device, host, str(path)


def clip(evs, start, end):
    return [Event(e.name, e.category, max(e.ts, start),
                  min(e.end, end) - max(e.ts, start), e.stream, e.origin)
            for e in evs if min(e.end, end) > max(e.ts, start)]


def summarize(root, csv_unit, w_start_ms, w_end_ms, min_gap_ms, no_trace=False):
    device, kernel_path = read_kernels(root, csv_unit)
    trace_device, host, trace_path = ([], [], None) if no_trace else read_trace(root)
    if not device:
        device = trace_device
        if not device:
            raise ValueError(f"No device time axis found under {root}. Need kernel_details.csv or trace_view.json")
        source = trace_path
    else:
        source = kernel_path
    earliest = min(x.ts for x in device)
    start = earliest + (w_start_ms or 0) * 1000
    end = earliest + w_end_ms * 1000 if w_end_ms is not None else max(x.end for x in device)
    if end <= start:
        raise ValueError("Bad window")
    dev = clip(device, start, end)
    if not dev:
        raise ValueError(f"No device events in window under {root}")
    categorized = {c: [e for e in dev if e.category == c] for c in
                   ("GEMM", "Attention", "HCCL", "Other NPU")}
    h = categorized["HCCL"]
    trace_hccl = clip([e for e in trace_device if e.category == "HCCL"], start, end)
    groups = collections.defaultdict(list)
    for x in h:
        key = ("SEND" if re.search(r"hcom.*send|\bsend\b", x.name, re.I) else
               "RECEIVE" if re.search(r"hcom.*recv|receive", x.name, re.I) else
               "ALLREDUCE" if re.search(r"allreduce", x.name, re.I) else
               "ALLGATHER" if re.search(r"allgather", x.name, re.I) else "OTHER HCCL")
        groups[key].append(x)
    span_ms = (end - start) / 1000
    occupied = union((e.ts, e.end) for e in dev)
    gaps = [(b, a) for (_, b), (a, _) in zip(occupied, occupied[1:]) if a > b]
    gap_details = [{"start_ms": round((s-start)/1000, 3), "end_ms": round((e-start)/1000, 3),
                    "duration_ms": round((e-s)/1000, 3)} for s, e in gaps if e-s >= min_gap_ms*1000]
    largest_gaps = sorted(gap_details, key=lambda x: -x["duration_ms"])[:10]
    sums = {k: stats(v) for k, v in categorized.items()}
    hsum = stats(h)
    cpu_host_valid = False
    markers = {}
    launch = {}
    if host:
        hs, he = min(x.ts for x in host), max(x.end for x in host)
        intersection = max(0, min(he, end) - max(hs, start))
        cpu_host_valid = intersection > min(he-hs, end-start) * 0.01
        if cpu_host_valid:
            host = clip(host, start, end)
            markers = {k: stats([x for x in host if re.search(pattern, x.name, re.I)])
                       for k, pattern in {
                           "segment_or_executor": r"segment|executor|tpr.*(push|pop|visit)",
                           "forward_backward": r"forward|backward|autograd",
                           "python": r"python",
                       }.items()}
            launch = stats([x for x in host if re.search(r"launchkernel|rtkernellaunch|aclrtlaunch|\brtlaunch", x.name, re.I)])
    hot = collections.defaultdict(list)
    for x in dev:
        hot[(x.category, x.name)].append(x)
    tops = sorted(({"category": category, "kernel": name, **stats(ev)} for (category, name), ev in hot.items()),
                  key=lambda x: -x["sum_ms"])[:16]
    cross = same_stream_conflicts(dev)
    report = {"root": str(root), "source": source, "trace_file": trace_path,
              "window_ms": round(span_ms, 3), "events": len(dev),
              "by_category": sums, "hccl_detail": {k: stats(v) for k, v in sorted(groups.items())},
              "hccl_union_ms": hsum["union_ms"],
              "trace_hccl_stats": stats(trace_hccl),
              "trace_hccl_max_concurrent": concurrency(trace_hccl),
              "trace_hccl_stream_conflicts": same_stream_conflicts(trace_hccl),
              "all_device_busy_union_ms": length_ms(occupied),
              "npu_gap_total_ms": sum(e-s for s,e in gaps)/1000,
              "gaps_over_threshold": sum(e-s >= min_gap_ms*1000 for s,e in gaps),
              "largest_device_idle_gaps": largest_gaps, "max_concurrent_hccl": concurrency(h),
              "same_stream_overlap_examples": cross, "top_kernels": tops,
              "host_clock_compatible": cpu_host_valid, "host_markers": markers, "host_launch": launch,
              "warnings": []}
    if report["trace_hccl_stream_conflicts"]:
        report["warnings"].append("Communication timeline has >1us same-stream overlapping HCCL intervals")
    if cross:
        report["warnings"].append("Same-stream NPU tasks overlap by >1us: timestamp/stream attribution is suspect")
    if hsum["union_ms"] > 0 and hsum["sum_ms"] / hsum["union_ms"] > 16:
        report["warnings"].append("HCCL sum/union >16x; inspect streams and source clock before interpreting union")
    if kernel_path and trace_path and not cpu_host_valid:
        report["warnings"].append("Host trace and CSV device time windows do not overlap; CPU scheduling attribution disabled")
    return report


def print_report(r, label):
    print(f"\n=== {label} ===")
    print(f"source={r['source']} window_ms={r['window_ms']} events={r['events']}")
    print("Category       count  sum_ms  union_ms   mean_us")
    for name, d in r["by_category"].items():
        print(f"{name:<14} {d['count']:>5} {d['sum_ms']:>8.2f} {d['union_ms']:>9.2f} {d['mean_us']:>9.2f}")
    print(f"DEVICE_ACTIVITY_UNION_MS={r['all_device_busy_union_ms']:.3f}  DEVICE_IDLE_INTERNAL_GAPS_MS={r['npu_gap_total_ms']:.3f}")
    print(f"HCCL max_simultaneous={r['max_concurrent_hccl']} same_stream_conflicts={len(r['same_stream_overlap_examples'])}")
    if r["trace_hccl_stats"]["count"]:
        print("Communication TRACE HCCL (separate timeline, do NOT add to CSV):",
              r["trace_hccl_stats"], "max_concurrent=",r["trace_hccl_max_concurrent"],
              "same_stream_conflicts=",len(r["trace_hccl_stream_conflicts"]))
    for name,d in r['hccl_detail'].items():
        print(f" HCCL {name}: count={d['count']} sum_ms={d['sum_ms']:.3f} union_ms={d['union_ms']:.3f}")
    print("Largest DEVICE gaps (not automatically Host bottlenecks):",r['largest_device_idle_gaps'][:5])
    print(f"Host clock compatible={r['host_clock_compatible']} host markers={r['host_markers']} launch={r['host_launch']}")
    print("Top kernels (summed durations; overlap not subtracted):")
    for item in r['top_kernels'][:12]:
        print(f"  {item['category']:<11} n={item['count']:<5} sum={item['sum_ms']:>8.2f}ms {item['kernel'][:95]}")
    for w in r['warnings']:
        print("WARNING:",w)
    for x in r['same_stream_overlap_examples'][:4]:
        print("WARNING example:", x)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--full',type=Path,required=True,help='single rank FULL ASCEND_PROFILER_OUTPUT')
    p.add_argument('--tpr',type=Path,required=True,help='same rank TPR ASCEND_PROFILER_OUTPUT')
    p.add_argument('--csv-start-unit',choices=('ns','us','ms','s'),default='us',
                   help='numeric kernel_details.csv Start Time unit; NPU Duration assumed us')
    p.add_argument('--window-start-ms',type=float,default=0,help='relative to first device kernel in each profile')
    p.add_argument('--window-end-ms',type=float,help='relative to first device kernel, omit for whole trace')
    p.add_argument('--min-gap-ms',type=float,default=.1)
    p.add_argument('--no-trace',action="store_true",help="skip possibly huge trace_view.json; device-only CSV analysis")
    p.add_argument('--json-out',type=Path)
    args=p.parse_args()
    full=summarize(args.full,args.csv_start_unit,args.window_start_ms,args.window_end_ms,args.min_gap_ms,args.no_trace)
    tpr=summarize(args.tpr,args.csv_start_unit,args.window_start_ms,args.window_end_ms,args.min_gap_ms,args.no_trace)
    print_report(full,'CP4 FULL')
    print_report(tpr,'CP4 TPR')
    print("\n=== Full minus TPR (positive means Full more costly; not additive across categories) ===")
    for k in ('GEMM','Attention','HCCL','Other NPU'):
        a,b=full['by_category'][k],tpr['by_category'][k]
        print(f"{k:<12} count_delta={a['count']-b['count']:+d} sum_ms_delta={a['sum_ms']-b['sum_ms']:+.3f} union_ms_delta={a['union_ms']-b['union_ms']:+.3f}")
    print(f"NPU idle gaps delta={full['npu_gap_total_ms']-tpr['npu_gap_total_ms']:+.3f} ms")
    print('IMPORTANT: categories can overlap. Their unions are NOT additive; device gap is between all device tasks.')
    print('IMPORTANT: compare the same rank, same step, same window, and backend. CPU/segment attribution needs aligned host markers.')
    if args.json_out:
        args.json_out.write_text(json.dumps({'full':full,'tpr':tpr},ensure_ascii=False,indent=2),encoding='utf-8')
        print('json_out=',args.json_out)


if __name__=='__main__':
    main()
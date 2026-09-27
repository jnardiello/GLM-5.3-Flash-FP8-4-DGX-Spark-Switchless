#!/usr/bin/env python3
"""Per-rank MemAvailable sampler over SSH for fidelity measurements; stdlib only.

Every --interval seconds, read MemAvailable from each host (rank order) with
`ssh -o BatchMode=yes <host> 'grep MemAvailable /proc/meminfo'`, append one JSONL
record per rank to --out, and track per-rank minima. When rank 0 drops below
--abort-gib, create --abort-file so the collectors stop between requests.
`--summary FILE` prints the per-rank minima of an existing JSONL file.
"""

from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path
import re
import signal
import subprocess
import sys
import time


MEM_RE = re.compile(r"^MemAvailable:\s+(\d+)\s+kB\s*$", re.MULTILINE)
REMOTE = "grep MemAvailable /proc/meminfo"
KIB_PER_GIB = 1024 * 1024


def parse_meminfo(text: str):
    match = MEM_RE.search(text or "")
    return int(match.group(1)) if match else None


def sample(hosts: list[str], ssh: str, timeout: float) -> list[dict]:
    """One round: all hosts in parallel, results in rank order."""
    ts = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="milliseconds")
    procs = []
    for host in hosts:
        try:
            procs.append(subprocess.Popen([ssh, "-o", "BatchMode=yes", host, REMOTE],
                                          stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True))
        except OSError as error:
            procs.append(error)
    records = []
    deadline = time.monotonic() + timeout
    for rank, (host, proc) in enumerate(zip(hosts, procs)):
        record = {"ts": ts, "rank": rank, "host": host, "mem_available_kib": None}
        if isinstance(proc, OSError):
            record["error"] = type(proc).__name__
        else:
            try:
                out, _ = proc.communicate(timeout=max(0.1, deadline - time.monotonic()))
                record["mem_available_kib"] = parse_meminfo(out)
                if record["mem_available_kib"] is None:
                    record["error"] = f"exit {proc.returncode}"
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.communicate()
                record["error"] = "timeout"
        records.append(record)
    return records


def minima(records) -> dict:
    out = {}
    for record in records:
        value = record.get("mem_available_kib")
        if value is None:
            continue
        key = (record["rank"], record["host"])
        if key not in out or value < out[key]["mem_available_kib"]:
            out[key] = {"rank": record["rank"], "host": record["host"],
                        "mem_available_kib": value, "ts": record["ts"]}
    return {rank: item for (rank, _), item in sorted(out.items())}


def print_minima(found: dict, stream=None) -> None:
    stream = stream or sys.stdout
    for rank, item in found.items():
        print(f"rank {rank} {item['host']}: min MemAvailable {item['mem_available_kib'] / KIB_PER_GIB:.2f} GiB "
              f"({item['mem_available_kib']} KiB) at {item['ts']}", file=stream)


def summarize(path: Path) -> int:
    records = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    found = minima(records)
    if not found:
        print("no samples with MemAvailable", file=sys.stderr)
        return 1
    print_minima(found)
    return 0


def run(args) -> int:
    stop = {"flag": False}

    def handle(signum, frame):
        stop["flag"] = True

    signal.signal(signal.SIGTERM, handle)
    signal.signal(signal.SIGINT, handle)
    threshold = args.abort_gib * KIB_PER_GIB
    seen, rounds, aborted = [], 0, False
    args.out.parent.mkdir(parents=True, exist_ok=True)
    while not stop["flag"]:
        started = time.monotonic()
        records = sample(args.hosts, args.ssh, args.ssh_timeout)
        with open(args.out, "a", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, separators=(",", ":")) + "\n")
        seen.extend(r for r in records if r.get("mem_available_kib") is not None)
        head = records[0].get("mem_available_kib")
        if head is not None and head < threshold and not aborted:
            if args.abort_file:
                args.abort_file.parent.mkdir(parents=True, exist_ok=True)
                args.abort_file.write_text(f"{records[0]['ts']} rank0 MemAvailable {head} KiB\n")
            print(f"ABORT: rank 0 MemAvailable {head / KIB_PER_GIB:.2f} GiB < {args.abort_gib} GiB; "
                  f"created {args.abort_file}", file=sys.stderr, flush=True)
            aborted = True
        for record in records:
            if record.get("error"):
                print(f"warning: rank {record['rank']} {record['host']}: {record['error']}",
                      file=sys.stderr, flush=True)
        rounds += 1
        if args.count and rounds >= args.count:
            break
        remaining = args.interval - (time.monotonic() - started)
        while remaining > 0 and not stop["flag"]:
            time.sleep(min(remaining, 0.5))
            remaining -= 0.5
    print_minima(minima(seen), sys.stderr)
    return 2 if aborted else 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--hosts", nargs="+", help="SSH hosts in rank order (rank 0 first)")
    p.add_argument("--out", type=Path, help="JSONL file to append to")
    p.add_argument("--interval", type=float, default=5.0)
    p.add_argument("--abort-gib", type=float, default=1.0)
    p.add_argument("--abort-file", type=Path)
    p.add_argument("--ssh", default="ssh", help="ssh executable")
    p.add_argument("--ssh-timeout", type=float, default=4.0)
    p.add_argument("--count", type=int, help="stop after this many rounds")
    p.add_argument("--summary", type=Path, help="print per-rank minima of a JSONL file and exit")
    args = p.parse_args(argv)
    if args.summary:
        return summarize(args.summary)
    if not args.hosts or not args.out:
        p.error("--hosts and --out are required unless --summary")
    return run(args)


if __name__ == "__main__":
    sys.exit(main())

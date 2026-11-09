"""Run Lumen while recording CPU and memory, then print figures for the technical note.

Usage (from msrc/):  python tools/measure.py
Go through the demo, then close the MuJoCo window. Samples go to logs/resources.csv.
CPU is a percentage of the WHOLE machine (100% = every core busy).
Latency comes from the turns this run added to logs/turns.csv.
"""
import argparse
import csv
import os
import statistics
import subprocess
import sys
import time

import psutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "logs", "resources.csv")
TURNS = os.path.join(ROOT, "logs", "turns.csv")


def read_turns():
    if not os.path.exists(TURNS):
        return []
    with open(TURNS, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def percentile(values, p):
    values = sorted(values)
    return values[min(len(values) - 1, int(round(p / 100 * (len(values) - 1))))]


def sample(root, tracked, cores):
    """(cpu %, RSS MB) of the process and its children. On Windows a venv's python.exe is a launcher."""
    cpu, rss = 0.0, 0
    try:
        live = [root] + root.children(recursive=True)
    except psutil.Error:
        return None
    for p in live:
        p = tracked.setdefault(p.pid, p)   # cpu_percent needs the same object between calls
        try:
            cpu += p.cpu_percent(None)
            rss += p.memory_info().rss
        except psutil.Error:
            pass
    return cpu / cores, rss / 1e6


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("script", nargs="?", default=os.path.join(ROOT, "src", "main.py"))
    parser.add_argument("--warmup", type=float, default=20.0,
                        help="seconds of model loading left out of the CPU figures")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    cores = psutil.cpu_count()
    turns_before = len(read_turns())
    proc = subprocess.Popen([sys.executable, args.script], cwd=ROOT)
    root = psutil.Process(proc.pid)
    tracked = {}
    samples = []
    t0 = time.monotonic()
    sample(root, tracked, cores)

    try:
        while proc.poll() is None:
            time.sleep(1.0)
            result = sample(root, tracked, cores)
            if result:
                samples.append((time.monotonic() - t0, *result))
    except KeyboardInterrupt:
        proc.terminate()
    proc.wait()

    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["seconds", "cpu_percent_of_machine", "rss_mb"])
        w.writerows((round(t, 1), round(c, 1), round(m)) for t, c, m in samples)

    running = [c for t, c, _ in samples if t >= args.warmup]
    print(f"\n===== Measurements ({len(samples)} s) =====")
    print(f"Machine: {cores} logical cores, {psutil.virtual_memory().total / 1e9:.1f} GB RAM")
    if running:
        print(f"CPU after {args.warmup:.0f} s warmup: mean {statistics.mean(running):.0f}%  "
              f"p95 {percentile(running, 95):.0f}%  peak {max(running):.0f}% of whole machine")
    else:
        print("CPU: run was shorter than the warmup, no figures")
    if samples:
        print(f"Memory (RSS): mean {statistics.mean(m for _, _, m in samples):.0f} MB  "
              f"peak {max(m for _, _, m in samples):.0f} MB")

    turns = read_turns()[turns_before:]
    for kind in ("talk", "look", "seek"):
        for column in ("total_ms", "model_ms"):
            values = [float(r[column]) for r in turns if r.get("kind") == kind]
            if values:
                print(f"{kind:>4} {column}: n={len(values)}  median {statistics.median(values):.0f}  "
                      f"p95 {percentile(values, 95):.0f}  max {max(values):.0f}")
    print(f"Samples saved to {OUT}")


if __name__ == "__main__":
    main()

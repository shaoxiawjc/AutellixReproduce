#!/usr/bin/env python3
"""Review Autellix baseline experiment results.

Usage:
  python scripts/review_baseline.py [results_dir]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> None:
    results_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "results/autellix_baseline")
    summaries = []
    for path in sorted(results_dir.glob("*_async_summary.json")):
        summaries.append(json.loads(path.read_text()))

    if not summaries:
        print(f"No async summary files found in {results_dir}")
        return

    by_workload: dict[str, list[dict]] = {}
    for s in summaries:
        by_workload.setdefault(s["workload"], []).append(s)

    for workload, entries in sorted(by_workload.items()):
        print(f"\n{'='*80}")
        print(f"  Workload: {workload.upper()}")
        print(f"{'='*80}")

        order = {"fcfs": 0, "mlfq": 1, "plas": 2, "atlas": 3}
        entries.sort(key=lambda s: (
            order.get(s["policy"], 99),
            0 if s.get("prefix_caching", True) else 1,
        ))

        fcfs_lat = next(
            (s["avg_program_token_latency_s"] for s in entries
             if s["policy"] == "fcfs" and s.get("prefix_caching", True)),
            1.0,
        )

        hdr = (f"  {'Policy':<22} {'Progs':>5} {'Calls':>5} "
               f"{'Elapsed':>9} {'Prog/s':>7} "
               f"{'TokLat':>10} {'P95Tok':>10} {'P99Tok':>10} {'vsFCFS':>7}")
        print(hdr)
        print(f"  {'-'*len(hdr)}")

        for s in entries:
            policy = s["policy"]
            prefix = "no-cache" if not s.get("prefix_caching", True) else ""
            label = f"{policy} {prefix}".strip()
            avg = s["avg_program_token_latency_s"]
            p95 = s["p95_program_token_latency_s"]
            p99 = s["p99_program_token_latency_s"]
            speedup = fcfs_lat / avg if avg > 0 else 0
            marker = " ← baseline" if (policy == "fcfs" and prefix == "") else ""

            print(
                f"  {label:<22} {s['programs']:>5} {s['calls']:>5} "
                f"{s['elapsed_s']:>8.1f}s {s['throughput_programs_per_s']:>7.4f} "
                f"{avg:>10.6f} {p95:>10.6f} {p99:>10.6f} "
                f"{speedup:>6.2f}x{marker}"
            )

        # Best among policies WITH prefix caching (apples-to-apples)
        cached = [s for s in entries if s.get("prefix_caching", True)]
        if cached:
            best = min(cached, key=lambda s: s["avg_program_token_latency_s"])
            best_lat = best["avg_program_token_latency_s"]
            delta = (fcfs_lat - best_lat) / fcfs_lat * 100
            print(f"\n  → Best (cached): {best['policy']} "
                  f"({best_lat:.6f} s/tok, {delta:+.1f}% vs FCFS)")

    # Cross-workload summary
    print(f"\n{'='*80}")
    print(f"  CROSS-WORKLOAD (cached policies only)")
    print(f"{'='*80}")
    print(f"  {'Workload':<12} {'FCFS':>10} {'MLFQ':>10} {'PLAS':>10} {'ATLAS':>10} {'Best':>14}")
    print(f"  {'-'*62}")
    for workload, entries in sorted(by_workload.items()):
        cached = {s["policy"]: s for s in entries if s.get("prefix_caching", True)}
        fcfs = cached.get("fcfs", {}).get("avg_program_token_latency_s", 0)
        row = f"  {workload:<12}"
        for p in ["fcfs", "mlfq", "plas", "atlas"]:
            v = cached.get(p, {}).get("avg_program_token_latency_s")
            row += f" {v:>10.6f}" if v is not None else f" {'N/A':>10}"
        if cached:
            best_p = min(cached, key=lambda p: cached[p]["avg_program_token_latency_s"])
            best_v = cached[best_p]["avg_program_token_latency_s"]
            gain = (fcfs - best_v) / fcfs * 100 if fcfs > 0 else 0
            row += f" {best_p:>8} {gain:>+5.1f}%"
        print(row)


if __name__ == "__main__":
    main()

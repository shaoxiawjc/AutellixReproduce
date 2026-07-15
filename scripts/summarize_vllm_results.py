from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def main() -> None:
    args = parse_args()
    rows = []
    for summary_path in sorted(Path(args.results_dir).glob("*_summary.json")):
        if summary_path.name.endswith("_async_summary.json"):
            continue
        rows.append(load_summary_row(summary_path))
    for summary_path in sorted(Path(args.results_dir).glob("*_async_summary.json")):
        rows.append(load_summary_row(summary_path))

    rows = sorted(rows, key=lambda r: (r["mode"], r["workload"], r["policy"]))
    if args.csv:
        write_csv(rows, Path(args.csv))
    print_table(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir")
    parser.add_argument("--csv")
    return parser.parse_args()


def load_summary_row(path: Path) -> dict[str, Any]:
    summary = json.loads(path.read_text())
    mode = summary.get("mode") or ("async" if "_async_" in path.name else "wave")
    calls_path = matching_calls_path(path)
    call_stats = load_call_stats(calls_path) if calls_path.exists() else {}
    return {
        "mode": mode,
        "workload": summary["workload"],
        "policy": summary["policy"],
        "programs": summary["programs"],
        "calls": summary["calls"],
        "elapsed_s": summary["elapsed_s"],
        "programs_per_s": summary["throughput_programs_per_s"],
        "output_tokens_per_s": summary["throughput_output_tokens_per_s"],
        "avg_program_latency_s": summary["avg_program_latency_s"],
        "p95_program_latency_s": summary["p95_program_latency_s"],
        **call_stats,
    }


def matching_calls_path(summary_path: Path) -> Path:
    name = summary_path.name
    if name.endswith("_async_summary.json"):
        return summary_path.with_name(name.replace("_async_summary.json", "_async_calls.jsonl"))
    return summary_path.with_name(name.replace("_summary.json", "_calls.jsonl"))


def load_call_stats(path: Path) -> dict[str, float]:
    call_latencies = []
    priorities_by_policy_program: dict[str, list[int]] = defaultdict(list)
    metric_waits = []
    metric_first_token = []
    with path.open() as handle:
        for line in handle:
            record = json.loads(line)
            call_latencies.append(record["finish_time"] - record["submit_time"])
            priorities_by_policy_program[record["program_id"]].append(record["priority"])
            metrics = record.get("vllm_metrics") or {}
            queued_ts = metrics.get("queued_ts")
            scheduled_ts = metrics.get("scheduled_ts")
            if queued_ts and scheduled_ts:
                metric_waits.append(scheduled_ts - queued_ts)
            first_token_latency = metrics.get("first_token_latency")
            if first_token_latency:
                metric_first_token.append(first_token_latency)

    return {
        "avg_call_latency_s": mean(call_latencies),
        "p95_call_latency_s": percentile(call_latencies, 0.95),
        "avg_metric_wait_s": mean(metric_waits),
        "p95_metric_wait_s": percentile(metric_waits, 0.95),
        "avg_first_token_latency_s": mean(metric_first_token),
        "priority_changes": sum(
            len(set(priorities)) > 1 for priorities in priorities_by_policy_program.values()
        ),
    }


def print_table(rows: list[dict[str, Any]]) -> None:
    columns = [
        "mode",
        "workload",
        "policy",
        "programs",
        "calls",
        "elapsed_s",
        "programs_per_s",
        "avg_program_latency_s",
        "p95_program_latency_s",
        "avg_call_latency_s",
        "avg_metric_wait_s",
    ]
    print(",".join(columns))
    for row in rows:
        print(",".join(format_value(row.get(column, "")) for column in columns))


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    columns = list(rows[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * q)))
    return ordered[index]


if __name__ == "__main__":
    main()

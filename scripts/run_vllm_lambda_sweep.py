from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    args = parse_args()
    script = Path(__file__).with_name("run_vllm_async_experiment.py")
    extra_args = args.extra_args
    if extra_args and extra_args[0] == "--":
        extra_args = extra_args[1:]
    for workload in args.workloads:
        for policy in args.policies:
            for arrival_rate in args.arrival_rates:
                cmd = [
                    sys.executable,
                    str(script),
                    "--workload",
                    workload,
                    "--policy",
                    policy,
                    "--arrival-rate",
                    str(arrival_rate),
                    "--arrival-seed",
                    str(args.arrival_seed),
                    "--output-dir",
                    args.output_dir,
                    "--max-programs",
                    str(args.max_programs),
                    "--max-calls-per-program",
                    str(args.max_calls_per_program),
                    "--max-tokens",
                    str(args.max_tokens),
                    "--max-model-len",
                    str(args.max_model_len),
                    "--gpu-memory-utilization",
                    str(args.gpu_memory_utilization),
                    "--max-num-seqs",
                    str(args.max_num_seqs),
                    "--max-num-batched-tokens",
                    str(args.max_num_batched_tokens),
                ]
                if args.disable_prefix_caching:
                    cmd.append("--disable-prefix-caching")
                if extra_args:
                    cmd.extend(extra_args)
                print(" ".join(cmd), flush=True)
                if not args.dry_run:
                    subprocess.run(cmd, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workloads",
        nargs="+",
        default=["sharegpt", "bfcl", "lats"],
        choices=["sharegpt", "bfcl", "lats"],
    )
    parser.add_argument(
        "--policies",
        nargs="+",
        default=["fcfs", "mlfq", "plas", "atlas"],
        choices=["fcfs", "mlfq", "plas", "atlas"],
    )
    parser.add_argument(
        "--arrival-rates",
        nargs="+",
        type=float,
        default=[0.2, 0.5, 1.0, 1.5, 2.0],
    )
    parser.add_argument("--arrival-seed", type=int, default=0)
    parser.add_argument("--output-dir", default="results/vllm_async")
    parser.add_argument("--max-programs", type=int, default=128)
    parser.add_argument("--max-calls-per-program", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.75)
    parser.add_argument("--max-num-seqs", type=int, default=32)
    parser.add_argument("--max-num-batched-tokens", type=int, default=4096)
    parser.add_argument("--disable-prefix-caching", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("extra_args", nargs=argparse.REMAINDER)
    return parser.parse_args()


if __name__ == "__main__":
    main()

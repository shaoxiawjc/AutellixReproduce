from __future__ import annotations

import argparse
from dataclasses import asdict

from autellix_reproduce.metrics import summarize
from autellix_reproduce.simulator import SimulationConfig, Simulator
from autellix_reproduce.trace_io import load_programs_jsonl
from autellix_reproduce.workloads import (
    figure2_workload,
    synthetic_chatbot_workload,
    synthetic_mcts_workload,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", choices=["fcfs", "mlfq", "plas", "atlas"], default="plas")
    parser.add_argument("--workload", choices=["chatbot", "mcts", "figure2"], default="chatbot")
    parser.add_argument("--programs", type=int, default=100)
    parser.add_argument("--arrival-rate", type=float, default=0.8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--trace", help="Load workload from a JSONL trace file.")
    args = parser.parse_args()

    if args.trace:
        programs = load_programs_jsonl(args.trace)
    elif args.workload == "figure2":
        programs = figure2_workload()
    elif args.workload == "mcts":
        programs = synthetic_mcts_workload(args.programs, args.arrival_rate, args.seed)
    else:
        programs = synthetic_chatbot_workload(args.programs, args.arrival_rate, args.seed)

    config = SimulationConfig(policy=args.policy, batch_size=args.batch_size)
    result = Simulator(programs, config).run()
    summary = summarize(result)
    for key, value in asdict(summary).items():
        print(f"{key}: {value}")

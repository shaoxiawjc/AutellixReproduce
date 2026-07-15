# AutellixReproduce

CPU-only reproduction harness for the scheduling ideas in **Autellix: An Efficient Serving Engine for LLM Agents as General Programs**.

This repository starts with a local simulator because the current development machine has no GPU. The simulator validates program-level scheduling behavior before integrating with the forked vLLM repository.

## Local Setup

```bash
uv sync --dev
uv run pytest
uv run ruff check .
```

## Run A Small Simulation

```bash
uv run autellix-sim --policy plas --programs 200 --arrival-rate 0.8
uv run autellix-sim --policy mlfq --workload figure2
uv run autellix-sim --policy atlas --trace data/traces/example.jsonl
```

## Current Scope

- Program-level trace model.
- Process table with service and waiting time.
- CPU-only scheduling simulator.
- FCFS, MLFQ, PLAS, and ATLAS policies.
- Synthetic workloads for smoke testing and paper-style examples.

## Later GPU Scope

After GPU server access is available:

- Clone `git@github.com:shaoxiawjc/vllm.git`.
- Add program/session metadata to vLLM requests.
- Port PLAS/ATLAS/MLFQ into vLLM scheduler hooks.
- Replay ShareGPT/BFCL/LATS-like workloads against real models.
- Add multi-engine routing and, later, swap-kernel ablations.

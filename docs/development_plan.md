# Development Plan

## Stage 1: CPU-only scheduler reproduction

This stage is runnable on macOS without GPU resources.

Deliverables:

- Program-level trace schema.
- Process table with per-program service, waiting, and critical-path state.
- FCFS, MLFQ, PLAS, and ATLAS schedulers.
- Discrete-event simulator with batching and preemption.
- Synthetic Chatbot and MCTS-like workloads.
- Program-level metrics: throughput, token latency, P95/P99, makespan, wait/service breakdown.

Validation:

```bash
uv run pytest
uv run ruff check .
uv run autellix-sim --policy plas --workload figure2
uv run autellix-sim --policy atlas --workload mcts --programs 10 --batch-size 4
```

## Stage 2: Trace loaders

Add real workload preparation while still staying CPU-only:

- ShareGPT conversations: one conversation is one program; each assistant generation is one LLM call.
- BFCL tasks: one function-calling task is one program; each assistant/tool turn sequence becomes dependent calls.
- LATS/MCTS: use a trace exporter if available; otherwise generate MCTS-like DAGs with calibrated call counts and token distributions.

Target trace format:

```json
{
  "program_id": "p0",
  "arrival_time": 0.0,
  "calls": [
    {
      "call_id": "c0",
      "thread_id": "main",
      "parents": [],
      "prefill_tokens": 512,
      "decode_tokens": 80
    }
  ]
}
```

## Stage 3: vLLM integration

Use the fork:

```bash
git clone git@github.com:shaoxiawjc/vllm.git external/vllm
```

Initial integration tasks:

- Identify the vLLM scheduler hook for sequence-group ordering and preemption.
- Add request metadata: `program_id`, `session_id`, `thread_id`, `parent_call_ids`.
- Mirror the simulator's `ProcessTable` inside the vLLM scheduler path.
- Implement policy switch: `fcfs`, `mlfq`, `plas`, `atlas`.
- Preserve vLLM-opt features first: prefix caching, chunked prefill, and multi-step scheduling.

Defer until single-engine scheduling works:

- Multi-engine `AsyncMultiLLMEngine` equivalent.
- Locality-aware routing.
- CUDA swap-kernel changes.

## Stage 4: GPU experiments

Start small after GPU access is provided:

- Small model first: TinyLlama, Qwen2.5-0.5B/1.5B, or another model that fits easily.
- Single GPU, single engine.
- Small ShareGPT subset and synthetic traces.
- Compare FCFS, vLLM-opt, MLFQ, PLAS.

Then expand:

- BFCL and MCTS-like workloads.
- ATLAS for multi-threaded DAG traces.
- Multi-engine routing.
- Swap-time ablation.

## Reporting

Results should be tagged by scope:

- `simulated`: CPU-only simulator results.
- `small-real`: small-model vLLM results.
- `paper-scale`: large-GPU experiments comparable to the paper.

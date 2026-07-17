# AutellixReproduce

Experiment scripts for evaluating the Autellix program-aware scheduler in a
forked vLLM engine.

## Prerequisites

- GPU server with the forked vLLM installed (see `../vllm`)
- ShareGPT and BFCL datasets on disk
- Python 3.11+

## Quick start

```bash
# Single experiment — ShareGPT with PLAS at moderate arrival rate
python scripts/run_vllm_async_experiment.py \
    --workload sharegpt --policy plas \
    --arrival-rate 0.5 --max-programs 128

# Full pipeline (4 phases: smoke → baseline → lambda sweep → prefix ablation)
AUTELLIX_SMOKE=1 AUTELLIX_BASELINE=1 AUTELLIX_LAMBDA=1 \
    bash run_autellix_experiments.sh
```

## Experiment scripts

| Script | Purpose |
|---|---|
| `run_vllm_async_experiment.py` | Single experiment — one workload × policy × arrival rate |
| `run_autellix_experiments.sh` | Orchestrates the full 4-phase pipeline |
| `summarize_vllm_results.py` | Aggregates result JSON files into a CSV summary |

## Key parameters

| Parameter | Meaning |
|---|---|
| `--workload` | `sharegpt`, `bfcl`, or `lats` (synthetic DAG) |
| `--policy` | `fcfs`, `mlfq`, `plas`, or `atlas` |
| `--arrival-rate` | Poisson program arrival rate (programs/s); 0 = all at once |
| `--max-programs` | Number of programs in the experiment |
| `--max-calls-per-program` | Max LLM calls per program |
| `--disable-prefix-caching` | Turn off prefix cache (pure vLLM baseline) |

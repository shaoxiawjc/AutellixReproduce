# First GPU Results

Date: 2026-07-15

Environment:

- GPU: 1x NVIDIA GeForce RTX 4090
- Model: `/root/resources/models/qwen3_0d6b`
- vLLM source: `/root/autellix_reproduce_work/vllm`
- Datasets:
  - ShareGPT: `/root/resources/datasets/sharegpt/ShareGPT_V3_unfiltered_cleaned_split_no_imsorry.json`
  - BFCL: `/root/resources/datasets/BFCL/BFCL_v3_multi_turn_base.json`
- Qwen3 thinking mode: disabled with `enable_thinking=False`

Command shape:

```bash
cd /root/autellix_reproduce_work/AutellixReproduce
PYTHONPATH=/root/autellix_reproduce_work/vllm:$PWD/src \
/root/autellix_reproduce_work/.venv/bin/python scripts/run_vllm_core_experiment.py \
  --workload sharegpt \
  --policy plas \
  --max-programs 16 \
  --max-calls-per-program 3 \
  --max-tokens 32 \
  --max-model-len 2048 \
  --max-num-seqs 16 \
  --max-num-batched-tokens 2048 \
  --gpu-memory-utilization 0.65 \
  --output-dir results/vllm_core
```

Results:

| Workload | Policy | Programs | Calls | Elapsed s | Programs/s | Output tok/s | Avg latency s | P95 latency s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BFCL | FCFS | 16 | 45 | 0.615592 | 25.991 | 2319.7 | 0.579341 | 0.615592 |
| BFCL | MLFQ | 16 | 45 | 0.600211 | 26.657 | 2379.2 | 0.573450 | 0.600211 |
| BFCL | PLAS | 16 | 45 | 0.556376 | 28.758 | 2566.6 | 0.520909 | 0.556376 |
| ShareGPT | FCFS | 16 | 40 | 0.784196 | 20.403 | 1608.0 | 0.660768 | 0.784196 |
| ShareGPT | MLFQ | 16 | 40 | 0.801473 | 19.963 | 1573.4 | 0.670723 | 0.801473 |
| ShareGPT | PLAS | 16 | 40 | 0.757160 | 21.132 | 1665.4 | 0.633230 | 0.757160 |

Interpretation:

- PLAS is consistently best in this small run.
- The gain is modest because the current driver uses synchronized waves: one dependency-ready call per active program per wave.
- This validates the vLLM priority-scheduler path and dataset/model plumbing, but it is not yet a paper-scale asynchronous Autellix reproduction.

## Async Driver Results

The async driver runs one coroutine per program. Each program submits its next call immediately after its previous call completes, with priority recomputed from its current attained service.

| Workload | Policy | Programs | Calls | Elapsed s | Programs/s | Output tok/s | Avg latency s | P95 latency s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BFCL | FCFS | 16 | 45 | 0.730171 | 21.913 | 1955.7 | 0.693539 | 0.728611 |
| BFCL | MLFQ | 16 | 45 | 0.733548 | 21.812 | 1946.7 | 0.697173 | 0.732333 |
| BFCL | PLAS | 16 | 45 | 0.633444 | 25.259 | 2254.3 | 0.594216 | 0.631439 |
| ShareGPT | FCFS | 16 | 40 | 0.921374 | 17.365 | 1368.6 | 0.779949 | 0.916675 |
| ShareGPT | MLFQ | 16 | 40 | 0.950156 | 16.839 | 1327.2 | 0.808499 | 0.945828 |
| ShareGPT | PLAS | 16 | 40 | 0.875731 | 18.270 | 1439.9 | 0.741346 | 0.871546 |

Async interpretation:

- PLAS improves over FCFS and MLFQ on both workloads.
- BFCL PLAS improves program throughput by about 15.3% over FCFS and 15.8% over MLFQ.
- ShareGPT PLAS improves program throughput by about 5.2% over FCFS and 8.5% over MLFQ.
- The gains are still smaller than paper-scale results because this run uses Qwen3-0.6B, only 16 programs, short generations, one GPU, and vLLM's existing priority scheduler rather than a full custom preemptive Autellix scheduler.

Next step:

- Increase program counts and calls per program to create stronger program-level head-of-line blocking.
- Add trace-level per-request wait-time extraction from vLLM metrics.
- Patch vLLM's scheduler path directly if existing priority scheduling is insufficient for preemption and MLFQ queue semantics.

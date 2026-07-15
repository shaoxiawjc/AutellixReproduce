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

Next step:

- Move from synchronized waves to `AsyncLLM` or a lower-level engine loop so each program can submit its next call immediately after its prior call completes.
- Then rerun with larger program counts and stronger long-tail settings to expose program-level HoL blocking more clearly.

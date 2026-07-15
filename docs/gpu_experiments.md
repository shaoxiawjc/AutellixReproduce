# GPU Core Experiments

The first GPU experiment uses one RTX 4090 and Qwen3-0.6B. It intentionally skips multi-engine load balancing.

## Environment

```bash
cd /root/autellix_reproduce_work/AutellixReproduce
export PYTHONPATH=/root/autellix_reproduce_work/vllm:$PWD/src
/root/autellix_reproduce_work/.venv/bin/python scripts/run_vllm_core_experiment.py \
  --workload sharegpt \
  --policy plas \
  --max-programs 16 \
  --max-calls-per-program 4 \
  --max-tokens 64
```

Qwen3 thinking is disabled with:

```python
tokenizer.apply_chat_template(..., enable_thinking=False)
```

## Policies

- `fcfs`: vLLM FCFS scheduler.
- `mlfq`: vLLM priority scheduler with all newly released calls at priority 0. This is a placeholder for request-level priority without program attained service.
- `plas`: vLLM priority scheduler with priority equal to the program's attained generated-token service.

## Current Limitation

This driver releases one dependency-ready call per program per wave. It is enough for the first one-GPU sanity experiment, but it is more synchronized than Autellix's actual asynchronous program execution. The next step is to switch to `AsyncLLM` so that a program can release its next call immediately when its prior call completes.

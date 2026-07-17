#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python3}"
RESULTS_DIR="${RESULTS_DIR:-${ROOT_DIR}/results/vllm_paper_style}"
MAX_PROGRAMS="${MAX_PROGRAMS:-128}"
MAX_CALLS_PER_PROGRAM="${MAX_CALLS_PER_PROGRAM:-8}"
MAX_TOKENS="${MAX_TOKENS:-64}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.75}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-4096}"
ARRIVAL_SEED="${ARRIVAL_SEED:-0}"
DRY_RUN="${DRY_RUN:-0}"

WORKLOADS=(${WORKLOADS:-sharegpt bfcl lats})
POLICIES=(${POLICIES:-fcfs mlfq plas atlas})
ARRIVAL_RATES=(${ARRIVAL_RATES:-0.2 0.5 1.0 1.5 2.0})

run_cmd() {
  echo "$*"
  if [[ "${DRY_RUN}" != "1" ]]; then
    "$@"
  fi
}

run_async_one() {
  local workload="$1"
  local policy="$2"
  shift 2
  run_cmd \
    "${PYTHON}" "${ROOT_DIR}/scripts/run_vllm_async_experiment.py" \
    --workload "${workload}" \
    --policy "${policy}" \
    --output-dir "${RESULTS_DIR}" \
    --max-programs "${MAX_PROGRAMS}" \
    --max-calls-per-program "${MAX_CALLS_PER_PROGRAM}" \
    --max-tokens "${MAX_TOKENS}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --max-num-seqs "${MAX_NUM_SEQS}" \
    --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
    "$@"
}

# vLLM baseline: FCFS without prefix caching.
for workload in "${WORKLOADS[@]}"; do
  run_async_one "${workload}" fcfs \
    --disable-prefix-caching \
    --arrival-rate 0 \
    --arrival-seed "${ARRIVAL_SEED}"
done

# vLLM-opt baseline: FCFS with prefix caching.
for workload in "${WORKLOADS[@]}"; do
  run_async_one "${workload}" fcfs \
    --arrival-rate 0 \
    --arrival-seed "${ARRIVAL_SEED}"
done

# Throughput-latency curves: Poisson arrivals over policy x lambda.
for workload in "${WORKLOADS[@]}"; do
  for arrival_rate in "${ARRIVAL_RATES[@]}"; do
    for policy in "${POLICIES[@]}"; do
      run_async_one "${workload}" "${policy}" \
        --arrival-rate "${arrival_rate}" \
        --arrival-seed "${ARRIVAL_SEED}"
    done
  done
done

run_cmd \
  "${PYTHON}" "${ROOT_DIR}/scripts/summarize_vllm_results.py" \
  "${RESULTS_DIR}" \
  --csv "${RESULTS_DIR}/summary.csv"

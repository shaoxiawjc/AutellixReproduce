#!/usr/bin/env bash
set -euo pipefail
cd /root/autellix/AutellixReproduce
export PYTHONPATH=/root/autellix/vllm
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
PYTHON=/root/autellix/.venv/bin/python
mkdir -p results/autellix_logs

RUN_SMOKE="${AUTELLIX_SMOKE:-0}"
RUN_BASELINE="${AUTELLIX_BASELINE:-0}"
RUN_LAMBDA="${AUTELLIX_LAMBDA:-1}"
RUN_PREFIX_ABLATION="${AUTELLIX_PREFIX_ABLATION:-0}"

log_step() { echo "[$(date -Is)] $*"; }

run_one() {
  local out_dir=$1 workload=$2 policy=$3
  shift 3
  mkdir -p "$out_dir"
  log_step START "$out_dir" "$workload" "$policy" "$@"
  "$PYTHON" scripts/run_vllm_async_experiment.py \
    --workload "$workload" \
    --policy "$policy" \
    --output-dir "$out_dir" \
    --disable-log-stats \
    "$@"
  log_step DONE "$out_dir" "$workload" "$policy"
}

# ── Phase 1: Smoke ────────────────────────────────────────────────────────
# Quick correctness check — all workloads × all policies, tiny scale.

if [ "${RUN_SMOKE}" = "1" ]; then
  log_step PHASE_SMOKE

  for workload in sharegpt bfcl lats; do
    for policy in fcfs mlfq plas atlas; do
      run_one results/autellix_smoke "$workload" "$policy" \
        --max-programs 8 \
        --max-calls-per-program 2 \
        --max-tokens 16 \
        --max-model-len 2048 \
        --max-num-seqs 16 \
        --max-num-batched-tokens 2048 \
        --gpu-memory-utilization 0.65
    done
  done
fi

# ── Phase 2: Baseline at moderate load ─────────────────────────────────────
# Single arrival rate, enough programs to create queue pressure.
# Includes vLLM-opt (fcfs with prefix caching) and pure vLLM (fcfs without).
# NOTE: lats is excluded — load_lats() generates synthetic DAGs whose token
# distributions differ from the paper's HotpotQA MCTS traces. The synthetic
# DAG is still run in Phase 1 (smoke) for ATLAS correctness validation.
# ShareGPT and BFCL are both sequential (single-threaded chains), so ATLAS
# degenerates to PLAS on them.  A real DAG workload is needed for paper-style
# ATLAS evaluation.

if [ "${RUN_BASELINE}" = "1" ]; then
  log_step PHASE_BASELINE

  BASELINE_LAMBDA=0.5
  BASELINE_WORKLOADS="sharegpt bfcl"

  for workload in ${BASELINE_WORKLOADS}; do
    # vLLM (FCFS, no prefix caching)
    run_one results/autellix_baseline "$workload" fcfs \
      --disable-prefix-caching \
      --arrival-rate "${BASELINE_LAMBDA}" \
      --max-programs 128 \
      --max-calls-per-program 8 \
      --max-tokens 512 \
      --max-model-len 16384 \
      --max-num-seqs 8 \
      --max-num-batched-tokens 16384 \
      --gpu-memory-utilization 0.9

    # vLLM-opt (FCFS + prefix caching), MLFQ, PLAS, ATLAS
    for policy in fcfs mlfq plas atlas; do
      run_one results/autellix_baseline "$workload" "$policy" \
        --arrival-rate "${BASELINE_LAMBDA}" \
        --max-programs 128 \
        --max-calls-per-program 8 \
        --max-tokens 512 \
        --max-model-len 16384 \
        --max-num-seqs 8 \
        --max-num-batched-tokens 16384 \
        --gpu-memory-utilization 0.9
    done
  done
fi

# ── Phase 3: Lambda sweep ──────────────────────────────────────────────────
# Paper-style throughput-latency curves.
# Sweeps arrival rates across workloads × policies.
# NOTE: lats excluded — synthetic DAG token distributions differ from the
# paper's HotpotQA MCTS traces. Use real LATS traces when available.

if [ "${RUN_LAMBDA}" = "1" ]; then
  log_step PHASE_LAMBDA_SWEEP

  SWEEP_RATES="1.0 4.0 8.0"
  SWEEP_POLICIES="fcfs mlfq plas atlas"
  SWEEP_WORKLOADS="sharegpt bfcl"

  for workload in ${SWEEP_WORKLOADS}; do
    # vLLM baseline (FCFS, no prefix caching)
    for rate in ${SWEEP_RATES}; do
      run_one results/autellix_lambda_sweep "$workload" fcfs \
        --disable-prefix-caching \
        --arrival-rate "$rate" \
        --max-programs 128 \
        --max-calls-per-program 12 \
        --max-tokens 512 \
        --max-model-len 16384 \
        --max-num-seqs 8 \
        --max-num-batched-tokens 16384 \
        --gpu-memory-utilization 0.9
    done

    # vLLM-opt / MLFQ / PLAS / ATLAS (all with prefix caching)
    for policy in ${SWEEP_POLICIES}; do
      for rate in ${SWEEP_RATES}; do
        run_one results/autellix_lambda_sweep "$workload" "$policy" \
          --arrival-rate "$rate" \
          --max-programs 128 \
          --max-calls-per-program 12 \
          --max-tokens 512 \
          --max-model-len 16384 \
          --max-num-seqs 8 \
          --max-num-batched-tokens 16384 \
          --gpu-memory-utilization 0.9
      done
    done
  done
fi

# ── Phase 4: Prefix-caching ablation ───────────────────────────────────────
# Opt-in (set AUTELLIX_PREFIX_ABLATION=1). Measures the contribution of
# prefix caching alone by running fcfs with and without it at several rates.

if [ "${RUN_PREFIX_ABLATION}" = "1" ]; then
  log_step PHASE_PREFIX_ABLATION

  ABLATION_RATES="0.2 1.0 2.0"

  for workload in sharegpt bfcl; do
    for rate in ${ABLATION_RATES}; do
      run_one results/autellix_prefix_ablation "$workload" fcfs \
        --disable-prefix-caching \
        --arrival-rate "$rate" \
        --max-programs 128 \
        --max-calls-per-program 8 \
        --max-tokens 512 \
        --max-model-len 16384 \
        --max-num-seqs 8 \
        --max-num-batched-tokens 16384 \
        --gpu-memory-utilization 0.9

      run_one results/autellix_prefix_ablation "$workload" fcfs \
        --arrival-rate "$rate" \
        --max-programs 128 \
        --max-calls-per-program 8 \
        --max-tokens 512 \
        --max-model-len 16384 \
        --max-num-seqs 8 \
        --max-num-batched-tokens 16384 \
        --gpu-memory-utilization 0.9
    done
  done
fi

log_step PIPELINE_DONE

# ── Summarise ──────────────────────────────────────────────────────────────
if command -v "$PYTHON" &>/dev/null; then
  for results_dir in \
    results/autellix_smoke \
    results/autellix_baseline \
    results/autellix_lambda_sweep \
    results/autellix_prefix_ablation; do
    if [ -d "$results_dir" ]; then
      echo "=== $results_dir ==="
      "$PYTHON" scripts/summarize_vllm_results.py "$results_dir" || true
    fi
  done
fi

# Experiment Scripts vs Paper §6 — Gaps

Compares `AutellixReproduce/scripts/` against the evaluation methodology
described in *Autellix: An Efficient Serving Engine for LLM Agents as General
Programs* (§6).

---

## 🔴 Critical Gaps

### 1. Missing Poisson arrival process and arrival-rate sweep

**Paper (§6.1):**
> "We synthesize a trace by randomly sampling programs, not LLM calls, from
> the above workloads and generating programs' arrivals using a Poisson
> process λ, following established methodologies [36, 77]."

**Current code** (`run_vllm_async_experiment.py:156`):
```python
await asyncio.gather(*(run_one_program(program) for program in programs))
```

All programs are launched simultaneously (`t≈0`), equivalent to infinite
arrival rate. The paper sweeps multiple λ values to draw latency-vs-arrival-rate
curves (Figure 12) and compares throughput at the same latency across policies.
Without controlled arrivals, the throughput-latency trade-off cannot be
measured.

**Fix:** Pre-compute per-program arrival times with `random.expovariate(λ)`,
stagger `run_one_program` launches with `asyncio.sleep(arrival_time)`, and
sweep λ over a range (e.g. 0.2, 0.5, 1.0, 1.5, 2.0, … program/s).

---

### 2. Program latency is not normalized by token count

**Paper (§6.2):**
> "program-level token latency, defined as the total program response time
> divided by the number of tokens generated."
>
> "For multi-threaded programs, program-level token latency is computed as
> the critical path response time divided by the total tokens across all
> threads."

**Current code** (`run_vllm_async_experiment.py:244–248`):
```python
program_latencies = [
    (program.finished_at or 0.0) - (program.started_at or 0.0)
    for program in programs
    if program.started_at is not None and program.finished_at is not None
]
```

Raw wall-clock seconds are reported as "latency." Programs that generate more
tokens will naturally have higher raw latency; without normalizing by token
count the metric penalises longer programs and is not comparable across
workloads with different token distributions.

**Fix:**
```python
program_token_latencies = [
    (program.finished_at - program.started_at)
    / max(1, program.total_output_tokens)
    for program in programs
    if program.started_at is not None
    and program.finished_at is not None
    and program.total_output_tokens > 0
]
```

The same normalization applies to P95/P99 values. The `Program` dataclass
needs a `total_output_tokens` accumulator (summed per call in the async loop).

---

### 3. No LATS / MCTS workload — ATLAS is untestable

**Paper (§6.1):**
> "Monte Carlo Tree Search: LATS [91] … each program instance contains on
> average 159.7 LLM calls … prefill and decoding phase of each call averages
> 467.2 and 72.6 tokens respectively."

ATLAS is the scheduler policy designed specifically for multi-threaded DAG
programs. It computes priorities from critical-path service time and tracks
per-thread state in `AutellixProcessRecord.threads`. Without a DAG workload:

- `parent_call_ids` is always a single-element tuple (sequential chain).
- `AutellixProcessTable.completed_parent_path_service` reduces to the same
  value as `critical_path_service`.
- ATLAS behaves identically to PLAS in single-threaded traces.

**Fix:** Either:
1. Generate a synthetic MCTS trace with branching structure and export it as
   JSONL for the simulator and vLLM driver, or
2. Adapt an existing LATS trace (the paper ran MCTS on HotpotQA).

---

### 4. No vLLM-opt baseline

**Paper (§6.2):**
> - **vLLM** — FCFS, no prefix caching.
> - **vLLM-opt** — chunk-prefill, prefix-caching, multi-step scheduling.
> - **MLFQ** — vLLM-opt + multi-level feedback queue preemption.
> - **Autellix** — PLAS (single-threaded) or ATLAS (multi-threaded).

All four baselines are needed to attribute gains to each component (prefix
caching vs. preemption vs. program-level scheduling).

**Current code:** `enable_prefix_caching=True` is set for **all** policies,
including FCFS. There is no way to run a pure FCFS-without-caching baseline.
Consequently:
- The gain from prefix caching alone cannot be isolated.
- MLFQ is not an independent layer — it runs through the same autellix
  scheduler code path (with `initial_priority=0`), which embeds
  `MultiLevelPriorityRequestQueue` semantics that differ from pure MLFQ.

**Fix:** Add a `--disable-prefix-caching` flag and run the `fcfs` policy with
caching off to produce the vLLM baseline. Run `fcfs` with caching on to
produce vLLM-opt. The paper's MLFQ is preemption-only (always starts at Q1);
the current implementation's `mlfq` policy is close because `initial_priority`
returns 0 → `queue_for_priority(0)` → Q0, matching the paper's "new calls
start at Q1."

---

## 🟡 Secondary Gaps

### 5. Tail latency not reported as curves

**Paper (§6.3, Figure 13):** P95 and P99 program-level token latency plotted
against arrival rate for each workload.

**Current code:** P95/P99 are computed as single scalars in
`summarize_vllm_results.py`. They are not broken down by workload×policy×λ.

**Fix:** After fixing items 1 and 2, collect P95/P99 per (workload, policy, λ)
and produce the equivalent of Figure 13.

---

### 6. No multi-engine experiments

**Paper (§6.4):** 4×8B replicas and 2×70B replicas with ShareGPT and LATS,
comparing Round Robin, Least Used, and Autellix load-balancer.

**Current code:** Single-engine only. The load balancer (§4.3, Algorithm 2) is
not implemented in the Python driver.

**Fix:** Requires `AsyncMultiLLMEngine` (the paper built this atop
`AsyncLLMEngine` in vLLM v0.6.1) or a multi-process wrapper. Defer until
single-engine results are solid.

---

### 7. No offline batch inference experiment

**Paper (§6.5.1, Figure 16):** All programs submitted at t=0; makespan
measured across 1k–4k programs.

**Current code:** The async driver runs all programs concurrently but programs
arrive at t=0 (no controlled staggering). The makespan is reported but not
compared across policies at varying scales.

**Fix:** Run the async driver with `--max-programs` sweeping 1000/2000/4000
and record makespan per policy. This is relatively straightforward once
larger workloads are available.

---

### 8. No timing breakdown

**Paper (§6.5.2, Figure 17):** Execution / Scheduler / Swap / Wait time
breakdown per policy.

**Current code:** `CallRecord.vllm_metrics` captures `queued_ts` and
`scheduled_ts`, which allows computing queue wait time. But there is no swap
time or scheduler overhead tracking in the result summary.

**Fix:** Extract `queued_ts`, `scheduled_ts`, `first_token_ts`, `last_token_ts`
from vLLM metrics and categorise them into wait, scheduler, execution, and
swap buckets. This is diagnostics-only and can be deferred.

---

### 9. No comparison to optimal (SRPT)

**Paper (§6.5.3, Figure 18):** Simulation comparing FCFS, Round-Robin, MLFQ,
Autellix, and SRPT (clairvoyant optimum).

**Current code:** The `autellix_reproduce` CPU simulator (`simulator.py`)
already supports this. It is not linked to the vLLM experiment pipeline.

**Fix:** Run `uv run autellix-sim --policy srpt` and add an SRPT policy to the
simulator's `POLICIES` dict. This is simulation-only and independent of the
vLLM scripts.

---

## ✅ Correct

| Item | Notes |
|------|-------|
| ShareGPT multi-turn handling | Each assistant turn = one dependent LLM call; `parent_call_ids` chained |
| BFCL tool-use handling | Function signatures in system prompt; multi-step calls with dependencies |
| `autellix` metadata passthrough | `sampling_params.extra_args["autellix"]` correctly populated with policy, program_id, call_id, thread_id, parent_call_ids |
| Runtime history accumulation | `history_mode=model` uses Qwen's actual output as assistant context; `history_mode=user_only` accumulates user/tool prompts only |
| FCFS/MLFQ/PLAS/ATLAS policy switch | `vllm_scheduling_policy()` maps to vLLM's `SchedulerPolicy` enum |
| Per-call timing | `submit_time`, `finish_time`, and vLLM internal metrics recorded |
| Async driver architecture | One coroutine per program; next call submitted immediately after prior call completes — matches Autellix's execution model |
| Qwen3 thinking disabled | `enable_thinking=False` in `apply_chat_template` |

---

## Priority

| # | Item | Impact |
|---|------|--------|
| 2 | Latency normalisation (s/tok) | **Blocker.** Current metric is not comparable to the paper. |
| 1 | Poisson arrival + λ sweep | **Blocker.** Cannot draw throughput-latency curves without it. |
| 3 | LATS workload | **Blocker** for ATLAS validation. |
| 4 | vLLM-opt baseline | High. Needed to isolate prefix-caching gains from scheduling gains. |
| 5 | Tail latency curves | Medium. Follows from items 1+2. |
| 7 | Offline batch | Medium. Easy to add once workloads scale. |
| 6 | Multi-engine | Defer. Requires multi-engine infrastructure. |
| 8 | Timing breakdown | Defer. Diagnostics only. |
| 9 | SRPT comparison | Defer. Simulation only, already supported in CPU simulator. |

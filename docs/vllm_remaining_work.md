# vLLM Autellix — Remaining Work

Status as of 2026-07-16.

## What Is Done

| Component | How |
|-----------|-----|
| Multi-level FIFO queue (`MultiLevelPriorityRequestQueue`) | Wired into `create_request_queue` — `self.waiting` and `self.skipped_waiting` are multi-level deques |
| Queue-level assignment from ProcessTable | `AutellixQueueConfig.queue_for_priority()` used in both `assign_initial_queue` and `_level_for` |
| Queue-index ordering for queue selection | `_select_waiting_queue_for_scheduling` uses `_autellix_queue_order_key` |
| Proactive preemption (waiting > running) | `_maybe_preempt_for_autellix_waiting` swaps out lowest running when higher-priority request is waiting |
| Preempt re-prioritisation | `_preempt_request` calls `_prepare_autellix_request` before re-enqueuing |
| Anti-starvation (program-level) | `_starved` uses `record.waiting_time / (record.service_time + per-request service)` vs β |
| PLAS/ATLAS/MLFQ initial priority logic | `initial_priority()` in `autellix.py` |
| Critical-path tracking for ATLAS | `completed_path_service` updated in `on_call_finished` |
| Request metadata (`program_id`, `call_id`, `thread_id`, `parent_call_ids`) | `request.py` reads from `sampling_params.extra_args["autellix"]` |

---

## Remaining Work

### 🔴 1. Quantum decrement is never called — demotion is dead

**Files:** `autellix.py:135–145`, `scheduler.py:401–410`

`_update_autellix_priorities` no longer calls `on_scheduled`. The
`num_scheduled_tokens` parameter is received but unused.

```python
# scheduler.py:401-410 — current
def _update_autellix_priorities(self, num_scheduled_tokens):
    ...
    self.autellix_process_table.add_waiting(self.waiting)
    self.autellix_process_table.add_waiting(self.skipped_waiting)
    # ❌ on_scheduled is never called — quantum_left never decreases
    if self.autellix_process_table.on_step_end(self.running):
        self.running.sort(key=self._autellix_queue_order_key)
```

**What this breaks:** `request.autellix_quantum_left` stays at its initial value
(64, 128, 256, …) forever. In `on_step_end`, the check `quantum_left <= 0` is
always false — requests are never demoted to a lower-priority queue. The only
queue movement that still works is anti-starvation promotion to Q1.

**Net effect:** the scheduler behaves like MLFQ with correct initial queue
placement but infinite quanta per queue.

**Fix:** Call `on_scheduled` with the count of **actual decode tokens produced**
(not `num_scheduled_tokens` which is the token budget allocated in one
scheduling step). The right place is `update_from_output`, where
`new_token_ids` are available per request:

```python
# In update_from_output, for each running request:
actual_decode_tokens = len(new_token_ids_for_this_request)
if actual_decode_tokens > 0:
    self.autellix_process_table.on_scheduled(request, actual_decode_tokens)
```

Then call `on_step_end` separately in `schedule()` (already done) to apply
demotion/promotion based on updated quantum values.

---

### 🔴 2. Proactive preemption does not reserve blocks for the waiting request

**File:** `scheduler.py:378–399, 520–521, 529–649, 729–738`

The current scheduling order in `schedule()`:

```
Step 0: _maybe_preempt_for_autellix_waiting()   → victim freed, blocks released
Step 1: Schedule RUNNING requests                → may consume the freed blocks
Step 2: Schedule WAITING requests                → high-priority waiting may still get nothing
```

`_maybe_preempt_for_autellix_waiting` preempts the lowest-priority running
request when a higher-priority request is waiting. But the freed KV-cache
blocks can be consumed in Step 1 by other running requests before the
high-priority waiting request is reached in Step 2.

Paper Algorithm 1 allocates blocks in strict queue-index order — a Q1 waiting
request gets blocks before a Q3 running request.

**Fix:** Either:

1. **(Simpler)** After preempting in `_maybe_preempt_for_autellix_waiting`,
   immediately pop the waiting request and attempt to allocate blocks for it.
   If allocation succeeds, move it to `self.running`. If not, preempt another
   running request and retry.

2. **(More thorough)** Reverse the order: schedule WAITING requests before
   RUNNING requests when an autellix policy is active. High-priority waiting
   requests get first access to the block pool.

---

### 🟡 3. Wait-time accumulation only counts one step per program per scheduling step

**File:** `autellix.py:82–92`

```python
def add_waiting(self, requests, steps=1):
    seen_programs: set[str] = set()
    for request in requests:
        ...
        if program_id in seen_programs:
            continue        # ← one step counted regardless of how many calls are queued
        seen_programs.add(program_id)
```

This is consistent with `_starved` (which also uses program-level waiting
time), but it means a program with 10 concurrent calls queued accumulates
waiting time at the same rate as a program with 1 call. The paper is ambiguous
on this point (it refers to "total waiting time W_total = W_p + W_c").

**Decision needed:** Is this the intended semantics (program-level wait, not
call-level), or should every queued call contribute to the program's waiting
total? Run experiments both ways and pick the one matching paper results.

---

### 🟢 4. `self.running` is a flat list sorted after every step

**File:** `scheduler.py:186, 409–410`

```python
self.running: list[Request] = []
# …
if self.autellix_process_table.on_step_end(self.running):
    self.running.sort(key=self._autellix_queue_order_key)
```

Functionally correct at current scale. If profiling shows scheduler overhead
from repeated sorts under high request counts, convert `self.running` to a
per-level `list[deque[Request]]` to make queue-order iteration O(1).

---

### 🟢 5. Multi-step scheduling

Paper §4.2.2: "running the scheduler once every N decoding steps rather than
at every step." Reduces swap frequency and scheduling overhead. Requires
over-provisioning (schedule more requests than needed, anticipating early
completions).

**Prerequisite:** item 1 (quantum decrement) must be fixed first so demotion
works correctly over multi-step intervals.

---

### 🟢 6. Bulk GPU-CPU swap kernel

Paper §5, §6.5.4: gather all KV blocks into a contiguous host buffer, transfer
in one `cudaMemcpy`. Reports up to 18× fewer swap operations and 3–7× less
swap time.

This is a CUDA/C++ change orthogonal to the scheduling logic. Only worth
pursuing after the Python-level scheduler is correct and swap-time shows up as
a bottleneck in profiling.

---

## Priority

| # | Item | Impact |
|---|------|--------|
| 1 | Quantum decrement → demotion | **Blocker.** Without it, the scheduler is not multi-level preemptive — it's MLFQ with better initial placement. |
| 2 | Block reservation for preempted-in requests | **Blocker.** Proactive preemption is implemented but can be ineffective because freed blocks are not reserved. |
| 3 | Wait-time semantics | Minor correctness question; verify with experiments. |
| 4 | Running queue structure | Code-quality improvement; defer until profiling shows need. |
| 5 | Multi-step scheduling | Performance optimisation; defer. |
| 6 | Bulk swap kernel | CUDA optimisation; defer. |

Items 1 and 2 are the two changes standing between the current code and a
faithful implementation of Algorithm 1.

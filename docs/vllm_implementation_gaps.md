# vLLM Multi-Level Preemptive Scheduler — Implementation Gaps

This document catalogs the remaining work to turn the current vLLM Autellix
patch into a faithful implementation of the multi-level preemptive scheduler
described in the paper *Autellix: An Efficient Serving Engine for LLM Agents as
General Programs* (§4.2 and Algorithm 1).

## Current State

| Component | Status |
|-----------|--------|
| `AutellixQueueConfig` (bounds, quanta, anti-starvation β) | ✅ `autellix.py` |
| `AutellixProcessTable` (per-program service/wait/thread/path tracking) | ✅ `autellix.py` |
| `AutellixProcessRecord` / `AutellixThreadRecord` data structures | ✅ `autellix.py` |
| `MultiLevelPriorityRequestQueue` class | ✅ `request_queue.py` — **but not wired in** |
| `Request` metadata fields (program_id, call_id, queue_index, quantum, …) | ✅ `request.py` |
| Scheduler hooks (`_prepare_autellix_request`, `_update_autellix_priorities`) | ✅ `scheduler.py` |
| Priority-based preemption fallback (on KV-cache allocation failure) | ✅ `scheduler.py` |
| End-to-end experiment scripts (wave + async driver) | ✅ `scripts/` |

**Key observation:** The current integration uses vLLM’s `PriorityRequestQueue`
(a heap) for both `self.waiting` and `self.skipped_waiting`, and the scheduler
only preempts when KV-cache allocation fails. The autellix hooks compute
program-level priorities and re-sort `self.running`, but this is not equivalent
to the paper's multi-level preemptive scheduler with discrete FIFO queues,
quantum-based demotion, and proactive preemption.

---

## Gap 1 — `MultiLevelPriorityRequestQueue` is not wired in 🔴 P0

**File:** `vllm/v1/core/sched/request_queue.py:283–297`

```python
# Current — returns a heap-based PriorityRequestQueue for autellix policies:
elif policy in (
    SchedulingPolicy.AUTELLIX, SchedulingPolicy.MLFQ,
    SchedulingPolicy.PLAS,    SchedulingPolicy.ATLAS,
):
    return PriorityRequestQueue()   # ❌

# Required — return the multi-level FIFO queue:
elif policy in (
    SchedulingPolicy.AUTELLIX, SchedulingPolicy.MLFQ,
    SchedulingPolicy.PLAS,    SchedulingPolicy.ATLAS,
):
    return MultiLevelPriorityRequestQueue()  # ✅
```

### Sub-issue: `_level_for` disagrees with `AutellixQueueConfig.queue_for_priority`

`MultiLevelPriorityRequestQueue._level_for` currently computes the level as
`priority // quantum_tokens`, while `AutellixQueueConfig.queue_for_priority`
uses a **bounds-based** bucketing (e.g. `(0, 64, 128, 256, 512, 1024)`).

These two schemes produce different level assignments for the same priority
value. They must be unified.

**Fix:** Inject `AutellixQueueConfig` into `MultiLevelPriorityRequestQueue`
and replace `_level_for` with `config.queue_for_priority(request.priority)`.

```python
class MultiLevelPriorityRequestQueue(RequestQueue):
    def __init__(self, queue_config: AutellixQueueConfig | None = None) -> None:
        self.config = queue_config or AutellixQueueConfig()
        self._queues: list[deque[Request]] = [
            deque() for _ in range(self.config.num_queues)
        ]

    def _level_for(self, request: Request) -> int:
        return self.config.queue_for_priority(
            max(0, int(getattr(request, "priority", 0)))
        )
```

---

## Gap 2 — `_select_waiting_queue_for_scheduling` does not use multi-level ordering 🔴 P0

**File:** `vllm/v1/core/sched/scheduler.py:1946–1956`

The current method compares heads of `self.waiting` and `self.skipped_waiting`
using `<` on `Request` objects. When `self.waiting` becomes a
`MultiLevelPriorityRequestQueue`, `peek_request()` returns the head of the
lowest-index non-empty queue — which is correct. However, the comparison
between the two queue heads must use **queue index first, then FIFO order**,
not the raw `priority` value.

**Fix:** Either:

1. **(Preferred)** Merge `waiting` and `skipped_waiting` into a **single**
   `MultiLevelPriorityRequestQueue`. When a skipped request becomes unblocked,
   re-enqueue it at its correct level. This eliminates the two-queue head
   comparison entirely.

   The `skipped_waiting` queue exists for requests that are temporarily blocked
   (waiting for remote KV, structured output grammar, or streaming input). With
   a single multi-level queue, blocked requests can simply be held aside in a
   separate structure (a set or a small list) rather than a second queue that
   participates in scheduling decisions.

2. **(Simpler interim fix)** Modify `_select_waiting_queue_for_scheduling` for
   autellix policies to pick the queue whose head has the lower queue index,
   falling back to FIFO order within the same index.

```python
def _select_waiting_queue_for_scheduling(self) -> RequestQueue | None:
    if self.policy == SchedulingPolicy.FCFS:
        return self.skipped_waiting or self.waiting or None

    if self._is_autellix_policy():
        # Multi-level: pick queue with lower-index head
        w = self.waiting.peek_request() if self.waiting else None
        s = self.skipped_waiting.peek_request() if self.skipped_waiting else None
        if w and s:
            w_idx = getattr(w, "autellix_queue_index", 0)
            s_idx = getattr(s, "autellix_queue_index", 0)
            return self.waiting if w_idx <= s_idx else self.skipped_waiting
        return self.waiting or self.skipped_waiting or None

    # Original PRIORITY path...
```

---

## Gap 3 — No proactive preemption based on queue priority 🔴 P0

**File:** `vllm/v1/core/sched/scheduler.py:591–635`

**This is the most critical missing piece.** The current scheduler only
preempts running requests when KV-cache allocation for a new request fails
(reactive preemption). The paper's Algorithm 1 requires **proactive preemption**:
at each scheduling step, if a waiting request has a higher effective priority
(lower queue index) than the lowest-priority running request, the scheduler
should swap out the running request to make room.

### What the paper says (Algorithm 1, lines 31–38)

```
B_out = []
for c ∈ {Q1, Q2, ..., QK} do       // iterate queues high→low priority
    if engine.can_fit(c) then
        B_out.append(c)
    else
        break                         // stop — lower queues get nothing
    end if
end for
```

The scheduler drains the highest-priority queue first, then the next, and so
on. If a request from a lower-priority queue is already running but a
higher-priority request is waiting, the running request should be preempted
(its KV cache swapped to CPU) so the higher-priority request can run.

### What needs to change

1. **Before the scheduling loop**, compute the ordered set of requests to run
   by merging `self.waiting` pops with `self.running` survivors — always
   favoring higher-priority queues.

2. **When a running request should be displaced**, call `_preempt_request` on
   the lowest-priority running request(s) until the higher-priority waiting
   request can be accommodated.

3. **At step end**, re-enqueue displaced requests via `_prepare_autellix_request`
   so they resume at their correct (possibly demoted/promoted) level.

### Pseudo-code for the modified scheduling loop

```python
# --- At the start of each scheduling step ---

# 1. Gather all runnable requests in priority order.
candidates: list[Request] = []
# Running requests keep their slots unless displaced.
for req in self.running:
    candidates.append(req)  # Already sorted by queue_index (see Gap 4)

# Newly ready requests from waiting queues.
while self.waiting:
    req = self.waiting.peek_request()
    candidates.append(req)  # Will be inserted at correct position below

# 2. Sort candidates by (queue_index, arrival_time).
candidates.sort(key=lambda r: (
    getattr(r, "autellix_queue_index", 0),
    r.arrival_time,
    r.request_id,
))

# 3. Allocate slots in priority order; preempt when allocation fails.
scheduled = []
for req in candidates:
    new_blocks = self.kv_cache_manager.allocate_slots(req, ...)
    if new_blocks is not None:
        scheduled.append(req)
        # if req was in waiting, pop it; if already running, keep it
    else:
        # Cannot fit — preempt the lowest-priority scheduled request
        victim = max(scheduled, key=lambda r: (
            getattr(r, "autellix_queue_index", 0), r.arrival_time
        ))
        if victim_is_lower_priority_than(req):
            scheduled.remove(victim)
            self._preempt_request(victim, timestamp)
            # Retry allocation for req...
        else:
            # req is lower priority than all scheduled; skip it
            break
```

### Dependency: KV-cache swap must work

Preempting a running request requires swapping its KV cache blocks from GPU to
CPU memory (so another request can use the GPU memory). vLLM already has this
mechanism — `_preempt_request` calls `_free_request_blocks`. The swap happens
implicitly via vLLM's block manager. With `MultiLevelPriorityRequestQueue`
preemption, the swap frequency will increase; the bulk swap kernel (Gap 10)
becomes relevant at scale.

---

## Gap 4 — `self.running` is a flat list, not a multi-level structure 🟡 P2

**File:** `vllm/v1/core/sched/scheduler.py:186`

```python
self.running: list[Request] = []
```

The current code re-sorts `self.running` by `autellix_queue_index` after each
`on_step_end`:

```python
# scheduler.py:374-381
if self.autellix_process_table.on_step_end(self.running):
    self.running.sort(key=lambda req: (
        int(getattr(req, "autellix_queue_index", 0)),
        req.arrival_time,
        req.request_id,
    ))
```

This is functionally adequate *if* proactive preemption (Gap 3) also sorts the
merged candidate list. However, a multi-level structure inside `self.running`
would make it easier to:

- Find the lowest-priority running request for preemption (O(1) with per-level
  deques).
- Maintain FIFO order within each level under demotion/promotion.
- Unify the data structure with `self.waiting`.

**Recommendation:** Defer until Gap 3 is implemented. If the sort-based
approach proves sufficient in benchmarks, this gap can remain a code-quality
note. If the scheduler loop shows measurable overhead from repeated sorting,
convert `self.running` to also use `MultiLevelPriorityRequestQueue` (or a
lighter per-level `list[deque[Request]]`).

---

## Gap 5 — Quantum is decremented by `num_scheduled_tokens`, not actual output tokens 🟡 P1

**Files:** `autellix.py:138–148`, `scheduler.py:370–372`

```python
# Current — uses token budget allocated in one scheduling step:
def on_scheduled(self, request, scheduled_tokens):
    request.autellix_quantum_left -= scheduled_tokens

# Called from scheduler:
for request in self.running:
    self.autellix_process_table.on_scheduled(
        request, num_scheduled_tokens.get(request.request_id, 0)
    )
```

`num_scheduled_tokens` is the number of tokens *allocated* to the request in
the current scheduling step. For a decoding request in continuous batching,
this is typically 1. For prefill, it can be many tokens.

The paper's quantum represents **decode tokens generated** — the request is
allowed to produce up to `quantum` output tokens in its current queue before
being considered for demotion. Prefill tokens are not counted against the
quantum.

### Fix

Track quantum consumption in `update_from_output`, where the actual number of
generated token IDs is known:

```python
# In Scheduler.update_from_output or similar post-execution hook:
for request in self.running:
    actual_output_tokens = len(output_token_ids_for_request)
    if actual_output_tokens > 0:
        self.autellix_process_table.on_scheduled(request, actual_output_tokens)
```

And remove the quantum decrement from the pre-execution `_update_autellix_priorities` path.

Alternatively, keep the pre-step hook for prefill accounting but use a separate
field (`autellix_decode_steps`) for the quantum, decrementing only by actual
decode steps observed post-execution.

---

## Gap 6 — Preempted requests are re-enqueued without priority recalculation 🟡 P1

**File:** `vllm/v1/core/sched/scheduler.py:1212–1234`

```python
def _preempt_request(self, request, timestamp):
    ...
    self.waiting.prepend_request(request)   # ← priority/queue_index not updated
```

When a request is preempted and put back into `self.waiting`, its
`autellix_queue_index` and `priority` are stale — they reflect the values from
when it was last scheduled. Meanwhile, its program's `service_time` in the
process table may have increased (other calls from the same program completed).

### Fix

Call `_prepare_autellix_request` at the end of `_preempt_request` (before or
instead of `prepend_request`) so the request's queue assignment is recomputed
from the current process table state:

```python
def _preempt_request(self, request, timestamp):
    ...
    # Recompute priority from current process-table state.
    self._prepare_autellix_request(request, timestamp)
    self.waiting.add_request(request)       # add_request respects queue_index
    # or: self.waiting.prepend_request(request) after prepare sets queue_index
```

Also ensure that `MultiLevelPriorityRequestQueue.prepend_request` respects the
request's current `autellix_queue_index` rather than recomputing it from
`priority` (they should be consistent after `_prepare_autellix_request`).

---

## Gap 7 — `on_scheduled` and `on_step_end` are called together, blurring their semantics 🟢 P3

**File:** `vllm/v1/core/sched/scheduler.py:362–381`

The current `_update_autellix_priorities` calls both `on_scheduled` (quantum
decrement) and `on_step_end` (demotion/promotion check) in one pass over
`self.running`. Conceptually these are separate phases:

| Phase | When | What |
|-------|------|------|
| `on_scheduled` | After model execution, when actual output tokens are known | Decrement quantum by actual decode tokens |
| `on_step_end` | End of scheduling step, after all quantum updates | Check demotion (quantum ≤ 0) and anti-starvation promotion |

**Fix (low priority, correctness-neutral):** Split into two calls — call
`on_scheduled` from `update_from_output` (post-execution, with actual output
token counts) and `on_step_end` from `schedule` (after all requests for this
step have been processed). This is primarily a code-clarity improvement and
becomes more important when Gap 5 (correct token tracking) is addressed.

---

## Gap 8 — Anti-starvation waiting-time accumulation has per-program dedup 🟡 P1

**File:** `vllm/v1/core/sched/autellix.py:82–95`

```python
def add_waiting(self, requests, steps=1):
    seen_programs: set[str] = set()
    for request in requests:
        ...
        if program_id in seen_programs:
            continue              # ← only counts 1 step per program per call
        ...
        seen_programs.add(program_id)
```

This means that in a single scheduling step, each program's `waiting_time` is
incremented by at most 1 step, regardless of how many of its calls are queued.
The paper's anti-starvation check (Algorithm 1, lines 24–30) uses total program
waiting time divided by total program service time:

```
if wait / service ≥ β then  → promote to Q1
```

### Analysis

The per-program dedup is arguably correct for **program-level** anti-starvation
(the paper states: "for a program p, Autellix promotes call c to Q1 if the
ratio of total waiting time to service time exceeds a threshold β"). If
multiple calls from the same program are queued, the program experiences one
step of waiting per scheduling step — the dedup prevents double-counting.

**However**, the `_starved` check in `autellix.py:210–218` also uses per-call
waiting (`request.autellix_scheduler_wait_steps`), which is accumulated
per-request without dedup. This is inconsistent with the per-program dedup in
`add_waiting`.

### Fix

Align the two paths. Either:

1. **Per-program only (paper semantics):** Remove per-call
   `autellix_scheduler_wait_steps` accumulation; use only
   `record.waiting_time` in `_starved`.

2. **Per-call only:** Remove the `seen_programs` dedup in `add_waiting`; let
   each queued call accumulate its own wait steps. This is more aggressive
   about anti-starvation (more calls waiting → faster promotion).

Recommendation: go with option 1 (per-program) as it matches the paper's
description most closely: "for a program p, Autellix promotes call c to Q1 if
the ratio of total waiting time (W_total = W_p + W_c) to service time (T_total
= T_p + T_c) exceeds a threshold β."

---

## Gap 9 — Missing multi-step scheduling 🟢 P3 (optimization)

**Paper reference:** §4.2.2, "Memory Management"

> Autellix reduces total swaps by adopting multi-step scheduling, running the
> scheduler once every N decoding steps rather than at every step.

The current implementation calls `schedule()` at every step. Multi-step
scheduling amortizes the scheduling overhead and reduces swap frequency by
batching decode steps. It also requires over-provisioning (scheduling more
requests than strictly needed, anticipating early completions).

**Implementation sketch:**

```python
# In SchedulerConfig or AutellixQueueConfig:
autellix_schedule_interval: int = 1  # 1 = every step; N > 1 = multi-step

# In Scheduler.schedule():
if self.current_step % self.autellix_schedule_interval != 0:
    # Reuse last batch; only run update_from_output
    return self._last_scheduler_output
```

**Prerequisite:** Gap 3 (proactive preemption) must be working first, since
multi-step scheduling delays preemption decisions. Without proactive
preemption, multi-step scheduling offers little benefit.

---

## Gap 10 — Missing bulk GPU-CPU swap kernel 🟢 P3 (optimization)

**Paper reference:** §4.2.2, §5, §6.5.4

The paper describes an optimized swap kernel that:

1. Gathers all KV blocks to be swapped into a contiguous host buffer.
2. Transfers them in a single `cudaMemcpy` operation rather than one per block.
3. Reports up to 18× reduction in swap count and 3–7× reduction in swap time.

This is a CUDA/C++ change, not Python. It is fully orthogonal to the
scheduling logic and can be implemented independently once proactive preemption
(Gap 3) creates sufficient swap pressure to justify it.

**Location:** vLLM's block manager / KV cache manager (`kv_cache_manager.py` or
the C++/CUDA kernel files).

---

## Priority Summary

| Priority | Gaps | Why |
|----------|------|-----|
| **P0 — Blocking** | 1, 2, 3 | Without multi-level queues (Gap 1), correct queue selection (Gap 2), and proactive preemption (Gap 3), the scheduler is not doing multi-level preemptive scheduling at all. The current implementation is priority-hint + reactive-preemption, which the 64-program experiment already showed is insufficient. |
| **P1 — Correctness** | 5, 6, 8 | Quantum miscounting (Gap 5) breaks demotion. Stale priority on re-enqueue (Gap 6) breaks fairness after preemption. Inconsistent wait-time semantics (Gap 8) affects anti-starvation. These can cause observable correctness regressions under load. |
| **P2 — Structure** | 4 | Running-queue data structure. The sort-based workaround is adequate; a dedicated structure becomes worthwhile if profiling shows scheduler overhead. |
| **P3 — Optimizations** | 7, 9, 10 | Code clarity (Gap 7), multi-step scheduling (Gap 9), and bulk swap kernel (Gap 10). Ship after P0/P1 are solid and benchmarks indicate these are the next bottleneck. |

## Minimal Viable Implementation

Gaps **1 + 2 + 3** together constitute the minimum set of changes needed to
turn the current priority-hint approximation into a genuine multi-level
preemptive scheduler:

1. Wire `MultiLevelPriorityRequestQueue` into `create_request_queue` with
   unified `AutellixQueueConfig.queue_for_priority` level assignment.
2. Fix `_select_waiting_queue_for_scheduling` to use queue-index ordering.
3. Add proactive preemption: when a higher-queue-index (lower-priority) running
   request blocks a lower-queue-index (higher-priority) waiting request, swap
   out the running request.

With only these three changes, the scheduler would correctly implement the
paper's Algorithm 1 scheduling loop (lines 31–38). Gaps 5, 6, and 8 should
follow immediately to ensure correctness under load.

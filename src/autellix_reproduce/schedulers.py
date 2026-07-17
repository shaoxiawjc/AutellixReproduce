from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field

from autellix_reproduce.models import CallState
from autellix_reproduce.process_table import ProcessTable


@dataclass(frozen=True)
class QueueConfig:
    bounds: tuple[int, ...] = (0, 4, 8, 16, 32, 64)
    quanta: tuple[int, ...] = (2, 4, 8, 16, 32, 64)
    anti_starvation_beta: float = 16.0

    @property
    def num_queues(self) -> int:
        return len(self.quanta)

    def queue_for_priority(self, priority: int) -> int:
        index = 0
        for bound in self.bounds[1:]:
            if priority < bound:
                return index
            index += 1
        return min(index, self.num_queues - 1)


class Scheduler(ABC):
    def __init__(self, process_table: ProcessTable, config: QueueConfig | None = None) -> None:
        self.process_table = process_table
        self.config = config or QueueConfig()

    @abstractmethod
    def add_call(self, call: CallState, now: int) -> None:
        raise NotImplementedError

    @abstractmethod
    def next_batch(self, batch_size: int, now: int) -> list[CallState]:
        raise NotImplementedError

    @abstractmethod
    def on_step_end(self, running: list[CallState], now: int) -> None:
        raise NotImplementedError

    @abstractmethod
    def has_pending(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def queued_calls(self) -> list[CallState]:
        raise NotImplementedError


class FCFSScheduler(Scheduler):
    def __init__(self, process_table: ProcessTable, config: QueueConfig | None = None) -> None:
        super().__init__(process_table, config)
        self.queue: deque[CallState] = deque()

    def add_call(self, call: CallState, now: int) -> None:
        self.process_table.on_call_ready(call, now)
        self.queue.append(call)

    def next_batch(self, batch_size: int, now: int) -> list[CallState]:
        batch = []
        while self.queue and len(batch) < batch_size:
            batch.append(self.queue.popleft())
        return batch

    def on_step_end(self, running: list[CallState], now: int) -> None:
        for call in running:
            if not call.done:
                self.queue.appendleft(call)

    def has_pending(self) -> bool:
        return bool(self.queue)

    def queued_calls(self) -> list[CallState]:
        return list(self.queue)


class MultiLevelScheduler(Scheduler):
    def __init__(self, process_table: ProcessTable, config: QueueConfig | None = None) -> None:
        super().__init__(process_table, config)
        self.queues: list[deque[CallState]] = [deque() for _ in range(self.config.num_queues)]

    def add_call(self, call: CallState, now: int) -> None:
        self.process_table.on_call_ready(call, now)
        priority = self.initial_priority(call)
        call.inherited_priority = priority
        call.queue_index = self.config.queue_for_priority(priority)
        call.quantum_left = self.config.quanta[call.queue_index]
        self.queues[call.queue_index].append(call)

    def next_batch(self, batch_size: int, now: int) -> list[CallState]:
        batch = []
        for queue in self.queues:
            while queue and len(batch) < batch_size:
                batch.append(queue.popleft())
            if len(batch) >= batch_size:
                break
        return batch

    def on_step_end(self, running: list[CallState], now: int) -> None:
        for call in running:
            if call.done:
                continue
            call.quantum_left -= 1
            if self._starved(call):
                call.queue_index = 0
                call.quantum_left = self.config.quanta[0]
                call.scheduler_wait_steps = 0
                call.scheduler_service_steps = 0
            elif call.quantum_left <= 0:
                call.queue_index = min(call.queue_index + 1, self.config.num_queues - 1)
                call.quantum_left = self.config.quanta[call.queue_index]
            self.queues[call.queue_index].append(call)

    def has_pending(self) -> bool:
        return any(self.queues)

    def queued_calls(self) -> list[CallState]:
        return [call for queue in self.queues for call in queue]

    def _starved(self, call: CallState) -> bool:
        record = self.process_table.get(call.program_id)
        service = record.service_time + call.scheduler_service_steps
        wait = record.waiting_time + call.scheduler_wait_steps
        return service > 0 and wait / service >= self.config.anti_starvation_beta

    @abstractmethod
    def initial_priority(self, call: CallState) -> int:
        raise NotImplementedError


class MLFQScheduler(MultiLevelScheduler):
    def initial_priority(self, call: CallState) -> int:
        return 0


class PLASScheduler(MultiLevelScheduler):
    def initial_priority(self, call: CallState) -> int:
        return self.process_table.get(call.program_id).service_time


class ATLASScheduler(MultiLevelScheduler):
    def initial_priority(self, call: CallState) -> int:
        record = self.process_table.get(call.program_id)
        if call.spec.parents:
            try:
                return self.process_table.completed_parent_path_service(call)
            except KeyError:
                return record.critical_path_service
        return record.critical_path_service


POLICIES = {
    "fcfs": FCFSScheduler,
    "mlfq": MLFQScheduler,
    "plas": PLASScheduler,
    "atlas": ATLASScheduler,
}


@dataclass
class SchedulerStats:
    completed_calls: int = 0
    policy: str = ""
    extra: dict[str, float] = field(default_factory=dict)

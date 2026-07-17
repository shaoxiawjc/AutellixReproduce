from __future__ import annotations

from dataclasses import dataclass, field

from autellix_reproduce.models import CallState, ProgramSpec


@dataclass
class ThreadRecord:
    thread_id: str
    service_time: int = 0
    active_call_ids: set[str] = field(default_factory=set)


@dataclass
class ProcessRecord:
    program_id: str
    service_time: int = 0
    waiting_time: int = 0
    critical_path_service: int = 0
    engine_id: int | None = None
    threads: dict[str, ThreadRecord] = field(default_factory=dict)
    completed_calls: set[str] = field(default_factory=set)
    completed_path_service: dict[str, int] = field(default_factory=dict)
    most_recent_arrival: int | None = None
    most_recent_completion: int | None = None


class ProcessTable:
    def __init__(self) -> None:
        self.records: dict[str, ProcessRecord] = {}

    def add_program(self, program: ProgramSpec) -> None:
        self.records.setdefault(program.program_id, ProcessRecord(program_id=program.program_id))

    def get(self, program_id: str) -> ProcessRecord:
        return self.records[program_id]

    def on_call_ready(self, call: CallState, now: int) -> None:
        record = self.get(call.program_id)
        record.most_recent_arrival = now
        thread = record.threads.setdefault(call.thread_id, ThreadRecord(thread_id=call.thread_id))
        thread.active_call_ids.add(call.call_id)

    def add_waiting(self, call: CallState, steps: int = 1) -> None:
        record = self.get(call.program_id)
        record.waiting_time += steps
        call.wait_steps += steps
        call.scheduler_wait_steps += steps

    def completed_parent_path_service(self, call: CallState) -> int:
        record = self.get(call.program_id)
        if not call.spec.parents:
            return 0
        return max(record.completed_path_service[parent] for parent in call.spec.parents)

    def on_call_finished(self, call: CallState, now: int) -> None:
        record = self.get(call.program_id)
        record.service_time += call.service_steps
        record.completed_calls.add(call.call_id)
        record.most_recent_completion = now

        parent_path = self.completed_parent_path_service(call)
        own_path = parent_path + call.service_steps
        record.completed_path_service[call.call_id] = own_path
        record.critical_path_service = max(record.critical_path_service, own_path)

        thread = record.threads.setdefault(call.thread_id, ThreadRecord(thread_id=call.thread_id))
        thread.service_time += call.service_steps
        thread.active_call_ids.discard(call.call_id)

    def completed(self, program_id: str, call_ids: set[str]) -> bool:
        return call_ids.issubset(self.get(program_id).completed_calls)

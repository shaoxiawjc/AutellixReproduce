from __future__ import annotations

from dataclasses import dataclass

from autellix_reproduce.models import CallState, ProgramResult, ProgramSpec
from autellix_reproduce.process_table import ProcessTable
from autellix_reproduce.schedulers import POLICIES, QueueConfig, Scheduler


@dataclass(frozen=True)
class SimulationConfig:
    policy: str = "plas"
    batch_size: int = 2
    max_steps: int = 1_000_000
    queue_config: QueueConfig = QueueConfig()


@dataclass
class SimulationResult:
    config: SimulationConfig
    program_results: dict[str, ProgramResult]
    process_table: ProcessTable
    makespan: int


class Simulator:
    def __init__(self, programs: list[ProgramSpec], config: SimulationConfig) -> None:
        if config.policy not in POLICIES:
            raise ValueError(f"Unknown policy {config.policy!r}; choices={sorted(POLICIES)}")
        self.programs = sorted(programs, key=lambda p: p.arrival_time)
        self.config = config
        self.process_table = ProcessTable()
        for program in self.programs:
            self.process_table.add_program(program)
        scheduler_cls = POLICIES[config.policy]
        self.scheduler: Scheduler = scheduler_cls(self.process_table, config.queue_config)
        self._all_calls = {
            call.call_id: call for program in self.programs for call in program.calls
        }
        self._program_by_call = {
            call.call_id: program for program in self.programs for call in program.calls
        }
        self._pending_call_ids = set(self._all_calls)
        self._created_calls: dict[str, CallState] = {}
        self._completed_calls: dict[str, CallState] = {}

    def run(self) -> SimulationResult:
        now = 0
        while now <= self.config.max_steps:
            self._release_ready_calls(now)
            if not self._pending_call_ids and not self.scheduler.has_pending():
                return self._build_result(now)
            self._account_waiting_time()
            batch = self.scheduler.next_batch(self.config.batch_size, now)
            self._run_one_step(batch, now)
            self.scheduler.on_step_end([call for call in batch if not call.done], now)
            now += 1
        raise RuntimeError(f"Simulation exceeded max_steps={self.config.max_steps}")

    def _release_ready_calls(self, now: int) -> None:
        ready = []
        for call_id in sorted(self._pending_call_ids):
            spec = self._all_calls[call_id]
            program = self._program_by_call[call_id]
            absolute_arrival = program.arrival_time + spec.arrival_offset
            if absolute_arrival > now:
                continue
            if not self.process_table.completed(spec.program_id, set(spec.parents)):
                continue
            ready.append(call_id)

        for call_id in ready:
            spec = self._all_calls[call_id]
            program = self._program_by_call[call_id]
            call = CallState(
                spec=spec,
                arrival_time=program.arrival_time + spec.arrival_offset,
                remaining_steps=spec.model_steps,
                ready_time=now,
            )
            self._created_calls[call_id] = call
            self._pending_call_ids.remove(call_id)
            self.scheduler.add_call(call, now)

    def _account_waiting_time(self) -> None:
        for call in self.scheduler.queued_calls():
            self.process_table.add_waiting(call)

    def _run_one_step(self, batch: list[CallState], now: int) -> None:
        for call in batch:
            if call.start_time is None:
                call.start_time = now
            if call.remaining_steps <= 0:
                continue
            call.remaining_steps -= 1
            call.service_steps += 1
            call.scheduler_service_steps += 1
            if call.remaining_steps == 0:
                call.finish_time = now + 1
                self.process_table.on_call_finished(call, now + 1)
                self._completed_calls[call.call_id] = call

    def _build_result(self, makespan: int) -> SimulationResult:
        results = {}
        for program in self.programs:
            calls = {
                call.call_id: self._completed_calls[call.call_id]
                for call in program.calls
            }
            finish_time = max(call.finish_time or 0 for call in calls.values())
            total_tokens = sum(call.spec.decode_tokens for call in calls.values())
            critical_path_finish_time = self._critical_path_finish(program, calls)
            results[program.program_id] = ProgramResult(
                program_id=program.program_id,
                arrival_time=program.arrival_time,
                finish_time=finish_time,
                critical_path_finish_time=critical_path_finish_time,
                total_tokens=total_tokens,
                calls=calls,
            )
        return SimulationResult(
            config=self.config,
            program_results=results,
            process_table=self.process_table,
            makespan=makespan,
        )

    def _critical_path_finish(
        self,
        program: ProgramSpec,
        calls: dict[str, CallState],
    ) -> int:
        children = {call.call_id: [] for call in program.calls}
        for call in program.calls:
            for parent in call.parents:
                children[parent].append(call.call_id)
        leaves = [call_id for call_id, child_ids in children.items() if not child_ids]
        return max(calls[call_id].finish_time or 0 for call_id in leaves)

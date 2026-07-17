from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class LLMCallSpec:
    call_id: str
    program_id: str
    thread_id: str
    arrival_offset: float
    prefill_tokens: int
    decode_tokens: int
    parents: tuple[str, ...] = ()

    @property
    def model_steps(self) -> int:
        return self.decode_tokens


@dataclass(frozen=True)
class ProgramSpec:
    program_id: str
    arrival_time: float
    calls: tuple[LLMCallSpec, ...]

    def call_map(self) -> dict[str, LLMCallSpec]:
        return {call.call_id: call for call in self.calls}


@dataclass
class CallState:
    spec: LLMCallSpec
    arrival_time: float
    remaining_steps: int
    service_steps: int = 0
    wait_steps: int = 0
    scheduler_service_steps: int = 0
    scheduler_wait_steps: int = 0
    queue_index: int = 0
    quantum_left: int = 0
    ready_time: int | None = None
    start_time: int | None = None
    finish_time: int | None = None
    inherited_priority: int = 0

    @property
    def call_id(self) -> str:
        return self.spec.call_id

    @property
    def program_id(self) -> str:
        return self.spec.program_id

    @property
    def thread_id(self) -> str:
        return self.spec.thread_id

    @property
    def done(self) -> bool:
        return self.remaining_steps <= 0


@dataclass
class ProgramResult:
    program_id: str
    arrival_time: float
    finish_time: int
    total_tokens: int
    critical_path_finish_time: int
    calls: dict[str, CallState] = field(default_factory=dict)

    @property
    def response_time(self) -> float:
        return self.finish_time - self.arrival_time

    @property
    def token_latency(self) -> float:
        if self.total_tokens == 0:
            return 0.0
        return self.response_time / self.total_tokens

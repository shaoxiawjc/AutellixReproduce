from __future__ import annotations

import random

from autellix_reproduce.models import LLMCallSpec, ProgramSpec


def figure2_workload() -> list[ProgramSpec]:
    specs = {
        "A": [4, 3, 1, 1],
        "B": [3, 3, 4],
        "C": [1, 2],
        "D": [4],
    }
    programs = []
    for program_id, decode_steps in specs.items():
        calls = []
        parents: tuple[str, ...] = ()
        for index, steps in enumerate(decode_steps, start=1):
            call_id = f"{program_id}{index}"
            calls.append(
                LLMCallSpec(
                    call_id=call_id,
                    program_id=program_id,
                    thread_id="main",
                    arrival_offset=0,
                    prefill_tokens=0,
                    decode_tokens=steps,
                    parents=parents,
                )
            )
            parents = (call_id,)
        programs.append(ProgramSpec(program_id=program_id, arrival_time=0, calls=tuple(calls)))
    return programs


def synthetic_chatbot_workload(
    programs: int,
    arrival_rate: float,
    seed: int = 0,
) -> list[ProgramSpec]:
    rng = random.Random(seed)
    now = 0.0
    traces = []
    for program_index in range(programs):
        now += rng.expovariate(arrival_rate)
        call_count = 1 + min(80, int(rng.paretovariate(1.8)))
        calls = []
        parent: tuple[str, ...] = ()
        for call_index in range(call_count):
            call_id = f"p{program_index}_c{call_index}"
            decode_tokens = max(1, int(rng.lognormvariate(4.5, 0.8)))
            calls.append(
                LLMCallSpec(
                    call_id=call_id,
                    program_id=f"p{program_index}",
                    thread_id="main",
                    arrival_offset=0,
                    prefill_tokens=max(1, int(rng.lognormvariate(5.0, 0.6))),
                    decode_tokens=decode_tokens,
                    parents=parent,
                )
            )
            parent = (call_id,)
        traces.append(
            ProgramSpec(
                program_id=f"p{program_index}",
                arrival_time=now,
                calls=tuple(calls),
            )
        )
    return traces


def synthetic_mcts_workload(
    programs: int,
    arrival_rate: float,
    seed: int = 0,
    branches: int = 4,
    depth: int = 4,
) -> list[ProgramSpec]:
    rng = random.Random(seed)
    now = 0.0
    traces = []
    for program_index in range(programs):
        now += rng.expovariate(arrival_rate)
        program_id = f"p{program_index}"
        calls = []
        frontier: list[str] = []
        root_id = f"{program_id}_root"
        calls.append(
            LLMCallSpec(
                call_id=root_id,
                program_id=program_id,
                thread_id="root",
                arrival_offset=0,
                prefill_tokens=512,
                decode_tokens=max(1, int(rng.lognormvariate(4.0, 0.5))),
            )
        )
        frontier.append(root_id)
        for level in range(depth):
            new_frontier = []
            for parent in frontier:
                for branch in range(branches):
                    call_id = f"{parent}_{level}_{branch}"
                    calls.append(
                        LLMCallSpec(
                            call_id=call_id,
                            program_id=program_id,
                            thread_id=f"t{level}_{branch}",
                            arrival_offset=0,
                            prefill_tokens=max(1, int(rng.lognormvariate(6.0, 0.4))),
                            decode_tokens=max(1, int(rng.lognormvariate(4.0, 0.5))),
                            parents=(parent,),
                        )
                    )
                    new_frontier.append(call_id)
            frontier = new_frontier
        traces.append(ProgramSpec(program_id=program_id, arrival_time=now, calls=tuple(calls)))
    return traces

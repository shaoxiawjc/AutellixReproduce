from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from autellix_reproduce.models import LLMCallSpec, ProgramSpec


def load_programs_jsonl(path: str | Path) -> list[ProgramSpec]:
    programs = []
    with Path(path).open() as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
                programs.append(_program_from_dict(payload))
            except Exception as exc:
                raise ValueError(f"Invalid trace record at {path}:{line_number}") from exc
    return programs


def dump_programs_jsonl(programs: list[ProgramSpec], path: str | Path) -> None:
    with Path(path).open("w") as handle:
        for program in programs:
            handle.write(json.dumps(_program_to_dict(program), sort_keys=True) + "\n")


def _program_from_dict(payload: dict[str, Any]) -> ProgramSpec:
    program_id = str(payload["program_id"])
    arrival_time = float(payload.get("arrival_time", 0.0))
    calls = []
    for call_payload in payload["calls"]:
        call_id = str(call_payload["call_id"])
        calls.append(
            LLMCallSpec(
                call_id=call_id,
                program_id=str(call_payload.get("program_id", program_id)),
                thread_id=str(call_payload.get("thread_id", "main")),
                arrival_offset=float(call_payload.get("arrival_offset", 0.0)),
                prefill_tokens=int(call_payload.get("prefill_tokens", 0)),
                decode_tokens=int(call_payload["decode_tokens"]),
                parents=tuple(str(parent) for parent in call_payload.get("parents", ())),
            )
        )
    return ProgramSpec(program_id=program_id, arrival_time=arrival_time, calls=tuple(calls))


def _program_to_dict(program: ProgramSpec) -> dict[str, Any]:
    return {
        "program_id": program.program_id,
        "arrival_time": program.arrival_time,
        "calls": [
            {
                "call_id": call.call_id,
                "program_id": call.program_id,
                "thread_id": call.thread_id,
                "arrival_offset": call.arrival_offset,
                "prefill_tokens": call.prefill_tokens,
                "decode_tokens": call.decode_tokens,
                "parents": list(call.parents),
            }
            for call in program.calls
        ],
    }

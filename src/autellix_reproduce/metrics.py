from __future__ import annotations

from dataclasses import dataclass

from autellix_reproduce.simulator import SimulationResult


@dataclass(frozen=True)
class MetricSummary:
    programs: int
    makespan: int
    throughput: float
    avg_latency: float
    p95_latency: float
    p99_latency: float
    avg_wait_steps: float
    avg_service_steps: float


def summarize(result: SimulationResult) -> MetricSummary:
    program_results = list(result.program_results.values())
    latencies = sorted(program.token_latency for program in program_results)
    all_calls = [
        call
        for program in program_results
        for call in program.calls.values()
    ]
    return MetricSummary(
        programs=len(program_results),
        makespan=result.makespan,
        throughput=len(program_results) / result.makespan if result.makespan else 0.0,
        avg_latency=sum(latencies) / len(latencies) if latencies else 0.0,
        p95_latency=_percentile(latencies, 0.95),
        p99_latency=_percentile(latencies, 0.99),
        avg_wait_steps=(
            sum(call.wait_steps for call in all_calls) / len(all_calls) if all_calls else 0.0
        ),
        avg_service_steps=(
            sum(call.service_steps for call in all_calls) / len(all_calls) if all_calls else 0.0
        ),
    )


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    index = min(len(values) - 1, max(0, int(round((len(values) - 1) * q))))
    return values[index]

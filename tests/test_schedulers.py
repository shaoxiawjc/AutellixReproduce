from autellix_reproduce.metrics import summarize
from autellix_reproduce.models import LLMCallSpec, ProgramSpec
from autellix_reproduce.simulator import SimulationConfig, Simulator
from autellix_reproduce.workloads import figure2_workload, synthetic_mcts_workload


def run(policy: str, programs: list[ProgramSpec], batch_size: int = 2):
    return Simulator(programs, SimulationConfig(policy=policy, batch_size=batch_size)).run()


def test_figure2_plas_completes_short_programs_earlier_than_mlfq() -> None:
    programs = figure2_workload()
    mlfq = run("mlfq", programs)
    plas = run("plas", programs)

    assert plas.program_results["C"].finish_time <= mlfq.program_results["C"].finish_time
    assert plas.program_results["D"].finish_time <= mlfq.program_results["D"].finish_time


def test_plas_assigns_later_long_program_call_lower_priority() -> None:
    programs = figure2_workload()
    result = run("plas", programs)
    a2 = result.program_results["A"].calls["A2"]

    assert a2.inherited_priority >= 4
    assert a2.queue_index >= 1


def test_atlas_runs_mcts_like_workload() -> None:
    programs = synthetic_mcts_workload(programs=3, arrival_rate=1.0, seed=1, branches=2, depth=2)
    result = run("atlas", programs, batch_size=4)
    summary = summarize(result)

    assert summary.programs == 3
    assert summary.makespan > 0
    assert all(program.response_time > 0 for program in result.program_results.values())


def test_program_dependencies_are_respected() -> None:
    program = ProgramSpec(
        program_id="p0",
        arrival_time=0,
        calls=(
            LLMCallSpec("c0", "p0", "t0", 0, 0, 3),
            LLMCallSpec("c1", "p0", "t0", 0, 0, 1, parents=("c0",)),
        ),
    )
    result = run("fcfs", [program], batch_size=4)
    c0 = result.program_results["p0"].calls["c0"]
    c1 = result.program_results["p0"].calls["c1"]

    assert c0.finish_time is not None
    assert c1.start_time is not None
    assert c1.start_time >= c0.finish_time

from autellix_reproduce.trace_io import dump_programs_jsonl, load_programs_jsonl
from autellix_reproduce.workloads import figure2_workload


def test_jsonl_roundtrip(tmp_path) -> None:
    path = tmp_path / "trace.jsonl"
    programs = figure2_workload()

    dump_programs_jsonl(programs, path)
    loaded = load_programs_jsonl(path)

    assert len(loaded) == len(programs)
    assert loaded[0].program_id == programs[0].program_id
    assert loaded[0].calls[0].decode_tokens == programs[0].calls[0].decode_tokens
    assert loaded[0].calls[1].parents == programs[0].calls[1].parents

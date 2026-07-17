from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_MODEL = "/root/rivermind-data/resources/models/qwen3_0d6b"
DEFAULT_SHAREGPT = (
    "/root/rivermind-data/resources/datasets/sharegpt/"
    "ShareGPT_V3_unfiltered_cleaned_split_no_imsorry.json"
)
DEFAULT_BFCL = (
    "/root/rivermind-data/resources/datasets/BFCL/BFCL_v3_multi_turn_base.json"
)
DEFAULT_VLLM_ROOT = "/root/autellix_reproduce_work/vllm"


@dataclass
class ProgramCall:
    call_id: str
    messages: list[dict[str, str]]
    new_messages: list[dict[str, str]] = field(default_factory=list)
    thread_id: str = "main"
    parents: tuple[str, ...] = ()


@dataclass
class Program:
    program_id: str
    calls: list[ProgramCall]
    next_call: int = 0
    attained_service: int = 0
    total_output_tokens: int = 0
    arrival_delay_s: float = 0.0
    started_at: float | None = None
    finished_at: float | None = None

    @property
    def done(self) -> bool:
        return self.next_call >= len(self.calls)


@dataclass
class CallRecord:
    workload: str
    policy: str
    program_id: str
    call_id: str
    wave: int
    priority: int
    prompt_tokens: int
    output_tokens: int
    submit_time: float
    finish_time: float
    text_preview: str = ""
    request_id: str | None = None
    vllm_metrics: dict[str, Any] = field(default_factory=dict)


def main() -> None:
    args = parse_args()
    if args.workload == "lats":
        raise SystemExit(
            "LATS requires the async driver; use run_vllm_async_experiment.py"
        )
    prepare_imports(args.vllm_root)

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    programs = load_workload(args, tokenizer)
    if not programs:
        raise RuntimeError("No programs were loaded.")

    llm = LLM(
        model=args.model,
        tokenizer=args.model,
        trust_remote_code=True,
        dtype=args.dtype,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        enable_prefix_caching=not args.disable_prefix_caching,
        enforce_eager=args.enforce_eager,
        scheduling_policy=vllm_scheduling_policy(args.policy),
    )
    sampling_params = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=0.0,
    )

    records = run_waves(
        llm=llm,
        tokenizer=tokenizer,
        sampling_params=sampling_params,
        programs=programs,
        workload=args.workload,
        policy=args.policy,
    )

    label = result_label(args)
    details_path = out_dir / f"{label}_calls.jsonl"
    with details_path.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record.__dict__, ensure_ascii=False) + "\n")

    summary = summarize(programs, records, args)
    summary_path = out_dir / f"{label}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workload", choices=["sharegpt", "bfcl", "lats"], required=True
    )
    parser.add_argument(
        "--policy", choices=["fcfs", "mlfq", "plas", "atlas"], required=True
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--sharegpt-path", default=DEFAULT_SHAREGPT)
    parser.add_argument("--bfcl-path", default=DEFAULT_BFCL)
    parser.add_argument("--vllm-root", default=DEFAULT_VLLM_ROOT)
    parser.add_argument("--output-dir", default="results/vllm_core")
    parser.add_argument("--max-programs", type=int, default=32)
    parser.add_argument("--max-calls-per-program", type=int, default=6)
    parser.add_argument("--lats-branching-factor", type=int, default=3)
    parser.add_argument("--lats-depth", type=int, default=4)
    parser.add_argument("--max-prompt-chars", type=int, default=6000)
    parser.add_argument("--max-prompt-tokens", type=int)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.75)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-seqs", type=int, default=32)
    parser.add_argument("--max-num-batched-tokens", type=int, default=4096)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--disable-prefix-caching", action="store_true")
    return parser.parse_args()


def result_label(args: argparse.Namespace) -> str:
    parts = [args.workload, args.policy]
    if args.disable_prefix_caching:
        parts.append("noprefix")
    return "_".join(parts)


def prepare_imports(vllm_root: str) -> None:
    if vllm_root not in sys.path:
        sys.path.insert(0, vllm_root)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def load_workload(args: argparse.Namespace, tokenizer: Any) -> list[Program]:
    if args.workload == "sharegpt":
        return load_sharegpt(args, tokenizer)
    if args.workload == "bfcl":
        return load_bfcl(args, tokenizer)
    return load_lats(args, tokenizer)


def load_sharegpt(args: argparse.Namespace, tokenizer: Any) -> list[Program]:
    with open(args.sharegpt_path) as handle:
        raw = json.load(handle)

    programs: list[Program] = []
    for item in raw:
        messages: list[dict[str, str]] = []
        calls: list[ProgramCall] = []
        conversations = item.get("conversations") or []
        for turn in conversations:
            role = turn.get("from")
            value = (turn.get("value") or "").strip()
            if not value:
                continue
            if role == "human":
                messages.append({"role": "user", "content": value})
                if prompt_len(tokenizer, messages) <= max_prompt_tokens(args):
                    calls.append(
                        ProgramCall(
                            call_id=f"{item.get('id', len(programs))}_{len(calls)}",
                            messages=list(messages),
                            new_messages=[{"role": "user", "content": value}],
                            parents=((calls[-1].call_id,) if calls else ()),
                        )
                    )
            if len(calls) >= args.max_calls_per_program:
                break
        if calls:
            programs.append(
                Program(program_id=str(item.get("id", len(programs))), calls=calls)
            )
        if len(programs) >= args.max_programs:
            break
    return programs


def load_bfcl(args: argparse.Namespace, tokenizer: Any) -> list[Program]:
    programs: list[Program] = []
    with open(args.bfcl_path) as handle:
        for line in handle:
            item = json.loads(line)
            calls: list[ProgramCall] = []
            context: list[dict[str, str]] = []
            system = bfcl_system_prompt(item)
            if system:
                context.append({"role": "system", "content": system})
            for turn_index, turn_messages in enumerate(item.get("question", [])):
                for msg in turn_messages:
                    content = (msg.get("content") or "").strip()
                    if not content:
                        continue
                    context.append({"role": msg.get("role", "user"), "content": content})
                if prompt_len(tokenizer, context) <= max_prompt_tokens(args):
                    calls.append(
                        ProgramCall(
                            call_id=f"{item.get('id', len(programs))}_{turn_index}",
                            messages=list(context),
                            new_messages=[
                                {
                                    "role": msg.get("role", "user"),
                                    "content": (msg.get("content") or "").strip(),
                                }
                                for msg in turn_messages
                                if (msg.get("content") or "").strip()
                            ],
                            parents=((calls[-1].call_id,) if calls else ()),
                        )
                    )
                if len(calls) >= args.max_calls_per_program:
                    break
            if calls:
                programs.append(
                    Program(
                        program_id=str(item.get("id", len(programs))),
                        calls=calls,
                    )
                )
            if len(programs) >= args.max_programs:
                break
    return programs


def load_lats(args: argparse.Namespace, tokenizer: Any) -> list[Program]:
    """Build a synthetic MCTS/LATS-style DAG workload.

    The real paper uses LATS traces from HotpotQA. This generator preserves the
    important scheduler-facing structure: a root call followed by multiple
    dependent branches whose children become ready when their parent completes.
    """
    programs: list[Program] = []
    branching = max(1, args.lats_branching_factor)
    depth = max(1, args.lats_depth)
    max_calls = max(1, args.max_calls_per_program)
    for program_index in range(args.max_programs):
        calls: list[ProgramCall] = []
        frontier: list[tuple[str, int, str]] = [("", 0, "root")]
        while frontier and len(calls) < max_calls:
            parent_id, node_depth, branch_name = frontier.pop(0)
            call_id = f"lats_{program_index}_{len(calls)}"
            question = (
                "Solve the HotpotQA-style multi-hop question by proposing the "
                f"next reasoning/action step. Program {program_index}, "
                f"branch {branch_name}, depth {node_depth}."
            )
            messages = [{"role": "user", "content": question}]
            if prompt_len(tokenizer, messages) <= max_prompt_tokens(args):
                calls.append(
                    ProgramCall(
                        call_id=call_id,
                        messages=messages,
                        new_messages=messages,
                        thread_id=branch_name,
                        parents=((parent_id,) if parent_id else ()),
                    )
                )
            if node_depth + 1 < depth:
                for child in range(branching):
                    frontier.append(
                        (
                            call_id,
                            node_depth + 1,
                            f"{branch_name}.{child}",
                        )
                    )
            if len(calls) >= max_calls:
                break
        if calls:
            programs.append(Program(program_id=f"lats_{program_index}", calls=calls))
    return programs


def bfcl_system_prompt(item: dict[str, Any]) -> str:
    functions = item.get("function")
    if not functions:
        return ""
    return (
        "You are a function-calling assistant. Select appropriate function calls "
        "when needed. Available functions:\n"
        + json.dumps(functions, ensure_ascii=False)
    )


def prompt_len(tokenizer: Any, messages: list[dict[str, str]]) -> int:
    prompt = render_prompt(tokenizer, messages)
    return len(tokenizer.encode(prompt))


def max_prompt_tokens(args: argparse.Namespace) -> int:
    explicit = args.max_prompt_tokens
    context_limit = max(1, args.max_model_len - args.max_tokens)
    legacy_limit = args.max_prompt_chars
    if explicit is None:
        return min(legacy_limit, context_limit)
    return min(explicit, context_limit)


def render_prompt(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def run_waves(
    llm: Any,
    tokenizer: Any,
    sampling_params: Any,
    programs: list[Program],
    workload: str,
    policy: str,
) -> list[CallRecord]:
    if has_dag_work(programs):
        raise ValueError(
            "Wave driver only supports sequential-chain workloads. "
            "Use run_vllm_async_experiment.py for --workload lats/DAG traces."
        )
    records: list[CallRecord] = []
    wave = 0
    while any(not program.done for program in programs):
        batch_programs = [program for program in programs if not program.done]
        prompts = [
            render_prompt(tokenizer, program.calls[program.next_call].messages)
            for program in batch_programs
        ]
        priorities = [priority_for(policy, program) for program in batch_programs]
        submit_time = time.perf_counter()
        for program in batch_programs:
            if program.started_at is None:
                program.started_at = submit_time

        request_sampling_params = [
            sampling_params_with_autellix_metadata(
                sampling_params=sampling_params,
                policy=policy,
                program=program,
                call=program.calls[program.next_call],
            )
            for program in batch_programs
        ]

        outputs = llm.generate(
            prompts,
            sampling_params=request_sampling_params,
            priority=None if policy == "fcfs" else priorities,
            use_tqdm=False,
        )
        finish_time = time.perf_counter()

        for program, output, priority in zip(
            batch_programs, outputs, priorities, strict=True
        ):
            call = program.calls[program.next_call]
            generated_tokens = sum(
                len(completion.token_ids) for completion in output.outputs
            )
            prompt_tokens = len(output.prompt_token_ids or [])
            text = output.outputs[0].text if output.outputs else ""
            program.attained_service += max(1, generated_tokens)
            program.total_output_tokens += generated_tokens
            program.next_call += 1
            if program.done:
                program.finished_at = finish_time
            records.append(
                CallRecord(
                    workload=workload,
                    policy=policy,
                    program_id=program.program_id,
                    call_id=call.call_id,
                    wave=wave,
                    priority=priority,
                    prompt_tokens=prompt_tokens,
                    output_tokens=generated_tokens,
                    submit_time=submit_time,
                    finish_time=finish_time,
                    text_preview=text[:160],
                    request_id=output.request_id,
                    vllm_metrics=metrics_to_dict(output.metrics),
                )
            )
        wave += 1
    return records


def has_dag_work(programs: list[Program]) -> bool:
    return any(
        len(call.parents) != (1 if index > 0 else 0)
        or (index > 0 and call.parents != (program.calls[index - 1].call_id,))
        for program in programs
        for index, call in enumerate(program.calls)
    )


def priority_for(policy: str, program: Program) -> int:
    if policy == "fcfs":
        return 0
    if policy == "mlfq":
        return 0
    if policy == "plas":
        return program.attained_service
    if policy == "atlas":
        return program.attained_service
    raise ValueError(policy)


def vllm_scheduling_policy(policy: str) -> str:
    if policy == "fcfs":
        return "fcfs"
    if policy in {"mlfq", "plas", "atlas"}:
        return policy
    raise ValueError(policy)


def sampling_params_with_autellix_metadata(
    sampling_params: Any,
    policy: str,
    program: Program,
    call: ProgramCall,
) -> Any:
    params = sampling_params.clone()
    extra_args = dict(params.extra_args or {})
    extra_args["autellix"] = {
        "policy": policy,
        "program_id": program.program_id,
        "call_id": call.call_id,
        "thread_id": call.thread_id,
        "parent_call_ids": list(call.parents),
    }
    params.extra_args = extra_args
    return params


def metrics_to_dict(metrics: Any) -> dict[str, Any]:
    if metrics is None:
        return {}
    keys = [
        "arrival_time",
        "queued_ts",
        "scheduled_ts",
        "first_token_ts",
        "last_token_ts",
        "first_token_latency",
        "num_generation_tokens",
    ]
    return {key: getattr(metrics, key) for key in keys if hasattr(metrics, key)}


def summarize(
    programs: list[Program],
    records: list[CallRecord],
    args: argparse.Namespace,
) -> dict[str, Any]:
    program_latencies = [
        (program.finished_at or 0.0) - (program.started_at or 0.0)
        for program in programs
        if program.started_at is not None and program.finished_at is not None
    ]
    program_token_latencies = [
        ((program.finished_at or 0.0) - (program.started_at or 0.0))
        / max(1, program.total_output_tokens)
        for program in programs
        if program.started_at is not None
        and program.finished_at is not None
        and program.total_output_tokens > 0
    ]
    total_output_tokens = sum(record.output_tokens for record in records)
    total_prompt_tokens = sum(record.prompt_tokens for record in records)
    start = min(record.submit_time for record in records)
    end = max(record.finish_time for record in records)
    elapsed = end - start
    return {
        "workload": args.workload,
        "policy": args.policy,
        "model": args.model,
        "programs": len(programs),
        "calls": len(records),
        "waves": max(record.wave for record in records) + 1 if records else 0,
        "elapsed_s": elapsed,
        "throughput_programs_per_s": len(programs) / elapsed if elapsed > 0 else 0.0,
        "throughput_output_tokens_per_s": total_output_tokens / elapsed
        if elapsed > 0
        else 0.0,
        "avg_program_latency_s": mean(program_latencies),
        "p95_program_latency_s": percentile(program_latencies, 0.95),
        "p99_program_latency_s": percentile(program_latencies, 0.99),
        "avg_program_token_latency_s": mean(program_token_latencies),
        "p95_program_token_latency_s": percentile(program_token_latencies, 0.95),
        "p99_program_token_latency_s": percentile(program_token_latencies, 0.99),
        "total_prompt_tokens": total_prompt_tokens,
        "total_output_tokens": total_output_tokens,
        "max_tokens": args.max_tokens,
        "max_programs": args.max_programs,
        "max_calls_per_program": args.max_calls_per_program,
        "prefix_caching": not args.disable_prefix_caching,
        "thinking": "disabled via tokenizer chat_template enable_thinking=False",
    }


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * q)))
    return ordered[index]


if __name__ == "__main__":
    main()

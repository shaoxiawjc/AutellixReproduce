from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_MODEL = "/root/autodl-tmp/resources/models/qwen3_0d6b"
DEFAULT_SHAREGPT = (
    "/root/autodl-tmp/resources/datasets/sharegpt/"
    "ShareGPT_V3_unfiltered_cleaned_split_no_imsorry.json"
)
DEFAULT_BFCL = (
    "/root/autodl-tmp/resources/datasets/BFCL/BFCL_v3_multi_turn_base.json"
)
DEFAULT_VLLM_ROOT = "/root/autellix/vllm"


@dataclass
class ProgramCall:
    call_id: str
    new_messages: list[dict[str, str]] = field(default_factory=list)
    thread_id: str = "main"
    parents: tuple[str, ...] = ()
    truncated: bool = False


@dataclass
class Program:
    program_id: str
    calls: list[ProgramCall]
    next_call: int = 0
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
    prepare_imports(args.vllm_root)
    asyncio.run(async_main(args))


async def async_main(args: argparse.Namespace) -> None:
    from transformers import AutoTokenizer
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.sampling_params import SamplingParams
    from vllm.v1.engine.async_llm import AsyncLLM

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    programs = load_workload(args, tokenizer)
    if not programs:
        raise RuntimeError("No programs were loaded.")

    engine_args = AsyncEngineArgs(
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
        disable_log_stats=args.disable_log_stats,
    )
    engine = AsyncLLM.from_engine_args(engine_args)
    sampling_params = SamplingParams(max_tokens=args.max_tokens, temperature=0.0)

    try:
        records = await run_programs_async(
            engine=engine,
            tokenizer=tokenizer,
            sampling_params=sampling_params,
            programs=programs,
            workload=args.workload,
            policy=args.policy,
            history_mode=args.history_mode,
            arrival_rate=args.arrival_rate,
            arrival_seed=args.arrival_seed,
        )
    finally:
        engine.shutdown()

    label = result_label(args)
    if args.arrival_rate > 0:
        label = f"{label}_lambda{args.arrival_rate:g}"
    details_path = out_dir / f"{label}_async_calls.jsonl"
    with details_path.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record.__dict__, ensure_ascii=False) + "\n")

    summary = summarize_async(programs, records, args)
    summary_path = out_dir / f"{label}_async_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


# ── CLI ──────────────────────────────────────────────────────────────────────


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
    parser.add_argument("--output-dir", default="results/vllm_async")
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
    parser.add_argument("--disable-log-stats", action="store_true")
    parser.add_argument("--disable-prefix-caching", action="store_true")
    parser.add_argument(
        "--arrival-rate",
        type=float,
        default=0.0,
        help="Poisson program arrival rate in programs/s; <=0 submits all at t=0.",
    )
    parser.add_argument("--arrival-seed", type=int, default=0)
    parser.add_argument(
        "--history-mode",
        choices=["model", "user_only"],
        default="model",
        help=(
            "model: append output as assistant history; user_only: "
            "accumulate only user/tool prompts."
        ),
    )
    return parser.parse_args()


# ── Helpers ───────────────────────────────────────────────────────────────────


def prepare_imports(vllm_root: str) -> None:
    if vllm_root not in sys.path:
        sys.path.insert(0, vllm_root)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def vllm_scheduling_policy(policy: str) -> str:
    if policy == "fcfs":
        return "fcfs"
    if policy in {"mlfq", "plas", "atlas"}:
        return policy
    raise ValueError(policy)


def result_label(args: argparse.Namespace) -> str:
    parts = [args.workload, args.policy]
    if args.disable_prefix_caching:
        parts.append("noprefix")
    return "_".join(parts)


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


# ── Workload loading ─────────────────────────────────────────────────────────


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
        calls: list[ProgramCall] = []
        conversations = item.get("conversations") or []
        accumulated_user: list[dict[str, str]] = []
        for turn in conversations:
            role = turn.get("from")
            value = (turn.get("value") or "").strip()
            if not value:
                continue
            if role == "human":
                user_msg = {"role": "user", "content": value}
                accumulated_user.append(user_msg)
                # Estimate token count: accumulated user messages only.
                # At runtime the model's own assistant outputs will be prepended
                # by update_runtime_history; we can't predict their length here.
                if prompt_len(tokenizer, accumulated_user) <= max_prompt_tokens(args):
                    calls.append(
                        ProgramCall(
                            call_id=f"{item.get('id', len(programs))}_{len(calls)}",
                            new_messages=[user_msg],
                            parents=((calls[-1].call_id,) if calls else ()),
                        )
                    )
                else:
                    break
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
            accumulated: list[dict[str, str]] = []
            system = bfcl_system_prompt(item)
            if system:
                accumulated.append({"role": "system", "content": system})
            for turn_index, turn_messages in enumerate(item.get("question", [])):
                new_msgs: list[dict[str, str]] = []
                for msg in turn_messages:
                    content = (msg.get("content") or "").strip()
                    if not content:
                        continue
                    role = msg.get("role", "user")
                    m = {"role": role, "content": content}
                    accumulated.append(m)
                    new_msgs.append(m)
                if prompt_len(tokenizer, accumulated) <= max_prompt_tokens(args):
                    calls.append(
                        ProgramCall(
                            call_id=f"{item.get('id', len(programs))}_{turn_index}",
                            new_messages=new_msgs,
                            parents=((calls[-1].call_id,) if calls else ()),
                        )
                    )
                else:
                    break
                if len(calls) >= args.max_calls_per_program:
                    break
            if calls:
                programs.append(
                    Program(program_id=str(item.get("id", len(programs))), calls=calls)
                )
            if len(programs) >= args.max_programs:
                break
    return programs


def load_lats(args: argparse.Namespace, tokenizer: Any) -> list[Program]:
    """Build a synthetic MCTS/LATS-style DAG workload.

    The paper uses LATS traces from HotpotQA. This generator preserves the
    scheduler-facing DAG structure: a root call followed by multiple dependent
    branches whose children become ready when their parent completes.
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
            new_msgs = [{"role": "user", "content": question}]
            if prompt_len(tokenizer, new_msgs) <= max_prompt_tokens(args):
                calls.append(
                    ProgramCall(
                        call_id=call_id,
                        new_messages=new_msgs,
                        thread_id=branch_name,
                        parents=((parent_id,) if parent_id else ()),
                    )
                )
            if node_depth + 1 < depth:
                for child in range(branching):
                    frontier.append((call_id, node_depth + 1, f"{branch_name}.{child}"))
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


# ── Async experiment runner ──────────────────────────────────────────────────


async def run_programs_async(
    engine: Any,
    tokenizer: Any,
    sampling_params: Any,
    programs: list[Program],
    workload: str,
    policy: str,
    history_mode: str,
    arrival_rate: float,
    arrival_seed: int,
) -> list[CallRecord]:
    records: list[CallRecord] = []
    lock = asyncio.Lock()
    assign_arrival_delays(programs, arrival_rate, arrival_seed)

    async def run_one_call(
        program: Program,
        call_index: int,
        call: ProgramCall,
        runtime_history: list[dict[str, str]],
    ) -> tuple[int, str]:
        prompt_messages = prompt_messages_for_call(
            call=call,
            runtime_history=runtime_history,
            history_mode=history_mode,
        )
        prompt = render_prompt(tokenizer, prompt_messages)
        submit_time = time.perf_counter()
        if program.started_at is None:
            program.started_at = submit_time

        request_id = f"{program.program_id}:{call.call_id}:{call_index}"
        final_output = None
        async for output in engine.generate(
            prompt=prompt,
            sampling_params=sampling_params_with_autellix_metadata(
                sampling_params=sampling_params,
                policy=policy,
                program=program,
                call=call,
            ),
            request_id=request_id,
            priority=0,
        ):
            final_output = output
        finish_time = time.perf_counter()
        if final_output is None:
            raise RuntimeError(f"Request {request_id} produced no output.")

        generated_tokens = sum(
            len(completion.token_ids) for completion in final_output.outputs
        )
        prompt_tokens = len(final_output.prompt_token_ids or [])
        text = final_output.outputs[0].text if final_output.outputs else ""
        program.total_output_tokens += generated_tokens

        record = CallRecord(
            workload=workload,
            policy=policy,
            program_id=program.program_id,
            call_id=call.call_id,
            wave=call_index,
            priority=0,
            prompt_tokens=prompt_tokens,
            output_tokens=generated_tokens,
            submit_time=submit_time,
            finish_time=finish_time,
            text_preview=text[:160],
            request_id=final_output.request_id,
            vllm_metrics=metrics_to_dict(final_output.metrics),
        )
        async with lock:
            records.append(record)
        return call_index, text

    async def run_one_program(program: Program) -> None:
        if program.arrival_delay_s > 0:
            await asyncio.sleep(program.arrival_delay_s)
        runtime_history_by_thread: dict[str, list[dict[str, str]]] = {}
        completed: set[str] = set()
        submitted: set[int] = set()
        pending: dict[asyncio.Task[tuple[int, str]], int] = {}

        while len(completed) < len(program.calls):
            made_progress = False
            for call_index, call in enumerate(program.calls):
                if call_index in submitted:
                    continue
                if any(parent not in completed for parent in call.parents):
                    continue
                submitted.add(call_index)
                made_progress = True
                runtime_history = runtime_history_by_thread.get(call.thread_id, [])
                pending[
                    asyncio.create_task(
                        run_one_call(program, call_index, call, runtime_history)
                    )
                ] = call_index

            if not pending:
                unresolved = [
                    call.call_id
                    for index, call in enumerate(program.calls)
                    if index not in submitted
                ]
                raise RuntimeError(
                    f"Program {program.program_id} has unsatisfied DAG parents: "
                    f"{unresolved}"
                )

            done_tasks, _ = await asyncio.wait(
                pending.keys(), return_when=asyncio.FIRST_COMPLETED
            )
            for task in done_tasks:
                call_index = pending.pop(task)
                _, text = task.result()
                call = program.calls[call_index]
                completed.add(call.call_id)
                program.next_call = max(program.next_call, len(completed))
                runtime_history_by_thread[call.thread_id] = update_runtime_history(
                    prompt_messages=prompt_messages_for_call(
                        call=call,
                        runtime_history=runtime_history_by_thread.get(
                            call.thread_id, []
                        ),
                        history_mode=history_mode,
                    ),
                    output_text=text,
                    history_mode=history_mode,
                )
            if not made_progress and not done_tasks:
                await asyncio.sleep(0)

        program.finished_at = time.perf_counter()

    async def run_one_program_sequential(program: Program) -> None:
        if program.arrival_delay_s > 0:
            await asyncio.sleep(program.arrival_delay_s)
        runtime_history: list[dict[str, str]] = []
        while not program.done:
            call_index = program.next_call
            call = program.calls[call_index]
            prompt_messages = prompt_messages_for_call(
                call=call,
                runtime_history=runtime_history,
                history_mode=history_mode,
            )
            prompt = render_prompt(tokenizer, prompt_messages)
            submit_time = time.perf_counter()
            if program.started_at is None:
                program.started_at = submit_time

            request_id = f"{program.program_id}:{call.call_id}:{call_index}"
            final_output = None
            async for output in engine.generate(
                prompt=prompt,
                sampling_params=sampling_params_with_autellix_metadata(
                    sampling_params=sampling_params,
                    policy=policy,
                    program=program,
                    call=call,
                ),
                request_id=request_id,
                priority=0,
            ):
                final_output = output
            finish_time = time.perf_counter()
            if final_output is None:
                raise RuntimeError(f"Request {request_id} produced no output.")

            generated_tokens = sum(
                len(completion.token_ids) for completion in final_output.outputs
            )
            prompt_tokens = len(final_output.prompt_token_ids or [])
            text = final_output.outputs[0].text if final_output.outputs else ""
            program.total_output_tokens += generated_tokens
            program.next_call += 1
            runtime_history = update_runtime_history(
                prompt_messages=prompt_messages,
                output_text=text,
                history_mode=history_mode,
            )
            if program.done:
                program.finished_at = finish_time

            record = CallRecord(
                workload=workload,
                policy=policy,
                program_id=program.program_id,
                call_id=call.call_id,
                wave=call_index,
                priority=0,
                prompt_tokens=prompt_tokens,
                output_tokens=generated_tokens,
                submit_time=submit_time,
                finish_time=finish_time,
                text_preview=text[:160],
                request_id=final_output.request_id,
                vllm_metrics=metrics_to_dict(final_output.metrics),
            )
            async with lock:
                records.append(record)

    runner = run_one_program if has_dag_work(programs) else run_one_program_sequential
    await asyncio.gather(*(runner(program) for program in programs))
    return sorted(records, key=lambda r: (r.submit_time, r.finish_time, r.program_id))


def has_dag_work(programs: list[Program]) -> bool:
    return any(
        len(call.parents) != (1 if index > 0 else 0)
        or (index > 0 and call.parents != (program.calls[index - 1].call_id,))
        for program in programs
        for index, call in enumerate(program.calls)
    )


def assign_arrival_delays(
    programs: list[Program], arrival_rate: float, arrival_seed: int
) -> None:
    if arrival_rate <= 0:
        for program in programs:
            program.arrival_delay_s = 0.0
        return
    rng = random.Random(arrival_seed)
    current = 0.0
    for program in programs:
        current += rng.expovariate(arrival_rate)
        program.arrival_delay_s = current


def prompt_messages_for_call(
    call: ProgramCall,
    runtime_history: list[dict[str, str]],
    history_mode: str,
) -> list[dict[str, str]]:
    if history_mode == "model":
        return runtime_history + call.new_messages
    if history_mode == "user_only":
        return runtime_history + [
            msg for msg in call.new_messages if msg.get("role") != "assistant"
        ]
    raise ValueError(history_mode)


def update_runtime_history(
    prompt_messages: list[dict[str, str]],
    output_text: str,
    history_mode: str,
) -> list[dict[str, str]]:
    if history_mode == "model":
        return prompt_messages + [{"role": "assistant", "content": output_text}]
    if history_mode == "user_only":
        return prompt_messages
    raise ValueError(history_mode)


# ── Summary ──────────────────────────────────────────────────────────────────


def summarize_async(
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
        "mode": "async",
        "workload": args.workload,
        "policy": args.policy,
        "model": args.model,
        "programs": len(programs),
        "calls": len(records),
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
        "history_mode": args.history_mode,
        "arrival_rate_programs_per_s": args.arrival_rate,
        "arrival_seed": args.arrival_seed,
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

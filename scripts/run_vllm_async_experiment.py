from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_vllm_core_experiment import (  # noqa: E402
    DEFAULT_BFCL,
    DEFAULT_MODEL,
    DEFAULT_SHAREGPT,
    DEFAULT_VLLM_ROOT,
    CallRecord,
    Program,
    load_workload,
    mean,
    metrics_to_dict,
    percentile,
    prepare_imports,
    priority_for,
    render_prompt,
    sampling_params_with_autellix_metadata,
    result_label,
    vllm_scheduling_policy,
)


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
            "model: append Qwen output as assistant history; user_only: "
            "accumulate only user/tool prompts."
        ),
    )
    return parser.parse_args()


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
        call: Any,
        runtime_history: list[dict[str, str]],
    ) -> tuple[int, str]:
        prompt_messages = prompt_messages_for_call(
            call=call,
            runtime_history=runtime_history,
            history_mode=history_mode,
        )
        prompt = render_prompt(tokenizer, prompt_messages)
        priority = priority_for(policy, program)
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
            priority=priority,
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
        program.attained_service += max(1, generated_tokens)
        program.total_output_tokens += generated_tokens

        record = CallRecord(
            workload=workload,
            policy=policy,
            program_id=program.program_id,
            call_id=call.call_id,
            wave=call_index,
            priority=priority,
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
            priority = priority_for(policy, program)
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
                priority=priority,
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
            program.attained_service += max(1, generated_tokens)
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
                priority=priority,
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
    call: Any,
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


if __name__ == "__main__":
    main()

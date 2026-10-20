"""Bounded A01 diagnostic: what makes the agent runner's first model request slow.

Both protocols were declared before their results were evaluated. The prompt, tool
definitions and chat options are imported from the agent runner for the selected task, so
they cannot drift from what `agentwatch_agent run` sends. Every request is recorded; nothing
is discarded or retried. Results recorded before 2026-10-20 used the removed LLM demo's
single-tool request and are not comparable with later runs.

cold-tools: for each of REPETITIONS, run two blocks in an order that alternates by
repetition. A block unloads the model, then sends the runner's exact first request once cold
and WARM_REQUESTS times warm, either with or without the runner's tool definitions.
"Cold" means the model was unloaded through keep_alive=0. That restarts Ollama's model
runner but not the container, so it does not reproduce a freshly created container.

warm-state: with the model loaded, for each of REPETITIONS, prime with the runner request,
then send it again after each IDLE_GAPS_SECONDS pause, then send a different prompt and the
runner request immediately afterwards. This separates idle time from prompt-cache contents.

Usage: OLLAMA_URL=http://127.0.0.1:11435 uv run --package agentwatch-agent \
           python deploy/diagnostics/first_request_timing.py --protocol cold-tools
"""

import argparse
import json
import math
import os
import time
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

from agentwatch_agent.config import DEFAULT_CHAT_OPTIONS
from agentwatch_agent.prompts import SYSTEM_PROMPT, task_message
from agentwatch_agent.tasks import Task, load_task
from agentwatch_agent.tools import tool_definitions
from agentwatch_worker.ollama import (
    model_placement,
    observe_model_placement,
    ollama_base_url,
    ollama_chat,
    ollama_loaded_models,
)

REPETITIONS = 3
WARM_REQUESTS = 2
UNLOAD_TIMEOUT_SECONDS = 30
UNLOAD_POLL_SECONDS = 0.5
DURATION_FIELDS = ("total_duration", "load_duration", "prompt_eval_duration", "eval_duration")
# 10 s approximates the gap before the 2026-10-13 warm run's first request.
IDLE_GAPS_SECONDS = (0, 10, 60)
OTHER_PROMPT = (
    "Before running anything, list in one sentence what the project's tests check. "
    "Do not call a tool."
)


def unload(url: str, model: str) -> None:
    request = urllib.request.Request(
        url.rstrip("/") + "/api/generate",
        data=json.dumps({"model": model, "keep_alive": 0}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        response.read()
    deadline = time.monotonic() + UNLOAD_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if model_placement(ollama_loaded_models(url), model)["status"] == "unavailable":
            return
        time.sleep(UNLOAD_POLL_SECONDS)
    raise RuntimeError(f"{model} was still loaded after {UNLOAD_TIMEOUT_SECONDS} s")


def duration_seconds(nanoseconds: object) -> float | None:
    """Convert an Ollama nanosecond duration, or return None when it is absent or invalid."""
    if isinstance(nanoseconds, bool) or not isinstance(nanoseconds, int | float):
        return None
    if not math.isfinite(nanoseconds) or nanoseconds < 0:
        return None
    return nanoseconds / 1e9


def first_messages(task: Task) -> list[dict]:
    """The runner's first request for `task`: its system prompt and task message."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task_message(task)},
    ]


def timed_request(
    url: str, model: str, task: Task, with_tools: bool, user_prompt: str | None = None
) -> dict:
    messages = first_messages(task)
    if user_prompt is not None:
        messages[-1]["content"] = user_prompt
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "think": False,
        "options": dict(DEFAULT_CHAT_OPTIONS),
    }
    if with_tools:
        payload["tools"] = list(tool_definitions(task))
    started_at = datetime.now(UTC)
    started = time.monotonic_ns()
    response = ollama_chat(url, payload)
    wall_seconds = (time.monotonic_ns() - started) / 1e9
    placement = observe_model_placement(lambda: ollama_loaded_models(url), model)
    return {
        "started_at": started_at.isoformat(),
        "wall_seconds": wall_seconds,
        # Missing or invalid Ollama fields stay null rather than becoming a measured zero.
        **{f"{field}_seconds": duration_seconds(response.get(field)) for field in DURATION_FIELDS},
        "prompt_eval_count": response.get("prompt_eval_count"),
        "eval_count": response.get("eval_count"),
        "requested_tool": bool(response["message"].get("tool_calls")),
        "gpu_memory_fraction": placement.get("gpu_memory_fraction"),
        "placement_status": placement["status"],
    }


class Recorder:
    def __init__(self, path: Path):
        self.stream = path.open("w")

    def record(self, labels: dict, measurement: dict) -> None:
        record = labels | measurement
        self.stream.write(json.dumps(record) + "\n")
        self.stream.flush()
        described = " ".join(f"{key}={value}" for key, value in labels.items())
        print(
            f"{described} wall={record['wall_seconds']:.3f}s "
            f"load={record['load_duration_seconds']}s "
            f"prompt={record['prompt_eval_duration_seconds']}s/{record['prompt_eval_count']}tok "
            f"eval={record['eval_duration_seconds']}s/{record['eval_count']}tok "
            f"gpu_mem={record['gpu_memory_fraction']}",
            flush=True,
        )


def cold_tools(url: str, model: str, task: Task, recorder: Recorder) -> None:
    for repetition in range(REPETITIONS):
        order = (True, False) if repetition % 2 == 0 else (False, True)
        for with_tools in order:
            unload(url, model)
            for position in range(1 + WARM_REQUESTS):
                labels = {
                    "repetition": repetition,
                    "with_tools": with_tools,
                    "state": "cold" if position == 0 else "warm",
                    "position": position,
                }
                recorder.record(labels, timed_request(url, model, task, with_tools))


def warm_state(url: str, model: str, task: Task, recorder: Recorder) -> None:
    for repetition in range(REPETITIONS):
        recorder.record(
            {"repetition": repetition, "condition": "prime"}, timed_request(url, model, task, True)
        )
        for gap_seconds in IDLE_GAPS_SECONDS:
            time.sleep(gap_seconds)
            recorder.record(
                {"repetition": repetition, "condition": f"after_{gap_seconds}s_idle"},
                timed_request(url, model, task, True),
            )
        recorder.record(
            {"repetition": repetition, "condition": "other_prompt"},
            timed_request(url, model, task, True, OTHER_PROMPT),
        )
        recorder.record(
            {"repetition": repetition, "condition": "after_other_prompt"},
            timed_request(url, model, task, True),
        )


PROTOCOLS = {"cold-tools": cold_tools, "warm-state": warm_state}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", choices=sorted(PROTOCOLS), required=True)
    parser.add_argument("--task", default="fix-mean-empty", help="Agent runner task name")
    parser.add_argument("--model", default=os.environ.get("AGENTWATCH_LLM_MODEL", "qwen3:1.7b"))
    parser.add_argument(
        "--url",
        type=ollama_base_url,
        default=os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434"),
    )
    args = parser.parse_args()
    task = load_task(args.task)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output = Path(".data") / f"a01-first-request-{args.protocol}-{stamp}"
    output.mkdir(parents=True, exist_ok=False)
    (output / "protocol.json").write_text(
        json.dumps(
            {
                "protocol": args.protocol,
                "request_source": "agent runner first request",
                "task_id": task.task_id,
                "task_content_sha256": task.content_sha256,
                "model": args.model,
                "url": args.url,
                "repetitions": REPETITIONS,
                "warm_requests_per_block": WARM_REQUESTS,
                "idle_gaps_seconds": IDLE_GAPS_SECONDS,
                "other_prompt": OTHER_PROMPT,
                "options": DEFAULT_CHAT_OPTIONS,
                "cold_definition": "model unloaded with keep_alive=0; container not restarted",
                "wall_clock": "time.monotonic_ns around the HTTP request",
            },
            indent=2,
        )
        + "\n"
    )
    recorder = Recorder(output / "requests.jsonl")
    try:
        PROTOCOLS[args.protocol](args.url, args.model, task, recorder)
    except Exception as error:
        # Keep why the declared protocol stopped next to the requests it did complete.
        (output / "aborted.json").write_text(
            json.dumps({"aborted_at": datetime.now(UTC).isoformat(), "error": repr(error)}) + "\n"
        )
        raise
    finally:
        recorder.stream.close()
    print(f"Evidence: {output}")


if __name__ == "__main__":
    main()

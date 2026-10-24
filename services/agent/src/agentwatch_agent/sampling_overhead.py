"""Measure whether resource sampling distorts the inference timing it is meant to explain.

Protocol (declared before measurement, see `PROTOCOL`): the workload is the runner's first
model request for one task, sent unchanged to the real model server. Warm-up requests are
discarded so the model is loaded, then unsampled and sampled requests alternate (ABAB) so
drift affects both conditions equally. A sampled request runs the run's CPU and GPU probes
on their own thread at the run's interval, writing every record to a journal exactly as a
live run does; live delivery is excluded (agent-overhead-v1 measures it). Each sampled
request starts its own sampler, which reads once immediately, so sampling is front-loaded
relative to a long run.

Reported per condition: median and 90th-percentile runner wall time and Ollama's reported
total, prompt-evaluation and generation durations, the generated token counts (identical
output is not guaranteed at temperature 0), and the sampler's own per-round cost. No
pass/fail threshold is applied: the project has not declared an overhead budget. A failed or
interrupted measurement still returns everything measured, with its status and error.
"""

import hashlib
import json
import platform
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from agentwatch_sdk.live import ResourceRegistration
from agentwatch_worker.nvml import NvmlDevice

from agentwatch_agent.clock import MonotonicClock
from agentwatch_agent.config import DEFAULT_CHAT_OPTIONS, DEFAULT_SAMPLING_INTERVAL_SECONDS
from agentwatch_agent.delivery import Journal
from agentwatch_agent.model import chat_payload
from agentwatch_agent.overhead import merged_availability, summarize
from agentwatch_agent.prompts import SYSTEM_PROMPT, task_message
from agentwatch_agent.resources import CpuProbe, GpuProbe, ResourceSampler
from agentwatch_agent.tasks import Task
from agentwatch_agent.tools import tool_definitions

PROTOCOL = {
    "version": "resource-sampling-overhead-v1",
    "workload": "the runner's first chat request for the task, against the real model server",
    "warmup_requests": 3,
    "repetitions_per_condition": 30,
    "order": "alternating unsampled/sampled after warm-up",
    "sampled": "CPU and GPU probes at the run interval, journaled records, no live delivery",
    "statistics": "median and p90 of runner wall and Ollama durations; sampler round cost",
    "threshold": None,
}
CONDITIONS = ("unsampled", "sampled")
# Ollama's reported durations compared between conditions, in nanoseconds.
SERVER_DURATIONS = ("total_duration", "prompt_eval_duration", "eval_duration")


def first_request(task: Task, model: str) -> dict:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task_message(task)},
    ]
    return chat_payload(model, messages, list(tool_definitions(task)), DEFAULT_CHAT_OPTIONS)


def reply_digest(response: dict) -> str:
    """Identify what the model said: its text and called functions.

    Ollama gives every tool call a new random `id`, so hashing the whole message would make
    identical replies look different.
    """
    message = response.get("message") if isinstance(response, dict) else None
    if isinstance(message, dict):
        calls = message.get("tool_calls")
        message = {
            "content": message.get("content"),
            "tool_calls": [
                call.get("function") if isinstance(call, dict) else call for call in calls
            ]
            if isinstance(calls, list)
            else calls,
        }
    canonical = json.dumps(message, sort_keys=True, allow_nan=False)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def measure_sampling_overhead(
    task: Task,
    model: str,
    chat: Callable[[dict], dict],
    output_dir: Path,
    *,
    open_gpu: Callable[[], NvmlDevice] | None = NvmlDevice.open,
    repetitions: int = PROTOCOL["repetitions_per_condition"],
    warmup: int = PROTOCOL["warmup_requests"],
    sampling_interval_seconds: float = DEFAULT_SAMPLING_INTERVAL_SECONDS,
    proc: Path = Path("/proc"),
    cgroup_root: Path = Path("/sys/fs/cgroup"),
) -> dict:
    if repetitions < 1 or warmup < 0:
        raise ValueError("repetitions must be positive and warmup nonnegative")
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = first_request(task, model)
    clock = MonotonicClock()
    suffix = uuid4().hex[:12]
    host = ResourceRegistration(
        resource_id=f"tool-host-{suffix}",
        service_id="agentwatch-sampling-overhead",
        resource_class="cpu_sandbox",
    )
    gpu = ResourceRegistration(
        resource_id=f"gpu-{suffix}", service_id="nvidia-nvml", resource_class="inference"
    )
    measurements: list[dict] = []
    sampling: list[dict] = []
    journal = Journal(output_dir / "sampling-journal.jsonl")
    run_id = uuid4()  # identifies journaled records only; nothing is registered live

    def timed(condition: str, index: int) -> dict:
        started = time.monotonic_ns()
        response = chat(payload)
        wall = (time.monotonic_ns() - started) / 1e9
        return {
            "index": index,
            "condition": condition,
            "wall_seconds": wall,
            **{key: response.get(key) for key in SERVER_DURATIONS},
            "prompt_eval_count": response.get("prompt_eval_count"),
            "eval_count": response.get("eval_count"),
            "reply_digest": reply_digest(response),
        }

    def sampled(index: int) -> dict:
        probes = [CpuProbe(host, proc, cgroup_root)]
        if open_gpu is not None:
            probes.append(GpuProbe(gpu, open_gpu))
        sampler = ResourceSampler(
            lambda record: journal.write("record", record),
            run_id=run_id,
            task_id=task.task_id,
            clock=clock,
            probes=probes,
            interval_seconds=sampling_interval_seconds,
        )
        with sampler:
            measurement = timed("sampled", index)
        # A dead sampler would silently remove the work being measured.
        if sampler.error is not None:
            raise sampler.error
        sampling.append(sampler.summary())
        return measurement

    warmups: list[dict] = []
    status, error = "completed", None
    try:
        for index in range(warmup):
            warmups.append(timed("warmup", -1 - index))
        for index in range(2 * repetitions):
            condition = CONDITIONS[index % 2]
            measurements.append(
                sampled(index) if condition == "sampled" else timed(condition, index)
            )
    # Kept as failed or cancelled evidence instead of discarding what was measured.
    except KeyboardInterrupt:
        status, error = "cancelled", "Interrupted by the operator"
    except Exception as failure:  # noqa: BLE001
        status, error = "failed", f"{type(failure).__name__}: {failure}"[:500]
    finally:
        journal.close()
    return _report(
        task, model, payload, warmups, measurements, sampling, sampling_interval_seconds
    ) | {"status": status, "error": error}


def _seconds(measurements: list[dict], key: str) -> dict | None:
    values = [m[key] / 1e9 for m in measurements if isinstance(m.get(key), int)]
    return summarize(values) if values else None


def _report(
    task: Task,
    model: str,
    payload: dict,
    warmups: list[dict],
    measurements: list[dict],
    sampling: list[dict],
    sampling_interval_seconds: float,
) -> dict:
    by_condition = {c: [m for m in measurements if m["condition"] == c] for c in CONDITIONS}
    conditions = {}
    for condition, rows in by_condition.items():
        if not rows:
            conditions[condition] = None
            continue
        conditions[condition] = {
            "wall_seconds": summarize([m["wall_seconds"] for m in rows]),
            **{f"{key}_seconds": _seconds(rows, key) for key in SERVER_DURATIONS},
            "eval_counts": sorted({m["eval_count"] for m in rows}, key=str),
            "reply_digests": sorted({m["reply_digest"] for m in rows}),
        }
    walls = {
        c: conditions[c]["wall_seconds"]["median"] if conditions[c] else None for c in CONDITIONS
    }
    difference = None if None in walls.values() else walls["sampled"] - walls["unsampled"]
    costs = [s["round_cost"] for s in sampling]
    rounds = sum(s["sample_rounds"] for s in sampling)
    return {
        "protocol": PROTOCOL,
        "recorded_at": datetime.now(UTC).isoformat(),
        "task_id": task.task_id,
        "model": model,
        "request_sha256": hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest(),
        "host": {"system": platform.system(), "machine": platform.machine()},
        "sampling_interval_seconds": sampling_interval_seconds,
        "conditions": conditions,
        "median_wall_difference_seconds": difference,
        "median_wall_difference_fraction": (
            None if difference is None else difference / walls["unsampled"]
        ),
        "sampling": {
            "rounds": rounds,
            "availability_counts": merged_availability(sampling),
            "resources": sampling[0]["resources"] if sampling else [],
            "thread_cpu_total_seconds": sum(c["thread_cpu_total_seconds"] for c in costs),
            "thread_cpu_per_round_seconds": (
                sum(c["thread_cpu_total_seconds"] for c in costs) / rounds if rounds else None
            ),
            "round_wall_max_seconds": max(
                (c["wall_max_seconds"] for c in costs if c["wall_max_seconds"] is not None),
                default=None,
            ),
        },
        "warmups": warmups,
        "measurements": measurements,
        "limitations": [
            "One host, one model and one request; this is not an overhead budget",
            "Live delivery is excluded; agent-overhead-v1 measures it on the tool workload",
            (
                "Each sampled request starts its own sampler, which reads immediately, so "
                "sampling is denser at request start than in a long run"
            ),
            "Temperature 0 does not guarantee identical replies; compare eval counts",
        ],
    }

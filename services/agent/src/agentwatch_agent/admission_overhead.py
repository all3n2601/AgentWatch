"""Measure what runner-process admission adds to the model request it admits (O06).

Protocol (declared before measurement, see `PROTOCOL`): the workload is the runner's first
model request for one task, sent unchanged to the real model server, as `sampling-overhead`
sends it. Warm-up requests are discarded so the model is loaded, then `admission_off` and
`admission_on` requests alternate (ABAB) so drift affects both conditions equally.

- `admission_off` sends the request alone, as a run with `--no-policy` does.
- `admission_on` adds what a run that reads the policy adds around a model request: a policy
  watcher polling the policy service, and a permit acquired from the admission controller
  before the request and released after it. In a run the watcher has been polling since the
  run began, so a request starts at an arbitrary point of the poll cycle and overlaps a poll
  with a probability of about its duration over the poll interval. To reproduce that, each
  `admission_on` request starts its own watcher, waits an untimed, uniformly random phase of
  one poll interval (seeded and recorded), and only then starts the timer; the watcher stops
  after the timer. Each measurement records its phase and the polls that ran while it was
  timed.

The active policy is read once, before the first request, and applied to the controller; the
limits stay unchanged for the whole measurement (the contract's admission overhead gate is
"with unchanged limits"). Requests are sent one at a time, so a permit never waits for another.
A policy that changes during the measurement fails it. No resource sampling and no live
delivery run in either condition, so the difference between them is admission alone.

The measurement applies no pass/fail threshold; saved results are judged against the
contract's admission overhead budget by `agentwatch_data.overhead_verdict`. A failed or
interrupted measurement still returns everything measured, with its status and error.
"""

import hashlib
import json
import platform
import random
import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime
from uuid import uuid4

from agentwatch_agent.admission import (
    DEFAULT_POLICY_POLL_SECONDS,
    AdmissionController,
    AdmissionSettings,
    PolicySource,
    PolicyWatcher,
    load_policy,
)
from agentwatch_agent.clock import MonotonicClock
from agentwatch_agent.config import RUNNER_VERSION
from agentwatch_agent.delivery import RetryPolicy
from agentwatch_agent.sampling_overhead import (
    SERVER_DURATIONS,
    first_request,
    generated_tokens,
    reply_digest,
)
from agentwatch_agent.stats import summarize
from agentwatch_agent.tasks import Task

PROTOCOL = {
    "version": "admission-overhead-v1",
    "workload": "the runner's first chat request for the task, against the real model server",
    "warmup_requests": 3,
    "repetitions_per_condition": 30,
    "order": "alternating admission_off/admission_on after warm-up",
    "admission_on": (
        "a policy watcher polling the policy service, and an inference permit acquired before "
        "and released after the request, under the active policy, limits unchanged; the timed "
        "request starts at a seeded, uniformly random phase of the watcher's poll cycle"
    ),
    "statistics": "median and p90 of runner wall and server durations; admission waits",
    "threshold": None,
}
CONDITIONS = ("admission_off", "admission_on")
CONTROL = "inference_requests"
# Long enough for any model request; a permit never waits when requests are sequential.
PERMIT_DEADLINE_SECONDS = 600.0


class PolicyChanged(RuntimeError):
    """The active policy changed during the measurement; limits must stay unchanged."""


def measure_admission_overhead(
    task: Task,
    model: str,
    chat: Callable[[dict], dict],
    source: PolicySource,
    *,
    inference_service_id: str,
    repetitions: int = PROTOCOL["repetitions_per_condition"],
    warmup: int = PROTOCOL["warmup_requests"],
    policy_poll_seconds: float = DEFAULT_POLICY_POLL_SECONDS,
    settings: AdmissionSettings | None = None,
    retry: RetryPolicy | None = None,
    phase_seed: int | None = None,
    pause: Callable[[float], None] = time.sleep,
) -> dict:
    """Run the protocol; `source` is the policy service a run would read."""
    if repetitions < 1 or warmup < 0:
        raise ValueError("repetitions must be positive and warmup nonnegative")
    if phase_seed is None:
        phase_seed = random.SystemRandom().randrange(2**32)
    phases = random.Random(phase_seed)
    payload = first_request(task, model)
    controller = AdmissionController(
        settings or AdmissionSettings(), inference_service_id=inference_service_id
    )
    clock = MonotonicClock()
    measurements: list[dict] = []
    warmups: list[dict] = []
    admissions: list[dict] = []
    watchers: list[dict] = []
    policy = None

    def timed(condition: str, index: int, around: Callable[[Callable[[], dict]], dict]) -> dict:
        started = time.monotonic_ns()
        response = around(lambda: chat(payload))
        wall = (time.monotonic_ns() - started) / 1e9
        return {
            "index": index,
            "condition": condition,
            "wall_seconds": wall,
            **{key: response.get(key) for key in SERVER_DURATIONS},
            "prompt_eval_count": response.get("prompt_eval_count"),
            "eval_count": generated_tokens(response),
            "reply_digest": reply_digest(response),
        }

    def alone(send: Callable[[], dict]) -> dict:
        return send()

    def admitted_measurement(index: int) -> dict:
        watcher = PolicyWatcher(source, controller, policy_poll_seconds)
        held = []

        def admitted(send: Callable[[], dict]) -> dict:
            permit = controller.acquire(
                CONTROL,
                step_id=f"request-{index}",
                span_id=f"span-{index}",
                resource_id="model-server",
                service_id=inference_service_id,
                clock=clock,
                cancel=threading.Event(),
                deadline_monotonic=time.monotonic() + PERMIT_DEADLINE_SECONDS,
            )
            held.append(permit)
            try:
                return send()
            finally:
                permit.release()

        phase = phases.uniform(0, policy_poll_seconds)
        with watcher:
            # Untimed: the request then starts where a run's request would, at an arbitrary
            # point of the watcher's poll cycle (its first poll comes one interval after start).
            pause(phase)
            polls_before = watcher.polls
            measurement = timed("admission_on", index, admitted)
            polls_during = watcher.polls - polls_before
        measurement |= {"poll_phase_seconds": phase, "polls_during_request": polls_during}
        watchers.append(watcher.summary())
        if watcher.error is not None:
            raise watcher.error
        if controller.policy_version != policy.version:
            raise PolicyChanged(
                f"Policy went from version {policy.version} to {controller.policy_version}"
            )
        admissions.append(held[0].snapshot())
        return measurement

    status, error = "completed", None
    try:
        policy = load_policy(source, retry or RetryPolicy())
        controller.apply(policy)
        for index in range(warmup):
            warmups.append(timed("warmup", -1 - index, alone))
        for index in range(2 * repetitions):
            condition = CONDITIONS[index % 2]
            measurements.append(
                admitted_measurement(index)
                if condition == "admission_on"
                else timed(condition, index, alone)
            )
        # A change between two watcher polls would otherwise go unseen: read once more.
        final = load_policy(source, retry or RetryPolicy())
        if final.version != policy.version:
            raise PolicyChanged(f"Policy went from version {policy.version} to {final.version}")
    # Kept as failed or cancelled evidence instead of discarding what was measured.
    except KeyboardInterrupt:
        status, error = "cancelled", "Interrupted by the operator"
    except Exception as failure:  # noqa: BLE001
        status, error = "failed", f"{type(failure).__name__}: {failure}"[:500]
    return _report(task, model, payload, controller, policy, warmups, measurements) | {
        "policy_poll_seconds": policy_poll_seconds,
        "poll_phase_seed": phase_seed,
        "admissions": admissions,
        "policy_watchers": watchers,
        "status": status,
        "error": error,
    }


def _seconds(measurements: list[dict], key: str) -> dict | None:
    values = [m[key] / 1e9 for m in measurements if isinstance(m.get(key), int)]
    return summarize(values) if values else None


def _report(task, model, payload, controller, policy, warmups, measurements) -> dict:
    by_condition = {c: [m for m in measurements if m["condition"] == c] for c in CONDITIONS}
    conditions = {}
    for condition, rows in by_condition.items():
        conditions[condition] = (
            {
                "wall_seconds": summarize([m["wall_seconds"] for m in rows]),
                **{f"{key}_seconds": _seconds(rows, key) for key in SERVER_DURATIONS},
                "eval_counts": sorted({m["eval_count"] for m in rows}, key=str),
                "reply_digests": sorted({m["reply_digest"] for m in rows}),
            }
            if rows
            else None
        )
    if conditions["admission_on"]:
        conditions["admission_on"]["requests_overlapping_a_poll"] = sum(
            1 for m in by_condition["admission_on"] if m.get("polls_during_request")
        )
    walls = {
        c: conditions[c]["wall_seconds"]["median"] if conditions[c] else None for c in CONDITIONS
    }
    difference = None if None in walls.values() else walls["admission_on"] - walls["admission_off"]
    summary = controller.summary()
    # A policy without an inference entry fails `apply`; the report must still be written.
    named = policy.controlled_services.get(CONTROL) if policy is not None else None
    return {
        "protocol": PROTOCOL,
        "recorded_at": datetime.now(UTC).isoformat(),
        "task_id": task.task_id,
        "model": model,
        "request_sha256": hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest(),
        "runner_version": RUNNER_VERSION,
        "host": {"system": platform.system(), "machine": platform.machine()},
        "policy": None
        if policy is None
        else {
            "version": policy.version,
            "limits": policy.limits.model_dump(exclude_none=True),
            "inference_service": named.service_id if named is not None else None,
            "enforced": summary["controls"][CONTROL]["enforced"],
            "not_enforced_reason": summary["controls"][CONTROL]["not_enforced_reason"],
        },
        "controller": {"id": controller.controller_id, "open_permits": summary["open_permits"]},
        "conditions": conditions,
        "median_wall_difference_seconds": difference,
        "median_wall_difference_fraction": (
            None if difference is None else difference / walls["admission_off"]
        ),
        "warmups": warmups,
        "measurements": measurements,
        "measurement_id": str(uuid4()),
        "limitations": [
            "One host, one model and one request; this is not an overhead budget",
            (
                "Sequential requests never wait for a permit: queueing under contention is the "
                "experiment's subject, not this measurement's"
            ),
            (
                "Polling cost is sampled: a request overlaps a poll only when one falls inside "
                "it, so with requests shorter than the poll interval most admission_on requests "
                "include no poll; polls_during_request shows how many did"
            ),
            "Temperature 0 does not guarantee identical replies; compare eval counts",
            (
                "Each pair runs admission_off before admission_on (ABAB, not randomized), as in "
                "the A07 overhead protocols; drift within a pair favours neither by design but "
                "is not randomized away"
            ),
        ],
    }

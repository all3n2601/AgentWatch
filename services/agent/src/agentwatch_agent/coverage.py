"""Report how one run's resource samples cover its steps, from what the runner journaled.

This is producer-side evidence for checking sampling (method `coverage-v2`), not the
service's window join (B02). It reads `live-journal.jsonl` and `agent-evidence.json` and
applies the live contracts' rules:

- Only the run's own records count: a record for another run makes the journal unusable,
  and records in a clock domain the run never registered are excluded and counted.
- A step's window is `[started_tick_ns, finished_tick_ns)` in its own clock domain; a sample
  exactly at the finish tick is outside it, and samples in another domain never join.
- A step joins only its own registered resource (`own`). Other sampled resources are
  reported as `context`: host-wide PSI, the runner's whole cgroup and the whole GPU describe
  what else was happening, never the step's exclusive use.

Per resource, `status` uses the FeatureRow telemetry vocabulary: `present` (a round inside
the window), `no_overlap` (rounds in the domain, none inside), `unaligned` (rounds only in
another domain), `empty` (sampled by the runner, but no rounds were journaled) and `missing`
(the resource has no registered metric, such as the model server or the docs index).

A sampling round is one probe reading of one resource (one tick). Expected rounds are the
window's duration over the declared interval, rounded down; a step shorter than one interval
is flagged and has no coverage fraction rather than a misleading 0 or 1. Aggregates use
measured readings only: gauges report min/mean/max, and cumulative counters report their
increase across the window, or null if they decreased anywhere in it (a reset).

Each executed step also reports its observed dispatch queue (submission to start, runner
side) apart from service time where one was measured: Ollama's reported total for model
calls, whose own queue is unavailable, and the search alone for retrieval, whose index has
no queue. The remainder of execution is the runner's own handling, an estimate.
"""

import json
import statistics
from collections import defaultdict
from itertools import pairwise
from pathlib import Path

from agentwatch_sdk.live import ResourceSample, StepEvent
from pydantic import ValidationError

from agentwatch_agent.resources import COUNTER_METRICS, interval_summary

COVERAGE_METHOD = "coverage-v2"
RECORD_MODELS = {"resource_sample": ResourceSample, "step_event": StepEvent}
# The live contracts' timing-breakdown tolerance.
TIMING_TOLERANCE_SECONDS = 1e-6
Rounds = dict[tuple[str, int], list[dict]]


class CoverageError(ValueError):
    """The journal or evidence cannot support a coverage report."""


def read_journal(path: Path) -> tuple[dict | None, list[dict], int]:
    """The run's registration, its valid records deduplicated by identity, and the count of
    unusable lines.

    Raises CoverageError when one identity carries different content: which copy is true
    cannot be decided, so the journal cannot support a report.
    """
    registration = None
    records: dict[str, dict] = {}
    unusable = 0
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                unusable += 1
                continue
            if not isinstance(entry, dict):
                unusable += 1
                continue
            if entry.get("kind") == "registration" and isinstance(entry.get("payload"), dict):
                registration = entry["payload"]
                continue
            if entry.get("kind") != "record":
                continue
            payload = entry.get("payload")
            model = (
                RECORD_MODELS.get(payload.get("record_type")) if isinstance(payload, dict) else None
            )
            try:
                if model is None:
                    raise ValueError("not a step event or resource sample")
                model.model_validate(payload)
            except (ValidationError, ValueError):
                unusable += 1
                continue
            identity = str(payload.get("sample_id") or payload.get("event_id"))
            if records.setdefault(identity, payload) != payload:
                # Exact duplicates are retries; changed content under one identity is not.
                raise CoverageError(f"Journal has conflicting records for identity {identity}")
    return registration, list(records.values()), unusable


def _aggregate(metric: str, values: list[float]) -> dict:
    if not values:
        return {"measured": 0}
    if metric in COUNTER_METRICS:
        # A counter that went down anywhere in the window was reset: no delta, even if it
        # later climbed back above its first value, and never negative load.
        reset = any(later < earlier for earlier, later in pairwise(values))
        return {"measured": len(values), "increase": None if reset else values[-1] - values[0]}
    return {
        "measured": len(values),
        "min": min(values),
        "mean": statistics.fmean(values),
        "max": max(values),
    }


def _latest_events(records: list[dict]) -> dict[str, dict]:
    latest: dict[str, dict] = {}
    for record in records:
        if record.get("record_type") != "step_event":
            continue
        current = latest.get(record["step_id"])
        if current is None or record["sequence"] > current["sequence"]:
            latest[record["step_id"]] = record
    return latest


def _resource_window(
    rounds: Rounds | None, sampled: bool, clock_id: str, started: int, finished: int, expected: int
) -> dict:
    if not rounds:
        return {"status": "empty" if sampled else "missing"}
    in_domain = {tick: readings for (clock, tick), readings in rounds.items() if clock == clock_id}
    if not in_domain:
        return {"status": "unaligned"}
    inside = sorted(
        (tick, readings) for tick, readings in in_domain.items() if started <= tick < finished
    )
    by_metric: dict[str, list[float]] = defaultdict(list)
    availability: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for _tick, readings in inside:
        for reading in readings:
            availability[reading["metric"]][reading["availability"]] += 1
            if reading["availability"] == "measured":
                by_metric[reading["metric"]].append(reading["value"])
    return {
        "status": "present" if inside else "no_overlap",
        "rounds_in_window": len(inside),
        "coverage": min(len(inside) / expected, 1.0) if expected else None,
        "availability": {metric: dict(counts) for metric, counts in availability.items()},
        "measured": {metric: _aggregate(metric, by_metric[metric]) for metric in availability},
    }


def _difference(total: float | None, part: float | None) -> float | None:
    if total is None or part is None:
        return None
    difference = total - part
    return difference if difference >= -TIMING_TOLERANCE_SECONDS else None


def _timing(event: dict, call_timing: dict | None) -> dict:
    """Observed queue (runner dispatch) apart from the service time measured for the call."""
    submitted, started = event["submitted_tick_ns"], event["started_tick_ns"]
    execution = (event["finished_tick_ns"] - started) / 1e9
    timing = {
        "dispatch_queue_seconds": (started - submitted) / 1e9,
        "execution_seconds": execution,
    }
    if call_timing is None:
        return timing
    if event["step_type"] == "retrieval":
        service = call_timing.get("service_wall_seconds")
        queue = {
            "service_queue_seconds": None,
            "service_queue_availability": call_timing.get("index_queue_availability"),
            "service_queue_reason": call_timing.get("index_queue_reason"),
        }
    else:
        service = call_timing.get("server_total_seconds")
        queue = {
            "service_queue_seconds": call_timing.get("observed_queue_seconds"),
            "service_queue_availability": call_timing.get("observed_queue_availability"),
            "service_queue_reason": call_timing.get("observed_queue_reason"),
        }
    return timing | {
        "service_method": call_timing.get("method"),
        "service_seconds": service,
        **queue,
        # Execution outside the service: runner handling, an estimate across two timers.
        "outside_service_seconds": _difference(execution, service),
    }


def _call_timings(evidence: dict) -> dict[str, dict]:
    timings = {}
    for call in evidence.get("model_calls", []):
        if isinstance(call, dict) and isinstance(call.get("timing"), dict):
            timings[call.get("step_id")] = call["timing"]
    for call in evidence.get("tool_calls", []):
        details = call.get("evidence") if isinstance(call, dict) else None
        if isinstance(details, dict) and isinstance(details.get("timing"), dict):
            timings[call.get("step_id")] = details["timing"]
    return timings


def _step_coverage(
    event: dict,
    rounds: dict[str, Rounds],
    sampled: dict[str, str],
    declared: float,
    call_timing: dict | None,
) -> dict:
    started, finished = event.get("started_tick_ns"), event.get("finished_tick_ns")
    base = {
        "step_id": event["step_id"],
        "step_type": event["step_type"],
        "state": event["state"],
        "resource_id": event.get("resource_id"),
    }
    if started is None or finished is None:
        return base | {"window": None, "status": "not_executed"}
    # Integer nanoseconds: float floor division would expect 2 rounds in 0.3 s at 0.1 s.
    expected = (finished - started) // round(declared * 1e9)
    window = {
        "clock_id": event["clock_id"],
        "started_tick_ns": started,
        "finished_tick_ns": finished,
    }

    def resource(resource_id: str) -> dict:
        return _resource_window(
            rounds.get(resource_id),
            resource_id in sampled,
            event["clock_id"],
            started,
            finished,
            expected,
        )

    own_id = event.get("resource_id")
    own = (
        {"status": "no_resource"} if own_id is None else {"resource_id": own_id} | resource(own_id)
    )
    context = {
        resource_id: {"scope": scope} | resource(resource_id)
        for resource_id, scope in sampled.items()
        if resource_id != own_id
    }
    return base | {
        "window": window,
        "duration_seconds": (finished - started) / 1e9,
        "expected_rounds": expected,
        "shorter_than_interval": expected == 0,
        "status": own["status"],
        "own": own,
        "context": context,
        "timing": _timing(event, call_timing),
    }


def coverage_report(run_dir: Path) -> dict:
    evidence_path = run_dir / "agent-evidence.json"
    journal_path = run_dir / "live-journal.jsonl"
    for path in (evidence_path, journal_path):
        if not path.is_file():
            raise CoverageError(f"{path.name} is missing from {run_dir.name}")
    try:
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        declared = evidence["configuration"]["sampling_interval_seconds"]
    except (json.JSONDecodeError, KeyError, TypeError) as error:
        raise CoverageError(
            f"agent-evidence.json has no declared sampling interval: {error}"
        ) from error
    if isinstance(declared, bool) or not isinstance(declared, (int, float)) or declared <= 0:
        raise CoverageError("The declared sampling interval is not a positive number")
    run_id = evidence.get("run_id")
    if not isinstance(run_id, str):
        raise CoverageError("agent-evidence.json has no run ID (the run was never registered)")
    registration, records, unusable = read_journal(journal_path)
    if registration is None or not isinstance(registration.get("clock_ids"), list):
        raise CoverageError("live-journal.jsonl has no run registration with clock IDs")
    foreign = {record["run_id"] for record in records} - {run_id}
    if foreign:
        raise CoverageError(f"Journal has records for {len(foreign)} other run(s)")
    clocks = set(registration["clock_ids"])
    unregistered = [record for record in records if record["clock_id"] not in clocks]
    records = [record for record in records if record["clock_id"] in clocks]

    rounds: dict[str, Rounds] = defaultdict(lambda: defaultdict(list))
    for record in records:
        if record.get("record_type") == "resource_sample":
            rounds[record["resource_id"]][(record["clock_id"], record["tick_ns"])].append(record)
    sampling_summary = evidence.get("resource_sampling") or {}
    sampled = {
        entry["resource_id"]: entry.get("scope")
        for entry in sampling_summary.get("resources", [])
        if isinstance(entry, dict) and isinstance(entry.get("resource_id"), str)
    }
    # A resource with journaled samples counts as sampled even if the evidence omits it.
    sampled |= {resource_id: sampled.get(resource_id) for resource_id in rounds}
    sampling = {
        resource_id: interval_summary(sorted(tick for _clock, tick in by_tick), declared)
        | {"metrics": sorted({r["metric"] for group in by_tick.values() for r in group})}
        for resource_id, by_tick in rounds.items()
    }
    timings = _call_timings(evidence)
    steps = [
        _step_coverage(event, rounds, sampled, declared, timings.get(event["step_id"]))
        for event in sorted(_latest_events(records).values(), key=lambda e: e["submitted_tick_ns"])
    ]
    return {
        "method": COVERAGE_METHOD,
        "run_id": run_id,
        "task_id": evidence.get("task", {}).get("task_id"),
        "run_status": evidence.get("status"),
        "declared_interval_seconds": declared,
        "unusable_journal_lines": unusable,
        "unregistered_clock_records": len(unregistered),
        "registered_resources": [
            {"resource_id": r.get("resource_id"), "resource_class": r.get("resource_class")}
            for r in registration.get("resources", [])
            if isinstance(r, dict)
        ],
        "sampling": sampling,
        "steps": steps,
    }

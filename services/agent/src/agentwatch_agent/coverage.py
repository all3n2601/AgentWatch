"""Report how one run's resource samples cover its steps, from what the runner journaled.

This is producer-side evidence for checking sampling (method `coverage-v1`), not the
service's window join (B02). It reads only `live-journal.jsonl` and the run's declared
sampling interval, and applies the live contracts' rules: a step's window is
`[started_tick_ns, finished_tick_ns)` in the step's own clock domain, a sample joins only
inside that domain, and a sample exactly at the finish tick is outside the window.

A sampling round is one probe reading of one resource (one tick). Expected rounds are the
window's duration over the declared interval, rounded down, so steps shorter than one
interval have no coverage fraction rather than a misleading 0 or 1. Aggregates use measured
readings only: gauges report min/mean/max, and the cumulative throttling counter reports its
increase across the window, or null after a reset. Host-wide PSI, the runner's cgroup counter
and whole-device GPU readings are context for a step, never a step-exclusive measurement.
"""

import json
import statistics
from collections import defaultdict
from itertools import pairwise
from pathlib import Path

from agentwatch_sdk.live import ResourceSample, StepEvent
from pydantic import ValidationError

from agentwatch_agent.resources import THROTTLED_METRIC, interval_summary

COVERAGE_METHOD = "coverage-v1"
COUNTER_METRICS = {THROTTLED_METRIC[0]}
RECORD_MODELS = {"resource_sample": ResourceSample, "step_event": StepEvent}


class CoverageError(ValueError):
    """The journal or evidence cannot support a coverage report."""


def read_journal(path: Path) -> tuple[list[dict], int]:
    """Valid journaled records, deduplicated by identity, and the count of unusable lines.

    Raises CoverageError when one identity carries different content: which copy is true
    cannot be decided, so the journal cannot support a report.
    """
    records: dict[str, dict] = {}
    unreadable = 0
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                unreadable += 1
                continue
            if not isinstance(entry, dict) or entry.get("kind") != "record":
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
                unreadable += 1
                continue
            identity = str(payload.get("sample_id") or payload.get("event_id"))
            if records.setdefault(identity, payload) != payload:
                # Exact duplicates are retries; changed content under one identity is not.
                raise CoverageError(f"Journal has conflicting records for identity {identity}")
    return list(records.values()), unreadable


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


def _step_coverage(event: dict, samples: dict[str, dict[int, list[dict]]], declared: float) -> dict:
    started, finished = event.get("started_tick_ns"), event.get("finished_tick_ns")
    base = {
        "step_id": event["step_id"],
        "step_type": event["step_type"],
        "state": event["state"],
        "resource_id": event.get("resource_id"),
    }
    if started is None or finished is None:
        return base | {"window": None, "status": "not_executed"}
    duration = (finished - started) / 1e9
    # Integer nanoseconds: float floor division would expect 2 rounds in 0.3 s at 0.1 s.
    expected = (finished - started) // round(declared * 1e9)
    resources = {}
    for resource_id, rounds in samples.items():
        inside = sorted(
            (tick, readings)
            for tick, readings in rounds.items()
            if readings[0]["clock_id"] == event["clock_id"] and started <= tick < finished
        )
        by_metric: dict[str, list[float]] = defaultdict(list)
        availability: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for _tick, readings in inside:
            for reading in readings:
                availability[reading["metric"]][reading["availability"]] += 1
                if reading["availability"] == "measured":
                    by_metric[reading["metric"]].append(reading["value"])
        resources[resource_id] = {
            "rounds_in_window": len(inside),
            "coverage": min(len(inside) / expected, 1.0) if expected else None,
            "availability": {metric: dict(counts) for metric, counts in availability.items()},
            "measured": {
                metric: _aggregate(metric, by_metric.get(metric, [])) for metric in availability
            },
        }
    if expected == 0:
        status = "shorter_than_interval"
    elif any(entry["rounds_in_window"] for entry in resources.values()):
        status = "present"
    else:
        status = "empty"
    return base | {
        "window": {
            "clock_id": event["clock_id"],
            "started_tick_ns": started,
            "finished_tick_ns": finished,
        },
        "duration_seconds": duration,
        "expected_rounds": expected,
        "status": status,
        "resources": resources,
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
    records, unreadable = read_journal(journal_path)
    rounds: dict[str, dict[int, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        if record.get("record_type") == "resource_sample":
            rounds[record["resource_id"]][record["tick_ns"]].append(record)
    sampling = {
        resource_id: interval_summary(sorted(ticks), declared)
        | {"metrics": sorted({r["metric"] for group in ticks.values() for r in group})}
        for resource_id, ticks in rounds.items()
    }
    steps = [
        _step_coverage(event, rounds, declared)
        for event in sorted(_latest_events(records).values(), key=lambda e: e["submitted_tick_ns"])
    ]
    return {
        "method": COVERAGE_METHOD,
        "run_id": evidence.get("run_id"),
        "task_id": evidence.get("task", {}).get("task_id"),
        "run_status": evidence.get("status"),
        "declared_interval_seconds": declared,
        "unusable_journal_lines": unreadable,
        "sampling": sampling,
        "steps": steps,
    }

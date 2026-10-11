"""Turn validated runs into one row per step, with resource readings for each step window.

Rows carry identity, feature, rule, and provenance columns. Labels and release metadata are
added later by the labeling and build stages.
"""

import re
import statistics
from itertools import pairwise

from agentwatch_data.extract import TELEMETRY_PRESENT, RawRun
from agentwatch_data.schema import SCHEMA_VERSION

BASELINE_STEP_ID = re.compile(r"baseline-\d+")
NO_OVERLAP = "no_overlap"  # the run has samples, but none inside any of its steps


class FlattenError(ValueError):
    """A validated run whose structure cannot be turned into trustworthy step rows."""


def task_id(run: RawRun) -> str:
    """Interim task identity until agent step events carry their own task_id."""
    experiment = run.result["experiment"]
    return f"{run.kind}:{experiment['workload']}:{experiment['iterations']}"


def conditioned_steps(result: dict) -> list[tuple[str, dict]]:
    """Each step with the condition given by its position in the result.

    The step_id must agree with that position. Deriving the condition from the name alone
    would let a renamed contended step silently become a baseline reference.
    """
    steps = [("baseline", step) for step in result["baseline"]["steps"]]
    steps += [("clean", result["clean"]), ("contended", result["contended"])]
    for condition, step in steps:
        step_id = step["step_id"]
        matches = (
            BASELINE_STEP_ID.fullmatch(step_id) if condition == "baseline" else step_id == condition
        )
        if not matches:
            raise FlattenError(f"Step {step_id!r} is recorded in the {condition} position")
    return steps


def throttled_delta_usec(counters: list[int | None]) -> int | None:
    """Throttling gained across ordered counter readings, or None if it was not measured.

    A missing reading breaks the chain, and a decrease between consecutive readings means
    the counter reset; either way the window's total is unknown.
    """
    if len(counters) < 2 or None in counters:
        return None
    if any(later < earlier for earlier, later in pairwise(counters)):
        return None
    return counters[-1] - counters[0]


def sampling_interval_seconds(samples: list[dict]) -> float | None:
    """The run's actual sampling interval: the median gap between consecutive samples."""
    gaps = [(b["timestamp_ns"] - a["timestamp_ns"]) / 1e9 for a, b in pairwise(samples)]
    gaps = [gap for gap in gaps if gap > 0]
    return statistics.median(gaps) if gaps else None


def window_readings(
    samples: list[dict],
    submitted_ns: int,
    finished_ns: int,
    interval_seconds: float | None = None,
) -> dict:
    """Summarize samples inside the step window.

    Unavailable readings stay None with a False flag; they are never reported as zero.
    Unlike CpuSampler.summarize, every consecutive counter pair is checked for resets.
    """
    rows = sorted(
        (s for s in samples if submitted_ns <= s["timestamp_ns"] <= finished_ns),
        key=lambda s: s["timestamp_ns"],
    )
    pressures = [
        row["cpu_pressure"]["some"]["avg10"]
        for row in rows
        if row["cpu_pressure"] and "avg10" in row["cpu_pressure"].get("some", {})
    ]
    counters = [(row["cgroup_cpu_stat"] or {}).get("throttled_usec") for row in rows]
    throttled = throttled_delta_usec(counters)
    coverage = None
    if interval_seconds:
        expected = (finished_ns - submitted_ns) / 1e9 / interval_seconds
        coverage = min(1.0, len(rows) / expected) if expected > 0 else None
    return {
        "cpu_pressure_max_avg10": max(pressures) if pressures else None,
        "cpu_pressure_available": bool(pressures),
        "cgroup_throttled_usec_delta": throttled,
        "cgroup_available": throttled is not None,
        "telemetry_sample_count": len(rows),
        "telemetry_coverage": coverage,
    }


def failed_checks(result: dict) -> list[str]:
    return sorted(name for name, ok in result["checks"].items() if name != "passed" and not ok)


def agent_model(result: dict) -> str | None:
    agent = result.get("agent")
    model = agent.get("model") if isinstance(agent, dict) else None
    return model if isinstance(model, str) and model.strip() else None


def telemetry_status(run: RawRun, steps: list[tuple[str, dict]]) -> str:
    """present only when some sample falls inside some step's own window.

    The span from the first submission to the last finish also covers the idle gaps
    between steps, so samples taken only in those gaps describe no step.
    """
    if run.telemetry_status != TELEMETRY_PRESENT:
        return run.telemetry_status
    windows = [(step["submitted_ns"], step["finished_ns"]) for _, step in steps]
    overlaps = any(
        start <= sample["timestamp_ns"] <= end for sample in run.samples for start, end in windows
    )
    return TELEMETRY_PRESENT if overlaps else NO_OVERLAP


def flatten_run(run: RawRun) -> list[dict]:
    result = run.result
    experiment = result["experiment"]
    steps = conditioned_steps(result)
    failed = failed_checks(result)
    status = telemetry_status(run, steps)
    interval = sampling_interval_seconds(run.samples) if status == TELEMETRY_PRESENT else None
    shared = {
        "run_id": run.run_id,
        "task_id": task_id(run),
        "kind": run.kind,
        "source": run.source,
        "recorded_at": run.recorded_at,
        "workload_iterations": experiment["iterations"],
        "tests_per_step": experiment.get("tests_per_step"),
        "worker_capacity": experiment["worker_capacity"],
        "workload": experiment["workload"],
        "agent_model": agent_model(result),
        "run_passed": result["checks"]["passed"],
        "failed_checks": ",".join(failed) or None,
        "telemetry_status": status,
        "schema_version": SCHEMA_VERSION,
        "source_sha256": run.source_sha256,
    }
    return [
        shared
        | {
            "step_id": step["step_id"],
            "condition": condition,
            "queue_seconds": step["queue_seconds"],
            "execution_seconds": step["execution_seconds"],
            "duration_seconds": step["duration_seconds"],
            "cpu_seconds": step["cpu_seconds"],
            "queue_fraction": step["queue_fraction"],
            "rule_prediction": step["resource"],
            "rule_version": step["rule_version"],
        }
        | window_readings(run.samples, step["submitted_ns"], step["finished_ns"], interval)
        for condition, step in steps
    ]

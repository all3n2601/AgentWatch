"""Turn validated runs into one row per step, with resource readings for each step window.

Rows carry identity, feature, rule, and provenance columns. Labels and release metadata are
added later by the labeling and build stages.
"""

import logging
import math
import re
import statistics
from itertools import pairwise

from agentwatch_features.legacy import window_readings

from agentwatch_data.extract import TELEMETRY_PRESENT, RawRun, Rejection
from agentwatch_data.schema import SCHEMA_VERSION, brief, finite_number

logger = logging.getLogger(__name__)

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

    The worker runs steps one after another (baselines, then clean, then contended) on the
    same fixed work, so overlapping windows or differing output checksums mean the evidence
    cannot be compared: a baseline overlapping contention would bias the reference, and a
    step that did different work explains its own slowdown. The producer's
    same_workload_output check covers only clean against contended, so all are checked here.
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
    for (_, earlier), (_, later) in pairwise(steps):
        if earlier["finished_ns"] > later["submitted_ns"]:
            raise FlattenError(
                f"Step {later['step_id']!r} starts before {earlier['step_id']!r} finishes"
            )
    checksums = {step["checksum"] for _, step in steps}
    if len(checksums) != 1 or not next(iter(checksums)).strip():
        raise FlattenError(
            f"Steps report {len(checksums)} different output checksums; they did not all do "
            "the same work"
        )
    return steps


def declared_interval_seconds(result: dict) -> float | None:
    """The sampler's configured interval, as recorded in the step telemetry summaries.

    Step telemetry is an untyped extra field in the ingestion contract, so each recorded value
    is validated before any comparison: a list or table must become a FlattenError (a
    per-run rejection), never a TypeError that aborts the whole batch.
    """
    declared = []
    for name in ("clean", "contended"):
        telemetry = result[name].get("telemetry")
        if not isinstance(telemetry, dict) or "interval_seconds" not in telemetry:
            continue
        value = telemetry["interval_seconds"]
        interval = finite_number(value)
        if interval is None or interval <= 0:
            raise FlattenError(
                f"{name} declares sampling interval {brief(value)}, not a positive number"
            )
        # Coverage divides whole nanoseconds by it, so it must be at least 1 ns and finite
        # when scaled; 5e-324 passes the check above but its expected sample count is inf.
        if not math.isfinite(interval * 1e9) or interval * 1e9 < 1:
            raise FlattenError(
                f"{name} declares sampling interval {brief(value)}, outside 1 ns and above"
            )
        declared.append(interval)
    if len(set(declared)) > 1:
        raise FlattenError(f"Steps declare different sampling intervals: {sorted(set(declared))}")
    return declared[0] if declared else None


def sampling_interval_seconds(samples: list[dict]) -> float | None:
    """The run's measured sampling interval: the median gap between consecutive samples.

    Recorded for diagnosis only. It cannot detect uniform sample loss (gaps simply grow),
    so coverage is judged against the declared interval instead.
    """
    gaps = [(b["timestamp_ns"] - a["timestamp_ns"]) / 1e9 for a, b in pairwise(samples)]
    gaps = [gap for gap in gaps if gap > 0]
    return statistics.median(gaps) if gaps else None


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
    declared = declared_interval_seconds(result)
    present = status == TELEMETRY_PRESENT
    measured = sampling_interval_seconds(run.samples) if present else None
    interval = declared if present else None
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
        "sampling_interval_seconds": declared,
        "measured_sampling_interval_seconds": measured,
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


def flatten_runs(runs: list[RawRun]) -> tuple[list[dict], list[Rejection]]:
    """Flatten every run; a run whose structure is untrustworthy is rejected, not fatal."""
    rows, rejections = [], []
    for run in runs:
        try:
            rows.extend(flatten_run(run))
        except FlattenError as error:
            logger.warning("Rejected run %s at flattening: %s", run.run_id, error)
            rejections.append(Rejection(run.origin or f"run {run.run_id}", str(error)))
    return rows, rejections

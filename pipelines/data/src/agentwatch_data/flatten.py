"""Turn validated runs into one row per step, with resource readings for each step window.

Rows carry identity, feature, rule, and provenance columns. Labels and release metadata are
added later by the labeling and build stages.
"""

from agentwatch_data.extract import RawRun


def task_id(run: RawRun) -> str:
    """Interim task identity until agent step events carry their own task_id."""
    experiment = run.result["experiment"]
    return f"{run.kind}:{experiment['workload']}:{experiment['iterations']}"


def condition(step_id: str) -> str:
    return "baseline" if step_id.startswith("baseline-") else step_id


def window_readings(samples: list[dict], submitted_ns: int, finished_ns: int) -> dict:
    """Summarize samples inside the step window, matching CpuSampler.summarize.

    Unavailable readings stay None with a False flag; they are never reported as zero.
    """
    rows = [s for s in samples if submitted_ns <= s["timestamp_ns"] <= finished_ns]
    pressures = [
        row["cpu_pressure"]["some"]["avg10"]
        for row in rows
        if row["cpu_pressure"] and "avg10" in row["cpu_pressure"].get("some", {})
    ]
    stats = [row["cgroup_cpu_stat"] for row in rows if row["cgroup_cpu_stat"]]
    throttled = None
    if len(stats) >= 2 and all("throttled_usec" in stat for stat in (stats[0], stats[-1])):
        delta = stats[-1]["throttled_usec"] - stats[0]["throttled_usec"]
        throttled = delta if delta >= 0 else None  # a counter reset is not a measurement
    return {
        "cpu_pressure_max_avg10": max(pressures) if pressures else None,
        "cpu_pressure_available": bool(pressures),
        "cgroup_throttled_usec_delta": throttled,
        "cgroup_available": throttled is not None,
        "telemetry_sample_count": len(rows),
    }


def flatten_run(run: RawRun) -> list[dict]:
    result = run.result
    experiment = result["experiment"]
    steps = [*result["baseline"]["steps"], result["clean"], result["contended"]]
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
        "source_sha256": run.source_sha256,
    }
    return [
        shared
        | {
            "step_id": step["step_id"],
            "condition": condition(step["step_id"]),
            "queue_seconds": step["queue_seconds"],
            "execution_seconds": step["execution_seconds"],
            "duration_seconds": step["duration_seconds"],
            "cpu_seconds": step["cpu_seconds"],
            "queue_fraction": step["queue_fraction"],
            "rule_prediction": step["resource"],
            "rule_version": step["rule_version"],
        }
        | window_readings(run.samples, step["submitted_ns"], step["finished_ns"])
        for step in steps
    ]

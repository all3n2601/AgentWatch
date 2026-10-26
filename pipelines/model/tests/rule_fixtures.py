"""Synthetic step rows for rule-baseline unit tests.

These rows are made up to exercise the rules, metrics and gates. They are not recorded evidence
and never appear in a dataset release.
"""

from itertools import count

_steps = count()


def row(
    label: str = "none",
    condition: str = "clean",
    task_id: str = "task-a",
    *,
    queue_fraction: float = 0.0,
    psi: float | None = None,
    throttled_usec: int | None = None,
    duration_seconds: float = 1.0,
    excess: float = 0.0,
    run_id: str = "run-1",
) -> dict:
    return {
        "run_id": run_id,
        "step_id": f"{condition}-{next(_steps)}",
        "task_id": task_id,
        "dataset_version": "steps-v1.2+synthetic",
        "label_version": "labels-v1.1+synthetic",
        "condition": condition,
        "label_resource": label,
        "label_excess_fraction": excess,
        "queue_fraction": queue_fraction,
        "duration_seconds": duration_seconds,
        "cpu_pressure_max_avg10": psi,
        "cpu_pressure_available": psi is not None,
        "cgroup_throttled_usec_delta": throttled_usec,
        "cgroup_available": throttled_usec is not None,
    }

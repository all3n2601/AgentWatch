"""Synthetic step rows for the rules tests.

These rows are made up to exercise the thresholds. They are not recorded evidence.
"""


def row(
    *,
    queue_fraction: float = 0.0,
    psi: float | None = None,
    throttled_usec: int | None = None,
    duration_seconds: float = 1.0,
) -> dict:
    return {
        "queue_fraction": queue_fraction,
        "duration_seconds": duration_seconds,
        "cpu_pressure_max_avg10": psi,
        "cpu_pressure_available": psi is not None,
        "cgroup_throttled_usec_delta": throttled_usec,
        "cgroup_available": throttled_usec is not None,
    }

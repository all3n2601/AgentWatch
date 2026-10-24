"""`steps-v1.2` step-window readings over completed-run CPU sampler rows.

These definitions are frozen by the released `steps-v1.2` dataset (see docs/data-card.md):
windows run from submission to finish, both inclusive, over host sampler rows. Live records
use different window semantics and are versioned separately in `agentwatch_features.live`.
"""

from itertools import pairwise

LEGACY_FEATURE_VERSION = "steps-v1.2"


def throttled_delta_usec[Reading: (int, float)](counters: list[Reading | None]) -> Reading | None:
    """Throttling gained across ordered counter readings, or None if it was not measured.

    A missing reading breaks the chain, and a decrease between consecutive readings means
    the counter reset; either way the window's total is unknown.
    """
    if len(counters) < 2 or None in counters:
        return None
    if any(later < earlier for earlier, later in pairwise(counters)):
        return None
    return counters[-1] - counters[0]


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
    # As with the throttling counter, a reading missing from any sample voids the window:
    # a maximum over the samples that happened to carry PSI is not the step's maximum.
    if len(pressures) != len(rows):
        pressures = []
    counters = [(row["cgroup_cpu_stat"] or {}).get("throttled_usec") for row in rows]
    throttled = throttled_delta_usec(counters)
    coverage = None
    if interval_seconds:
        # Whole intervals inside the window: a healthy sampler delivers at least this many.
        # A step shorter than one interval cannot show whether samples were lost. Integer
        # nanoseconds, because floor(0.3 / 0.1) is 2 in floating point.
        expected = (finished_ns - submitted_ns) // round(interval_seconds * 1e9)
        coverage = min(1.0, len(rows) / expected) if expected > 0 else None
    return {
        "cpu_pressure_max_avg10": max(pressures) if pressures else None,
        "cpu_pressure_available": bool(pressures),
        "cgroup_throttled_usec_delta": throttled,
        "cgroup_available": throttled is not None,
        "telemetry_sample_count": len(rows),
        "telemetry_coverage": coverage,
    }

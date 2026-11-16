"""Judge achieved sampling against the declared coverage limits (A07).

The limits are declared in docs/instrumentation-overhead-protocol.md
(`instrumentation-overhead-acceptance-v1`) before any A07 measurement; a change needs a new
version. Inputs are `interval_summary` dicts, one per sampler (or per sampled target and clock
domain in a coverage report), so a verdict can always be recomputed from saved evidence.

Pooling across samplers: the median check uses the median of each sampler's own median gap
(gaps themselves are not saved), the maximum check the largest gap of any sampler, and the
missed fraction all missed rounds over all expected rounds. A sampler with fewer than two
rounds has no gap and is counted, not judged. With no gap at all the verdict is
`not_evaluated`, never `met`.
"""

import statistics

LIMITS = {
    "version": "instrumentation-overhead-acceptance-v1",
    # The achieved median gap stays within this fraction of the declared interval.
    "median_tolerance_fraction": 0.10,
    # No gap longer than this multiple of the declared interval.
    "max_interval_factor": 2.0,
    # Missed rounds as a share of the rounds the declared cadence expects.
    "max_missed_fraction": 0.01,
}
MET, NOT_MET, NOT_EVALUATED = "met", "not_met", "not_evaluated"
# The `interval_summary` fields a verdict needs; measurement reports save these per sampler.
INTERVAL_FIELDS = (
    "sample_rounds",
    "observed_interval_median_seconds",
    "observed_interval_p90_seconds",
    "observed_interval_max_seconds",
    "late_rounds",
    "early_rounds",
    "expected_rounds",
    "missed_rounds",
)


def interval_fields(summary: dict) -> dict:
    """One sampler's interval summary without its probes, availability or cost."""
    return {field: summary[field] for field in INTERVAL_FIELDS}


def sampling_record(summaries: list[dict], declared_seconds: float) -> dict:
    """Each sampler's intervals and their verdict, for a measurement report."""
    samplers = [interval_fields(summary) for summary in summaries]
    return {"samplers": samplers, "limits": judge_sampling(samplers, declared_seconds)}


def judge_sampling(summaries: list[dict], declared_seconds: float) -> dict:
    if not declared_seconds > 0:
        raise ValueError("The declared sampling interval must be positive")
    medians = [
        s["observed_interval_median_seconds"]
        for s in summaries
        if s.get("observed_interval_median_seconds") is not None
    ]
    maxima = [
        s["observed_interval_max_seconds"]
        for s in summaries
        if s.get("observed_interval_max_seconds") is not None
    ]
    base = {
        "limits_version": LIMITS["version"],
        "declared_interval_seconds": declared_seconds,
        "samplers": len(summaries),
        "samplers_without_interval": len(summaries) - len(medians),
        "late_rounds": sum(s.get("late_rounds", 0) for s in summaries),
    }
    if not medians:
        return base | {"verdict": NOT_EVALUATED, "checks": None, "failed_checks": []}
    tolerance = LIMITS["median_tolerance_fraction"] * declared_seconds
    low, high = declared_seconds - tolerance, declared_seconds + tolerance
    median = statistics.median(medians)
    longest = max(maxima)
    longest_allowed = LIMITS["max_interval_factor"] * declared_seconds
    missed = sum(s.get("missed_rounds", 0) for s in summaries)
    expected = sum(s.get("expected_rounds", 0) for s in summaries)
    missed_fraction = missed / expected if expected else None
    checks = {
        "interval_median": {
            "statistic": "median_of_sampler_medians",
            "value": median,
            "limit": [low, high],
            "met": low <= median <= high,
        },
        "interval_max": {
            "value": longest,
            "limit": longest_allowed,
            "met": longest <= longest_allowed,
        },
        "missed_fraction": {
            "value": missed_fraction,
            "missed_rounds": missed,
            "expected_rounds": expected,
            "limit": LIMITS["max_missed_fraction"],
            "met": missed_fraction is not None and missed_fraction <= LIMITS["max_missed_fraction"],
        },
    }
    failed = [name for name, check in checks.items() if not check["met"]]
    return base | {"verdict": NOT_MET if failed else MET, "checks": checks, "failed_checks": failed}


def short_steps(steps: list[dict]) -> dict:
    """Executed steps shorter than one declared interval: they get no coverage fraction."""
    executed = [step for step in steps if step.get("window") is not None]
    short = sum(bool(step.get("shorter_than_interval")) for step in executed)
    return {
        "executed": len(executed),
        "shorter_than_interval": short,
        "share": short / len(executed) if executed else None,
    }

"""Judge achieved sampling against the declared coverage limits (A07).

The limits are declared in docs/instrumentation-overhead-protocol.md
(`instrumentation-overhead-acceptance-v1`) before any A07 measurement; a change needs a new
version. Inputs are `interval_summary` dicts, one per sampler (or per sampled target and clock
domain in a coverage report), so a verdict can always be recomputed from saved evidence.

Pooling across samplers: the median check uses the median of each sampler's own median gap
(gaps themselves are not saved), the maximum check the largest gap of any sampler, and the
missed fraction all missed rounds over all expected rounds. A sampler with fewer than two
rounds has no gap and is counted, not judged. With fewer than `min_measured_gaps` gaps in
total the verdict is `not_evaluated`, never `met`. A resource that was configured for sampling
but recorded no sample makes the verdict `not_met`.
"""

import statistics
from collections.abc import Iterable

LIMITS = {
    "version": "instrumentation-overhead-acceptance-v1",
    # The achieved median gap stays within this fraction of the declared interval.
    "median_tolerance_fraction": 0.10,
    # No gap longer than this multiple of the declared interval.
    "max_interval_factor": 2.0,
    # Missed rounds as a share of the rounds the declared cadence expects.
    "max_missed_fraction": 0.01,
    # Fewer gaps than this, across all samplers, cannot support a verdict: `not_evaluated`.
    "min_measured_gaps": 10,
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


def judge_sampling(
    summaries: list[dict], declared_seconds: float, unsampled_resources: Iterable[str] = ()
) -> dict:
    """Judge interval summaries; `unsampled_resources` were configured but never sampled.

    Fields are read strictly: a summary without one is an error, never a measured zero.
    """
    if not declared_seconds > 0:
        raise ValueError("The declared sampling interval must be positive")
    measured = [s for s in summaries if s["observed_interval_median_seconds"] is not None]
    for summary in measured:
        if summary["observed_interval_max_seconds"] is None or summary["sample_rounds"] < 2:
            raise ValueError("An interval summary with a median gap lacks its maximum or rounds")
    gaps = sum(summary["sample_rounds"] - 1 for summary in measured)
    unsampled = sorted(set(unsampled_resources))
    base = {
        "limits_version": LIMITS["version"],
        "declared_interval_seconds": declared_seconds,
        "samplers": len(summaries),
        "samplers_without_interval": len(summaries) - len(measured),
        "measured_gaps": gaps,
        "unsampled_resources": unsampled,
        "late_rounds": sum(summary["late_rounds"] for summary in summaries),
    }
    checks = _checks(measured, summaries, declared_seconds) if measured else None
    failed = [name for name, check in (checks or {}).items() if not check["met"]]
    if unsampled:
        # A configured probe that never sampled is a coverage failure, whatever the others did.
        return base | {
            "verdict": NOT_MET,
            "reason": f"{len(unsampled)} configured resource(s) recorded no sample",
            "checks": checks,
            "failed_checks": ["unsampled_resources", *failed],
        }
    if gaps < LIMITS["min_measured_gaps"]:
        return base | {
            "verdict": NOT_EVALUATED,
            "reason": f"{gaps} measured gaps, fewer than {LIMITS['min_measured_gaps']}",
            "checks": checks,
            "failed_checks": [],
        }
    return base | {
        "verdict": NOT_MET if failed else MET,
        "reason": None,
        "checks": checks,
        "failed_checks": failed,
    }


def _checks(measured: list[dict], summaries: list[dict], declared_seconds: float) -> dict:
    tolerance = LIMITS["median_tolerance_fraction"] * declared_seconds
    low, high = declared_seconds - tolerance, declared_seconds + tolerance
    median = statistics.median(s["observed_interval_median_seconds"] for s in measured)
    longest = max(s["observed_interval_max_seconds"] for s in measured)
    longest_allowed = LIMITS["max_interval_factor"] * declared_seconds
    missed = sum(s["missed_rounds"] for s in summaries)
    expected = sum(s["expected_rounds"] for s in summaries)
    # A sampler that measured a gap expects at least one round, so this never divides by zero.
    missed_fraction = missed / expected
    return {
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
            "met": missed_fraction <= LIMITS["max_missed_fraction"],
        },
    }


def short_steps(steps: list[dict]) -> dict:
    """Executed steps shorter than one declared interval: they get no coverage fraction."""
    executed = [step for step in steps if step.get("window") is not None]
    short = sum(bool(step.get("shorter_than_interval")) for step in executed)
    return {
        "executed": len(executed),
        "shorter_than_interval": short,
        "share": short / len(executed) if executed else None,
    }

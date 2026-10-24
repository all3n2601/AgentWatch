"""Order statistics shared by every evidence summary, so each "p90" uses one definition.

The p90 is the sorted element at index `round(0.9 * (n - 1))`, as `agent-overhead-v1`
published it first. It is an observed value, never interpolated.
"""

import statistics


def p90(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[round(0.9 * (len(ordered) - 1))]


def summarize(values: list[float]) -> dict:
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "median": statistics.median(ordered),
        "p90": p90(ordered),
        "min": ordered[0],
        "max": ordered[-1],
    }

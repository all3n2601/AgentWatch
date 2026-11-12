"""Median of latency readings for the dashboard."""


def median(values):
    """Return the median of values."""
    ordered = sorted(values)
    middle = len(ordered) // 2
    return ordered[middle]

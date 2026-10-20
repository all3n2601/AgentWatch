"""Summary statistics used by the reporting service."""


def total(values):
    """Return the sum of values."""
    return sum(values)


def mean(values):
    """Return the arithmetic mean of values."""
    return sum(values) / len(values)

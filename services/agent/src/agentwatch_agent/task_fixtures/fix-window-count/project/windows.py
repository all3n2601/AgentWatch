"""Sliding windows over a sequence of readings."""


def windows(values, size):
    """Return every contiguous window of `size` values, in order."""
    if size <= 0:
        raise ValueError("size must be positive")
    return [values[start : start + size] for start in range(len(values) - size)]

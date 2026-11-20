"""The attribution class contract shared by training, evaluation and serving."""

import math
from typing import TypeGuard

RESOURCE_CLASSES = ("none", "cpu_sandbox", "inference", "retrieval")
NONE = "none"


def is_number(value: object) -> TypeGuard[int | float]:
    """A finite int or float; booleans are not numbers here."""
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)

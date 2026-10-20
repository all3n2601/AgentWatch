"""The runner's monotonic clock domain and wall-time observations."""

import time
from datetime import UTC, datetime
from uuid import uuid4

# live-contracts-v1 ticks must stay within JavaScript's safe integer range (~104 days of ns).
MAX_TICK_NS = 9_007_199_254_740_991


class MonotonicClock:
    """One clock domain per runner process lifetime, as live-contracts-v1 requires.

    Ticks are monotonic nanoseconds relative to this domain's origin. They are comparable only
    with other ticks from the same `clock_id`; `observed_at` wall time is a separate reading.
    """

    def __init__(self, clock_id: str | None = None):
        self.clock_id = clock_id or f"agent-runner-{uuid4()}"
        self._origin_ns = time.monotonic_ns()

    def tick(self) -> int:
        elapsed = time.monotonic_ns() - self._origin_ns
        if elapsed > MAX_TICK_NS:
            raise OverflowError("Clock domain exceeded the live tick range; start a new run")
        return elapsed

    def now(self) -> datetime:
        return datetime.now(UTC)

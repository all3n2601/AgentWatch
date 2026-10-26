"""Run one or more injectors together for one replay: isolated or mixed contention.

Entering starts every injector, then waits `ramp_seconds` so load is established before the
replay starts. If any injector fails to start, the ones already started are stopped before
the error propagates. Exiting stops every injector, even when an earlier one fails to stop;
an unconfirmed cleanup is raised after all of them were attempted, because contention that
may still be running must stop the experiment.
"""

import threading
from collections.abc import Sequence

from agentwatch_injectors.base import (
    BoundedInjector,
    InjectionClock,
    InjectorCleanupError,
    InjectorError,
)

SET_LOG_VERSION = "injection-set-v1"
MAX_RAMP_SECONDS = 30.0


class InjectionCancelled(InjectorError):
    """Cancellation was requested before contention finished ramping up."""


def stop_reason(exc_type: type[BaseException] | None) -> str:
    if exc_type is None:
        return "requested"
    if issubclass(exc_type, (KeyboardInterrupt, InjectionCancelled)):
        return "cancelled"
    return "failure"


class InjectionSet:
    def __init__(
        self,
        clock: InjectionClock,
        injectors: Sequence[BoundedInjector],
        *,
        ramp_seconds: float = 0.0,
        cancel: threading.Event | None = None,
    ):
        kinds = [injector.kind for injector in injectors]
        if len(set(kinds)) != len(kinds):
            raise ValueError(f"each injector kind may appear once, got {kinds}")
        if any(injector.clock is not clock for injector in injectors):
            raise ValueError("every injector must record ticks in the set's clock domain")
        if not 0 <= ramp_seconds <= MAX_RAMP_SECONDS:
            raise ValueError(f"ramp_seconds must be from 0 to {MAX_RAMP_SECONDS:g}")
        self.clock = clock
        self.injectors = tuple(injectors)
        self.ramp_seconds = ramp_seconds
        self.cancel = cancel or threading.Event()
        self._log: dict = {
            "log_version": SET_LOG_VERSION,
            "clock_id": clock.clock_id,
            "injectors": list(kinds),
            "ramp_seconds": ramp_seconds,
            "ramp_completed_tick_ns": None,
        }

    def __enter__(self):
        started: list[BoundedInjector] = []
        try:
            for injector in self.injectors:
                injector.start()
                started.append(injector)
            # Interruptible, so cancelling the experiment never waits out the ramp.
            if self.injectors and self.cancel.wait(self.ramp_seconds):
                raise InjectionCancelled("Cancelled while contention was ramping up")
        except BaseException as error:
            self._stop_all(started, stop_reason(type(error)), error)
            raise
        self._log["ramp_completed_tick_ns"] = self.clock.tick()
        return self

    def __exit__(self, exc_type, exc, _traceback):
        self._stop_all(self.injectors, stop_reason(exc_type), exc)

    def _stop_all(
        self, injectors: Sequence[BoundedInjector], reason: str, pending: BaseException | None
    ) -> None:
        failures: list[InjectorCleanupError] = []
        for injector in injectors:
            try:
                injector.stop(reason)
            except InjectorCleanupError as error:
                failures.append(error)
        if failures:
            if pending is not None:
                failures[0].add_note(f"Raised while handling {type(pending).__name__}: {pending}")
            raise failures[0]

    def log(self) -> dict:
        """The set's log with each injector's own `injection-log-v1` record."""
        return self._log | {
            "injections": [injector.log() for injector in self.injectors],
            "cleanup_confirmed": all(
                injector.log().get("cleanup", {}).get("confirmed") is True
                for injector in self.injectors
            ),
        }

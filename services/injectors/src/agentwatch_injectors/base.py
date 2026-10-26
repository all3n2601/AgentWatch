"""The lifecycle, bounds and evidence every contention injector shares.

An injector applies load to one designated resource from `start()` until `stop()`, and never
longer than its declared `max_duration_seconds`: a watchdog stops it at that bound even if the
caller never does. `start()` returns only once the load is running (`active_tick_ns`), waiting
at most `READY_TIMEOUT_SECONDS` for its workers, and the bound counts from then, so a slow
process start-up cannot silently shorten the contention. Stopping always runs the injector's
cleanup and records whether the cleanup was confirmed (every child process or thread is gone). An unconfirmed cleanup raises
`InjectorCleanupError` from `stop()`, every time it is called, so contention can never silently
outlive an experiment.

Each injector writes an `injection-log-v1` record. Its ticks come from the clock passed in,
normally the replayed run's own monotonic clock domain, so the injection window can be compared
with that run's step windows. The log is labeling metadata: it never becomes a model feature.
"""

import math
import threading
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Protocol

LOG_VERSION = "injection-log-v1"
# Absolute cap on any one injection; an experiment declares a shorter bound per replay.
MAX_INJECTION_SECONDS = 1800.0
# How long a stopped injector's workers get to report and exit before they are terminated,
# and how long each termination step then waits before escalating.
STOP_GRACE_SECONDS = 2.0
KILL_GRACE_SECONDS = 2.0
# Longest wait for an injector's workers to start applying load before start() fails.
READY_TIMEOUT_SECONDS = 15.0
# A child process stops itself this long after its own hard deadline (counted from its own
# start, so it also covers the ready wait), so load ends even if the supervising process was
# killed before it could stop the child.
BACKSTOP_GRACE_SECONDS = READY_TIMEOUT_SECONDS + 5.0
STOP_REASONS = ("requested", "duration_bound", "failure", "cancelled")


class InjectionClock(Protocol):
    """A monotonic clock domain: ticks compare only with ticks of the same `clock_id`."""

    clock_id: str

    def tick(self) -> int: ...
    def now(self) -> datetime: ...


class InjectorError(RuntimeError):
    """An injector was misused or could not start."""


class InjectorCleanupError(InjectorError):
    """An injector's workers could not be confirmed stopped; its capacity may still be loaded."""


def require_seconds(name: str, value: object, maximum: float) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0 < value <= maximum
    ):
        raise ValueError(f"{name} must be finite seconds in (0, {maximum:g}]")
    return float(value)


def require_count(name: str, value: object, maximum: int) -> int:
    if type(value) is not int or not 1 <= value <= maximum:
        raise ValueError(f"{name} must be an integer from 1 to {maximum}")
    return value


class BoundedInjector(ABC):
    """Start/stop lifecycle with a hard duration bound and confirmed cleanup.

    Subclasses declare `kind`, `method` and `resource_class`, return their bounds and target
    from `bounds()`/`target()`, start their workers in `_launch()`, and stop them in `_halt()`,
    which returns the measured activity and the cleanup evidence.
    """

    kind: str
    method: str
    resource_class: str

    def __init__(self, clock: InjectionClock, max_duration_seconds: float):
        self.clock = clock
        self.max_duration_seconds = require_seconds(
            "max_duration_seconds", max_duration_seconds, MAX_INJECTION_SECONDS
        )
        self._lock = threading.Lock()
        self._state = "idle"
        self._watchdog: threading.Timer | None = None
        self._log: dict = {}

    @abstractmethod
    def bounds(self) -> dict: ...

    @abstractmethod
    def target(self) -> dict: ...

    @abstractmethod
    def _launch(self) -> None:
        """Start every worker and return once all of them apply load; raise if they cannot."""

    @abstractmethod
    def _halt(self) -> tuple[dict, dict]:
        """Stop every worker; return (activity, cleanup) where cleanup has `confirmed`."""

    def start(self) -> None:
        with self._lock:
            if self._state != "idle":
                raise InjectorError(f"{self.kind} injector was already started")
            self._state = "active"
            self._log = {
                "log_version": LOG_VERSION,
                "injector": self.kind,
                "method": self.method,
                "contended_resource_class": self.resource_class,
                "target": self.target(),
                "bounds": {"max_duration_seconds": self.max_duration_seconds} | self.bounds(),
                "clock_id": self.clock.clock_id,
                "started_at": self.clock.now().isoformat(),
                "started_tick_ns": self.clock.tick(),
                "errors": [],
            }
            try:
                self._launch()
            except BaseException as error:
                self._log["errors"].append(f"start: {type(error).__name__}: {error}"[:500])
                self._shutdown("failure")
                if not self._log["cleanup"]["confirmed"]:
                    # Load may still be running: that outranks the start failure itself.
                    raise InjectorCleanupError(
                        f"{self.kind} injector failed to start and its cleanup was not "
                        f"confirmed: {self._log['cleanup']}"
                    ) from error
                raise
            self._log["active_tick_ns"] = self.clock.tick()
            self._watchdog = threading.Timer(self.max_duration_seconds, self._bound_reached)
            self._watchdog.name = f"{self.kind}-injector-watchdog"
            self._watchdog.daemon = True
            self._watchdog.start()

    def stop(self, reason: str = "requested") -> dict:
        """Stop (idempotently) and return the log; raises if cleanup was not confirmed."""
        if reason not in STOP_REASONS:
            raise ValueError(f"stop reason must be one of {STOP_REASONS}")
        with self._lock:
            if self._state == "idle":
                raise InjectorError(f"{self.kind} injector was never started")
            if self._state == "active":
                watchdog = self._watchdog
                if watchdog is not None and watchdog is not threading.current_thread():
                    watchdog.cancel()
                self._shutdown(reason)
            log = self.log()
        if not log["cleanup"]["confirmed"]:
            raise InjectorCleanupError(
                f"{self.kind} injector cleanup was not confirmed: {log['cleanup']}"
            )
        return log

    def log(self) -> dict:
        return dict(self._log)

    def _bound_reached(self) -> None:
        try:
            self.stop("duration_bound")
        # The watchdog thread has no caller to raise to; the unconfirmed cleanup stays in the
        # log and the caller's own stop() raises it.
        except InjectorCleanupError:
            pass

    def _shutdown(self, reason: str) -> None:
        self._log["stop_reason"] = reason
        self._log["stop_requested_tick_ns"] = self.clock.tick()
        try:
            activity, cleanup = self._halt()
        # Recorded as unconfirmed cleanup, which stop() raises; never treated as stopped.
        except Exception as error:  # noqa: BLE001
            self._log["errors"].append(f"stop: {type(error).__name__}: {error}"[:500])
            activity, cleanup = {}, {"confirmed": False}
        self._log |= {
            "stopped_tick_ns": self.clock.tick(),
            "stopped_at": self.clock.now().isoformat(),
            "activity": activity,
            "cleanup": cleanup,
        }
        self._state = "stopped"

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, _exc, _traceback):
        if exc_type is None:
            self.stop("requested")
        else:
            self.stop("cancelled" if issubclass(exc_type, KeyboardInterrupt) else "failure")

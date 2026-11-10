"""Runner-process admission limits for model requests and test jobs (O03, ADR 0010).

The O02 policy service stores versioned limits for the `coding-agent` target. This module
enforces them inside one runner process: every model request and test job asks the process's
`AdmissionController` for a permit before it starts, and gives it back on every terminal path.

- **Scope:** one controller per runner process counts that process's work only. Several
  runner processes each enforce the limit separately, so their combined concurrency can
  exceed it; nothing here is, or claims to be, a global limit.
- **Applicability:** a policy control applies only to the service it names. The inference
  limit controls the service the policy records (Ollama); a runner that sends its model
  requests elsewhere (vLLM) records the inference limit as not applicable instead of applying
  it to an unrelated workload.
- **Fairness and bounds:** each control has one first-in, first-out queue. An arrival is
  admitted only when nobody is queued ahead of it and a permit is free, so no request can
  overtake another, and the queue's length and each wait are bounded.
- **Changes:** a new policy version changes future admissions only. Lowering a limit never
  stops admitted work; new admissions wait until enough permits are returned. Disable or
  restore to the baseline removes the limit, and waiters are admitted in order.

Admission wait is recorded on the runner's clock, separately from the step's remaining
dispatch, the model server's own queue and execution time.
"""

import math
import os
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Literal, Protocol
from uuid import uuid4

from agentwatch_sdk.policy import RUNNER_SERVICE_ID, PolicyVersion

from agentwatch_agent.clock import MonotonicClock
from agentwatch_agent.delivery import RetryPolicy, is_retryable

ADMISSION_METHOD = "runner-admission-v1"
POLICY_TARGET = "coding-agent"
ENFORCEMENT_SCOPE = "runner_process"
Control = Literal["inference_requests", "test_jobs"]
CONTROLS: tuple[Control, ...] = ("inference_requests", "test_jobs")
RefusalOutcome = Literal["queue_full", "wait_exceeded", "cancelled", "run_deadline"]
DEFAULT_MAX_QUEUE_LENGTH = 64
MAX_QUEUE_LENGTH = 1024
# Waits longer than any model request's socket timeout (300 s) plus a test job, so a queue of
# a few runs at limit 1 is admissible; each run's own time limit still bounds its wait.
DEFAULT_MAX_WAIT_SECONDS = 600.0
MAX_WAIT_SECONDS = 3600.0
DEFAULT_POLICY_POLL_SECONDS = 1.0
# How often a waiter re-checks cancellation and its deadlines while no permit is returned.
WAIT_POLL_SECONDS = 0.05


class PolicyError(ValueError):
    """The policy service returned a policy this runner cannot enforce faithfully."""


class AdmissionRefused(Exception):
    """No permit was granted; `record` is the admission's evidence."""

    def __init__(self, outcome: RefusalOutcome, reason: str, record: dict):
        super().__init__(reason)
        self.outcome = outcome
        self.reason = reason
        self.record = record


@dataclass(frozen=True)
class QueueBounds:
    """One control's queue bounds, declared before a run (opt-contract-v1).

    `max_queue_length` counts requests waiting, not those admitted; 0 means a request that
    cannot be admitted at once is refused instead of queued.
    """

    max_queue_length: int = DEFAULT_MAX_QUEUE_LENGTH
    max_wait_seconds: float = DEFAULT_MAX_WAIT_SECONDS

    def __post_init__(self):
        if type(self.max_queue_length) is not int or not (
            0 <= self.max_queue_length <= MAX_QUEUE_LENGTH
        ):
            raise ValueError(f"max_queue_length must be an integer from 0 to {MAX_QUEUE_LENGTH}")
        if (
            isinstance(self.max_wait_seconds, bool)
            or not isinstance(self.max_wait_seconds, (int, float))
            or not math.isfinite(self.max_wait_seconds)
            or not 0 < self.max_wait_seconds <= MAX_WAIT_SECONDS
        ):
            raise ValueError(
                f"max_wait_seconds must be finite, positive and at most {MAX_WAIT_SECONDS:g}"
            )


@dataclass(frozen=True)
class AdmissionSettings:
    inference_requests: QueueBounds = field(default_factory=QueueBounds)
    test_jobs: QueueBounds = field(default_factory=QueueBounds)

    def bounds(self, control: Control) -> QueueBounds:
        return self.inference_requests if control == "inference_requests" else self.test_jobs


class PolicySource(Protocol):
    """The subset of agentwatch_sdk.LiveClient that reads the active policy."""

    def get_policy(self, target: str = POLICY_TARGET) -> PolicyVersion: ...


def load_policy(source: PolicySource, retry: RetryPolicy) -> PolicyVersion:
    """Read the active policy, retrying only transient failures (the read is idempotent)."""
    return retry.call(lambda: source.get_policy(POLICY_TARGET))


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass
class _ControlState:
    # None is no limit: the policy omits this control, or it does not apply to this runner.
    limit: int | None = None
    enforced: bool = False
    not_enforced_reason: str | None = "No policy applied yet"
    active: int = 0
    queue: deque = field(default_factory=deque)
    admitted: int = 0
    refused: dict[str, int] = field(default_factory=dict)
    longest_wait_seconds: float | None = None

    def has_capacity(self) -> bool:
        return self.limit is None or self.active < self.limit


class Permit:
    """One admitted request or test job; give it back with `release`.

    Release is idempotent. Work that continues on another thread after its caller stopped
    waiting (an abandoned model request) claims the permit first, so it is returned when that
    work actually ends and the slot is never handed out twice.
    """

    def __init__(self, controller: "AdmissionController", record: dict, clock: MonotonicClock):
        self._controller = controller
        self._record = record
        self._clock = clock
        self._admitted_ns = time.monotonic_ns()
        self._state: Literal["held", "claimed", "released"] = "held"

    @property
    def control(self) -> Control:
        return self._record["control"]

    def claim(self) -> bool:
        """Hand the permit to the thread doing the work; False if it was already returned."""
        with self._controller._condition:
            if self._state != "held":
                return False
            self._state = "claimed"
            return True

    def release(self) -> None:
        with self._controller._condition:
            if self._state == "released":
                return
            self._state = "released"
            self._record["released_tick_ns"] = self._clock.tick()
            self._record["hold_seconds"] = (time.monotonic_ns() - self._admitted_ns) / 1e9
            self._controller._return(self)

    def release_unless_claimed(self) -> None:
        """Return a permit whose work never started; claimed work returns it itself."""
        with self._controller._condition:
            if self._state == "held":
                self.release()

    def snapshot(self) -> dict:
        """A copy of this admission's evidence; the permit may still be held by claimed work."""
        with self._controller._condition:
            released = self._state == "released"
            return self._record | {
                "released": released,
                "held_by_unfinished_work": self._state == "claimed",
            }


class AdmissionController:
    """Process-wide admission for the `coding-agent` policy target; thread-safe.

    `inference_service_id` is the service this process sends model requests to; test jobs run
    in the runner itself. Create one controller per runner process and share it between that
    process's runs.
    """

    def __init__(
        self,
        settings: AdmissionSettings | None = None,
        *,
        inference_service_id: str,
        wait_poll_seconds: float = WAIT_POLL_SECONDS,
    ):
        self.settings = settings or AdmissionSettings()
        self.controller_id = f"admission-{uuid4()}"
        self.pid = os.getpid()
        self.governed_services: dict[Control, str] = {
            "inference_requests": inference_service_id,
            "test_jobs": RUNNER_SERVICE_ID,
        }
        self.wait_poll_seconds = wait_poll_seconds
        self._condition = threading.Condition()
        # Serializes reading and applying the policy, so versions are applied in the order the
        # service served them; admissions never wait on it.
        self._refresh_lock = threading.Lock()
        self._controls: dict[Control, _ControlState] = {c: _ControlState() for c in CONTROLS}
        self._policy: PolicyVersion | None = None
        self._history: list[dict] = []
        self._open: set[Permit] = set()

    # Policy ------------------------------------------------------------------------------

    @property
    def policy_version(self) -> int | None:
        with self._condition:
            return self._policy.version if self._policy is not None else None

    def apply(self, policy: PolicyVersion) -> bool:
        """Apply the active policy version; True if it changed the version in force."""
        if policy.target != POLICY_TARGET:
            raise PolicyError(f"Policy is for target {policy.target!r}, not {POLICY_TARGET!r}")
        if policy.enforcement_scope != ENFORCEMENT_SCOPE:
            raise PolicyError(f"Unsupported enforcement scope {policy.enforcement_scope!r}")
        missing = [c for c in CONTROLS if c not in policy.controlled_services]
        if missing:
            raise PolicyError(f"Policy names no controlled service for {', '.join(missing)}")
        with self._condition:
            current = self._policy
            if current is not None:
                if policy.version == current.version:
                    if policy != current:
                        raise PolicyError(f"Policy version {policy.version} changed content")
                    return False
                if policy.version < current.version:
                    # Versions only grow; an older active version means a reset service.
                    raise PolicyError(
                        f"Active policy went from version {current.version} back to "
                        f"{policy.version}"
                    )
            applied = {}
            for control in CONTROLS:
                state = self._controls[control]
                named = policy.controlled_services[control].service_id
                governed = self.governed_services[control]
                configured = getattr(policy.limits, control)
                if named == governed:
                    state.limit, state.enforced, state.not_enforced_reason = configured, True, None
                else:
                    state.limit, state.enforced = None, False
                    state.not_enforced_reason = (
                        f"Policy controls service {named!r}; this runner sends this work to "
                        f"{governed!r}"
                    )
                applied[control] = {
                    "configured_limit": configured,
                    "enforced": state.enforced,
                    "limit": state.limit,
                    "not_enforced_reason": state.not_enforced_reason,
                }
            self._policy = policy
            self._history.append(
                {
                    "version": policy.version,
                    "action": policy.action,
                    "restored_from": policy.restored_from,
                    "applied_at": _utc_now(),
                    "controls": applied,
                }
            )
            # A raised or removed limit may admit waiters; they re-check in order.
            self._condition.notify_all()
            return True

    def refresh(self, source: PolicySource, retry: RetryPolicy | None = None) -> bool:
        """Read the active policy and apply it; True if it changed the version in force."""
        with self._refresh_lock:
            policy = load_policy(source, retry) if retry else source.get_policy(POLICY_TARGET)
            return self.apply(policy)

    def ensure_policy(self, source: PolicySource, retry: RetryPolicy) -> None:
        """Apply the active policy unless one already is, before a run admits anything."""
        with self._refresh_lock:
            if self.policy_version is None:
                self.apply(load_policy(source, retry))

    # Admission ---------------------------------------------------------------------------

    def acquire(
        self,
        control: Control,
        *,
        step_id: str,
        span_id: str,
        resource_id: str | None,
        service_id: str | None,
        clock: MonotonicClock,
        cancel: threading.Event,
        deadline_monotonic: float,
    ) -> Permit:
        """Wait in order for a permit, or raise AdmissionRefused with the admission's record.

        Cancellation and the run deadline end the wait as well as the control's queue bounds.
        A refused or abandoned wait leaves the queue, so it never blocks the requests behind it.
        """
        if control not in CONTROLS:
            raise ValueError(f"Unknown admission control {control!r}")
        if service_id != self.governed_services[control]:
            raise ValueError(
                f"{control} admission is for service {self.governed_services[control]!r}, "
                f"not {service_id!r}"
            )
        bounds = self.settings.bounds(control)
        arrived_ns = time.monotonic_ns()
        wait_deadline = time.monotonic() + bounds.max_wait_seconds
        with self._condition:
            if self._policy is None:
                raise RuntimeError("No policy has been applied to this admission controller")
            state = self._controls[control]
            record = {
                "method": ADMISSION_METHOD,
                "controller_id": self.controller_id,
                "pid": self.pid,
                "enforcement_scope": ENFORCEMENT_SCOPE,
                "control": control,
                "step_id": step_id,
                "span_id": span_id,
                "resource_id": resource_id,
                "service_id": service_id,
                "clock_id": clock.clock_id,
                "arrived_tick_ns": clock.tick(),
                "policy_version_at_arrival": self._policy.version,
                "limit_at_arrival": state.limit,
                "active_at_arrival": state.active,
                "queued_ahead_at_arrival": len(state.queue),
                "max_queue_length": bounds.max_queue_length,
                "max_wait_seconds": bounds.max_wait_seconds,
            }
            if cancel.is_set():
                self._refuse(state, record, arrived_ns, clock, "cancelled")
            if time.monotonic() >= deadline_monotonic:
                self._refuse(state, record, arrived_ns, clock, "run_deadline")
            if not state.queue and state.has_capacity():
                return self._admit(state, record, arrived_ns, clock)
            if len(state.queue) >= bounds.max_queue_length:
                self._refuse(state, record, arrived_ns, clock, "queue_full")
            waiter = object()
            state.queue.append(waiter)
            try:
                while True:
                    if cancel.is_set():
                        self._refuse(state, record, arrived_ns, clock, "cancelled")
                    now = time.monotonic()
                    if now >= deadline_monotonic:
                        self._refuse(state, record, arrived_ns, clock, "run_deadline")
                    if state.queue[0] is waiter and state.has_capacity():
                        state.queue.popleft()
                        return self._admit(state, record, arrived_ns, clock)
                    if now >= wait_deadline:
                        self._refuse(state, record, arrived_ns, clock, "wait_exceeded")
                    remaining = min(deadline_monotonic, wait_deadline) - now
                    self._condition.wait(min(self.wait_poll_seconds, remaining))
            finally:
                # Refused or interrupted (including by an exception): leave the queue so the
                # next waiter becomes the head, and let it re-check.
                if waiter in state.queue:
                    state.queue.remove(waiter)
                self._condition.notify_all()

    def _admit(
        self, state: _ControlState, record: dict, arrived_ns: int, clock: MonotonicClock
    ) -> Permit:
        wait_seconds = (time.monotonic_ns() - arrived_ns) / 1e9
        state.active += 1
        state.admitted += 1
        if state.longest_wait_seconds is None or wait_seconds > state.longest_wait_seconds:
            state.longest_wait_seconds = wait_seconds
        record |= {
            "outcome": "admitted",
            "admitted_tick_ns": clock.tick(),
            "admission_wait_seconds": wait_seconds,
            "policy_version": self._policy.version,
            "limit": state.limit,
            "enforced": state.enforced,
            "not_enforced_reason": state.not_enforced_reason,
            "released_tick_ns": None,
            "hold_seconds": None,
        }
        permit = Permit(self, record, clock)
        self._open.add(permit)
        # With a limit above 1, the next waiter may be admissible too.
        self._condition.notify_all()
        return permit

    def _refuse(
        self,
        state: _ControlState,
        record: dict,
        arrived_ns: int,
        clock: MonotonicClock,
        outcome: RefusalOutcome,
    ) -> None:
        state.refused[outcome] = state.refused.get(outcome, 0) + 1
        control = record["control"]
        reasons = {
            "queue_full": (
                f"Admission refused: the {control} queue is full "
                f"({record['max_queue_length']} waiting)"
            ),
            "wait_exceeded": (
                f"Admission refused: waited longer than {record['max_wait_seconds']:g} s "
                f"for a {control} permit"
            ),
            "cancelled": f"Cancelled while waiting for a {control} permit",
            "run_deadline": f"Run time limit reached while waiting for a {control} permit",
        }
        refused = record | {
            "outcome": outcome,
            "refused_tick_ns": clock.tick(),
            "admission_wait_seconds": (time.monotonic_ns() - arrived_ns) / 1e9,
            "policy_version": self._policy.version,
            "limit": state.limit,
            "enforced": state.enforced,
            "not_enforced_reason": state.not_enforced_reason,
        }
        raise AdmissionRefused(outcome, reasons[outcome], refused)

    def _return(self, permit: Permit) -> None:
        # Called by Permit.release with the condition held.
        if permit in self._open:
            self._open.remove(permit)
            self._controls[permit.control].active -= 1
            self._condition.notify_all()

    # Evidence ----------------------------------------------------------------------------

    def configuration(self) -> dict:
        """The declared admission settings, part of a run's configuration digest."""
        return {
            "method": ADMISSION_METHOD,
            "policy_target": POLICY_TARGET,
            "enforcement_scope": ENFORCEMENT_SCOPE,
            "queue_bounds": {
                control: asdict(self.settings.bounds(control)) for control in CONTROLS
            },
        }

    def open_permits(self) -> int:
        with self._condition:
            return len(self._open)

    def wait_until_idle(self, timeout_seconds: float) -> bool:
        """Wait for every permit to be returned; False if some are still held at the timeout."""
        deadline = time.monotonic() + timeout_seconds
        with self._condition:
            while self._open:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._condition.wait(remaining)
            return True

    def summary(self) -> dict:
        """The process's admission state: limits, counts and every policy version applied."""
        with self._condition:
            return {
                "method": ADMISSION_METHOD,
                "controller_id": self.controller_id,
                "pid": self.pid,
                "enforcement_scope": ENFORCEMENT_SCOPE,
                "scope_note": (
                    "Counts this runner process only; other runner processes are not counted, "
                    "so limits are not global"
                ),
                "governed_services": dict(self.governed_services),
                "policy_version": self._policy.version if self._policy is not None else None,
                "policy_history": [dict(entry) for entry in self._history],
                "controls": {
                    control: {
                        "limit": state.limit,
                        "enforced": state.enforced,
                        "not_enforced_reason": state.not_enforced_reason,
                        "active": state.active,
                        "queued": len(state.queue),
                        "admitted": state.admitted,
                        "refused": dict(state.refused),
                        "longest_wait_seconds": state.longest_wait_seconds,
                    }
                    for control, state in self._controls.items()
                },
                "open_permits": len(self._open),
            }


class PolicyWatcher:
    """Poll the active policy and apply each new version to the controller.

    A failed poll keeps the version already in force and is counted: an unreachable policy
    service never silently becomes "no limits". A response the controller cannot apply, or a
    permanent error, stops polling and is kept in `error`, which fails the runs it serves.
    """

    def __init__(
        self,
        source: PolicySource,
        controller: AdmissionController,
        poll_seconds: float = DEFAULT_POLICY_POLL_SECONDS,
    ):
        if (
            isinstance(poll_seconds, bool)
            or not isinstance(poll_seconds, (int, float))
            or not math.isfinite(poll_seconds)
            or poll_seconds <= 0
        ):
            raise ValueError("poll_seconds must be finite and positive")
        self.source = source
        self.controller = controller
        self.poll_seconds = poll_seconds
        self.polls = 0
        self.poll_failures = 0
        self.last_error: str | None = None
        self.last_success_at: str | None = None
        self.error: BaseException | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="policy-watcher", daemon=True)

    def _run(self) -> None:
        while not self._stop.wait(self.poll_seconds):
            self.polls += 1
            try:
                self.controller.refresh(self.source)
            # Any other failure stops polling and is handed to the runs, which re-raise it.
            except Exception as error:  # noqa: BLE001
                if isinstance(error, PolicyError) or not is_retryable(error):
                    self.error = error
                    return
                # The version in force stays; the failure is counted and reported.
                self.poll_failures += 1
                self.last_error = f"{type(error).__name__}: {error}"[:500]
                continue
            self.last_success_at = _utc_now()

    def summary(self) -> dict:
        return {
            "poll_seconds": self.poll_seconds,
            "polls": self.polls,
            "poll_failures": self.poll_failures,
            "last_error": self.last_error,
            "last_success_at": self.last_success_at,
            "stopped_by_error": (
                f"{type(self.error).__name__}: {self.error}"[:500] if self.error else None
            ),
        }

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *_):
        self._stop.set()
        self._thread.join(timeout=30)


@dataclass(frozen=True)
class RunnerAdmission:
    """What a runner needs to admit its work: the process's controller, where its policy comes
    from, and the poller that keeps it current."""

    controller: AdmissionController
    source: PolicySource
    # None when the policy is read once and never refreshed (tests, fixed experiments).
    watcher: PolicyWatcher | None = None
    retry: RetryPolicy = field(default_factory=RetryPolicy)

    def ensure_policy(self) -> None:
        """Read and apply the active policy if no run of this process has yet."""
        self.controller.ensure_policy(self.source, self.retry)

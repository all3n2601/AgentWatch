"""Own the server-side run: registration, state changes and cancellation requests."""

import threading
from dataclasses import dataclass
from uuid import UUID, uuid4

from agentwatch_sdk.live import MAX_CURSOR, RunChange, RunCreate, RunSnapshot

from agentwatch_agent.delivery import (
    TRANSIENT_ERRORS,
    Journal,
    LiveRejectedError,
    LiveTransport,
    RetryPolicy,
    is_retryable,
)
from agentwatch_agent.redaction import redact_reason

TERMINAL_RUN_STATES = {"completed", "failed", "cancelled"}
REGISTRATION_UNCONFIRMED = "Runner could not confirm registration; no task work ran"


class RunSession:
    """A registered live run whose state this producer advances.

    State changes carry the expected revision. A 409 means another actor (normally a
    cancellation request) changed the run first; the session then adopts the server's state
    rather than retrying with a new identity.
    """

    def __init__(
        self,
        transport: LiveTransport,
        run_id: UUID,
        journal: Journal,
        retry: RetryPolicy,
    ):
        self.transport = transport
        self.run_id = run_id
        self.journal = journal
        self.retry = retry
        self.revision = 0
        self.state = "queued"
        self.reason: str | None = None

    @classmethod
    def open(
        cls,
        transport: LiveTransport,
        registration: RunCreate,
        journal: Journal,
        retry: RetryPolicy,
    ) -> "RunSession":
        # The same request_id is resent on every attempt: an uncertain first attempt that did
        # create the run returns the same run UUID instead of creating a duplicate.
        journal.write("registration", registration)
        try:
            created = retry.call(lambda: transport.create_run(registration))
        except TRANSIENT_ERRORS as error:
            if not is_retryable(error):
                raise
            # The service may have recorded the run before the response was lost. Journal a
            # closure so `resend`, whose exact-retry registration resolves that run, closes it
            # instead of leaving it queued forever. Revision 0 is the queued state.
            closure = RunChange(
                change_id=uuid4(),
                expected_revision=0,
                state="failed",
                reason=REGISTRATION_UNCONFIRMED,
            )
            journal.write("run_change", closure, deferred=True)
            raise
        journal.write("run_created", {"run_id": str(created.run_id), "created": created.created})
        return cls(transport, created.run_id, journal, retry)

    def change_state(self, state: str, reason: str | None) -> str:
        """Request a transition; return the run's resulting state.

        A 409 means the run changed first. If another actor closed it, that state is adopted.
        If only the revision moved (an earlier change applied but its response was lost), the
        transition is retried once at the refreshed revision under a new change identity.
        """
        for _ in range(2):
            if self.state in TERMINAL_RUN_STATES or self.state == state:
                return self.state
            change = self._change(state, reason)
            self.journal.write("run_change", change)
            try:
                result = self.retry.call(
                    lambda c=change: self.transport.change_state(self.run_id, c)
                )
            except LiveRejectedError as error:
                if error.status != 409:
                    raise
                self.refresh()
                if self.revision == change.expected_revision:
                    # Refused on content (e.g. unfinished steps), not a stale revision.
                    return self.state
                continue
            self.revision = result.revision
            self.state = state
            self.reason = change.reason
            return self.state
        return self.state

    def adopt_remote_closure(self, observed_state: str, observed_reason: str | None) -> str:
        """Adopt a closure another actor made, journaling it first so `resend` knows the target.

        The journal entry uses the state the watcher observed, so a refresh that then fails
        transiently still leaves a journal that `resend` can settle.
        """
        self.journal.write("remote_closure", {"state": observed_state, "reason": observed_reason})
        self.refresh()
        return self.state

    def defer_change(self, state: str, reason: str | None) -> None:
        """Journal a transition without sending it, for `resend` to apply after the records."""
        self.journal.write("run_change", self._change(state, reason), deferred=True)

    def _change(self, state: str, reason: str | None) -> RunChange:
        return RunChange(
            change_id=uuid4(),
            expected_revision=self.revision,
            state=state,
            reason=redact_reason(reason) if reason is not None else None,
        )

    def refresh(self) -> RunSnapshot:
        snapshot = self.retry.call(lambda: self.snapshot())
        self.revision = snapshot.revision
        self.state = snapshot.state
        self.reason = snapshot.reason
        return snapshot

    def snapshot(self) -> RunSnapshot:
        # Records after the largest cursor are empty, so this reads only the run and its steps.
        return self.transport.get_run(self.run_id, after=MAX_CURSOR, limit=1)


@dataclass(frozen=True)
class CancellationRequest:
    source: str
    reason: str
    # The run state the live API reported, for requests observed there.
    state: str | None = None


class CancellationState:
    """The first cancellation request wins; later ones keep the original source and reason."""

    def __init__(self):
        self.event = threading.Event()
        self.request: CancellationRequest | None = None
        # Reentrant, because a signal handler can interrupt the main thread inside cancel().
        self._lock = threading.RLock()
        self._signalled = False

    def cancel(self, source: str, reason: str, state: str | None = None) -> None:
        with self._lock:
            if self.request is None:
                self.request = CancellationRequest(source, reason, state)
        self.event.set()

    def cancel_from_signal(self, source: str, reason: str) -> None:
        """Signal-handler entry point. Only the first signal takes any lock.

        Event.set() uses a non-reentrant lock internally, so a repeated signal that interrupts
        the first handler must return without touching it. A plain attribute assignment is
        atomic with respect to Python signal handlers, which run between bytecodes.
        """
        if self._signalled:
            return
        self._signalled = True
        self.cancel(source, reason)


class CancellationWatcher:
    """Poll the run for a cancellation request and record it in a CancellationState.

    The service only records `cleanup_requested`; stopping work and releasing resources is
    this producer's job. Local operator signals use the same state directly.
    """

    def __init__(self, session: RunSession, state: CancellationState, poll_seconds: float):
        self.session = session
        self.state = state
        self.poll_seconds = poll_seconds
        self.poll_failures = 0
        self.last_error: str | None = None
        # A non-transient failure (e.g. a response that violates the contract) stops polling
        # and is re-raised by the runner instead of being hidden as "no cancellation".
        self.error: BaseException | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="cancel-watcher", daemon=True)

    def _run(self) -> None:
        while not self._stop.wait(self.poll_seconds) and not self.state.event.is_set():
            try:
                snapshot = self.session.snapshot()
            # Any other failure is handed to the runner, which re-raises it.
            except Exception as error:  # noqa: BLE001
                if not is_retryable(error):
                    self.error = error
                    return
                # A failed poll only delays cancellation; it is counted and reported.
                self.poll_failures += 1
                self.last_error = f"{type(error).__name__}: {error}"[:500]
                continue
            if snapshot.cleanup_requested or snapshot.state in {"failed", "cancelled"}:
                reason = snapshot.reason or f"Run was {snapshot.state} through the live API"
                self.state.cancel("live_api", reason, snapshot.state)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *_):
        self._stop.set()
        self._thread.join(timeout=30)

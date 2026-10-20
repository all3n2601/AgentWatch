"""Deliver live records with bounded retries, keeping a local journal of everything sent.

Every registration, record and run change is written to an append-only JSONL journal before
it is sent. The live API treats exact retries as idempotent, so a journal can be replayed
later (`python -m agentwatch_agent resend`) without duplicating evidence. Delivery problems
never invent or drop evidence silently: undelivered and rejected records are counted and
reported in the run evidence.
"""

import http.client
import json
import threading
import time
from collections.abc import Callable
from contextlib import closing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, TypeVar
from urllib.error import HTTPError, URLError
from uuid import UUID, uuid4

from agentwatch_sdk.live import (
    MAX_CURSOR,
    BatchResult,
    ChangeResult,
    CreateResult,
    LiveBatch,
    ResourceSample,
    RunChange,
    RunCreate,
    RunSnapshot,
    StepEvent,
)
from pydantic import BaseModel

MAX_BATCH_RECORDS = 256
MAX_PENDING_RECORDS = 20_000
MAX_ERROR_BODY_BYTES = 65_536
# 408/429/5xx can succeed on an identical retry; other statuses describe the request itself.
RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}

LiveRecord = StepEvent | ResourceSample
Result = TypeVar("Result")


class LiveTransport(Protocol):
    """The subset of agentwatch_sdk.LiveClient the runner uses."""

    def create_run(self, registration: RunCreate) -> CreateResult: ...
    def ingest(self, run_id: UUID, batch: LiveBatch) -> BatchResult: ...
    def change_state(self, run_id: UUID, change: RunChange) -> ChangeResult: ...
    def get_run(self, run_id: UUID, after: int = 0, limit: int = 256) -> RunSnapshot: ...


class LiveRejectedError(Exception):
    """The live API refused a request; resending identical content cannot succeed."""

    def __init__(self, status: int, detail: str):
        super().__init__(f"Live API returned {status}: {detail}")
        self.status = status
        self.detail = detail


def _error_detail(error: HTTPError) -> str:
    try:
        body = error.read(MAX_ERROR_BODY_BYTES)
        detail = json.loads(body).get("detail")
        return detail if isinstance(detail, str) else json.dumps(detail)[:2000]
    except (OSError, ValueError, AttributeError):
        return error.reason if isinstance(error.reason, str) else "no detail"


# Transport failures whose outcome is unknown; HTTPError (a URLError) is narrowed by status.
TRANSIENT_ERRORS = (URLError, TimeoutError, ConnectionError, http.client.HTTPException)


def is_retryable(error: BaseException) -> bool:
    if isinstance(error, HTTPError):
        return error.code in RETRYABLE_STATUS
    return isinstance(error, TRANSIENT_ERRORS)


@dataclass(frozen=True)
class RetryPolicy:
    attempts: int = 4
    initial_backoff_seconds: float = 0.25
    max_backoff_seconds: float = 4.0

    def __post_init__(self):
        if type(self.attempts) is not int or self.attempts < 1:
            raise ValueError("attempts must be a positive integer")
        if not 0 <= self.initial_backoff_seconds <= self.max_backoff_seconds:
            raise ValueError("backoff bounds must be ordered and nonnegative")

    def call(
        self, operation: Callable[[], Result], sleep: Callable[[float], None] = time.sleep
    ) -> Result:
        """Run `operation`, retrying only transient failures; the caller keeps identities fixed.

        Non-retryable HTTP statuses become LiveRejectedError. Response validation errors
        propagate unchanged: they mean client and server disagree on the contract.
        """
        backoff = self.initial_backoff_seconds
        for attempt in range(1, self.attempts + 1):
            try:
                return operation()
            except HTTPError as error:
                if not is_retryable(error):
                    raise LiveRejectedError(error.code, _error_detail(error)) from error
                if attempt == self.attempts:
                    raise
            except TRANSIENT_ERRORS:
                if attempt == self.attempts:
                    raise
            sleep(backoff)
            backoff = min(backoff * 2, self.max_backoff_seconds)
        raise AssertionError("unreachable")


class Journal:
    """Append-only JSONL of everything the runner sends, flushed line by line."""

    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()
        path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = path.open("a", encoding="utf-8")

    def write(self, kind: str, payload: BaseModel | dict, **extra) -> None:
        body = payload.model_dump(mode="json") if isinstance(payload, BaseModel) else payload
        line = json.dumps({"kind": kind, "payload": body} | extra, allow_nan=False)
        with self._lock:
            self._stream.write(line + "\n")
            self._stream.flush()

    def close(self) -> None:
        with self._lock:
            if not self._stream.closed:
                self._stream.close()


@dataclass
class DeliveryStats:
    emitted: int = 0
    delivered: int = 0
    pending: int = 0
    not_queued_overflow: int = 0
    rejected: int = 0
    transient_failures: int = 0
    last_error: str | None = None
    rejections: list[dict] | None = None

    @property
    def complete(self) -> bool:
        return self.pending == 0 and self.not_queued_overflow == 0 and self.rejected == 0


class LiveDelivery:
    """Queue records and deliver them in order, in batches, from any thread."""

    def __init__(
        self,
        transport: LiveTransport,
        run_id: UUID,
        journal: Journal,
        *,
        retry: RetryPolicy | None = None,
        sleep: Callable[[float], None] = time.sleep,
        max_pending: int = MAX_PENDING_RECORDS,
    ):
        self.transport = transport
        self.run_id = run_id
        self.journal = journal
        self.retry = retry or RetryPolicy()
        self._sleep = sleep
        self._max_pending = max_pending
        self._pending: list[LiveRecord] = []
        self._queue_lock = threading.Lock()
        self._send_lock = threading.Lock()
        self._stats = DeliveryStats(rejections=[])

    def emit(self, record: LiveRecord) -> None:
        self.journal.write("record", record)
        with self._queue_lock:
            self._stats.emitted += 1
            if len(self._pending) >= self._max_pending:
                # Memory stays bounded during a long outage; the journal still has the record.
                self._stats.not_queued_overflow += 1
                return
            self._pending.append(record)

    def flush(self) -> bool:
        """Send queued records in order. Return True when nothing remains queued.

        Stops at the first batch that still fails after bounded transient retries, keeping it
        and everything after it queued for the next flush. Batches are atomic, so a rejected
        batch is resent one record at a time: only the records the service actually refuses
        (e.g. a step first reported after a remote cancellation closed the run) are set aside
        and reported, and the rest of the evidence, such as late terminal events, still lands.
        """
        with self._send_lock:
            while True:
                with self._queue_lock:
                    batch = self._pending[:MAX_BATCH_RECORDS]
                if not batch:
                    return True
                try:
                    ingest_isolating(
                        self.transport,
                        self.run_id,
                        batch,
                        self.retry,
                        self._sleep,
                        on_delivered=self._settle,
                        on_rejected=lambda record, error: self._settle([record], rejected=error),
                    )
                except TRANSIENT_ERRORS as error:
                    return self._transient(error)

    def _transient(self, error: BaseException) -> bool:
        with self._queue_lock:
            self._stats.transient_failures += 1
            self._stats.last_error = f"{type(error).__name__}: {error}"[:500]
        return False

    def _settle(
        self, batch: list[LiveRecord], *, rejected: LiveRejectedError | None = None
    ) -> None:
        with self._queue_lock:
            # Only this method removes items, under the send lock, so the batch is the prefix.
            del self._pending[: len(batch)]
            if rejected is None:
                self._stats.delivered += len(batch)
                return
            self._stats.rejected += len(batch)
            self._stats.last_error = str(rejected)[:500]
            self._stats.rejections.append(
                {
                    "status": rejected.status,
                    "detail": rejected.detail[:500],
                    "record_ids": [str(_record_id(record)) for record in batch],
                }
            )
        self.journal.write(
            "rejected",
            {"status": rejected.status, "detail": rejected.detail},
            record_ids=[str(_record_id(record)) for record in batch],
        )

    def stats(self) -> DeliveryStats:
        with self._queue_lock:
            return DeliveryStats(
                emitted=self._stats.emitted,
                delivered=self._stats.delivered,
                pending=len(self._pending),
                not_queued_overflow=self._stats.not_queued_overflow,
                rejected=self._stats.rejected,
                transient_failures=self._stats.transient_failures,
                last_error=self._stats.last_error,
                rejections=list(self._stats.rejections),
            )


def ingest_isolating(
    transport: LiveTransport,
    run_id: UUID,
    records: list[LiveRecord],
    retry: RetryPolicy,
    sleep: Callable[[float], None],
    *,
    on_delivered: Callable[[list[LiveRecord]], None],
    on_rejected: Callable[[LiveRecord, LiveRejectedError], None],
) -> None:
    """Send records as one atomic batch; if the service refuses it, resend them one by one.

    Only records the service refuses on their own are reported as rejected, so one refused
    record cannot discard the rest of its batch. Progress is reported through the callbacks
    before a transient failure propagates, so callers can resume from the right record.
    """
    try:
        retry.call(lambda: transport.ingest(run_id, LiveBatch(records=records)), sleep)
    except LiveRejectedError as error:
        if len(records) == 1:
            on_rejected(records[0], error)
            return
        for record in records:
            try:
                retry.call(lambda r=record: transport.ingest(run_id, LiveBatch(records=[r])), sleep)
            except LiveRejectedError as record_error:
                on_rejected(record, record_error)
            else:
                on_delivered([record])
        return
    on_delivered(records)


def _record_id(record: LiveRecord) -> UUID:
    return record.event_id if isinstance(record, StepEvent) else record.sample_id


class PeriodicFlusher:
    """Deliver queued records in the background so instrumentation never blocks the agent."""

    def __init__(self, delivery: LiveDelivery, interval_seconds: float = 0.5):
        self.delivery = delivery
        self.interval_seconds = interval_seconds
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="live-flusher", daemon=True)
        self.error: BaseException | None = None

    def _run(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            try:
                self.delivery.flush()
            # Handed to the runner, which re-raises it; a dying thread must not hide the error.
            except Exception as error:  # noqa: BLE001
                self.error = error
                return

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *_):
        self.stop()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=30)


@dataclass
class ReplayReport:
    run_id: UUID | None = None
    records_sent: int = 0
    records_skipped_rejected: int = 0
    rejections: list[dict] = field(default_factory=list)
    changes_applied: int = 0
    changes_superseded: int = 0
    unresolved_changes: list[dict] = field(default_factory=list)
    expected_state: str | None = None
    final_state: str | None = None
    truncated_final_line: bool = False

    @property
    def complete(self) -> bool:
        """True when every record landed and the run is in the journaled (or a later) state."""
        run_settled = (
            self.expected_state is None
            or self.final_state == self.expected_state
            or (self.final_state in _TERMINAL and self.changes_superseded > 0)
        )
        return not self.rejections and not self.unresolved_changes and run_settled


_TERMINAL = {"completed", "failed", "cancelled"}


def _journal_entries(path: Path, report: ReplayReport) -> list[dict]:
    lines = path.read_text(encoding="utf-8").splitlines()
    entries = []
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            # Only an interrupted final write is tolerated; damage elsewhere is not guessed at.
            if number == len(lines):
                report.truncated_final_line = True
                continue
            raise ValueError(f"Journal line {number} is not valid JSON") from None
    return entries


def _replay_change(
    transport: LiveTransport,
    run_id: UUID,
    change: RunChange,
    retry: RetryPolicy,
    sleep: Callable[[float], None],
    journal: Journal,
    report: ReplayReport,
) -> None:
    """Apply one journaled run change, distinguishing why the service may refuse it."""
    for attempt in range(2):
        try:
            retry.call(lambda c=change: transport.change_state(run_id, c), sleep)
            report.changes_applied += 1
            return
        except LiveRejectedError as error:
            if error.status != 409:
                raise
            snapshot = retry.call(
                lambda: transport.get_run(run_id, after=MAX_CURSOR, limit=1), sleep
            )
            if snapshot.state == change.state:
                report.changes_applied += 1  # Applied earlier under another identity.
                return
            if snapshot.state in _TERMINAL:
                report.changes_superseded += 1  # Another actor, e.g. the dashboard, closed it.
                return
            if attempt == 0 and snapshot.revision != change.expected_revision:
                # Stale revision: an earlier change applied but its response was lost.
                change = RunChange(
                    change_id=uuid4(),
                    expected_revision=snapshot.revision,
                    state=change.state,
                    reason=change.reason,
                )
                journal.write("run_change", change, replayed=True)
                continue
            report.unresolved_changes.append(
                {"state": change.state, "status": error.status, "detail": error.detail[:500]}
            )
            return


def replay_journal(
    path: Path,
    transport: LiveTransport,
    retry: RetryPolicy | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> ReplayReport:
    """Resend a run journal with its original identities; already-saved evidence is a no-op.

    Records are resent like live delivery, isolating refused records instead of dropping
    their batch. Records the service already refused during the run are skipped. Afterwards
    the run's actual state is compared with the last journaled change, so a closure the
    service still refuses is reported rather than counted as done. Transient failures
    propagate after bounded retries; replay is idempotent, so it can simply be run again.
    """
    retry = retry or RetryPolicy()
    report = ReplayReport()
    entries = _journal_entries(path, report)
    rejected_ids = {
        record_id
        for entry in entries
        if entry.get("kind") == "rejected"
        for record_id in entry.get("record_ids", [])
    }
    pending: list[dict] = []

    def delivered(records: list[LiveRecord]) -> None:
        report.records_sent += len(records)

    def rejected(record: LiveRecord, error: LiveRejectedError) -> None:
        report.rejections.append(
            {
                "record_id": str(_record_id(record)),
                "status": error.status,
                "detail": error.detail[:500],
            }
        )

    def send_pending() -> None:
        while pending:
            chunk = pending[:MAX_BATCH_RECORDS]
            records = LiveBatch.model_validate({"records": chunk}).records
            ingest_isolating(
                transport,
                report.run_id,
                records,
                retry,
                sleep,
                on_delivered=delivered,
                on_rejected=rejected,
            )
            del pending[: len(chunk)]

    with closing(Journal(path)) as journal:
        for entry in entries:
            kind, payload = entry.get("kind"), entry.get("payload")
            if kind == "registration":
                registration = RunCreate.model_validate(payload)
                report.run_id = retry.call(
                    lambda r=registration: transport.create_run(r), sleep
                ).run_id
            elif kind == "run_created":
                if report.run_id is None or str(report.run_id) != payload.get("run_id"):
                    raise ValueError("Journal registration does not resolve to its recorded run")
            elif kind == "record":
                if report.run_id is None:
                    raise ValueError("Journal record precedes its run registration")
                record_id = payload.get("event_id") or payload.get("sample_id")
                if record_id in rejected_ids:
                    report.records_skipped_rejected += 1
                else:
                    pending.append(payload)
            elif kind == "run_change":
                send_pending()
                change = RunChange.model_validate(payload)
                report.expected_state = change.state
                _replay_change(transport, report.run_id, change, retry, sleep, journal, report)
            elif kind != "rejected":
                raise ValueError(f"Unknown journal entry kind {kind!r}")
        send_pending()
    if report.run_id is not None:
        snapshot = retry.call(
            lambda: transport.get_run(report.run_id, after=MAX_CURSOR, limit=1), sleep
        )
        report.final_state = snapshot.state
    return report

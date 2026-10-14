"""Bounded transactional live lifecycle; polling observes only validated step prefixes."""

from collections.abc import Iterator
from contextlib import contextmanager
from uuid import UUID, uuid4

from agentwatch_sdk.live import (
    TERMINAL_STATES,
    LiveBatch,
    ResourceSample,
    RunChange,
    RunCreate,
    StepEvent,
    step_timings,
    validate_step_identity,
    validate_transition,
)
from psycopg import Connection
from psycopg.types.json import Jsonb

from agentwatch_worker.store import ConflictError, RunStore

MAX_RECORDS = 100_000
MAX_STEPS = 1000
MAX_PENDING = 256
GAP_EXPIRY_SECONDS = 300


class MissingRunError(Exception):
    pass


class LiveStore:
    def __init__(self, store: RunStore):
        self.store = store

    @contextmanager
    def connect(self) -> Iterator[Connection]:
        with self.store.connect() as conn:
            conn.execute("SET LOCAL statement_timeout = '5s'")
            conn.execute("SET LOCAL lock_timeout = '3s'")
            yield conn

    @staticmethod
    def lock(conn: Connection, run_id: UUID) -> dict:
        row = conn.execute(
            "SELECT * FROM live_runs WHERE run_id=%s FOR UPDATE", (run_id,)
        ).fetchone()
        if row is None:
            raise MissingRunError("Live run not found")
        return row

    @staticmethod
    def expire(conn: Connection, run_id: UUID) -> None:
        # Expiry is durable evidence and is evaluated on reads/writes, never silently deleted.
        conn.execute(
            "UPDATE live_records SET disposition='expired' WHERE run_id=%s "
            "AND disposition='pending' AND received_at < now() - %s * interval '1 second'",
            (run_id, GAP_EXPIRY_SECONDS),
        )

    def create(self, registration: RunCreate) -> dict:
        payload = registration.model_dump(mode="json")
        with self.connect() as conn:
            # Serialize creation retries before assigning the server-owned run UUID.
            conn.execute("SELECT pg_advisory_xact_lock(730502)")
            row = conn.execute(
                "SELECT run_id,registration FROM live_runs WHERE request_id=%s",
                (registration.request_id,),
            ).fetchone()
            if row:
                if row["registration"] != payload:
                    raise ConflictError("Run creation request reused with different registration")
                return {"run_id": str(row["run_id"]), "created": False}
            run_id = uuid4()
            conn.execute(
                "INSERT INTO live_runs(run_id,request_id,registration) VALUES (%s,%s,%s)",
                (run_id, registration.request_id, Jsonb(payload)),
            )
        return {"run_id": str(run_id), "created": True}

    @staticmethod
    def validate_registration(row: dict, record: StepEvent | ResourceSample) -> None:
        registration = row["registration"]
        if str(record.run_id) != str(row["run_id"]) or record.task_id != registration["task_id"]:
            raise ConflictError("Record run/task identity mismatch")
        if record.clock_id not in registration["clock_ids"]:
            raise ValueError("Clock domain is not registered for this run")
        resources = {r["resource_id"]: r for r in registration["resources"]}
        services = {r["service_id"] for r in registration["resources"]}
        if record.service_id is not None and record.service_id not in services:
            raise ValueError("Service is not registered for this run")
        if record.resource_id is not None:
            resource = resources.get(record.resource_id)
            if resource is None or (
                record.service_id is not None and resource["service_id"] != record.service_id
            ):
                raise ValueError("Resource/service identity is not registered for this run")
            if (
                isinstance(record, ResourceSample)
                and resource["resource_class"] != record.resource_class
            ):
                raise ValueError("Sample resource class mismatch")

    @staticmethod
    def steps(conn: Connection, run_id: UUID) -> dict[str, StepEvent]:
        rows = conn.execute(
            "SELECT DISTINCT ON (step_id) step_id,payload FROM live_records "
            "WHERE run_id=%s AND record_type='step_event' AND disposition='applied' "
            "ORDER BY step_id,step_revision DESC",
            (run_id,),
        ).fetchall()
        return {r["step_id"]: StepEvent.model_validate(r["payload"]) for r in rows}

    @staticmethod
    def check_parents(conn: Connection, run_id: UUID) -> None:
        rows = conn.execute(
            "SELECT DISTINCT ON (step_id) step_id,payload->>'parent_step_id' AS parent "
            "FROM live_records WHERE run_id=%s AND record_type='step_event' "
            "ORDER BY step_id,step_revision",
            (run_id,),
        ).fetchall()
        parents = {r["step_id"]: r["parent"] for r in rows}
        if len(parents) > MAX_STEPS:
            raise ValueError("Run step limit exceeded")
        for step in parents:
            seen = set()
            current = step
            while current is not None:
                if current not in parents:
                    raise ValueError("Parent step must exist in the same run/batch")
                if current in seen:
                    raise ValueError("Step parent cycle")
                seen.add(current)
                current = parents[current]

    @staticmethod
    def apply_prefix(conn: Connection, run_id: UUID, step_id: str) -> None:
        rows = conn.execute(
            "SELECT record_id,payload,disposition FROM live_records "
            "WHERE run_id=%s AND step_id=%s ORDER BY step_revision",
            (run_id, step_id),
        ).fetchall()
        previous = None
        reference = None
        for row in rows:
            event = StepEvent.model_validate(row["payload"])
            if reference is not None:
                validate_step_identity(reference, event)
            if reference is None or event.started_tick_ns is not None:
                reference = event
        for row in rows:
            event = StepEvent.model_validate(row["payload"])
            expected = 0 if previous is None else previous.sequence + 1
            if event.sequence != expected or row["disposition"] == "expired":
                break
            validate_transition(previous, event)
            conn.execute(
                "UPDATE live_records SET disposition='applied' WHERE record_id=%s",
                (row["record_id"],),
            )
            previous = event

    def ingest(self, run_id: UUID, batch: LiveBatch) -> dict:
        with self.connect() as conn:
            row = self.lock(conn, run_id)
            self.expire(conn, run_id)
            known_steps = {
                r["step_id"]
                for r in conn.execute(
                    "SELECT DISTINCT step_id FROM live_records WHERE run_id=%s "
                    "AND record_type='step_event'",
                    (run_id,),
                ).fetchall()
            }
            added = 0
            applied_steps = self.steps(conn, run_id)
            ids = []
            touched_steps = set()
            for record in batch.records:
                self.validate_registration(row, record)
                payload = record.model_dump(mode="json")
                is_step = isinstance(record, StepEvent)
                record_id = record.event_id if is_step else record.sample_id
                existing = conn.execute(
                    "SELECT payload FROM live_records WHERE record_id=%s", (record_id,)
                ).fetchone()
                if existing:
                    if existing["payload"] != payload:
                        raise ConflictError("Record UUID reused with different evidence")
                    ids.append(record_id)
                    continue
                if is_step and (
                    row["state"] == "completed"
                    or (
                        record.step_id in applied_steps
                        and applied_steps[record.step_id].state in TERMINAL_STATES
                    )
                ):
                    raise ConflictError("Cannot revise completed step/run history")
                if (
                    is_step
                    and row["state"] in TERMINAL_STATES
                    and record.step_id not in known_steps
                ):
                    raise ConflictError("Cannot add a new step to a closed run")
                sample_key = (
                    None
                    if is_step
                    else [
                        record.resource_id,
                        record.service_id,
                        record.clock_id,
                        record.metric,
                        record.tick_ns,
                    ]
                )
                inserted = conn.execute(
                    "INSERT INTO live_records(record_id,run_id,record_type,step_id,step_revision,"
                    "sample_key,payload,disposition) VALUES (%s,%s,%s,%s,%s,%s,%s,%s) "
                    "ON CONFLICT DO NOTHING RETURNING record_id",
                    (
                        record_id,
                        run_id,
                        record.record_type,
                        record.step_id if is_step else None,
                        record.sequence if is_step else None,
                        Jsonb(sample_key) if sample_key is not None else None,
                        Jsonb(payload),
                        "pending" if is_step else "applied",
                    ),
                ).fetchone()
                if not inserted:
                    raise ConflictError("Record identity/revision/reading already exists")
                ids.append(record_id)
                added += 1
                if is_step:
                    touched_steps.add(record.step_id)
            self.check_parents(conn, run_id)
            for step_id in touched_steps:
                self.apply_prefix(conn, run_id, step_id)
            counts = conn.execute(
                "SELECT count(*) AS total,count(*) FILTER (WHERE disposition='pending') AS pending "
                "FROM live_records WHERE run_id=%s",
                (run_id,),
            ).fetchone()
            if counts["total"] > MAX_RECORDS or counts["pending"] > MAX_PENDING:
                raise ValueError("Run record/pending-buffer limit exceeded")
            if added:
                conn.execute("UPDATE live_runs SET updated_at=now() WHERE run_id=%s", (run_id,))
            acknowledgements = conn.execute(
                "SELECT record_id,disposition FROM live_records WHERE record_id=ANY(%s)", (ids,)
            ).fetchall()
            gaps = self.gaps(conn, run_id)
        return {
            "run_id": str(run_id),
            "added": added,
            "acknowledgements": acknowledgements,
            "gaps": gaps,
        }

    @staticmethod
    def gaps(conn: Connection, run_id: UUID) -> list[dict]:
        rows = conn.execute(
            "SELECT step_id,min(step_revision) AS first_pending,"
            "bool_or(disposition='expired') AS expired FROM live_records "
            "WHERE run_id=%s AND disposition IN ('pending','expired') "
            "GROUP BY step_id ORDER BY step_id",
            (run_id,),
        ).fetchall()
        states = LiveStore.steps(conn, run_id)
        return [
            {
                "step_id": r["step_id"],
                "missing_sequence": (
                    states[r["step_id"]].sequence + 1 if r["step_id"] in states else 0
                ),
                "first_pending_sequence": r["first_pending"],
                "expired": r["expired"],
            }
            for r in rows
        ]

    def change(self, run_id: UUID, change: RunChange) -> dict:
        payload = change.model_dump(mode="json")
        with self.connect() as conn:
            row = self.lock(conn, run_id)
            self.expire(conn, run_id)
            existing = conn.execute(
                "SELECT run_id,payload,revision FROM live_run_changes WHERE change_id=%s",
                (change.change_id,),
            ).fetchone()
            if existing:
                if str(existing["run_id"]) != str(run_id) or existing["payload"] != payload:
                    raise ConflictError("Run change UUID reused with different content")
                return {"run_id": str(run_id), "revision": existing["revision"], "changed": False}
            if row["revision"] != change.expected_revision:
                raise ConflictError("Run revision conflict; refresh the selected run")
            allowed = {"queued": {"running", "failed", "cancelled"}, "running": TERMINAL_STATES}
            if change.state not in allowed.get(row["state"], set()):
                raise ConflictError("Invalid run transition")
            steps = self.steps(conn, run_id)
            gaps = self.gaps(conn, run_id)
            unfinished = sorted(
                {s for s, e in steps.items() if e.state not in TERMINAL_STATES}
                | {g["step_id"] for g in gaps}
            )
            if change.state == "completed" and unfinished:
                raise ConflictError("Cannot complete a run with unfinished steps or event gaps")
            revision = row["revision"] + 1
            inserted = conn.execute(
                "INSERT INTO live_run_changes(change_id,run_id,payload,revision) VALUES (%s,%s,%s,%s) "
                "ON CONFLICT DO NOTHING RETURNING change_id",
                (change.change_id, run_id, Jsonb(payload), revision),
            ).fetchone()
            if inserted is None:
                raise ConflictError("Run change UUID/revision already exists")
            conn.execute(
                "UPDATE live_runs SET state=%s,revision=%s,reason=%s,cleanup_requested=%s,"
                "unfinished_steps=%s,updated_at=now() WHERE run_id=%s",
                (
                    change.state,
                    revision,
                    change.reason,
                    change.state == "cancelled",
                    Jsonb(unfinished if change.state in TERMINAL_STATES else []),
                    run_id,
                ),
            )
        return {"run_id": str(run_id), "revision": revision, "changed": True}

    def get(self, run_id: UUID, after: int = 0, limit: int = 256) -> dict:
        with self.connect() as conn:
            row = self.lock(conn, run_id)
            self.expire(conn, run_id)
            rows = conn.execute(
                "SELECT sequence,record_id,payload,disposition,received_at FROM live_records "
                "WHERE run_id=%s AND sequence>%s ORDER BY sequence LIMIT %s",
                (run_id, after, limit + 1),
            ).fetchall()
            changes = conn.execute(
                "SELECT payload,revision FROM live_run_changes WHERE run_id=%s ORDER BY revision",
                (run_id,),
            ).fetchall()
            states = self.steps(conn, run_id)
            return {
                **row,
                "cleanup_confirmed": None,
                "steps": [
                    {"event": e.model_dump(mode="json"), "timings": step_timings(e)}
                    for _, e in sorted(states.items())
                ],
                "changes": changes,
                "gaps": self.gaps(conn, run_id),
                "records": rows[:limit],
                "next_cursor": rows[limit - 1]["sequence"] if len(rows) > limit else None,
            }

    def list_runs(self, before: int | None, limit: int) -> dict:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT run_id,registration,state,revision,cleanup_requested,created_at,updated_at,"
                "sequence FROM live_runs WHERE (%s::bigint IS NULL OR sequence<%s) "
                "ORDER BY sequence DESC LIMIT %s",
                (before, before, limit + 1),
            ).fetchall()
        return {
            "runs": rows[:limit],
            "next_cursor": rows[limit - 1]["sequence"] if len(rows) > limit else None,
        }

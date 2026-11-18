"""Append-only step attributions of one live run, keyed by the configuration that served them.

A row holds one terminal step's attribution record (`build_attribution`) for one feature version
and one serving key: the rules version plus the served model's state, reason and identity. A new
rules file or model pair therefore appends rows beside the old ones instead of rewriting them.
A stored record is never rewritten. When a later pass would compute a different feature row
for the same step (a late sample inside its window), only `evidence_changed` is set.

Both stored timestamps come from the database clock: `step_terminal_received_at` is when the
step's terminal event was received, `committed_at` when the attribution row was written.
"""

import json
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime
from functools import partial
from uuid import UUID

from agentwatch_attribution.hashing import canonical_sha256
from agentwatch_attribution.rules import Rules
from agentwatch_features import FEATURE_VERSION, FeatureRow
from agentwatch_sdk.live import TERMINAL_STATES
from psycopg import Connection
from psycopg.types.json import Jsonb

from agentwatch_worker.attribution import build_attribution, predict_batch
from agentwatch_worker.attribution_model import ModelService
from agentwatch_worker.live_features import live_feature_rows
from agentwatch_worker.live_store import LiveStore, MissingRunError
from agentwatch_worker.store import RunStore

logger = logging.getLogger(__name__)
STEP_TERMINAL_STATES = sorted(TERMINAL_STATES)
SCORED_STEP_STATE = "completed"
# First key of the two-integer advisory lock serializing one run's inserts. The two-integer key
# space is separate from the single-bigint one the migration (730501), run-creation (730502) and
# policy (730503) locks use; the number is distinct so a grep finds one lock per number.
ATTRIBUTION_INSERT_LOCK_NAMESPACE = 730504
# Records reach the dashboard as strict JSON; a non-finite number must fail, not be stored.
strict_dumps = partial(json.dumps, allow_nan=False)


@dataclass(frozen=True)
class StepState:
    """A step's latest applied revision: its state and when that revision was received."""

    state: str
    received_at: datetime


@dataclass(frozen=True)
class StoredRow:
    """What a stored attribution row says about the evidence it was computed from."""

    sample_ids: frozenset[str]
    evidence_changed: bool


class AttributionStore:
    def __init__(self, store: RunStore, rules: Rules, model: ModelService) -> None:
        self.store = store
        self.live = LiveStore(store)
        self.rules = rules
        self.model = model
        identity = model.status.identity
        # Through JSON so tuples (class_order) are lists, as they are when read back.
        self.identity: dict[str, object] | None = (
            None if identity is None else json.loads(strict_dumps(asdict(identity)))
        )
        self.serving_key = canonical_sha256(
            {
                "rules": rules.version,
                "model": {
                    "state": model.status.state,
                    "reason": model.status.reason,
                    "identity": self.identity,
                },
            }
        )

    def model_status(self) -> dict[str, object]:
        """The serving model's status, as every attribution response reports it.

        `serving_key` is the key this store writes: rows carrying another key were served by a
        superseded rules version or model pair.
        """
        identity = self.model.status.identity
        return {
            "state": self.model.status.state,
            "reason": self.model.status.reason,
            "identity": None if self.identity is None else dict(self.identity),
            "not_promoted": None if identity is None else identity.blocks_promotion,
            "serving_key": self.serving_key,
        }

    def attribute_run(self, run_id: UUID, *, recheck_evidence: bool = True) -> int:
        """Store a row for every terminal step lacking one for this serving key; return how many.

        Features are computed only for the steps that need work: those lacking a row and, with
        `recheck_evidence`, those whose stored row is not yet flagged, so a late sample inside a
        stored step's window is noticed. Without it a pass costs only the pending steps. The
        feature read takes no lock on the run row (`live_feature_rows(lock=False)`), so neither a
        long pass nor a poll delays ingest.

        All pending rows are scored in one model call, outside any transaction. They are then
        inserted on one connection, each in its own short transaction: the insert's foreign-key
        check holds FOR KEY SHARE on the run row, which blocks `LiveStore.lock` (ingest, state
        changes, snapshots) only until that single insert commits. Concurrent passes on one run
        insert one at a time, so their rows commit in sequence order (see `insert`), and the
        unique key leaves one row per step. A step whose record cannot be built or stored
        (`ValueError`) is logged and left pending; the steps after it are still attributed. A
        failure of the feature read or of storage propagates; rows already committed stay.
        """
        interval = self.declared_interval_seconds(run_id)
        with self.session() as conn:
            with self.transaction(conn):
                steps = self.terminal_steps(conn, run_id)
                stored = self.stored_rows(conn, run_id)
            pending = [step_id for step_id in steps if step_id not in stored]
            audited = [
                step_id
                for step_id, row in stored.items()
                if recheck_evidence and step_id in steps and not row.evidence_changed
            ]
            wanted = sorted({*pending, *audited})
            if not wanted:
                return 0
            features = live_feature_rows(self.store, run_id, interval, step_ids=wanted, lock=False)
            new_rows = []
            for row in features.rows:
                known = stored.get(row.step_id)
                if known is None:
                    new_rows.append(row)
                elif {str(sample) for sample in row.source_sample_ids} != known.sample_ids:
                    with self.transaction(conn):
                        self.flag_evidence_changed(conn, run_id, row)
            return self.store_new_rows(conn, run_id, new_rows, steps)

    def store_new_rows(
        self, conn: Connection, run_id: UUID, rows: list[FeatureRow], steps: dict[str, StepState]
    ) -> int:
        scored = (
            {}
            if self.model.served is None
            else predict_batch(
                self.model.served,
                [row for row in rows if steps[row.step_id].state == SCORED_STEP_STATE],
            )
        )
        inserted = 0
        for row in rows:
            step = steps[row.step_id]
            try:
                record = build_attribution(
                    row,
                    step.state,
                    self.rules,
                    self.model.status,
                    self.model.served,
                    scored.get(row.step_id),
                )
                with self.transaction(conn):
                    inserted += self.insert(conn, run_id, row, record, step.received_at)
            except ValueError as error:
                logger.warning(
                    "Attribution skipped one step; it stays pending: type=%s", type(error).__name__
                )
        return inserted

    def has_pending_terminal_steps(self, run_id: UUID) -> bool:
        """Whether a terminal applied step lacks a row for this serving key; no feature work."""
        with self.live.connect() as conn:
            self.require_run(conn, run_id)
            row = conn.execute(
                "SELECT EXISTS (SELECT 1 FROM ("
                "SELECT DISTINCT ON (step_id) step_id,payload->>'state' AS state "
                "FROM live_records WHERE run_id=%s AND record_type='step_event' "
                "AND disposition='applied' ORDER BY step_id,step_revision DESC) step "
                "WHERE step.state=ANY(%s) AND NOT EXISTS ("
                "SELECT 1 FROM live_step_attributions a WHERE a.run_id=%s "
                "AND a.step_id=step.step_id AND a.feature_version=%s AND a.serving_key=%s)"
                ") AS pending",
                (run_id, STEP_TERMINAL_STATES, run_id, FEATURE_VERSION, self.serving_key),
            ).fetchone()
        return bool(row["pending"])

    def page(self, run_id: UUID, after: int, limit: int) -> dict:
        """Stored rows of the selected run after the `after` cursor, oldest first.

        Rows of every serving key are returned; each carries its own. `next_cursor` is set only
        when more rows exist.
        """
        if after < 0 or limit < 1:
            raise ValueError("after must be non-negative and limit at least 1")
        with self.live.connect() as conn:
            self.require_run(conn, run_id)
            rows = conn.execute(
                "SELECT sequence,serving_key,committed_at,step_terminal_received_at,"
                "evidence_changed,record FROM live_step_attributions "
                "WHERE run_id=%s AND sequence>%s ORDER BY sequence LIMIT %s",
                (run_id, after, limit + 1),
            ).fetchall()
        return {
            "run_id": str(run_id),
            "model": self.model_status(),
            "attributions": rows[:limit],
            "next_cursor": rows[limit - 1]["sequence"] if len(rows) > limit else None,
        }

    def declared_interval_seconds(self, run_id: UUID) -> float | None:
        """The registration's sampling interval, or None when the run was registered without it."""
        with self.live.connect() as conn:
            row = conn.execute(
                "SELECT registration->'sampling_interval_seconds' AS interval "
                "FROM live_runs WHERE run_id=%s",
                (run_id,),
            ).fetchone()
        if row is None:
            raise MissingRunError("Live run not found")
        interval = row["interval"]
        if interval is None:
            return None
        if isinstance(interval, bool) or not isinstance(interval, int | float):
            raise TypeError(f"Stored sampling_interval_seconds is not a number: {interval!r}")
        return float(interval)

    @staticmethod
    def require_run(conn: Connection, run_id: UUID) -> None:
        if conn.execute("SELECT 1 FROM live_runs WHERE run_id=%s", (run_id,)).fetchone() is None:
            raise MissingRunError("Live run not found")

    @contextmanager
    def session(self) -> Iterator[Connection]:
        """One connection for a whole pass; its transactions are explicit (`transaction`)."""
        with self.store.connect() as conn:
            conn.autocommit = True
            yield conn

    @staticmethod
    @contextmanager
    def transaction(conn: Connection) -> Iterator[Connection]:
        """A transaction with the live store's statement and lock timeouts."""
        with conn.transaction():
            LiveStore.bound_waits(conn)
            yield conn

    @staticmethod
    def terminal_steps(conn: Connection, run_id: UUID) -> dict[str, StepState]:
        """Each terminal step's latest applied revision: its state and when it was received."""
        rows = conn.execute(
            "SELECT DISTINCT ON (step_id) step_id,payload->>'state' AS state,received_at "
            "FROM live_records WHERE run_id=%s AND record_type='step_event' "
            "AND disposition='applied' ORDER BY step_id,step_revision DESC",
            (run_id,),
        ).fetchall()
        return {
            row["step_id"]: StepState(row["state"], row["received_at"])
            for row in rows
            if row["state"] in TERMINAL_STATES
        }

    def stored_rows(self, conn: Connection, run_id: UUID) -> dict[str, StoredRow]:
        """The stored rows of this serving key and feature version, by step."""
        rows = conn.execute(
            "SELECT step_id,evidence_changed,"
            "record->'feature_row'->'source_sample_ids' AS samples "
            "FROM live_step_attributions WHERE run_id=%s AND serving_key=%s AND feature_version=%s",
            (run_id, self.serving_key, FEATURE_VERSION),
        ).fetchall()
        return {
            row["step_id"]: StoredRow(frozenset(row["samples"]), row["evidence_changed"])
            for row in rows
        }

    def flag_evidence_changed(self, conn: Connection, run_id: UUID, row: FeatureRow) -> None:
        conn.execute(
            "UPDATE live_step_attributions SET evidence_changed=true WHERE run_id=%s "
            "AND step_id=%s AND feature_version=%s AND serving_key=%s AND NOT evidence_changed",
            (run_id, row.step_id, row.feature_version, self.serving_key),
        )

    def insert(
        self,
        conn: Connection,
        run_id: UUID,
        row: FeatureRow,
        record: dict[str, object],
        step_terminal_received_at: datetime,
    ) -> int:
        """1 if this pass wrote the row, 0 if a concurrent pass already had.

        Invariant: a run's rows commit in `sequence` order, so a poller that has read sequence n
        never later finds a smaller one. `sequence` is assigned at INSERT but visible only at
        COMMIT, so without it two passes on one run could commit out of order and a poller that
        advanced past the later row would skip the earlier one for good. The transaction first
        takes an advisory lock keyed on the run id, held until it commits, so the next insert
        for the run draws its sequence only after this one is visible. It never locks the
        `live_runs` row: ingest, state changes and polls are not delayed by it. Other runs use
        other keys; a hash collision only serializes two runs' inserts.
        """
        conn.execute(
            "SELECT pg_advisory_xact_lock(%s, hashtext(%s))",
            (ATTRIBUTION_INSERT_LOCK_NAMESPACE, str(run_id)),
        )
        inserted = conn.execute(
            "INSERT INTO live_step_attributions(run_id,step_id,feature_version,serving_key,"
            "record,step_terminal_received_at) VALUES (%s,%s,%s,%s,%s,%s) "
            "ON CONFLICT (run_id,step_id,feature_version,serving_key) DO NOTHING "
            "RETURNING sequence",
            (
                run_id,
                row.step_id,
                row.feature_version,
                self.serving_key,
                Jsonb(record, dumps=strict_dumps),
                step_terminal_received_at,
            ),
        ).fetchone()
        return 0 if inserted is None else 1

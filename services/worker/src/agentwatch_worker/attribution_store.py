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
from dataclasses import asdict
from datetime import datetime
from functools import partial
from uuid import UUID

from agentwatch_attribution.hashing import canonical_sha256
from agentwatch_attribution.rules import Rules
from agentwatch_features import FEATURE_VERSION, FeatureRow
from agentwatch_sdk.live import TERMINAL_STATES
from psycopg import Connection
from psycopg.types.json import Jsonb

from agentwatch_worker.attribution import build_attribution
from agentwatch_worker.attribution_model import ModelService
from agentwatch_worker.live_features import live_feature_rows
from agentwatch_worker.live_store import LiveStore, MissingRunError
from agentwatch_worker.store import RunStore

STEP_TERMINAL_STATES = sorted(TERMINAL_STATES)
# Records reach the dashboard as strict JSON; a non-finite number must fail, not be stored.
strict_dumps = partial(json.dumps, allow_nan=False)


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
        """The serving model's status, as every attribution response reports it."""
        identity = self.model.status.identity
        return {
            "state": self.model.status.state,
            "reason": self.model.status.reason,
            "identity": None if self.identity is None else dict(self.identity),
            "not_promoted": None if identity is None else identity.blocks_promotion,
        }

    def attribute_run(self, run_id: UUID) -> int:
        """Store a row for every terminal step lacking one for this serving key; return how many.

        Feature rows are computed first: `live_feature_rows` takes the run lock on its own
        connection, so it must not run while this store holds that lock elsewhere. The insert
        transaction relies on the unique key, so concurrent passes leave one row per step.
        Errors from building a record propagate and roll back the whole pass.
        """
        interval = self.declared_interval_seconds(run_id)
        features = live_feature_rows(self.store, run_id, interval)
        inserted = 0
        with self.live.connect() as conn:
            states = LiveStore.steps(conn, run_id)
            received = self.terminal_received_at(conn, run_id)
            stored = self.stored_sample_ids(conn, run_id)
            for row in features.rows:
                stored_samples = stored.get((row.step_id, row.feature_version))
                if stored_samples is not None:
                    if {str(s) for s in row.source_sample_ids} != stored_samples:
                        self.flag_evidence_changed(conn, run_id, row)
                    continue
                record = build_attribution(
                    row,
                    states[row.step_id].state,
                    self.rules,
                    self.model.status,
                    self.model.served,
                )
                inserted += self.insert(conn, run_id, row, record, received[row.step_id])
        return inserted

    def pending_terminal_steps(self, run_id: UUID) -> list[str]:
        """Terminal applied steps without a row for this serving key; no feature computation."""
        with self.live.connect() as conn:
            self.require_run(conn, run_id)
            rows = conn.execute(
                "SELECT step.step_id FROM ("
                "SELECT DISTINCT ON (step_id) step_id,payload->>'state' AS state "
                "FROM live_records WHERE run_id=%s AND record_type='step_event' "
                "AND disposition='applied' ORDER BY step_id,step_revision DESC) step "
                "WHERE step.state=ANY(%s) AND NOT EXISTS ("
                "SELECT 1 FROM live_step_attributions a WHERE a.run_id=%s "
                "AND a.step_id=step.step_id AND a.feature_version=%s AND a.serving_key=%s) "
                "ORDER BY step.step_id",
                (run_id, STEP_TERMINAL_STATES, run_id, FEATURE_VERSION, self.serving_key),
            ).fetchall()
        return [row["step_id"] for row in rows]

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

    @staticmethod
    def terminal_received_at(conn: Connection, run_id: UUID) -> dict[str, datetime]:
        """When each step's latest applied revision (its terminal one, once terminal) arrived."""
        rows = conn.execute(
            "SELECT DISTINCT ON (step_id) step_id,received_at FROM live_records "
            "WHERE run_id=%s AND record_type='step_event' AND disposition='applied' "
            "ORDER BY step_id,step_revision DESC",
            (run_id,),
        ).fetchall()
        return {row["step_id"]: row["received_at"] for row in rows}

    def stored_sample_ids(self, conn: Connection, run_id: UUID) -> dict[tuple[str, str], set[str]]:
        """The source sample ids of each stored row for this serving key."""
        rows = conn.execute(
            "SELECT step_id,feature_version,record->'feature_row'->'source_sample_ids' AS samples "
            "FROM live_step_attributions WHERE run_id=%s AND serving_key=%s",
            (run_id, self.serving_key),
        ).fetchall()
        return {(row["step_id"], row["feature_version"]): set(row["samples"]) for row in rows}

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
        """1 if this pass wrote the row, 0 if a concurrent pass already had."""
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

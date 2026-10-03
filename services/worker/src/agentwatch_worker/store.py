"""Transactional PostgreSQL history with immutable, idempotent evidence imports."""

from importlib.resources import files

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from agentwatch_worker.contracts import RunIngest, TelemetryBatch


class ConflictError(Exception):
    pass


class RunStore:
    def __init__(self, dsn: str):
        self.dsn = dsn

    def connect(self):
        return psycopg.connect(self.dsn, row_factory=dict_row, connect_timeout=3)

    def migrate(self):
        with self.connect() as conn:
            conn.execute("SELECT pg_advisory_xact_lock(730501)")
            conn.execute(
                "CREATE TABLE IF NOT EXISTS schema_migrations "
                "(version TEXT PRIMARY KEY, applied_at TIMESTAMPTZ DEFAULT now())"
            )
            for migration in sorted(
                files("agentwatch_worker").joinpath("migrations").iterdir(), key=lambda p: p.name
            ):
                if (
                    migration.name.endswith(".sql")
                    and not conn.execute(
                        "SELECT version FROM schema_migrations WHERE version=%s", (migration.name,)
                    ).fetchone()
                ):
                    conn.execute(migration.read_text(), prepare=False)
                    conn.execute(
                        "INSERT INTO schema_migrations(version) VALUES (%s)", (migration.name,)
                    )

    def ready(self):
        with self.connect() as conn:
            conn.execute("SELECT 1 FROM runs LIMIT 1")

    @staticmethod
    def insert_telemetry(conn, run_id, batch: TelemetryBatch):
        added = {"spans": 0, "samples": 0}
        for span in batch.spans:
            payload = span.model_dump(mode="json")
            params = (run_id, span.context.trace_id, span.context.span_id)
            inserted = conn.execute(
                "INSERT INTO spans(run_id, trace_id, span_id, payload) VALUES (%s,%s,%s,%s) "
                "ON CONFLICT DO NOTHING RETURNING span_id",
                (*params, Jsonb(payload)),
            ).fetchone()
            if inserted:
                added["spans"] += 1
            elif (
                conn.execute(
                    "SELECT payload FROM spans WHERE run_id=%s AND trace_id=%s AND span_id=%s",
                    params,
                ).fetchone()["payload"]
                != payload
            ):
                raise ConflictError("Span identity already exists with different evidence")
        for sample in batch.samples:
            payload = sample.model_dump(mode="json")
            params = (run_id, sample.timestamp_ns)
            inserted = conn.execute(
                "INSERT INTO samples(run_id, timestamp_ns, payload) VALUES (%s,%s,%s) "
                "ON CONFLICT DO NOTHING RETURNING timestamp_ns",
                (*params, Jsonb(payload)),
            ).fetchone()
            if inserted:
                added["samples"] += 1
            elif (
                conn.execute(
                    "SELECT payload FROM samples WHERE run_id=%s AND timestamp_ns=%s", params
                ).fetchone()["payload"]
                != payload
            ):
                raise ConflictError("Sample timestamp already exists with different evidence")
        return added

    def ingest(self, batch: RunIngest):
        run_id = str(batch.result.run_id)
        payload = batch.result.model_dump(mode="json")
        with self.connect() as conn:
            created = bool(
                conn.execute(
                    "INSERT INTO runs(run_id,kind,result) VALUES (%s,%s,%s) "
                    "ON CONFLICT DO NOTHING RETURNING run_id",
                    (run_id, batch.kind, Jsonb(payload)),
                ).fetchone()
            )
            row = conn.execute(
                "SELECT kind,result FROM runs WHERE run_id=%s FOR UPDATE", (run_id,)
            ).fetchone()
            if row["kind"] != batch.kind or row["result"] != payload:
                raise ConflictError("Run ID already exists with different results")
            added = self.insert_telemetry(conn, run_id, batch)
        return {"run_id": run_id, "created": created, "added": added}

    def list_runs(self, limit=50, before=None):
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT run_id, kind, result, created_at, sequence FROM runs "
                "WHERE (%s::bigint IS NULL OR sequence < %s) ORDER BY sequence DESC LIMIT %s",
                (before, before, limit + 1),
            ).fetchall()
        return {
            "runs": rows[:limit],
            "next_cursor": rows[limit - 1]["sequence"] if len(rows) > limit else None,
        }

    def get(self, run_id):
        with self.connect() as conn:
            return conn.execute(
                "SELECT run_id,kind,result,created_at,sequence FROM runs WHERE run_id=%s", (run_id,)
            ).fetchone()

    def append_telemetry(self, run_id, batch):
        with self.connect() as conn:
            row = conn.execute(
                "SELECT result FROM runs WHERE run_id=%s FOR UPDATE", (run_id,)
            ).fetchone()
            if row is None:
                return None
            result = row["result"]
            steps = [*result["baseline"]["steps"], result["clean"], result["contended"]]
            batch.validate_run(str(run_id), {s["step_id"] for s in steps})
            return self.insert_telemetry(conn, run_id, batch)

    def artifact(self, run_id, filename):
        with self.connect() as conn:
            if filename == "results.json":
                row = conn.execute("SELECT result FROM runs WHERE run_id=%s", (run_id,)).fetchone()
                return row["result"] if row else None
            if not conn.execute("SELECT run_id FROM runs WHERE run_id=%s", (run_id,)).fetchone():
                return None
            if filename == "spans.jsonl":
                rows = conn.execute(
                    "SELECT payload FROM spans WHERE run_id=%s "
                    "ORDER BY payload->>'start_time', span_id",
                    (run_id,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT payload FROM samples WHERE run_id=%s ORDER BY timestamp_ns", (run_id,)
                ).fetchall()
            return [row["payload"] for row in rows]

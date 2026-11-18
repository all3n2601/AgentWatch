"""Join one live run's stored step events and resource samples into step-features-v1 rows."""

from collections.abc import Collection
from uuid import UUID

from agentwatch_features import FeatureRows, compute_feature_rows
from agentwatch_sdk.live import ResourceSampleSnapshot, StepEventSnapshot

from agentwatch_worker.live_store import LiveStore, MissingRunError
from agentwatch_worker.store import RunStore


def live_feature_rows(
    store: RunStore,
    run_id: UUID,
    declared_interval_seconds: float | None,
    *,
    step_ids: Collection[str] | None = None,
    lock: bool = True,
) -> FeatureRows:
    """Feature rows for the selected run's terminal steps, from applied evidence only.

    Pending step revisions are not yet a validated lifecycle, so they never shape a window;
    such steps are reported as incomplete. The caller supplies the producer's declared sampling
    interval (the registration's optional `sampling_interval_seconds`) or None, which gives null
    coverage: a run registered without one has no coverage figure, never a guessed one.

    `step_ids` limits the rows to those steps; every resource sample is still read, since a
    window can reach any of them. With `lock` (the default) the read holds the run row lock and
    first applies gap expiry, as snapshots do. `lock=False` reads the applied records in one
    statement, so one consistent snapshot, without delaying ingest: expiry only turns pending
    records into expired ones, so it cannot change applied evidence.
    """
    live = LiveStore(store)
    with live.connect() as conn:
        if lock:
            live.lock(conn, run_id)
            live.expire(conn, run_id)
        elif conn.execute("SELECT 1 FROM live_runs WHERE run_id=%s", (run_id,)).fetchone() is None:
            raise MissingRunError("Live run not found")
        selection, params = "", (run_id,)
        if step_ids is not None:
            selection = " AND (record_type='resource_sample' OR step_id=ANY(%s))"
            params = (run_id, list(step_ids))
        records = conn.execute(
            "SELECT record_type,payload FROM live_records "
            f"WHERE run_id=%s AND disposition='applied'{selection} ORDER BY sequence",
            params,
        ).fetchall()
    events = [
        StepEventSnapshot.model_validate(r["payload"])
        for r in records
        if r["record_type"] == "step_event"
    ]
    samples = [
        ResourceSampleSnapshot.model_validate(r["payload"])
        for r in records
        if r["record_type"] == "resource_sample"
    ]
    return compute_feature_rows(events, samples, declared_interval_seconds)

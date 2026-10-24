"""Join one live run's stored step events and resource samples into step-features-v1 rows."""

from uuid import UUID

from agentwatch_features import FeatureRows, compute_feature_rows
from agentwatch_sdk.live import ResourceSampleSnapshot, StepEventSnapshot

from agentwatch_worker.live_store import LiveStore
from agentwatch_worker.store import RunStore


def live_feature_rows(
    store: RunStore, run_id: UUID, declared_interval_seconds: float | None
) -> FeatureRows:
    """Feature rows for the selected run's terminal steps, from applied evidence only.

    Pending step revisions are not yet a validated lifecycle, so they never shape a window;
    such steps are reported as incomplete. The run registration records no sampling interval,
    so the caller supplies the producer's declared value or None (null coverage).
    """
    live = LiveStore(store)
    with live.connect() as conn:
        live.lock(conn, run_id)
        live.expire(conn, run_id)
        records = conn.execute(
            "SELECT record_type,payload FROM live_records "
            "WHERE run_id=%s AND disposition='applied' ORDER BY sequence",
            (run_id,),
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

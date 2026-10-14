# 0001: Add independent live lifecycle persistence and polling

Date: October 14, 2026. Status: accepted for B01 implementation.

Completed CPU/coding results have an immutable experiment shape which cannot represent
active/failed steps. Mutating it into a lifecycle record would reinterpret saved evidence
and require unrelated dashboard/dataset changes.

Add live registrations, immutable run changes and event/sample records through ordered
migration 0002. Keep completed imports/endpoints intact. Lock each live run transactionally;
validate batches atomically and apply only contiguous step revisions. Retain bounded pending
events and expired gaps. Compute timing from one declared monotonic domain without estimating
missing measurements. Use bounded record pages plus full current-state snapshots for polling.

This gives agent/frontend owners an additive client contract without a new scheduler or
streaming service. Consumers must distinguish arrival cursors from changing dispositions,
poll cancellation requests, and avoid claiming cleanup confirmation. Producer instrumentation,
feature alignment, prediction serving and authenticated deployment remain their assigned tasks.
Storage recovery preserves all tables; application rollback never deletes live history.
See [the runbook](../live-ingestion.md) for limits, errors, migration and handoff details.

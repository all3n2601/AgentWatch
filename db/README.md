# Run history

PostgreSQL stores completed experiment results, OpenTelemetry spans, and CPU samples.
The Compose volume `agentwatch_postgres-data` preserves them across service restarts.

Ordered migrations are packaged under `services/worker/src/agentwatch_worker/migrations/` so
installed deployments can find them. API startup applies unapplied migrations transactionally
under a PostgreSQL advisory lock and records versions in `schema_migrations`.

`runs` has a UUID primary key and monotonic history sequence. `spans` deduplicates by run,
trace ID, and span ID; `samples` deduplicates by run and nanosecond timestamp. JSONB preserves
the validated evidence, including supported extra metadata. Duplicate identities with different
payloads cause conflicts, and the whole ingestion transaction rolls back.

This schema currently supports completed local experiment runs. A future migration will add
live run lifecycle and generalized agent steps rather than changing saved evidence in place.

# Persistent runs and ingestion

Start Docker, run `make infra-up`, then run `make dev-worker` in a terminal. Start the dashboard
with `make dev-web` in another terminal. The API binds to localhost on port 8090, with generated
OpenAPI documentation at `http://127.0.0.1:8090/docs`.

The default development database is `agentwatch` on localhost:5432. Export `DATABASE_URL` to
override the API connection, or configure Compose's `POSTGRES_DB`, `POSTGRES_USER`, and
`POSTGRES_PASSWORD` consistently. Values in `.env.example` are development defaults.

## Record and import

- `make import-runs`: save the two existing local experiments without rerunning their workloads.
- `make record-coding`: run the coding test tool and publish results, spans, and samples.
- `make record-cpu`: run the CPU workload and publish it.
- `make demo-coding` / `make demo-cpu`: continue to generate local artifacts without publishing.

An inconclusive experiment can still be published so its failed checks remain visible. Local
evidence remains on disk if publishing fails. Retry using the publisher instead of rerunning:

```sh
uv run --package agentwatch-worker python -m agentwatch_worker.publish --output .data/coding-demo --kind coding
```

Configure `AGENTWATCH_API_URL` for both the web server and publisher when the API uses another
address. The browser calls same-origin dashboard endpoints, so it does not need cross-origin access.

## HTTP contract

| Endpoint | Purpose |
|---|---|
| `POST /v1/runs` | Ingest `{kind, result, spans, samples}` atomically |
| `GET /v1/runs?limit=50&before=<cursor>` | List completed runs, newest first, with stable cursor pagination |
| `GET /v1/runs/<uuid>` | Read saved run results and metadata |
| `POST /v1/runs/<uuid>/telemetry` | Append a validated `{spans, samples}` batch for known steps |
| `GET /v1/runs/<uuid>/artifacts/results.json` | Download the saved result |
| `GET /v1/runs/<uuid>/artifacts/spans.jsonl` | Download saved OpenTelemetry spans |
| `GET /v1/runs/<uuid>/artifacts/samples.jsonl` | Download saved CPU samples |
| `GET /readyz` | Check database readiness |
| `GET /healthz` | Check the API process |

The result contract mirrors the current demo's `results.json`: UUID run ID, experiment settings,
unique completed steps, baseline measurements, evaluation results, checks, and limitations.
It validates timing order, duration breakdown, queue fractions, baseline repeat count, and
agreement between overall acceptance and individual checks. Nonfinite numeric evidence is rejected.

Spans use the OpenTelemetry SDK's JSON export shape: name, hexadecimal trace/span context,
timezone-aware start/end, attributes, and optional additional SDK fields. Run and step attributes
must refer to the saved experiment. This JSON endpoint is not an OTLP receiver. Samples retain
Linux pressure/cgroup measurements or explicit unavailable values with nanosecond timestamps.

New runs return 201. Repeated identical imports return 200 and counts of newly added telemetry.
Conflicting identities return 409, invalid contracts return 422, and missing runs return 404.
Database failures return 503 without exposing connection details. Empty optional telemetry batches
are permitted; list requests accept 1–100 runs. The dashboard initially loads 100 and offers
older-page loading when a cursor is available.

## Dashboard behavior

The run picker selects saved evidence by UUID. Comparisons only offer runs of the same kind and
identify different workload settings. Timing differences describe observations, not proven causal
improvements. Refresh preserves the selected run while bringing in newly recorded history.

If the API is unavailable, the dashboard labels its local-file fallback explicitly. A download
requested for a saved UUID never silently substitutes the newest local file.

## Verification and scope

Run database integration tests against an explicitly selected test connection:

```sh
AGENTWATCH_TEST_DATABASE_URL=postgresql://agentwatch:agentwatch_dev@localhost:5432/agentwatch uv run pytest
```

Each database test creates and removes its own uniquely named schema; it does not clear run history.
CI supplies a dedicated PostgreSQL service and runs these checks automatically.

This is a local development API for completed CPU/coding experiments, without authentication or
live run state. Generalized agent intake, running/completed lifecycle, automatic updates, and model
evaluation will be added next. The current UI still distinguishes observed queue delays from
hardware saturation and does not claim calibrated ML confidence.

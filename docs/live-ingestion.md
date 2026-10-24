# Live lifecycle ingestion (B01)

The worker serves `live-contracts-v1` at `http://127.0.0.1:8090/v1/live/runs`.
The Team Planner records owner-reported three-team acceptance of C01 on October 13, 2026.
The [shared record contract](shared-event-contracts.md) defines identity, clocks and evidence;
[live OpenAPI](../api/openapi/live.json) defines request/response shapes. Python producer
models/client live in `agentwatch_sdk`; generated TypeScript types are exported by
`@agentwatch/sdk`. Existing `/v1/runs` imports, history and downloads remain separate.

## Transport and producer flow

1. `POST /v1/live/runs` with a retained `request_id`, `task_id`, lowercase SHA-256
   `configuration_digest`, unique `clock_ids` and resource/service registrations. The digest
   identifies configuration; the service cannot verify the underlying inputs. Resources are
   immutable IDs for actual measurement/process lifetimes. Retain the server-created UUID
   for subsequent requests. Exact creation retries return that UUID (`200`, `created: false`);
   new creations return `201`.
2. `POST /v1/live/runs/{UUID}/state` with a retained `change_id`, `expected_revision`, `state`
   and explicit nullable `reason`. Revision zero starts queued. Allowed transitions are
   queued → running → completed/failed/cancelled, or queued → failed/cancelled. Failure and
   cancellation require a redacted reason; other states require null. Exact retries return
   their original revision, even after later transitions. Different content or a stale
   revision returns `409`; refresh before issuing a new command.
3. `POST /v1/live/runs/{UUID}/records` with `{"records": [...]}` containing StepEvent and/or
   ResourceSample. Run/task/clock/resource/service must match registration. The batch is
   atomic: one invalid record or identity conflict rolls back all additions. Record UUIDs
   share one namespace across live runs and record types.
4. Poll `GET /v1/live/runs/{UUID}?after=0&limit=256` for current run/validated step states,
   derived timings, closure/change history, gaps and a bounded record page. Drain pages using
   `next_cursor`; retain the largest received record `sequence` for later polls, including
   the final page when `next_cursor` is null. `after` filters immutable arrivals, not changes
   to disposition. Always consume the full `steps` and `gaps` snapshot: older pending records
   can become applied/expired without a new arrival. Reconnect with the selected UUID/cursor.
5. `GET /v1/live/runs?limit=50&before=...` lists live registrations and current run states.
   History cursors decrease; record cursors increase. A missing selected live UUID is `404`.
   Both cursor parameters are bounded by PostgreSQL's signed bigint maximum (`2**63 - 1`);
   out-of-range cursors return `422` and the SDK rejects them before sending a request.

The integration transport is polling, not SSE/WebSocket delivery. Producer instrumentation
is an A02 follow-up; the dashboard's live consumer (F02) now polls the live read API. Poll
intervals are caller choices; B01 makes no live-latency or instrumentation-overhead claim.

## Bounds, retries and gaps

| Bound | Limit |
| --- | --- |
| Raw live write body, including chunked requests | 1 MiB; 10 seconds to receive |
| Records per batch/page | 1–256 |
| Live history page | 1–100 runs |
| Registered resources/clock domains | Up to 64 each; at least one clock |
| Steps / retained records per run | 1,000 / 100,000 |
| Pending records per run | 256 after applying contiguous prefixes |
| Step revision | Integer 0–1023 |
| Pending expiry | 300 seconds from server receipt |
| Database connect / lock / statement timeout | 3 / 3 / 5 seconds |

Sequences start at queued revision zero. Only a contiguous validated prefix becomes the
step state. Future revisions are acknowledged `pending` with the missing sequence; retry
missing records using original UUIDs/content. A UUID with different content, an occupied
step revision, or a repeated sample key occupied by another UUID is `409`. Sample keys are
resource/service/clock/metric/tick within the run. JSON comparison ignores object key order;
equivalent timezone offsets normalize to UTC.
Numeric retry comparison uses PostgreSQL JSONB equality, preserving exact retries of large
finite floating-point readings even when JSONB renders their stored form as an integer.

Individually impossible events (including unstarted revision-two terminal events) return
`422` before storage. Every adjacent stored revision is checked even without a queued prefix;
an arriving record that contradicts existing step identity/timing/state returns `409` and
rolls back its whole batch. The previously accepted record remains immutable. If an older
server already retained an impossible pending history, close that run as failed and create
a new run; do not edit or delete the original evidence to obtain a successful lifecycle.

Expiry is evaluated transactionally on reads/writes of that run. Expired records remain
retained with a durable gap indicator. Filling earlier gaps never revives expired evidence;
retries report `expired`. Explicitly fail the run and start another when evidence cannot be
recovered. No timer fabricates completion. Parents must exist in the same batch or already
be known; cycles/self-parenting are rejected. Step identity, submission/domain and recorded
start ticks are checked even on pending evidence. Completed steps cannot receive new revisions;
exact retries remain valid. Completed runs cannot receive new step events; registered late
samples are permitted.

Errors: `400` unparseable body/invalid Content-Length, `413` body size, `408` body timeout,
`422` invalid/unsupported data or capacity bound, `409` identity/lifecycle conflict,
`503` database unavailable or lock/statement timeout, and `500` unexpected internal storage
failures. Oversized declared lengths are rejected before requesting a body/100 Continue.
Compressed bodies are not decompressed. Large integer/deep nesting parser limits can produce
`400`; data parsed successfully but incompatible with the request envelope produces `422`.
NUL/unpaired-surrogate text and timestamps outside the UTC datetime range are rejected as
`422`, without misreporting a database outage. Validation responses omit supplied input.
`ErrorResponse.detail` is a string for semantic/HTTP errors or a list of validation issues;
OpenAPI and TypeScript declare both forms. Record registration errors include the batch index.
After uncertain transport outcomes retry only with retained identities/content. The Python
client bounds responses to 8 MiB and defaults to a 10-second **socket inactivity timeout**.
This is not a total request deadline: a continuously delivering peer can take longer.
Callers requiring a total deadline must enforce it in their orchestration. The client validates
known response fields and propagates transport/API errors without automatic retries.

## Terminal states, cancellation and measurements

A terminal step never closes its run. Run completion requires no unfinished steps or
pending/expired gaps. Failure/cancellation retains closure-time `unfinished_steps`, unchanged
by late evidence. Known unfinished steps can receive late evidence after failed/cancelled
closure; new steps cannot be created there.

Cancellation persists `cleanup_requested: true` in the selected run snapshot. Producers must
poll, stop their designated work, release resources and emit actual terminal step evidence.
The service does not execute producer commands or terminate processes. `cleanup_confirmed`
remains null: cancellation receipt proves no cleanup. Isolation/cleanup enforcement is A03.

Ticks are relative monotonic nanoseconds in registered domains, within JavaScript's safe
integer range. Derived seconds use compatible ticks. Queue delay, execution wall time and
process CPU time remain distinct. Unstarted failed/cancelled steps have null queue/execution/CPU
and a total notification duration. Zero-duration steps have null queue fraction.

The metric registry (`METRICS` in `agentwatch_sdk.live`) accepts:

| Metric | Class | Unit, bounds | Meaning |
| --- | --- | --- | --- |
| `cpu.psi.some.avg10.v1` | `cpu_sandbox` | percent 0–100 | Host CPU PSI `some avg10` gauge |
| `cpu.cgroup.throttled_usec.v1` | `cpu_sandbox` | microseconds ≥ 0 | Cumulative cgroup throttling counter |
| `gpu.nvml.utilization.gpu.v1` | `inference` | percent 0–100 | Share of NVML's sample period with a kernel running |
| `gpu.nvml.utilization.memory.v1` | `inference` | percent 0–100 | Share of NVML's sample period with device memory busy |
| `gpu.nvml.memory.used_bytes.v1` | `inference` | bytes ≥ 0 | Device memory in use by every process |
| `gpu.nvml.power.usage_watts.v1` | `inference` | watts ≥ 0 | Device power draw gauge |
| `gpu.nvml.clock.sm_mhz.v1` | `inference` | megahertz ≥ 0 | Current SM clock |

GPU entries (A04, [ADR 0003](adr/0003-gpu-and-inference-measurement.md)) are whole-device
NVML gauges, never per process, and NVML utilization cannot resolve activity shorter than its
own period or establish saturation. New signals (DCGM, vLLM, retrieval) need new versioned
names; a definition never changes meaning in place. The change is additive, but an older
service refuses the GPU metrics, and **every** agent `run` sends them: it always registers the
GPU and reports it as `unsupported` where NVML is missing (macOS, CPU-only hosts). Deploy the
service first; a runner reporting to an older service ends every run with refused records
(exit 5). With 7 records per round, the longest `--run-timeout` the per-run sample budget
allows at 250 ms falls from 10,000 s to about 2,857 s on every host. All five availability
states survive history; absent readings never become measured zero. B02 alignment, deltas and
coverage are defined in [shared step features](step-features.md); `step-features-v1` reads only
the two CPU metrics, so GPU readings need a new feature version. No predictions or
host-saturation diagnosis are produced here.

## Producer and frontend handoff

```python
from uuid import uuid4
from agentwatch_sdk import LiveClient, RunCreate, RunChange

client = LiveClient("http://127.0.0.1:8090", timeout_seconds=10)
registration = RunCreate(
    contract_version="live-contracts-v1",
    request_id=uuid4(),  # Retain for retries, including uncertain creation responses.
    task_id=task_definition_id,
    configuration_digest=configuration_sha256,
    clock_ids=[producer_clock_id],
    resources=[],  # Register actual targets before emitting resource records.
)
run_id = client.create_run(registration).run_id
client.change_state(run_id, RunChange(
    change_id=uuid4(), expected_revision=0, state="running", reason=None,
))
snapshot = client.get_run(run_id)
```

The task/configuration/clock variables must come from actual producer settings; this snippet
creates no measured telemetry. Retain each change object before sending, for idempotent retries.
Benoy/Akshayaa can emit shared `StepEvent`, `ResourceSample` and `LiveBatch` models. Meghana can
consume `RunSnapshot`, `RunPage` and `BatchResult` types after validating external responses;
wire types alone cannot validate runtime measurements. Regenerate schemas/types with
`uv run python -m agentwatch_worker.export_live_contracts`; drift is regression-tested.

Producer request models remain strict and reject unknown fields/metrics. Response models
ignore additive fields at every nested level and retain newer metric names without applying
an older writer's metric registry; known types and availability/value semantics remain
validated. An unfamiliar returned metric is evidence to retain, not a metric an old client
can interpret or emit. Compatible response additions therefore do not prevent a producer
from polling cancellation. Changed meanings, required fields or incompatible enum/type changes
still need explicit contract/version coordination.

## Migration, recovery and verification

Startup applies ordered `0002_live_lifecycle.sql` under existing migration locking and
transactionality. It adds live tables/indexes without rewriting runs/spans/samples. Back up
before deployment. For application rollback, retain added tables and migration records;
the old application ignores them. Never drop history or rewrite an applied migration. A failed
migration transaction rolls back; repair its cause and restart. Operational backup/restore
verification remains D04, distinct from B01's populated-schema upgrade regression.
Live queries select explicit columns so additive future table columns do not leak into the
wire response or break an application rollback. Audit migration `0003_remove_duplicate_live_step_index.sql`
removes only the redundant step index; the unique constraint supplies the same index ordering.
The populated-0002 upgrade preserves evidence and retry/conflict behavior. Existing migration
files remain unchanged. An application rollback needs no index restoration because the unique
index remains available to both application versions.

Run `AGENTWATCH_TEST_DATABASE_URL=<dedicated test database> uv run pytest` and
`uv run ruff check .`. Live tests use isolated schemas for retries, ordering, identities,
rollback, cancellation, gaps, late evidence, availability, pagination, concurrent creation
and populated-schema upgrades. Synthetic fixtures are test-only evidence. This API inherits
the current local ingestion deployment boundary; authenticated shared deployment is B04 work.

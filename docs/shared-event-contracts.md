# Shared live telemetry and prediction contracts

Status: **accepted `live-contracts-v1`**, owner-reported three-team acceptance recorded on
the Team Planner October 13, 2026. Board task C01, owner Allen, partner reviewer Akshay.
The record semantics below preserve the October 11 proposal; references to proposal review
describe that historical process. B01 implements StepEvent/ResourceSample ingestion, lifecycle,
SDK and additive storage; see [live ingestion](live-ingestion.md) for transport and bounds.
B02 implements the window join and the `step-features-v1` FeatureRow allowlist; see
[shared step features](step-features.md). FeatureRow/Prediction serving remains B03 work.

The current [completed-run ingestion](ingestion-api.md) remains supported unchanged.
The [data card](data-card.md) defines the separate, implemented `steps-v1.2` dataset.
The [evaluation contract](evaluation-contract.md) controls model evaluation and promotion;
this proposal does not accept its proposed thresholds or claim evaluated model quality.

## Common representation and identity

All four records below are JSON objects. Every listed field is required; a nullable field
must be explicitly `null` when unknown. Unknown fields and unsupported contract versions
are rejected. Reject booleans where numbers are required, nonfinite numbers, invalid UUIDs,
blank identifiers, invalid enums, and timestamps without a timezone. Do not coerce strings
into numbers or fill missing observations with zero. Identifiers are at most 128 characters;
redacted explanatory reasons are at most 512 characters.

| Field | Type | Meaning |
| --- | --- | --- |
| `contract_version` | string | Exactly `live-contracts-v1` for this proposal |
| `record_type` | enum | `step_event`, `resource_sample`, `feature_row`, or `prediction` |
| `run_id` | UUID string | Server-created run; never inferred from the latest local artifact |
| `task_id` | nonblank string | Stable, versioned task definition shared by all related baselines/replays |

The run creator registers `task_id` and a workload/configuration digest before events arrive.
The service rejects records for unknown runs or inconsistent task identity. A task's inputs
and revision determine `task_id`; worker capacity, contention condition, workload repetition,
and sampling frequency belong to run configuration, not the task grouping key. Legacy
`kind:workload:iterations` placeholders must remain labeled as such and cannot establish
unseen-task evaluation. Historical results are not silently assigned new task identities.

`step_id` is unique within a run. `parent_step_id`, when present, names another step in the
same run; reject self-parenting and cycles. Event/sample UUIDs are unique across runs.
`trace_id` is 32 lowercase hexadecimal characters and `span_id` is 16, both nonzero;
these proposed unprefixed IDs need an explicit adapter for the current `0x`-prefixed span
format. `resource_id` and `service_id` name registered run resources/services. A resource
record maps an opaque ID to the host/container/GPU and process lifetime it actually measures;
restarting that resource creates a new ID. Do not export host paths or credentials as IDs.

## Clocks, timing and sample windows

Wall timestamps are RFC 3339 with an offset, normalized to UTC when stored. `observed_at`
means observation time at the producer; `received_at` is assigned separately by the server.
Their difference is not execution duration or a reliable network-delay measurement.

`clock_id` names one producer's monotonic clock domain for a process lifetime. Nanosecond
ticks are JSON integers in `[0, 9007199254740991]` for lossless TypeScript interoperability.
Emit ticks relative to a domain origin to keep this bound. Compare ticks only within the
same clock domain. A restart registers a new domain. No implicit comparison is allowed
between host wall clocks, monotonic domains, or process CPU clocks.

Timing fields use seconds and finite nonnegative numbers. For an executed terminal step,
`submitted_tick_ns <= started_tick_ns <= finished_tick_ns`. Compute directly measured
queue delay from submission to start, execution wall time from start to finish, and total
wall duration from submission to finish in the same domain. CPU time is process CPU time
with a declared process/child scope; it may exceed execution wall time under parallelism.
Never call execution wall time pure useful work, or queue contention host CPU saturation.

Zero-duration terminal steps are valid. Their queue fraction is `null`, not a division by
zero. Estimated excess latency is a separate prediction quantity, not measured queue time.
For failed/cancelled steps that never started, queue and execution seconds stay null; total
duration is submission to terminal notification, explicitly not execution time. Validate timing
breakdowns against ticks within a proposed absolute tolerance of `1e-6` seconds. All teams must
review this tolerance before freeze; no legacy timing tolerance changes with this proposal.

Step sample windows use `[started_tick_ns, finished_tick_ns)`; a sample exactly at finish
does not belong to that step. Join only matching run and registered resource/service identity.
Overlapping parent/child windows can share observations but must not sum them as independent
resource totals. Samples from a different clock domain remain unaligned unless a separately
versioned clock mapping with measured uncertainty exists. No such mapping is implemented today.

## StepEvent

In addition to common fields:

| Field | Type | Meaning |
| --- | --- | --- |
| `event_id` | UUID string | Retry identity; reuse with identical semantic content |
| `step_id`, `step_type` | string | Stable step identity and `model_call`, `tool`, `retrieval`, or `other` |
| `parent_step_id` | string or null | Parent relationship within the run |
| `trace_id`, `span_id` | hex strings | Trace and span identity, stable across this step's events |
| `resource_id`, `service_id` | string or null | Measured execution target; null if not known |
| `sequence` | integer >= 0 | Contiguous per-step revision: queued starts at 0 |
| `state` | enum | `queued`, `running`, `completed`, `failed`, or `cancelled` |
| `observed_at`, `clock_id` | strings | Wall observation timestamp and monotonic domain |
| `submitted_tick_ns` | integer | Submission in this clock domain |
| `started_tick_ns`, `finished_tick_ns` | integer or null | Known execution boundaries |
| `cpu_seconds` | number or null | Measured process CPU, never estimated from wall time |
| `cpu_scope` | enum or null | `process` or `process_and_children`; required iff CPU time exists |
| `reason` | string or null | Redacted failure/cancellation reason |

Allowed transitions are queued → running → completed/failed/cancelled, or queued →
failed/cancelled. Running requires a start and no finish. Completed requires start/finish
and a null reason. Failed/cancelled require finish and a nonblank reason; start may be null
when execution never began. A null start requires null CPU time and CPU scope. Queued requires
null start, finish, CPU, CPU scope and reason.
Nonterminal events have null CPU and reason. Identity, submitted tick, clock domain and
any already-recorded start tick cannot change across revisions. CPU time is null when no
process measurement exists, including a cancellation before start.

The server accepts exact retries idempotently. Reusing an event UUID with changed content,
or a `(run_id, step_id, sequence)` with different content, is a conflict. Semantic content
is the validated JSON value, independent of object key order. Reject unknown step transitions
and conflicting run identity atomically with the whole ingestion batch.

For out-of-order delivery, retain future revisions as pending evidence and report the missing
sequence; apply only the contiguous validated prefix. Do not display a future terminal event
as completed until preceding transitions have been validated. Batch bounds, pending-buffer
limits, expiry and acknowledgements must be specified in B01 before exposing a live endpoint.
Expiry preserves a gap/error indicator rather than fabricating a successful lifecycle.

Run state is server-owned and distinct from step state. It uses the same five state names;
completing one step does not complete the run. A terminal run closure records its reason and
any unfinished steps explicitly. Cancellation must request producer cleanup; receipt of a
cancellation event does not establish that processes were terminated. B01 must test retries,
gaps, failure, cancellation, atomic rollback and late evidence before claiming this behavior.

## ResourceSample

In addition to common fields:

| Field | Type | Meaning |
| --- | --- | --- |
| `sample_id` | UUID string | Idempotent sample identity |
| `resource_id`, `service_id` | nonblank strings | Registered resource and measurement service |
| `resource_class` | enum | `cpu_sandbox`, `inference`, or `retrieval`; not a diagnosis |
| `observed_at`, `clock_id`, `tick_ns` | timestamp, string, integer | Producer observation and monotonic tick |
| `metric`, `unit` | strings | Versioned metric identifier and exact registered unit |
| `value` | finite number or null | Actual reading, including measured zero |
| `availability` | enum | `measured`, `missing`, `unsupported`, `malformed`, or `unavailable` |
| `reason` | string or null | Null for measured values; nonblank otherwise |

Measured values require a number; all other availability states require null. Metric registry
entries define gauge/counter semantics, units, bounds, scope and aggregation. Initial CPU
entries map existing PSI `some avg10` to percent `[0,100]` and cgroup throttled time to a
nonnegative microsecond counter. Counter resets are unavailable deltas, never negative load.
Inference/GPU and retrieval entries require their owners' measured definitions under A04;
they are not assumed available because an enum exists. A04 registered five whole-device NVML
GPU gauges for `inference` resources and three cumulative CPU stall and usage counters for
`cpu_sandbox` resources ([the registry](live-ingestion.md)). No retrieval metric is
registered: the in-process index has nothing to sample, and its per-request service time is
local evidence (`retrieval-timing-v1`). Cumulative counters and host-wide
PSI must not be presented as step-exclusive duration. Sampling configuration records both
declared interval and observed gaps; duplicates must not inflate coverage.

Samples are resource-scoped rather than forced to one step. The service does window joining
under B02. Same UUID/same content is an exact retry; same UUID/different content or conflicting
readings for the same resource, metric and tick is a conflict. Preserve all unavailable states
through history and export. Missing files and empty sample streams remain distinguishable.

## FeatureRow

In addition to common fields:

| Field | Type | Meaning |
| --- | --- | --- |
| `step_id`, `step_type` | strings | Source step and its declared type |
| `feature_version` | nonblank string | Immutable feature-definition/aggregation version |
| `source_event_ids`, `source_sample_ids` | UUID arrays | Unique source lineage; event IDs must be nonempty |
| `window` | object | `clock_id`, `started_tick_ns`, `finished_tick_ns`; nullable boundaries for nonexecuted steps |
| `timings` | object | Nullable `queue_seconds`, `execution_seconds`, `duration_seconds`, `cpu_seconds` |
| `telemetry` | object | `status`, `sample_count`, `coverage`, `declared_interval_seconds`, `observed_interval_seconds` |
| `features` | object | Only the explicitly allowlisted numeric/bool/null inputs for `feature_version` |

Window boundaries must both exist and be ordered, or both be null. Telemetry status is
`present`, `empty`, `missing`, `no_overlap`, or `unaligned`; sample count is a nonnegative
integer. Coverage is null or a finite fraction `[0,1]`; intervals are null or finite positive
seconds. Counts refer to distinct observations inside the aligned step window, not all run
samples. Unaligned telemetry yields null coverage and null resource aggregates.

`FeatureRow` is a serving/training envelope, not a replacement Parquet schema. B02 defines
the immutable `step-features-v1` allowlist and common computation in
[shared step features](step-features.md), including how each telemetry status is assigned.
For legacy dataset extraction, preserve `steps-v1.2` definitions, including its documented
coverage denominator; changing window semantics requires a new version and compatibility
tests. IDs, versions, lineage, condition, injector configuration, rule predictions, baselines
used for labels and label metadata are never model inputs. Unknown feature keys fail validation.
Dataset releases retain their own `dataset_version`, `label_version` and split hashes outside
the feature map. Online rows do not invent dataset or label versions.

## Prediction

In addition to common fields:

| Field | Type | Meaning |
| --- | --- | --- |
| `prediction_id` | UUID string | Immutable result identity |
| `step_id`, `feature_version` | strings | Source step and consumed feature contract |
| `method`, `method_version` | enum, string | `rule` or `model`, and the exact rule/model artifact version |
| `model_version` | string or null | Required for model results; null for rules |
| `produced_at` | timezone-aware timestamp | Result production wall time |
| `decision` | enum | `attributed`, `none`, or `abstained` |
| `resource` | enum or null | `cpu_sandbox`, `inference`, `retrieval`, `none`, or null |
| `estimated_excess_fraction`, `estimated_excess_seconds` | number or null | Separate estimates, finite fraction `[0,1]` and nonnegative seconds |
| `confidence` | number or null | Calibrated confidence only, finite fraction `[0,1]` |
| `calibration_version` | string or null | Required iff confidence is provided |
| `telemetry_status`, `reason` | strings | Feature-row status and redacted explanation/limitations |

Attributed requires a resource class, none requires `resource: "none"`, and abstained requires
null resource, estimates and confidence. Reason is always nonblank. Missing evidence or an
unavailable model can yield abstention; `none` means sufficient evidence for no identified
bottleneck. Rule output can be stored separately when a model is unavailable, explicitly as
`method: "rule"`. Rule outputs have null model version, confidence and calibration version.
Uncalibrated scores are not exposed as confidence. Model availability does not imply promotion.

Estimates are either both null or both supplied. When supplied, the source step needs positive
measured duration and seconds equals fraction × duration within the feature contract's declared
numeric tolerance. Missing estimates are permitted even with a resource decision. Zero excess
does not imply measured zero queue time. Store rule/model predictions separately for compatible
comparisons; never replace original measurements or a previous immutable prediction in place.

## Illustrative fixtures for review

These examples are **synthetic specification fixtures**, not recorded-run evidence. The
following is a complete failed-step record; preceding queued/running revisions are required
before it becomes the applied state. A cancelled-before-start variant changes `event_id`,
`sequence` to 1, `state` to `cancelled`, `started_tick_ns` to null, and reason to
`Cancelled before a worker started`; its predecessor is the queued revision.

```json
{
  "contract_version": "live-contracts-v1",
  "record_type": "step_event",
  "run_id": "00000000-0000-4000-8000-000000000001",
  "task_id": "fixture-coding-task-v1",
  "event_id": "00000000-0000-4000-8000-000000000002",
  "step_id": "tool-1",
  "step_type": "tool",
  "parent_step_id": null,
  "trace_id": "00000000000000000000000000000001",
  "span_id": "0000000000000001",
  "resource_id": "fixture-worker-1",
  "service_id": "fixture-sandbox-1",
  "sequence": 2,
  "state": "failed",
  "observed_at": "2026-10-11T20:00:02Z",
  "clock_id": "fixture-process-1",
  "submitted_tick_ns": 0,
  "started_tick_ns": 100000000,
  "finished_tick_ns": 2000000000,
  "cpu_seconds": null,
  "cpu_scope": null,
  "reason": "Tool timed out; cleanup status is recorded separately"
}
```

A resource sample preserves unsupported telemetry without a zero reading:

```json
{
  "contract_version": "live-contracts-v1",
  "record_type": "resource_sample",
  "run_id": "00000000-0000-4000-8000-000000000001",
  "task_id": "fixture-coding-task-v1",
  "sample_id": "00000000-0000-4000-8000-000000000003",
  "resource_id": "fixture-worker-1",
  "service_id": "fixture-sandbox-1",
  "resource_class": "cpu_sandbox",
  "observed_at": "2026-10-11T20:00:01Z",
  "clock_id": "fixture-process-1",
  "tick_ns": 1000000000,
  "metric": "cpu.psi.some.avg10.v1",
  "unit": "percent",
  "value": null,
  "availability": "unsupported",
  "reason": "CPU PSI is not exposed on this platform"
}
```

A separate missing-stream variant of the same fixture preserves timing despite an absent sample
stream (it does not consume the unsupported sample above). Its feature map is empty
until B02 freezes an allowlist, so this fixture is not a usable model input:

```json
{
  "contract_version": "live-contracts-v1",
  "record_type": "feature_row",
  "run_id": "00000000-0000-4000-8000-000000000001",
  "task_id": "fixture-coding-task-v1",
  "step_id": "tool-1",
  "step_type": "tool",
  "feature_version": "proposal-only",
  "source_event_ids": ["00000000-0000-4000-8000-000000000002"],
  "source_sample_ids": [],
  "window": {
    "clock_id": "fixture-process-1",
    "started_tick_ns": 100000000,
    "finished_tick_ns": 2000000000
  },
  "timings": {
    "queue_seconds": 0.1,
    "execution_seconds": 1.9,
    "duration_seconds": 2.0,
    "cpu_seconds": null
  },
  "telemetry": {
    "status": "missing",
    "sample_count": 0,
    "coverage": null,
    "declared_interval_seconds": 0.25,
    "observed_interval_seconds": null
  },
  "features": {}
}
```

An abstention records insufficient evidence without asserting a clean run or model success:

```json
{
  "contract_version": "live-contracts-v1",
  "record_type": "prediction",
  "run_id": "00000000-0000-4000-8000-000000000001",
  "task_id": "fixture-coding-task-v1",
  "prediction_id": "00000000-0000-4000-8000-000000000004",
  "step_id": "tool-1",
  "feature_version": "proposal-only",
  "method": "rule",
  "method_version": "proposal-only",
  "model_version": null,
  "produced_at": "2026-10-11T20:00:03Z",
  "decision": "abstained",
  "resource": null,
  "estimated_excess_fraction": null,
  "estimated_excess_seconds": null,
  "confidence": null,
  "calibration_version": null,
  "telemetry_status": "missing",
  "reason": "Insufficient resource evidence; no attribution evaluated"
}
```

## Review and implementation handoff

| Reviewer/workstream | Required decision before freeze |
| --- | --- |
| Akshay, data | Confirm stable task grouping, leakage exclusions, feature lineage and dataset compatibility |
| Benoy/Akshayaa, agent | Confirm identity propagation, clock domains, metric registry, failure/cancellation and cleanup evidence |
| Meghana/Rohit, frontend/model | Confirm partial-state display, prediction/abstention fields and calibrated-confidence semantics |
| Allen, backend | Specify B01 transport, run creation/closure, bounded batches/gaps, conflict responses and atomic storage |

Do not mark C01 done or unlock its dependents based solely on merging this proposal. Record
each team's review and settled choices, then change its status to accepted with an immutable
version. Breaking meaning changes require a new contract version, explicit adapters and a
rollout plan; they must not silently reinterpret saved history.

B01 implements accepted schemas under `api/openapi/` with matching SDKs, runtime validators,
consumers, ordered migrations and regression tests. B02 versions the shared feature allowlist
and window computation. B03 implements immutable predictions and explicit model unavailability.
Before changing current consumers, retain completed-run endpoints and run PostgreSQL integration
tests on a dedicated test database with isolated schemas. No database or migration changes are
part of C01's proposal.

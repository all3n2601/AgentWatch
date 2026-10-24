# Shared step features (B02)

Status: `step-features-v1` implemented October 24, 2026 in
[`packages/features`](../packages/features/src/agentwatch_features/) (`agentwatch-features`).
Board task B02, owner Allen, partner reviewer Akshay. Training and serving import the same
functions; no second implementation exists. No prediction is produced here: serving rule and
model results is B03.

| Version | Module | Input | Window | Consumers |
| --- | --- | --- | --- | --- |
| `steps-v1.2` | `agentwatch_features.legacy` | Completed-run sampler rows | Submission to finish, both inclusive | `pipelines/data` flattening; released datasets |
| `step-features-v1` | `agentwatch_features.live` | Applied live StepEvents and ResourceSamples | `[started_tick_ns, finished_tick_ns)` in the step's clock domain | Worker join (`agentwatch_worker.live_features`); B03 serving; D03 dataset v1 |

`steps-v1.2` definitions are unchanged; see the [data card](data-card.md). The two versions use
different windows and inputs, so their values are not interchangeable. `step-features-v1` has
no workload-size or worker-capacity features: live registrations do not record them. A model
trained on `steps-v1.2` columns cannot consume `step-features-v1` rows without a new, explicitly
versioned mapping.

## Using it

```python
from agentwatch_features import compute_feature_rows

result = compute_feature_rows(step_events, resource_samples, declared_interval_seconds)
result.rows               # one validated FeatureRow per terminal step, ordered by step_id
result.incomplete_step_ids  # steps whose applied history is not terminal yet
```

Pass one run's **applied** step revisions (not pending ones) and that run's samples. The worker
helper `live_feature_rows(store, run_id, declared_interval_seconds)` loads them from PostgreSQL
for the selected run UUID and raises `MissingRunError` for an unknown run. The live registration
does not record a sampling interval; pass the producer's declared interval, or `None` to leave
coverage null. Never pass an observed interval as the declared one.

## Window join

A sample joins a step only when run, `resource_id`, `service_id` and `clock_id` all match the
step, and `started_tick_ns <= tick_ns < finished_tick_ns`. A sample exactly at finish belongs to
the next window. Samples in another clock domain are never compared with the step's ticks.
Only `cpu.psi.some.avg10.v1` and `cpu.cgroup.throttled_usec.v1` are read; newer metrics never
change a v1 row. Exact sample retries count once; a reused sample UUID with different content,
or two readings sharing a resource/service/clock/metric/tick, raises `ValueError`.

| Telemetry status | Meaning | Coverage |
| --- | --- | --- |
| `present` | At least one sample inside the window | Computed |
| `no_overlap` | Same-domain samples exist for the target, none inside the window | Computed (0 when an interval fits) |
| `empty` | The step's target has no v1-metric samples in this run | Computed (0 when an interval fits) |
| `unaligned` | The target's samples are all in other clock domains | Null |
| `missing` | The step never started, or has no measured resource/service identity | Null |

## Features

The feature map holds exactly these keys, in this order (`FEATURE_NAMES`, from the typed
`StepFeatures` model). Unknown or missing keys, and wrongly typed values such as a boolean
where a number is required, fail validation. Identity, lineage, run configuration, rule output and labels are never
features.

| Feature | Unit | Definition |
| --- | --- | --- |
| `queue_seconds` | s | Start − submission ticks; null if never started |
| `execution_seconds` | s | Finish − start ticks; null if never started. Not pure work |
| `duration_seconds` | s | Finish − submission ticks; for unstarted terminal steps, to the terminal notification |
| `cpu_seconds` | s | Producer-measured process CPU; null when not measured |
| `queue_fraction` | 0–1 | Queue ÷ duration; null for zero duration or no start |
| `cpu_pressure_max_avg10` | percent | Maximum host PSI `some avg10` inside the window; null unless every PSI reading there was measured |
| `cpu_pressure_available` | bool | Whether the maximum above was measured |
| `cgroup_throttled_usec_delta` | µs | Last − first throttled counter inside the window: a lower bound for the step. Null with fewer than two readings, any unmeasured reading, or any decrease (counter reset) |
| `cgroup_available` | bool | Whether the delta above was measured |
| `telemetry_sample_count` | count | Distinct sample ticks inside the window, across both metrics and all availability states |
| `telemetry_coverage` | 0–1 | Sample count ÷ whole declared intervals in the window (integer nanoseconds), capped at 1; null without a declared interval or when the window is shorter than one interval |

A measured zero stays `0.0` with its availability flag true; an unsupported, missing, malformed
or unavailable reading stays null with the flag false. Host-wide PSI and cumulative counters are
context, not step-exclusive load or host-saturation evidence. Overlapping parent/child windows
can share samples; never sum their aggregates as independent totals.

`FeatureRow` also carries the contract envelope: source event IDs (every applied revision),
source sample IDs (joined samples), the window, timings, and telemetry status, sample count,
coverage, declared interval and observed interval (median gap between distinct joined ticks).
Rows reject internal disagreement: feature timings or telemetry that differ from the
envelope, an availability flag that contradicts its value, a `queue_fraction` that is not
queue ÷ duration (within 1e-6), present telemetry without sample lineage, aggregates without
present telemetry, or a non-`missing` status for a step that never started.

## Verification and limitations

- `packages/features/tests/test_live_features.py`: 53 synthetic, test-only cases for measured
  zero versus unsupported, units, counter resets, half-open windows, incomplete/partial windows,
  cancellation before start, zero duration, identity and clock joins, retries, conflicts and row
  validation. 100% line and branch coverage of `live.py`.
- `services/worker/tests/test_live_feature_join.py`: 4 isolated-schema PostgreSQL cases for
  stored joins, pending revisions, late samples and a missing run.
- `pipelines/data/tests/` pass unchanged against the moved `steps-v1.2` functions.
- No HTTP endpoint serves feature rows yet; B03 decides how rows are served and stored.
- Coverage needs the producer's declared sampling interval, which live runs do not record.
- No real agent run has been joined: evidence so far is synthetic fixtures only. Sampling
  overhead and achieved intervals are not measured by this change.

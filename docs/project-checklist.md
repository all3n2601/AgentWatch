# AgentWatch project and testing checklist

Status snapshot: October 2, 2026. This is the current implementation and verification checklist;
[the implementation plan](implementation-plan.md) describes the larger MLOps target.

We have a working local CPU queue profiler, coding test-tool replay, persistent run history,
and an interactive dashboard. The next milestone is live run lifecycle and generalized telemetry
intake. A Docker Qwen3 tool-calling smoke test now invokes the fixed coding experiment;
general live-agent ingestion remains unfinished.

A checked item means the stated work or verification was completed. An unchecked test means it
still needs evidence, even when the corresponding feature exists. The test counts below reflect
completed verification sessions, with the Docker LLM follow-up described separately. No percentage
of project completion is assigned because the remaining ML and infrastructure work is substantial.

## Current working flow

Trusted CPU task or Python test tool → measured worker queue and execution timestamps →
OpenTelemetry JSON spans and CPU samples → validated FastAPI ingestion → PostgreSQL history →
Next.js dashboard with saved-run selection and comparisons.

Current attribution identifies CPU worker capacity contention using a timing threshold. It does
not establish host CPU saturation, provide a calibrated confidence score, or classify GPU and
retrieval bottlenecks. Execution wall time is not pure useful work. The test tool is a trusted
fixture now callable by a local LLM, rather than a secure execution sandbox. The ingestion endpoint accepts
completed CPU/coding experiment evidence rather than OTLP or arbitrary live agent workloads.

## Completed implementation

### Local profiling and experiments

- [x] Monorepo with a Python workspace and a Next.js application.
- [x] Fixed CPU workload with clean baselines and a controlled competing task.
- [x] Python test tool executed in a separate interpreter, with repeatable output checksums.
- [x] Direct measurement of enqueue, worker start, worker finish, and process CPU time.
- [x] Warm-up and repeated baselines with an execution variability check below 15%.
- [x] Versioned CPU queue attribution rule with a baseline-derived threshold.
- [x] OpenTelemetry spans carrying run IDs, step IDs, workload size, timings, and worker events.
- [x] Linux CPU pressure and cgroup v2 readers with a target 250 ms sampling interval.
- [x] Per-window sample summaries and explicit unavailable values on unsupported systems.
- [x] JSON results, JSONL spans/samples, and a standalone HTML experiment report.
- [x] Separate commands for local demos, recording to history, and importing existing evidence.

### Ingestion and persistence

- [x] FastAPI health, database readiness, and generated API documentation.
- [x] Validated completed-run contract covering timestamps, timing breakdown, queue fraction,
  unique step IDs, baseline repeat count, acceptance checks, and finite numeric evidence.
- [x] Span correlation checks tying telemetry to the recorded run and its known steps.
- [x] PostgreSQL run, span, and sample tables with packaged, versioned startup migrations.
- [x] Atomic ingestion with idempotent duplicates and conflicts for changed evidence.
- [x] Telemetry append endpoint for completed runs and their known steps.
- [x] Saved-run listing, individual lookup, and cursor pagination.
- [x] Evidence downloads tied to a saved run UUID.
- [x] Persistent Docker volume for the local PostgreSQL service.
- [x] Three actual experiment runs saved during the most recent implementation session.

### Docker LLM smoke test

- [x] Optional Compose profile with digest-pinned Ollama image and persistent model volume.
- [x] Qwen3 1.7B downloaded and executed on Docker CPU with real tool calling.
- [x] Allowlisted `run_tests` dispatcher rejects invented arguments and sends corrective feedback.
- [x] Real tool result returned to the model and followed by a final answer.
- [x] Agent conversation, inference timing/token metrics, and OpenTelemetry inference spans saved.
- [x] Model-invoked coding measurements saved in PostgreSQL and retrievable through the web API.
- [x] Rerun/start/stop commands documented in [the LLM guide](llm-demo.md).
- [x] Qwen3 agent timeline, inference request metrics, token counts, and expandable conversation
  displayed in a dedicated dashboard Agent view, with spans retained in saved result evidence.
- [ ] Generalized ingestion of arbitrary agent/inference traces beyond this fixed tool loop.
- [ ] Tool execution isolated inside its own container.
- [ ] GPU inference, retrieval tools, agent failures, and longer multi-tool tasks validated.

### Dashboard

- [x] Tailwind CSS and shadcn/ui components with Recharts visualizations.
- [x] Workspace-wide overview metrics, execution/queue chart by run, and measured resource coverage.
- [x] Resource cards showing GPU and retrieval as awaiting data.
- [x] Trace waterfall aligned to recorded step timestamps.
- [x] Span search, contention filter, evidence drawer, and sortable run-history columns.
- [x] Telemetry coverage and experiment integrity views.
- [x] Saved-run navigation, comparison tables, and observed timing deltas.
- [x] Matching-workload indication and caveat for comparisons with different settings.
- [x] Manual refresh and older-page loading control when history has another page.
- [x] Saved evidence exports and an explicitly labeled local-file fallback if history is unavailable.
- [x] Empty-state instructions and mobile navigation.
- [x] Seven distinct URL pages: overview, runs, traces, agents, resources, compare, settings.
- [x] Custom run navigators and searchable tables replace generic run/comparison dropdowns.
- [x] Recorded-time, workload, and status filters operate on measured history.
- [x] Resources page retrieves saved raw samples and displays actual coverage.
- [x] Browser-persisted density and automatic completed-history refresh settings.
- [x] Requested run links load records outside the first history page; invalid IDs warn visibly.

### Development support

- [x] Makefile commands and guides for demos, ingestion, persistence, and the dashboard.
- [x] Python and web test suites, linting, and web type checking.
- [x] CI configuration includes a PostgreSQL service for integration tests.
- [x] Database integration tests use separate temporary schemas without clearing saved history.

## Completed verification

### Automated tests from the last run

| Area | Passing tests | What is covered |
|---|---:|---|
| Python run contracts | 8 | Valid evidence, invalid timing/fraction, duplicate step IDs, wrong repeat count, contradictory checks, foreign run spans, nonfinite extra values |
| PostgreSQL API | 5 | Idempotent imports, retained history, cursor pagination, readiness, result conflicts, span append/deduplication, transaction rollback, invalid ingestion, artifact lookup |
| CPU queue attribution | 5 | Direct timestamp calculations, small dispatch delays, invalid timestamp order |
| CPU sampler | 5 | Linux file fixtures, window alignment, missing/malformed readings, cgroup path boundaries, counter resets |
| Coding test tool | 3 | Input edge cases, actual subprocess execution, propagated subprocess failure |
| Existing health/runtime scaffolds | 2 | Service health and runtime identity |
| LLM dispatcher | 6 | Tool-result feedback, invalid/missing calls, repeated calls, correction after invented arguments |
| Web evidence schema | 8 | Unavailable vs. zero measurements, invalid fractions/resources, inconsistent timing, missing baseline/status, tool-call preservation, inference metrics, agent spans, invalid optional agent evidence |
| Workspace aggregation | 4 | Separate tool/inference totals, missing token counts, recorded-time scopes, matching comparisons |
| **Total** | **46** | **34 Python tests and 12 web tests** |

- [x] Most recent Python suite passed with PostgreSQL integration tests enabled.
- [x] Most recent web schema suite passed.
- [x] Most recent Python and web lint checks passed.
- [x] Most recent web type checks and optimized production build passed.

Docker LLM follow-up: both published tool loops passed. Latest run
`99ca88a6-41c7-4861-a8d4-54fce55fa58c` measured clean queue time 0.000151 s
and contended queue time 1.094709 s. Dashboard history and its saved evidence download
were verified through the web API. The first attempted model call invented an argument and
was rejected without execution; corrective feedback is now supported and tested.
Both Python (including PostgreSQL) and web test suites were rerun for this follow-up;
web lint/typecheck/build results above remain from the previous UI implementation session.

Agent dashboard follow-up: fresh run `b5ebc2bd-d964-4cee-9a14-000d4a22cef8` passed;
four agent spans were retained in saved evidence and rendered as the measured model/tool timeline.
Expandable tool feedback and navigation back to the coding traces were checked in the browser.
The Agent view was checked at 390 px width with no page-level horizontal overflow.
Python and web tests, web lint, type checks, and production build passed for this integration.

Workspace redesign follow-up: 12 web tests, lint, type checking, and production build passed.
Browser verification covered run search/workload filters, span inspection, saved raw sample coverage,
matching/mismatched comparisons, persistent compact density, and automatic 15-second refresh.
All seven pages were checked at 390 px width; table overflow is contained within its own scroller.
The Python count above remains from the prior integration session; this UI-only change did not rerun it.

The Python run emitted a test-client dependency deprecation warning; the suite passed. Remote CI
execution has not been verified in this session, and passing these checks is not full-system coverage.

### Manual and local integration checks

- [x] CPU-only and coding test-tool demos passed their acceptance checks.
- [x] Seven OpenTelemetry spans from each demo were parsed and checked against their run ID.
- [x] Missing macOS CPU pressure readings appeared as unavailable, not synthetic zero values.
- [x] Search, attribution filters, run switching, evidence panel, waterfall, and telemetry view worked.
- [x] Responsive layout and mobile navigation were inspected in the browser.
- [x] The initial dashboard empty state was inspected before persistent history was added.
- [x] Two local experiments and a new coding run were ingested through the API.
- [x] Re-importing existing runs added no duplicate run, span, or sample records.
- [x] All three saved runs survived both database and API restarts.
- [x] Older-run evidence downloads remained correct after local artifacts were replaced.
- [x] Saved-run comparisons displayed the recorded timing differences.
- [x] History unavailability displayed a local-file fallback and history reconnected afterward.
- [x] Invalid web history cursors and unsupported artifact paths were rejected.
- [x] Browser checks reported no console errors or warnings in the final history view.

These browser and restart checks are manual evidence; they are not yet an automated end-to-end suite.

## Tests still needed for the existing foundation

These can be completed before adding an agent.

### Ingestion and database robustness

- [ ] Parallel identical imports: one saved run, consistent responses, and no duplicate evidence.
- [ ] Parallel conflicting imports: rejected changes with no partial writes or deadlocks.
- [ ] Conflicting sample payloads and sample append retries, including full-batch rollback.
- [ ] Explicit coverage of unknown step references, missing span attributes, timezone errors,
  invalid trace/span IDs, and nonfinite values in standalone telemetry append requests.
- [ ] Bounds and failure behavior for large batches, long metadata, and malformed request bodies.
- [ ] Database failure during reads and writes: correct 503 responses and usable recovery.
- [ ] Connection failures and interrupted publishing: local artifacts retained and retry succeeds.
- [ ] Migration startup from an empty database and simultaneous API startups.
- [ ] A future migration applied over populated history without losing prior evidence.
- [ ] Large history pagination with newly inserted runs, including the dashboard older-page control.
- [ ] Saved downloads during API failure never silently return the newest local file.
- [ ] Backup and restore into a separate database with results, spans, and samples intact.

### Dashboard and contract coverage

- [ ] Automated browser tests for overview → selected run → step evidence → export.
- [ ] Automated comparison tests, including different workload settings and removed selections.
- [ ] Automated refresh, reconnect, empty-history, invalid-history, and local-fallback tests.
- [ ] Confirm empty database instructions work with the recording command and ingestion API active.
- [ ] Sorting, combined filters, zero matches, older history pages, and long run/step labels.
- [ ] Keyboard navigation, focus restoration, screen-reader labels, and color contrast audit.
- [ ] Tooltip/detail readability on small screens and charts under reduced-motion preferences.
- [ ] Large-run performance, rendering limits, and table/waterfall virtualization if needed.
- [ ] Cross-browser checks beyond the in-app browser used so far.
- [ ] Full repository checks, including Go and shared packages, on the final integrated branch.
- [ ] Observe a successful remote CI run; configuration alone does not establish this.
- [ ] Contract compatibility between Python validation, web validation, and generated clients.

### Measurement validity

- [ ] Run the sampler on actual Linux hosts with CPU pressure and cgroup v2 enabled.
- [ ] Exercise supported and unavailable cgroup configurations, permissions, and malformed counters.
- [ ] Measure achieved sampling intervals, missed samples, and short-step coverage.
- [ ] Compare a genuinely CPU-starved task with a busy worker queue and a naturally long clean task.
- [ ] Verify boundaries under clock changes and specify monotonic timing for duration calculations.
- [ ] Measure profiler and sampler overhead against uninstrumented execution.
- [ ] Repeat experiments across more workload sizes and host loads; establish false-positive rates.
- [ ] Verify task timeout/cancellation cleans up worker and child processes under failure.

## Remaining implementation milestones

### Milestone 1 Live run lifecycle before an agent

- [ ] Add run creation and explicit queued, running, completed, failed, and cancelled states.
- [ ] Generalize step contracts beyond the completed CPU/coding experiment shape.
- [ ] Accept incremental step events and resource samples before a run is complete.
- [ ] Record parent/child span relationships and resource/service identity.
- [ ] Join samples to steps in the service rather than trusting only precomputed experiment summaries.
- [ ] Add automatic dashboard updates, connection status, last-update time, and stale-data indicators.
  Completed-history polling, source status, and last-refresh time now exist; live-run streaming and
  stale-event indicators remain unfinished.
- [ ] Display active steps, terminal failures, and incomplete telemetry without misleading totals.
- [ ] Provide an instrumentation client for controlled test workloads.

**Exit test:** A scripted workload creates a run, reports multiple steps while executing, and
finishes or fails. The dashboard updates automatically. Event retries, out-of-order delivery,
API interruption, and cancellation preserve a coherent lifecycle and readable evidence.

### Milestone 2 Three resource classes

- [ ] Add inference telemetry from vLLM queues/request timing and NVIDIA DCGM where available.
- [ ] Add retrieval telemetry from Qdrant requests and queue/service latency where observable.
- [ ] Add Prometheus collection and resource/service identity correlation.
- [ ] Extend rule attribution to CPU, inference, retrieval, none, and explicit uncertainty.
- [ ] Add CPU hardware saturation, inference, and retrieval views to the dashboard.
- [ ] Create bounded CPU, inference, retrieval, and mixed-contention injectors.

**Exit test:** Known isolated workloads exercise each resource and a clean control. Measurements
and attribution identify the affected class. Real GPU validation uses real inference hardware;
mock responses only validate plumbing. Missing signals and mixed contention remain explicit.

### Milestone 3 One instrumented coding agent

- [ ] Integrate one Python agent and its tool dispatcher using the generalized client.
- [ ] Propagate run/step/trace context through model calls, code execution, and retrieval.
- [ ] Freeze task inputs, model configuration, and replay state for repeatability.
- [ ] Add actual container isolation and resource limits for executed code.
- [ ] Bound task duration and preserve failure/cancellation evidence.

**Exit test:** One agent task produces a complete trace with real model/tool/retrieval activity.
Repeat it cleanly and with controlled contention. Verify correlation, cleanup, and measured overhead.

### Milestone 4 Reproducible dataset and strong rule baseline

Data pipeline progress (2026-10-06), in `pipelines/data` and [the data card](data-card.md):

- [x] Step dataset schema `steps-v1` with column roles and leakage guards (merged in #1).
- [x] Run extraction from the API and local evidence with contract validation (merged in #2).
- [x] Extraction integrity (2026-10-08): bounded retries, failure instead of partial
  extraction, cursor and API row validation, duplicate-run detection across sources, an
  evidence hash covering telemetry, and per-run telemetry status. Verified against the live
  ingestion API on a test database with seven real runs; 100% line and branch coverage of
  `pipelines/data`.
- [x] One row per step with per-window resource readings (merged in #3).
- [x] Row and label integrity (2026-10-09), `steps-v1.1` and `labels-v1.1`: conditions taken from
  the result structure, schema validation of every column (type, nullability, allowed values,
  bounds), telemetry coverage and status, run checks and injection method as lineage, refused
  labels for runs with different workload output, and a label fingerprint covering rule values
  and labeling code. Golden test on a recorded run in `pipelines/data/tests/fixtures/`; 100%
  line and branch coverage of `pipelines/data`.
- [x] Verification review fixes (2026-10-11), `steps-v1.2`: flattening and labeling reject bad
  runs individually instead of failing a whole build; strict JSON validation of evidence (no
  boolean or string coercion); evidence kind, check names, and LLM agent evidence verified;
  telemetry coverage judged against the declared sampling interval, with the measured interval
  recorded. Verified against the live ingestion API with seven real runs.
- [x] Dataset build (2026-10-11): `make dataset` writes a release named by a content-derived
  `dataset_version` with `steps.parquet` and `manifest.json`; rebuilds reproduce it byte for
  byte, failed builds write nothing, and manifests list every rejection with host paths
  redacted. Plan and dependencies in [the data pipeline plan](data-pipeline-plan.md).
- [x] Single-resource labels and the excess-latency target from a frozen, hashed rule file
  (`labels-v1`), with unit tests in `pipelines/data/tests/test_dataset_labels.py`.

- [ ] Define labels, slowdown thresholds, noise treatment, mixed-resource labels, and ties.
  Single-resource labels, thresholds, and noise treatment are defined in `labels-v1.1`; mixed
  contention and ties are not yet defined and fail labeling.
- [ ] Choose whether the regression target is observed queue time or estimated excess latency.
- [ ] Generate clean, isolated-contention, and mixed-contention replays over fixed tasks.
- [ ] Store versioned Parquet datasets with GCS/DVC or an explicitly chosen equivalent.
  Versioned Parquet releases are built locally; GCS/DVC storage is not yet set up.
- [ ] Validate data with TFDV or an explicitly chosen equivalent.
- [ ] Freeze splits grouped by task and related replay to prevent leakage.
- [ ] Hold out unseen tasks and at least one injection method.
- [x] Exclude injection configuration and label-generation metadata from model features
  (`LEAKY_COLUMNS` in `schema.py`; enforced by `test_leaky_columns_are_never_features`).
- [ ] Evaluate the expanded rule baseline before training the learned model.

**Exit test:** Rebuild labels and features from recorded evidence. Validate schema, missingness,
class balance, baseline noise, split independence, and feature leakage. Report rule metrics on the
frozen evaluation sets without tuning against them.

### Milestone 5 Learned attribution and serving

- [ ] Train LightGBM resource classification and latency-fraction regression.
- [ ] Track data versions, parameters, metrics, and artifacts in MLflow.
- [ ] Compare models and rules on the same held-out tasks and contention cases.
- [ ] Evaluate natural overload in addition to injector-driven experiments.
- [ ] Calibrate confidence, select an abstention threshold, and show uncertain results explicitly.
- [ ] Load a versioned registered model in the correlation service.
- [ ] Show model version, telemetry coverage, and estimated attribution in the dashboard.

**Exit test:** Report accuracy/macro F1, regression MAE, clean-run false positives, and calibration.
Validate missing features and model load failures. Measure time from step completion to visible
attribution. If ML does not outperform rules, report that result rather than weakening the baseline.

### Milestone 6 Monitoring and retraining

- [ ] Add operational metrics and repeated-bottleneck alerts.
- [ ] Run bounded canary evaluations on isolated test capacity.
- [ ] Monitor feature drift and known-label quality without presenting drift alone as accuracy loss.
- [ ] Build Airflow retraining with versioned inputs and tracked output models.
- [ ] Add CI promotion gates and rollback to a previous model.
- [ ] Integrate Grafana/Alertmanager if they add operational value alongside the product dashboard.

**Exit test:** A degraded candidate is rejected. Drift/canary failure triggers the intended workflow.
Retraining reproduces evaluation, promotion updates the served version, and rollback restores it.

### Milestone 7 Deployment and final demonstration

- [ ] Containerize the web application and ingestion/correlation service.
- [ ] Add Helm deployment, resource requests/limits, readiness checks, and service configuration.
- [ ] Implement GCP/GKE infrastructure, GPU quota/access, and artifact storage as needed.
- [ ] Add secrets management, trace redaction, retention rules, and access control for shared use.
- [ ] Document threat boundaries, backups, restoration, deployment, and incident procedures.
- [ ] Set cloud budgets and measure workload, instrumentation, and serving costs.
- [ ] Reconcile the unused Go control plane/CLI and placeholder SDK/Protobuf contracts with the
  adopted architecture; retire or implement them deliberately.
- [ ] Export/version the actual FastAPI contract and add clients/compatibility checks.
- [ ] Rehearse the reviewer-selected contention demo and preserve backup recordings/results.

**Exit test:** Start from a clean environment, deploy, record an agent task, identify a selected
bottleneck, inspect its evidence, and recover from a service/model failure. Validate secrets and
redaction, resource limits, backups, cost controls, and the full CI release path.

## Final project acceptance still to demonstrate

These are project targets, not achievements inferred from the current demo.

- [ ] All three resource classes work on the chosen workload.
- [ ] Clean evaluation steps are classified none at least 95% of the time on a declared test set.
- [ ] Learned attribution improves the chosen mixed-contention metric over a strong rule baseline;
  the original plan targets roughly 10 percentage points, subject to honest measured results.
- [ ] Attribution is visible within approximately 15 seconds of step completion under declared load.
- [ ] Instrumentation overhead meets a numerical budget chosen before final evaluation.
- [ ] Confidence is calibrated on held-out evidence and uncertainty is visible.
- [ ] An independent reviewer can choose a resource to contend and inspect the resulting diagnosis.
- [ ] Results, dataset/model versions, deployment instructions, and backup demonstrations are reproducible.

## Decisions to settle during the remaining work

- [ ] Define the duration/clock contract and what the UI may call queue, execution, or estimated work.
- [ ] Define which infrastructure telemetry can support a resource attribution versus contextual evidence.
- [ ] Choose the live transport and resource identity scheme.
- [ ] Decide the minimum role of Grafana/Prometheus alongside the custom dashboard.
- [ ] Set the primary evaluation metric, overhead budget, abstention criteria, and promotion gates.
- [ ] Confirm cloud budget and hardware access before adding deployment obligations.

## Commands and supporting evidence

```sh
# Local infrastructure and services, in separate terminals where needed
make infra-up
make dev-worker
make dev-web

# Import existing evidence or record a new run
make import-runs
make record-coding
make record-cpu

# Existing automated checks with PostgreSQL integration enabled
AGENTWATCH_TEST_DATABASE_URL=postgresql://agentwatch:agentwatch_dev@localhost:5432/agentwatch uv run pytest
pnpm --filter @agentwatch/web test
pnpm --filter @agentwatch/web lint
pnpm --filter @agentwatch/web typecheck
pnpm --filter @agentwatch/web build
uv run ruff check .
```

The database test connection above is the local development default. Each integration test uses its
own temporary schema. Without `AGENTWATCH_TEST_DATABASE_URL`, those database tests are skipped;
a passing reduced suite must not be reported as a complete database verification.

Implementation and test evidence lives in [the worker service](../services/worker/src/agentwatch_worker/),
[Python tests](../services/worker/tests/), [the test-tool runtime](../services/sandbox/),
[the dashboard](../apps/web/components/dashboard.tsx), and [web contract tests](../apps/web/tests/).
Setup details are in [the CPU demo guide](cpu-demo.md), [ingestion guide](ingestion-api.md),
and [database guide](../db/README.md).

## Keeping this checklist current

Mark implementation complete when its behavior exists and its stated exit test has evidence.
Keep remaining testing items open until they have been exercised. Record dates and artifact/test
references for new verification; distinguish manual checks from automation. Update this snapshot
when milestones change rather than converting all items into one overall progress percentage.

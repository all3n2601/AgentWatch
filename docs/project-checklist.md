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

Removed 2026-10-20 in PR #27: the single-tool demo (`agentwatch_worker.llm_demo`, `make
demo-llm`/`record-llm`) was replaced by the instrumented agent runner; its Ollama helpers moved
to `agentwatch_worker.ollama`, and the GPU diagnostics now drive the runner. The items below
record what the demo verified; its saved runs remain viewable in the dashboard Agent view.

- [x] Optional Compose profile with digest-pinned Ollama image and persistent model volume.
- [x] Qwen3 1.7B downloaded and executed on Docker CPU with real tool calling.
- [x] Allowlisted `run_tests` dispatcher rejects invented arguments and sends corrective feedback.
- [x] Real tool result returned to the model and followed by a final answer.
- [x] Agent conversation, inference timing/token metrics, and OpenTelemetry inference spans saved.
- [x] Model-invoked coding measurements saved in PostgreSQL and retrievable through the web API.
- [x] Rerun/start/stop commands documented in [the LLM guide](llm-demo.md), which now covers
  the model server for the agent runner.
- [x] Qwen3 agent timeline, inference request metrics, token counts, and expandable conversation
  displayed in a dedicated dashboard Agent view, with spans retained in saved result evidence.
- [x] Optional NVIDIA GPU Compose override and configurable Ollama host port; CPU-only default
  unchanged. Tool loop ran on one RTX 5070 laptop GPU, 2026-10-13, run
  `7afc596d-131f-4ba6-a374-4ffad0ccce01` ([GPU access record](gpu-measurement-access.md)).
- [x] Ollama model placement recorded from `/api/ps` instead of a hard-coded CPU label.
- [ ] Generalized ingestion of arbitrary agent/inference traces beyond this fixed tool loop.
- [ ] Tool execution isolated inside its own container.
- [ ] GPU inference, retrieval tools, agent failures, and longer multi-tool tasks validated.
  Only the fixed tool loop has run on a local laptop GPU; L4/GCP, vLLM, retrieval, failure,
  and multi-tool cases remain. The first request after creating the GPU container took
  53–64 s, and two first requests of later runs took 2.05–2.27 s, for unexplained reasons.
- [x] GPU resource samples recorded by AgentWatch itself (A04). A01 used an `nvidia-smi`
  diagnostic that is not ingested or correlated with steps. 2026-10-24, branch
  `feature/gpu-inference-telemetry` (not yet merged): the runner samples five versioned
  whole-device NVML gauges as live `inference` samples on the run's clock domain. Three real
  RTX 5070 Laptop runs (`a0709063…`, `2f8b2674…`, `3a4e8ce3…`) measured all metrics in every
  round, and the live API stored exactly the journaled samples
  ([A04 record](gpu-measurement-access.md#a04-gpu-and-inference-telemetry-recorded-by-agentwatch-october-24-2026)).
  Tests: `services/worker/tests/test_nvml.py`, `services/agent/tests/test_resources.py`.

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
  2026-10-13: PSI and cgroup v2 readings were available in WSL2 Ubuntu on the GPU laptop
  ([access record](gpu-measurement-access.md)); WSL2 measures its VM, so a non-WSL host remains.
  2026-10-24 (A04, branch `feature/cpu-retrieval-telemetry`, not yet merged): 24 real runs
  in two attempts sampled five CPU metrics in WSL2 and in Docker containers with their own
  cgroup, all `measured`. The containers share the WSL2 kernel, so a separate Linux host
  remains open ([A04 CPU record](gpu-measurement-access.md#a04-cpu-stall-cgroup-and-retrieval-timing-october-24-2026)).
- [ ] Establish a stable CPU baseline on the GPU laptop. 2026-10-13: 6 of 16 coding experiments
  passed the unchanged 15% variability gate (CVs 10.9–42.2%); validation is inconclusive there.
- [ ] Exercise supported and unavailable cgroup configurations, permissions, and malformed counters.
  2026-10-24 (A04): three real configurations: the shared WSL2 cgroup without a limit, and a
  container cgroup with `cpu.max` 1 CPU and 0.25 CPU. Throttling measured 0 where the limit
  did not bind, and 286–437 ms in test steps at 0.25 CPU. Missing PSI, cgroup or counters and
  malformed, negative and non-finite values are covered by fixtures in
  `services/agent/tests/test_resources.py` only. Permission failures, cgroup v1 and systemd
  per-service cgroups are not exercised.
- [ ] Measure achieved sampling intervals, missed samples, and short-step coverage.
  2026-10-24 (A04, GPU laptop/WSL2 only): runs record achieved intervals, late and missed
  rounds, and `coverage` reports per-step windows. Three GPU runs: median 0.250 s, max 0.290 s,
  no missed rounds; 0.24 s model calls were shorter than the 250 ms interval and one had no
  GPU reading. Open: other hosts, longer workloads and a choice of interval.
  2026-10-24 (A04 CPU/retrieval): `coverage-v3` joins each step only to its own resource and
  never reports a counter increase from one reading. 24 runs: median 0.250 s, max 0.299 s, no
  missed rounds; every retrieval and most tool steps were shorter than one interval, and
  about half of the model-call windows held fewer than two CPU readings.
- [ ] Compare a genuinely CPU-starved task with a busy worker queue and a naturally long clean task.
- [ ] Verify boundaries under clock changes and specify monotonic timing for duration calculations.
- [ ] Measure profiler and sampler overhead against uninstrumented execution.
  2026-10-24 (A04): `resource-sampling-overhead-v1`, two attempts of 30+30 real GPU requests:
  sampled minus unsampled median −0.5 ms and +3.1 ms, within a 54 ms shift between attempts,
  so no effect was resolved; sampler thread 2.8–3.6 ms CPU per round. Open: live delivery
  under sustained sampling, other hosts, and a declared overhead budget (O01).
  2026-10-24 (A04 CPU/retrieval): agent-overhead-v1 with five CPU metrics, two attempts each in
  WSL2 and a 1-CPU container: median wall −0.4% to +4.9% (sign varies, not resolved);
  runner CPU +4.2 to +5.3 ms per repetition; sampler thread 1.0–2.4% of a core in runs.
- [ ] Repeat experiments across more workload sizes and host loads; establish false-positive rates.
- [ ] Verify task timeout/cancellation cleans up worker and child processes under failure.

## Remaining implementation milestones

### Milestone 1 Live run lifecycle before an agent

- [x] Add run creation and explicit queued, running, completed, failed, and cancelled states.
- [x] Generalize step contracts beyond the completed CPU/coding experiment shape.
- [x] Accept incremental step events and resource samples before a run is complete.
- [x] Record parent/child step/span identity and registered resource/service identity.
  B01, October 14, 2026: [live API and handoff](live-ingestion.md), strict Python producer
  records/client, generated OpenAPI/TypeScript types and additive migration 0002. Isolated
  PostgreSQL regressions in `services/worker/tests/test_live_api.py` cover ordering, retries,
  conflicts, rollback, gaps, cancellation, late evidence and populated-schema upgrades.
  Cancellation requests producer cleanup; process termination remains unverified A03 work.
  Local verification: 578 Python tests passed with the dedicated PostgreSQL database (no skips),
  Ruff passed, and JavaScript lint/typecheck/13 tests/build passed. Go formatting/vet/tests and
  Compose validation passed. A real HTTP Python SDK smoke exercised synthetic test-database
  creation, missing revisions, prefix application, completion, retries and conflict rejection.
  PR CI/reviewer acceptance and actual producer/frontend integration remain separate gates.
  PR #25 audit follow-up, October 14, 2026: reject NUL/invalid Unicode and UTC overflow;
  reject impossible pending histories atomically; preserve large-float retries and bounded
  cursors; tolerate additive schema/response fields and newer read-side metrics; publish both
  error detail shapes. Migration 0003 removes only a redundant index, preserving its unique
  constraint and populated live evidence. Existing malformed pending evidence remains readable
  for explicit failure/restart. `services/worker/tests/test_live_audit.py` and upload/SDK regressions
  bring the suite to 635 passing Python tests with dedicated PostgreSQL and no skips (57 added).
  Ruff and JavaScript lint/typecheck/13 tests/build pass. Real uvicorn/SDK smoke verifies
  rollback, validation, numeric retries, cancellation polling and 413 before 100 Continue.
- [ ] Join samples to steps in the service rather than trusting only precomputed experiment summaries.
  B02, October 24, 2026 (branch `feature/step-sample-features`): shared
  [`step-features-v1`](step-features.md) joins applied live StepEvents to same-run,
  same-resource/service, same-clock samples in `[start, finish)` and validates FeatureRows;
  `agentwatch_worker.live_features` loads them from PostgreSQL. `steps-v1.2` window code moved
  unchanged into `agentwatch_features.legacy`, so training and serving share one implementation.
  Regressions: 55 synthetic cases in `packages/features/tests/test_live_features.py` (null vs
  zero, units, counter resets, half-open and incomplete windows, cancellation, clock/identity
  joins, retries, conflicts) and 4 isolated-schema PostgreSQL cases in
  `services/worker/tests/test_live_feature_join.py`. Local verification after merging `main` at e60984a: 1071 Python tests passed
  with a dedicated PostgreSQL 17 test database and no skips; Ruff passed. Open until B03 serves
  rows from a real run: no endpoint calls the join yet, live runs record no declared sampling
  interval (coverage stays null without one), and only synthetic fixtures have been joined.
  2026-10-24: the producer-side `coverage-v3` report applies the contract's join rules to a
  run's journal (own resource and service, registered clocks, `[started, finished)`), aligned
  with `step-features-v1`'s statuses and join key; it does not import the service join.
- [ ] Add automatic dashboard updates, connection status, last-update time, and stale-data indicators.
  Completed-history polling, source status, and last-refresh time now exist; live-run streaming and
  stale-event indicators remain unfinished.
  F01, October 18, 2026 (branch `feature/live-run-view`): the header shows the actual history
  source and last load time on all viewports. The `/live` view shows feed freshness (receiving,
  stale, closed, no records) and last-record age from the browser's monotonic clock only, so
  server/browser clock offsets cannot distort it (`apps/web/tests/live-replay.test.ts` covers
  ±60 s). Review follow-up, October 20, 2026: the age keeps advancing after a run closes, while
  freshness stays closed; the `no-records` replay demonstrates the no-records state with a null
  receipt; the freshness status is a polite live region so the stale transition is announced.
  Its only data source is a synthetic development replay; F02 must poll the B01 live read API
  before automatic live updates exist. Keep this open until then.
  F02, October 24, 2026 (PR #39, branch `feature/live-feed`, building on the #31 prototype;
  in review): `/live` polls the B01 live read API in the browser through validated web routes
  (`apps/web/app/api/live/runs/`) instead of `router.refresh()`. The selected run is read every
  2 s after the last record sequence received; a failed read keeps the last data and selection
  and retries with backoff (1–30 s, ±20% jitter); polling stops once a closed run is drained
  (with a check for late evidence that shows it is checking, then when it checked and how many
  new records it found, or its error) and pauses while the tab is hidden. A connection status
  shows connected, reconnecting (attempt, delay, cause) and stopped states. Feed age is anchored
  on the browser's monotonic clock when new records arrive, never on `received_at`. The
  `?replay=` path is removed; its scenarios are labeled synthetic browser-test fixtures
  (`apps/web/e2e/fixtures/live-scenarios.ts`). Verification: 69 web unit tests
  (`apps/web/tests/live-poll.test.ts` covers failure then recovery, backoff bounds and ±60 s
  skew), 24 Playwright tests (`apps/web/e2e/live-feed.spec.ts`: active to terminal, reconnect,
  failures, ±60 s skew, late-evidence checks; `live-scenarios.spec.ts`), lint, typecheck and build. Manual checks
  against the real stack (`make infra-up`, `dev-worker`, `dev-web`, `agent-overhead`): run
  selection, active to Completed with final totals, reconnect after a worker restart, not-found
  and invalid UUIDs. Final manual check on the real stack, October 24, 2026: every check was
  done first in a normal desktop browser window on a MacBook, then the live view at iPhone SE
  and 768 px widths, 200% zoom and keyboard navigation; "Check for late evidence"
  shows that it is checking, then its result; `?replay=` no longer shows a replay.
  PR #39 review fixes, October 24, 2026: jittered retry delays are clamped to the 1–30 s bounds,
  and cached record dispositions are reconciled with each snapshot's step state, so a pending
  completion applied later without redelivery ends its step on the timeline
  (`apps/web/tests/live-poll.test.ts`; browser regression in `apps/web/e2e/live-feed.spec.ts`).
  After merging current `main` (#38's CPU counters keyed by metric): 72 web unit tests, 25
  Playwright tests, lint, typecheck and build pass. Left unchecked for reviewers to confirm.
- [ ] Display active steps, terminal failures, and incomplete telemetry without misleading totals.
  F01, October 18, 2026: `/live` renders StepEvent records through one adapter
  (`apps/web/lib/step-events.ts`) that applies each step's contiguous revision prefix and
  reports gaps. It shows run state, active steps (state and start time only, no computed
  elapsed time), failed/cancelled steps with reasons, and step totals labeled final only when
  the run is closed, every step is terminal and no revision or record is missing; otherwise
  sums are labeled partial. CPU time is totalled only when fully measured in one scope.
  Development replay (`?replay=<scenario>`, non-production builds only, labeled synthetic)
  scripts active, completed, failed, cancelled-before-start, out-of-order, no-records, stale,
  missing-CPU and zero-duration runs from `apps/web/fixtures/step-event-replays.ts`.
  Verification: 52 web tests in `apps/web/tests/` (adapter, status helpers, each scenario,
  replay gating, browser-clock receipt age), lint,
  typecheck and build pass; a production build ignores `?replay=`. Headless Chrome screenshots
  were checked at 360, 375, 640 (200% zoom width), 768 and 1440 px with no page-level overflow.
  Manual browser checks at commit `a355f8d`: keyboard navigation (focus visible; replay
  scenarios open with Enter) and 200% browser zoom. Real live runs are not displayed until F02 connects the B01 read API,
  so this stays open.
  F02, October 24, 2026 (PR #39, branch `feature/live-feed`): step state, timings and gaps now come from
  the service's validated snapshot (`apps/web/lib/live-snapshot.ts`), so pending and expired
  revisions are never applied and expired gaps are reported as evidence that will never be
  applied (`apps/web/tests/live-snapshot.test.ts`: pending → expired → filled prefix). Totals
  no longer add a parent step to the children it contains: the span is the wall time covered
  by any step within one clock domain, and queue/execution/CPU sums cover leaf steps
  (`apps/web/tests/live-totals.test.ts`; a 5.44 s nested run no longer shows ~10.9 s). Live
  samples are charted per resource, service and metric, with not-measured readings counted by
  availability state and shown as gaps (`apps/web/tests/live-feed-data.test.ts`). A selected
  UUID is read directly, so runs older than the newest 50 are selectable. The view states that
  saved evidence is unavailable for live runs and offers a JSON export of the selected UUID's
  live records, read page by page with gaps and pending/expired records unchanged and labeled
  with its read time (`apps/web/tests/live-export.test.ts`). Browser tests in
  `apps/web/e2e/live-feed.spec.ts` cover unavailable samples, a multi-page UUID-correct export
  after switching runs mid-read, and a run outside the newest 50. Final manual check on the
  real stack, October 24, 2026: the downloaded JSON carries the selected run's UUID, and the
  saved-evidence unavailable state is shown. Left unchecked for reviewers to confirm.
- [ ] Link live runs to saved evidence. Live records are kept apart from saved run history
  (ADR 0001) and agent runs report only to the live API (ADR 0002), so a live UUID has no
  saved-run counterpart; F02 shows that as unavailable. Linking them needs its own API task.
- [x] Provide an instrumentation client for controlled test workloads.
  A02, October 16, 2026: [`services/agent`](../services/agent/README.md) records model,
  tool and retrieval steps through the B01 SDK with background batched delivery, bounded
  retries, per-record isolation of rejected batches and a replayable journal. Real uvicorn,
  PostgreSQL and `LiveClient` tests in `services/agent/tests/test_live_integration.py` cover a
  completed run, dashboard cancellation and idempotent replay. The dashboard's live consumer
  (F02) is separate.

**Exit test:** A scripted workload creates a run, reports multiple steps while executing, and
finishes or fails. The dashboard updates automatically. Event retries, out-of-order delivery,
API interruption, and cancellation preserve a coherent lifecycle and readable evidence.

### Milestone 2 Three resource classes

- [ ] Add inference telemetry from vLLM queues/request timing and NVIDIA DCGM where available.
  2026-10-24 (A04): NVML GPU gauges and per-call Ollama request timing
  (`ollama-request-timing-v1`) on the laptop; the inference queue is recorded as unavailable
  because Ollama reports none. vLLM and DCGM are not installed there and remain open.
- [ ] Add retrieval telemetry from Qdrant requests and queue/service latency where observable.
  No retrieval metric is registered; retrieval is in-process BM25. 2026-10-24 (A04): each
  search records `retrieval-timing-v1` (service time apart from dispatch, index queue
  `unsupported`); 12 real searches took 46–117 µs of service time against 0.9–12.9 ms of
  dispatch ([A04 CPU record](gpu-measurement-access.md#a04-cpu-stall-cgroup-and-retrieval-timing-october-24-2026)). Qdrant and networked retrieval remain open.
- [ ] Add Prometheus collection and resource/service identity correlation.
- [ ] Extend rule attribution to CPU, inference, retrieval, none, and explicit uncertainty.
- [ ] Add CPU hardware saturation, inference, and retrieval views to the dashboard.
- [ ] Create bounded CPU, inference, retrieval, and mixed-contention injectors.
  2026-10-25 (A05, [ADR 0006](adr/0006-bounded-injectors-and-agent-replay.md),
  [injectors README](../services/injectors/README.md)): `services/injectors` has CPU spin
  processes, competing loopback-only Ollama requests, and an in-process retrieval index flood,
  run alone or mixed. Each has validated duration and resource caps, a watchdog, a child
  backstop, terminate/kill escalation, and confirmed cleanup that raises when unconfirmed
  (57 tests with real processes and threads). A real GPU experiment had every injector active
  and cleaned up on request ([A05 record](gpu-measurement-access.md#a05-bounded-injectors-and-replay-experiment-october-25-2026)).
  The retrieval flood also slows the runner's other threads, including its sampler (0.30–0.39 s
  median instead of 0.25 s), so it is not retrieval-exclusive. Open until review and merge.

**Exit test:** Known isolated workloads exercise each resource and a clean control. Measurements
and attribution identify the affected class. Real GPU validation uses real inference hardware;
mock responses only validate plumbing. Missing signals and mixed contention remain explicit.

### Milestone 3 One instrumented coding agent

A02 progress (2026-10-16), [ADR 0002](adr/0002-agent-runner-evidence-and-tool-safety.md): the
runner exposes allowlisted `search_docs`, `read_file`, `apply_patch` (reviewed patches only) and
`run_tests` tools over two fixed tasks. 118 agent/sandbox tests cover rejected calls, tool and
model failures, limits, remote/local cancellation, cleanup, outages and replay. Local
verification on the final code: 749 Python tests with the dedicated PostgreSQL database and no
skips, Ruff, JavaScript lint/typecheck/13 tests/build (Node 22), Go format/vet/tests and both
Compose configurations passed. Agent tests passed 10 consecutive runs and the live integration
tests 15, after a test race in dashboard cancellation was fixed. Remote CI is not yet observed. Real Qwen3 1.7B runs on
the development Mac's Docker CPU, all kept: final-code run `30d40fd4-a062-4703-a690-aecc44390060`
**completed the task**: search, one rejected read, `read_file`, the correct patch, `run_tests`,
a final answer, and runner verification passing 5/5 tests. Its 13 steps closed with no gaps
and all 1,726 records were delivered; model calls took 17–68 s. Two earlier runs, on code
before the final edits, produced correlated live runs but did not finish:
`97fb140d-be6f-42f9-b31c-0d1ab305aabf` failed when a model request hit its 300 s socket
timeout, and `301307b7-b975-4e2a-ab9b-ce55f0b3ec32` (after lowering `num_ctx` to 4096) hit
the 900 s run limit. Their model calls took 38–199 s for about 22 output tokens, versus
17–68 s in the passing run with the same configuration; the cause of this variation is unknown.
Both failed runs searched, had one invalid read rejected, read `stats.py` and applied the
correct patch before stopping. All records were delivered and no step was left unfinished.
Overhead protocol `agent-overhead-v1` (30 alternating repetitions of the 55 ms fixture
suite): median 0.0547 s instrumented vs 0.0548 s uninstrumented, with +1.4 ms runner CPU per
repetition. The workload is shorter than the 250 ms sampling interval, so sustained sampling
cost is not characterized.

PR #27 review follow-up (2026-10-20): closures wait for delivery (otherwise they are
journaled for `resend`), cleanup actions run independently, setup failures close the run or
leave nothing behind, `resend` isolates refused records and verifies the final run state,
stale revisions are retried once, the root step fails with a failed run, signal handling is
lock-safe, placement reads honour cancellation and the deadline, overhead runs keep the task
ID and close on failure, and fixtures are pinned to LF. The `llm_demo` loop was migrated onto
the runner. 26 regression tests fail or hang on the previous head (c7cd269) and pass now. Local
verification: 766 Python tests with the dedicated PostgreSQL database and no skips (agent tests
10 consecutive runs), Ruff, JavaScript lint/typecheck/13 tests/build, Go, and both Compose
configurations passed. Real run `51b868fb-f838-4712-bf3e-cd1198d6000d` on this code completed
the task (verification 5/5, 13 steps, no gaps, placement measured on all 6 model calls). The
two changed dashboard empty-state messages were not checked in a browser.

GPU end-to-end validation (2026-10-20, reported by Benoy, who holds the evidence files): run
`63f8e154-e503-480c-845e-9c52ca612577` at commit `2dd827c` on an RTX 5070 Laptop GPU (8,151 MiB)
through Docker Desktop and WSL2, Ollama 0.35.1 with the GPU override, `qwen3:1.7b` (Q4_K_M,
context 4,096). It completed with verification 5/5, 13 steps, no unfinished steps, and 94 of 94
records delivered. Placement was measured on all 6 model calls with 100% of the model in VRAM.
This run predates the second and third review rounds below; their changes affect failure,
outage and replay paths, not the success path it exercised.
The first call took 4.89 s (4.29 s model load in a fresh container) and the others 0.19–0.54 s.
CPU PSI and the cgroup counter were measured in all 28 sampling rounds (median 0.250 s, max
0.253 s). One run was made; it was not repeated. This is a laptop GPU, not L4/GCP evidence.

Second review round (2026-10-20): the GPU probe's preflight uses `/readyz`; failed model calls
no longer time the placement read; overflowed records and refused completions defer the
closure; a dashboard closure is journaled so `resend` settles; an API outage at the start of a
run is recoverable (exit 3); a verification timeout is reported as such; local setup failures
are not attributed to the live API; refused records exit 5; and overhead runs use their own
`agentwatch-agent-overhead` service. 13 regression tests fail on `2dd827c` and pass now; a 14th,
for the overhead exit code, was written later and is counted in the third round.

Third round (pre-merge audit, 2026-10-20), from three independent reviews of every module:
`resend` maps outages to exit 3 and unusable journals to exit 2 instead of tracebacks; never
registered journals are refused, and an unconfirmed registration journals a closure so
`resend` closes the run; a journal without a closure exits 6, and `--fail-unclosed` closes it;
a completion refused after full delivery closes the run as failed (exit 5); after a queue
overflow, records stay journal-only so revisions arrive in order; an interrupted journal line
no longer swallows later entries; a remote closure is journaled before its refresh. In the
runner, steps end before shutdown and cleanup; a background-thread failure fails the run and
is listed in `background_errors`; the run limit has one wording, including for a tool cut
short by it; workspace copy failures leave no temporary directory; an unwritable output
directory is a setup failure; a remotely failed run fails its steps; the root step's queue
delay excludes runner start-up; and an abandoned request is recorded only if one was sent.
`overhead` validates its patch, maps registration failures to exit 4, cancels cleanly on
SIGINT/SIGTERM, keeps the measurement on a transient completion failure, and no longer
swallows closing-flush errors. `run_project_tests` now enforces timeouts shorter than its poll
interval (a suite finishing after its deadline was reported as passed). 27 regression tests
fail on the pre-round code, on behavior rather than imports, and pass now.

A fresh review of the third round's own changes found five more defects, now fixed: `resend`
after a refused completion stays at exit 5 on every rerun; an interrupted journal line stays
replayable after replay appends (it is marked when the journal is reopened); a background
error during shutdown fails the run; an adopted remote closure records the remote reason; and
`overhead` reports refused records with exit 5. Each has a regression test that fails when its
fix is undone (checked by reverting each fix in a scratch copy).

Verification on the final code (2026-10-20): 809 Python tests with the dedicated PostgreSQL
database and no skips, agent and sandbox tests in 5 consecutive runs, Ruff, JavaScript
lint/typecheck/13 tests/build, Go, and both Compose configurations. Real Qwen3 CPU run
`592ef981-f125-40e6-843a-5c2d8988765d` on the third-round code completed the task
(verification passed, 13 steps, no gaps, no cleanup or background errors, root-step queue delay
0.0002 s); the five final fixes do not touch the success path it exercised.

- [x] Integrate one Python agent and its tool dispatcher using the generalized client.
  Verified by the integration tests, the real CPU runs above, and Benoy's GPU run
  `63f8e154-e503-480c-845e-9c52ca612577`, each of which completed the task.
- [ ] Propagate run/step/trace context through model calls, code execution, and retrieval.
  Steps share run, trace and parent context inside the runner; trace context is not
  propagated into the Ollama server or the test subprocess.
- [ ] Freeze task inputs, model configuration, and replay state for repeatability.
  Task inputs are pinned by content hash and the configuration digest is registered. 2026-10-25
  (A05): `agent-replay-v1` replays a verified run's exact model inputs and tool calls, so the
  step path is fixed. Replays of one plan register one digest. On the GPU laptop, 5 clean
  baselines had an `agent-run` CV of 2.9% on the final code (3.4–4.3% earlier; gate ≤ 15%).
  Three of four source runs' first reply differed in every replay; the other replies matched. Only `fix-mean-empty` was
  replayed.
- [ ] Add actual container isolation and resource limits for executed code.
  A03 (2026-10-24), [ADR 0004](adr/0004-container-isolation-for-tool-execution.md) and the
  [sandbox README](../services/sandbox/README.md): `run_project_tests_isolated` runs a suite in a
  disposable Docker container. It has a private read-only staging copy and a read-only root, a
  bounded `noexec` `/tmp`, no `/dev/shm`, no network, uid 65534 with an empty capability
  bounding set and `no-new-privileges`, memory/CPU/process/file limits, and output drained into
  a bounded tail. A host deadline is enforced from creation onward, plus a backstop in a PID 1,
  non-dumpable wrapper the suite cannot signal, end or read. Leftovers are swept by a deadline
  label, and the image is digest-pinned and never pulled implicitly. Probe suites in
  `services/sandbox/tests/test_container.py` (60 tests) verify each boundary from inside the
  container. Each of 21 controls was also checked by removing it and confirming its probe fails.
  Every outcome (passed, failed, timed out, cancelled, resource limit, rejected request, start
  failure) is tested to remove its container and staging copy; a failed removal raises.
  Verified on Docker Desktop for macOS (arm64) and in CI on Docker Engine on Linux (amd64)
  (the PR #35 `python` job ran them with none skipped).
  Known limitation: a suite can exit 0 itself and appear to pass (see the threat model). Open
  until the agent runner switches to this executor (Benoy's integration); the agent still uses
  the trusted-fixture subprocess runner.
- [x] Bound task duration and preserve failure/cancellation evidence.
  Model-call, tool-call, test and run limits are enforced (real run `301307b7…` closed at the
  900 s limit), and cancellation ends every step with recorded cleanup. Service-confirmed
  cleanup remains A03.

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
  byte, failed builds publish nothing, and every build (new or reused) writes a report listing its
  sources and rejections with host paths redacted. Plan and dependencies in [the data pipeline plan](data-pipeline-plan.md).
- [x] Single-resource labels and the excess-latency target from a frozen, hashed rule file
  (`labels-v1`), with unit tests in `pipelines/data/tests/test_dataset_labels.py`.

- [ ] Define labels, slowdown thresholds, noise treatment, mixed-resource labels, and ties.
  Single-resource labels, thresholds, and noise treatment are defined in `labels-v1.1`; mixed
  contention and ties are not yet defined and fail labeling.
- [ ] Choose whether the regression target is observed queue time or estimated excess latency.
- [ ] Generate clean, isolated-contention, and mixed-contention replays over fixed tasks.
  2026-10-25 (A05): `make agent-experiment` writes clean, `cpu`, `inference`, `retrieval` and
  `mixed` replays, plus an `experiment.json` manifest and `injections/` logs that keep the
  conditions out of replay evidence. Four real runs of the protocol on one task are recorded; the second task,
  repeats and labels for these replays (D02) are open.
- [ ] Store versioned Parquet datasets with GCS/DVC or an explicitly chosen equivalent.
  DVC versioning is in place (2026-10-18): `make add-evidence` curates runs into an immutable
  evidence store, `make version-dataset` verifies and snapshots the evidence and dataset stores,
  and `pipelines/versioning/tests` shows a fresh clone rebuilding a versioned release byte for
  byte through a local remote. Open until the GCS remote exists (needs F's bucket).
- [x] Validate data with TFDV or an explicitly chosen equivalent (2026-10-12): `make
  validate-dataset` checks a release with TFDV 1.21 against the frozen schema in
  `pipelines/validation/schema/steps.pbtxt`; Parquet types that differ from the release's contract
  are refused, schema anomalies fail the gate, and drift against the previous passing release is
  reported. Each attempt commits its result and statistics as one directory (2026-10-13). Gate
  tests at 100% line and branch coverage.
- [x] Freeze splits grouped by task and related replay to prevent leakage (2026-10-13): `make
  split-dataset` splits only releases with a passing quality-gate verdict, 70/15/15 by hashed
  `task_id` from the evaluation contract, keeps non-measured runs out of test, records per-split
  row hashes, and runs the contract's single-feature AUC leakage check on the training split
  (`test_dataset_splits.py`, 100% line and branch coverage). Unseen tasks still depend on a real
  `task_id`.
- [x] Orchestrate the dataset stages with Airflow (2026-10-24): the `agentwatch_dataset` DAG in
  `pipelines/orchestration` runs acquire → build → validate → split → version → push →
  summarize, each in its stage's own environment, with only the DVC push retried
  (`pipelines/orchestration/tests`). Run end to end with `make pipeline-run` on the shared
  `steps-v1.2+5954f97ef8b2` evidence and with a new recorded run added. Alerts, committing
  `data/*.dvc` from the DAG, and the Gantt-chart optimization are open.
- [ ] Hold out unseen tasks and at least one injection method.
- [x] Exclude injection configuration and label-generation metadata from model features
  (`LEAKY_COLUMNS` in `schema.py`; enforced by `test_leaky_columns_are_never_features`).
- [ ] Evaluate the expanded rule baseline before training the learned model.
  2026-10-26 (M02): `make rule-baseline` evaluates `rules-v1` against `eval-contract-v1`
  ([rule baseline](rule-baseline.md), `pipelines/model/tests`). On the interim
  `steps-v1.2+5954f97ef8b2` split it reports blocked by the leakage check, with no metrics.
  Open until a release covering all four classes and mixed contention passes that check.

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

### Optimization extension planned October 13 2026

The [optimization layer plan](optimization-plan.md) adds the following future work. All items
are unimplemented/unverified; the planning update is not execution evidence. O labels match
task titles on the Team Planner; its generated card IDs are separate. The board stages run
from M3 through M6 and do not renumber the implementation milestones above.

- [ ] O01 — Rohit: accept policy/evaluation contracts and numerical gates after C01/A04/M01.
- [ ] O02 — Allen: validated versioned manual controls and audit history after O01/B01/B02.
- [ ] O03 — Akshayaa: bounded runner admission and cleanup after O02/A02/A03.
- [ ] O04 — Akshay: paired-run manifests and comparison validation after O01/D02.
- [ ] O05 — Meghana: manual controls and UUID-specific comparison UI after O02/F02.
- [ ] O06 — Benoy: real hardware evaluation and restore checks after O03/O04/O05/A05/A07.
- [ ] O07 — Rohit, stretch: shadow-first bounded adaptive rules after O06; accepted gates
  and operator opt-in required for actuation.
- [ ] X01 includes evaluated manual controls and honest pass/fail/inconclusive results after O06.
- [ ] X02 retains optimization contracts, manifests, runbooks and failed/inconclusive evidence.

**Exit test:** Compare preregistered fixed limits against baseline on compatible repeated
workloads; record completion time, throughput, tail latency, task quality, overhead and failures.
Prove cancellation/permit cleanup, missing-evidence handling and disable/restore. Keep numerical
gates proposed until O01 accepts them, preserve the existing CPU noise gate, and do not claim
improvement or enable adaptive control from an uncontrolled/inconclusive comparison.

### Core acceptance targets

These are project targets, not achievements inferred from the current demo.

- [ ] All three resource classes work on the chosen workload.
  2026-10-24 (A04): CPU and GPU samples and per-call inference and retrieval timing are
  recorded for the fixed agent tasks on one laptop. Retrieval has no sampled resource, and no
  contention has been attributed yet (A05, rule attribution).
- [ ] Clean evaluation steps are classified none at least 95% of the time on a declared test set.
- [ ] Learned attribution improves the chosen mixed-contention metric over a strong rule baseline;
  the original plan targets roughly 10 percentage points, subject to honest measured results.
- [ ] Attribution is visible within approximately 15 seconds of step completion under declared load.
- [ ] Instrumentation overhead meets a numerical budget chosen before final evaluation.
- [ ] Confidence is calibrated on held-out evidence and uncertainty is visible.
- [ ] An independent reviewer can choose a resource to contend and inspect the resulting diagnosis.
- [ ] Results, dataset/model versions, deployment instructions, and backup demonstrations are reproducible.

## Decisions to settle during the remaining work

C01 proposal (2026-10-11): [shared live contracts](shared-event-contracts.md) documents
StepEvent, ResourceSample, FeatureRow and Prediction, including identity, clocks, unavailable
readings, retries, failure/cancellation and abstention. This is a specification for team review,
not implemented live ingestion. All three teams' acceptance, API/SDK implementation and lifecycle
verification remain open; no milestone or verification checkbox is completed by this proposal.

- [ ] Define the duration/clock contract and what the UI may call queue, execution, or estimated work.
- [ ] Define which infrastructure telemetry can support a resource attribution versus contextual evidence.
- [ ] Choose the live transport and resource identity scheme.
- [ ] Decide the minimum role of Grafana/Prometheus alongside the custom dashboard.
- [ ] Set the primary evaluation metric, overhead budget, abstention criteria, and promotion gates.
  Proposed in `eval-contract-v1` (2026-10-09): [the evaluation contract](evaluation-contract.md)
  and `pipelines/model/evaluation-contract.toml`. Open until reviewed by the dataset and
  promotion-check owners.
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

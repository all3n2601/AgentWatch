# Attribution serving (B03)

The worker attributes each terminal step of a live run with the `rules-v1` verdict and, when a
model pair is configured, the M03/M04 model's calibrated prediction. The wire contract is in
[live ingestion](live-ingestion.md#attribution); the meaning of a prediction is in the
[model serving contract](model-serving-contract.md). This page covers configuration, model
states and the step-completion-to-visible-attribution latency measurement.

## Configuration

Three environment variables, read once when the worker starts (documented in `.env.example`):

| Variable | Meaning |
| --- | --- |
| `AGENTWATCH_MODEL_CANDIDATES_DIR` | Folder holding `classifier.txt`, `regressor.txt` and `preprocessing.json` |
| `AGENTWATCH_MODEL_CALIBRATION_DIR` | Folder holding `calibration.json` |
| `AGENTWATCH_ALLOW_UNPROMOTED_MODEL` | `1` or `true` serves a pair whose calibration has `blocks_promotion: true`, labelled not promoted |

Set both folders or neither; leaving both empty serves rules only. `make train-candidates
SPLIT=<split file>` writes the candidates folder and `make calibrate SPLIT=<the same split>` the
calibration folder (defaults `.data/candidates/` and `.data/calibration/`; see
[model training](model-training.md) and [model calibration](model-calibration.md)).

The current M03/M04 pair is **not promoted**: its `calibration.json` records
`blocks_promotion: true` because four `eval-contract-v3` gates are `not_evaluated`. The worker
refuses it (`not_promoted_refused`) unless `AGENTWATCH_ALLOW_UNPROMOTED_MODEL=1`, and then labels
every prediction not promoted. Promotion, rollback and registry loading are M05; the worker reads
local folders only and never changes a pair's promotion state.

```bash
AGENTWATCH_MODEL_CANDIDATES_DIR=.data/candidates \
AGENTWATCH_MODEL_CALIBRATION_DIR=.data/calibration \
AGENTWATCH_ALLOW_UNPROMOTED_MODEL=1 \
make dev-worker
```

## Model states

`GET /v1/live/attribution/model`, and the `model` object of every attributions page, report one
state:

| `state` | Meaning |
| --- | --- |
| `not_configured` | No folders set; rows carry the rules verdict and model outcome `unavailable` |
| `load_failed` | A load check failed; `reason` names it (`artifact_unreadable`, `hash_mismatch`, `version_mismatch`, `not_evaluated`, `not_fitted`, `columns_mismatch`, `features_mismatch`, `lightgbm_unavailable`, `model_load_failed`) |
| `not_promoted_refused` | The pair loaded but blocks promotion and the flag is not set; `identity` is reported, nothing is predicted |
| `ready_not_promoted` | That pair is served because the flag allows it; `not_promoted` is `true` |
| `ready_promoted` | A pair with `blocks_promotion: false` is served |

The pair is loaded once at startup; changing the files needs a restart, after which new rows
carry a new `serving_key` beside the old ones.

## API summary

- `GET /v1/live/runs/{UUID}/attributions?after=0&limit=50`: stored attributions of the selected
  run after the cursor, oldest first, with the serving model's status. Unknown UUID `404`; a run
  without terminal steps is an empty page.
- `GET /v1/live/attribution/model`: the model status on its own.

A terminal step event posted to `POST /v1/live/runs/{UUID}/records` starts a background pass
after the response; a read attributes any terminal step still lacking a row. Rows are
append-only. Field meanings, cursors and the late-sample flag are in
[live ingestion](live-ingestion.md#attribution).

## Measuring step completion to visible attribution

`agentwatch_worker.attribution_latency` drives a synthetic workload through a running worker's
live API and writes a JSON report:

```bash
make measure-attribution-latency BASE_URL=http://127.0.0.1:8090 STEPS=50 \
  POLL_INTERVAL_SECONDS=2.0 STEP_DURATION_SECONDS=0.2 VISIBLE_TIMEOUT_SECONDS=30 \
  OUTPUT=.data/attribution-latency/report.json
```

It registers one live run (task `synthetic-latency-v1`, one `cpu_sandbox` resource, declared
sampling interval of a quarter of the step duration) and moves it to running. Steps run one at a
time. Each step posts its queued and running events, sleeps for the step duration while taking
four made-up readings of `cpu.psi.some.avg10.v1` and `cpu.cgroup.throttled_usec.v1`, posts
those samples, then posts the terminal `completed` event alone and polls
`GET .../attributions` with the cursor until the step's row appears or the visible timeout
passes. The run is closed as completed afterwards (cancelled if a request fails).

- **Client latency** uses one `time.monotonic()` clock, from just before the request that
  delivers the step's terminal event to the poll response that contains its attribution. The
  first poll is sent one poll interval after that request starts, as for a dashboard whose
  previous poll had just returned, so each figure includes a whole poll interval. This is the
  figure compared with the limit.
- **Server-side latency** is `committed_at - step_terminal_received_at` from the attributions
  response, both database-clock times. It is reported separately and never compared with the
  client clock.
- **Never visible**: a step whose row does not appear within the visible timeout is a failure,
  listed by id and counted, not dropped. Any such step, or no visible step at all, makes
  `p95_within_limit` `null` (inconclusive), never `true`.
- The report records the workload, poll interval, step duration, timeout, the model status the
  worker reported (identity and `not_promoted` included), the model states seen on pages, model
  outcomes, the actual first-poll delay and poll spacing, the machine, the git revision (with a
  flag for modified tracked files), UTC start and finish times and `synthetic_workload: true`.
  It does not change any promotion state.
- **Status.** A report has `status: "completed"`, or `status: "failed"` when a request failed
  (the run is then cancelled). The command still writes a failed run's report to `--output` and
  exits `1`: the declared parameters, the run id if one was registered (else `null`), the model
  status if it was read, `error` (type, a message of at most 500 characters that never quotes a
  response body, and notes such as a failed cancel), `steps_completed` with each completed
  step's result in `per_step`, `failed_at`, `synthetic_workload: true` and
  `p95_within_limit: null`. A report it cannot write is reported on stderr with exit `1`.
  Settings rejected before any request exit `2` and write no report. `status` is an additive
  field of `attribution-latency-v1`: the three 2026-11-18 reports below predate it, have no
  `status` and are completed runs, so a reader treats a missing `status` as completed.

### Declared workload and acceptance

Declared on 2026-11-18 (UTC), committed before any measurement run.

- **Statistic and limit.** Nearest-rank p95 of client latency over every step of a run (poll
  interval included), against `max_attribution_latency_p95_seconds = 15.0` from
  `pipelines/model/evaluation-contract.toml` (`eval-contract-v3`, the "95th percentile time from
  step completion to a visible attribution"). A run is within the limit only if p95 is at most
  15.0 s and every step became visible within the timeout; a step never visible makes the run
  inconclusive.
- **Model pair.** The synthetic test pair (`write_synthetic_pair` from
  `packages/attribution/tests/pair_fixtures.py`, `blocks_promotion=True`, versions
  `training-v1+synthetic` and `calibration-v1+synthetic`), served with
  `AGENTWATCH_ALLOW_UNPROMOTED_MODEL=1`, so the expected state is `ready_not_promoted`. The real
  M03/M04 pair is not used: `.data/candidates` and `.data/calibration` do not exist in the main
  checkout, and DVC data is not pulled for this. Another state seen is recorded, not corrected.
- **Worker and database.** One `uvicorn agentwatch_worker.app:app` process (no reload) on
  `127.0.0.1:8091`, against a throwaway database `agentwatch_latency` created for the
  measurement in the local PostgreSQL 17.11 test cluster on port 55432 and dropped afterwards.
  The client, worker and database run on the same machine.
- **Machine.** Apple M4 Pro, 14 CPUs, 24 GiB memory, macOS 27.0.1 (arm64), Python 3.14.7. Other
  desktop applications stay open; nothing else is deliberately loaded.
- **Runs**, in this order, against the same worker process, each its own live run, each run
  exactly once:

  | Run | Steps | Step duration | Poll interval | Visible timeout | Role |
  | --- | --- | --- | --- | --- | --- |
  | A | 50 | 0.2 s | 2.0 s (the dashboard's `LIVE_POLL_INTERVAL_MS`) | 30 s | Gate measurement |
  | B | 50 | 0.2 s | 2.0 s | 30 s | Repeat of A, to show variability |
  | C | 50 | 0.2 s | 0.25 s | 30 s | Descriptive: shows the server-side share; not a gate run |

- **Acceptance.** The synthetic-workload result is recorded as within the limit only if runs A
  and B are both within it; a `false` or `null` in either is the result. Run C is reported but
  decides nothing. Every run is reported whatever it shows. A run that aborts with an error is
  recorded as failed with its error, not replaced. No run is repeated to improve a number.
- **Scope.** This is one input to the attribution-latency gate on a synthetic workload; it is
  not promotion evidence and changes no gate or promotion state.

### Results

Measured on 2026-11-18 (UTC) on the declared machine (Apple M4 Pro, 14 CPUs, macOS 27.0.1,
Python 3.14.7, PostgreSQL 17.11 on the same machine) at code revision `796c38e` (the
declaration commit; no tracked files modified). Model status on every page of every run:
`ready_not_promoted`, `not_promoted: true`, the synthetic pair `training-v1+synthetic` /
`calibration-v1+synthetic` (serving key `856f5f32…`, unchanged during the runs). The real
M03/M04 pair was not measured; it is not available locally. Each run was done once, in the
declared order; none failed or was repeated.

| Run | Poll interval | Client p50 / p95 / max (s) | Server-side p50 / p95 / max (s) | Never visible | Within 15.0 s |
| --- | --- | --- | --- | --- | --- |
| [A](attribution-latency/2026-11-18-run-A.json) | 2.0 s | 2.019 / 2.026 / 2.027 | 0.025 / 0.031 / 0.061 | 0 of 50 | `true` |
| [B](attribution-latency/2026-11-18-run-B.json) | 2.0 s | 2.024 / 2.029 / 2.032 | 0.026 / 0.031 / 0.033 | 0 of 50 | `true` |
| [C](attribution-latency/2026-11-18-run-C.json) | 0.25 s | 0.270 / 0.278 / 0.281 | 0.025 / 0.031 / 0.032 | 0 of 50 | `true` (descriptive) |

Under the declared acceptance rule, runs A and B are both within the limit, so the
synthetic-workload result is within the 15.0 s p95 limit. Every step's attribution was visible on
the first poll, and all 150 rows carry model outcome `predicted`. The actual first poll was sent
2.005 s (A and B, median) and 0.255 s (C) after the terminal request started, so the client
figure is the poll schedule plus one poll's response time (roughly 5 to 30 ms); it cannot show
where inside the interval the row was written. The server-side figure does: the row was
committed about 25 ms (median) after the terminal event was received. It rose from about 22 ms
over the first ten steps to about 28 to 31 ms over the last ten in every run, consistent with each
pass recomputing the feature rows of the whole run. Run A's 0.061 s maximum is one step
(`latency-step-0046`).

The reports are synthetic-workload evidence (`synthetic_workload: true`) and say nothing about
prediction quality.

### Limitations

- The steps and resource readings are synthetic: a client-side sleep with made-up CPU readings,
  no agent, model call or real contention, run one step at a time. The model pair is the
  tiny synthetic test pair (15 boosting rounds on 240 made-up rows), so its prediction time may
  differ from the real pair's.
- One machine, client, worker and database on loopback; no network between them, one worker
  process, no concurrent runs. Each pass recomputes the whole run's feature rows; runs longer
  than 50 steps (the bound is 1,000) were not measured.
- The client figure includes a whole poll interval by construction (first poll one interval
  after the terminal request), so it is a worst-phase figure for one poller, not the latency of
  a randomly timed dashboard poll.
- Three model inputs are always missing live (`workload_iterations`, `tests_per_step`,
  `worker_capacity`) and are passed to the model as missing.
- The current M03/M04 pair is not promoted, and four contract gates remain `not_evaluated`
  (held-out injection method, unseen tasks, instrumentation overhead and attribution latency).
  This synthetic result is one input to the attribution-latency gate, not promotion evidence;
  no gate or promotion state changes. A latency measurement with the real pair is still open.
  Promotion is M05.
- No deployment passes the `AGENTWATCH_MODEL_*` variables yet: Compose has no worker service
  (the worker runs on the host with `make dev-worker`), so containerized serving with a model is
  B04 work.
- A non-terminal batch that fills a gap and so applies an earlier-sent terminal event starts no
  background pass; that step's row is written by the next read, so its server-side latency can
  be up to one poll interval longer.
- A sample that arrives inside a step's window after its row was written is flagged
  (`evidence_changed`) only by the pass after the run closes, and a poller sees the flag only by
  re-reading from `after=0`; see [live ingestion](live-ingestion.md#attribution). Earlier passes
  compute features only for steps lacking a row, which keeps a pass proportional to the steps it
  attributes.
- A served pair that raises on one step leaves that step `unavailable` (`prediction_failed`) with
  its rules verdict; the other steps are scored in the same pass.

## Dashboard handoff (F03)

The dashboard reads `GET /v1/live/runs/{UUID}/attributions` for the selected run only, drains
`next_cursor`, then polls with the largest `sequence` received; `AttributionRecord` in
`@agentwatch/sdk` types each row's `record`. Show the rules verdict and the model outcome side
by side with the model state, versions and `not_promoted` label, `missing_inputs` beside the
prediction, an abstention as its own state and `excess_fraction_estimate` as an estimate (see
the [model serving contract](model-serving-contract.md)). `evidence_changed` can turn `true`
after a row was served, and an incremental poll never returns that row again: to show the flag
the dashboard must re-read the run from `after=0` periodically, for example on opening a run
and then on a slower schedule than the incremental poll. Rows of a superseded `serving_key`
stay on the page; compare each row's key with `model.serving_key`.

The packaging decision behind serving is [ADR 0014](adr/0014-attribution-serving-package.md).

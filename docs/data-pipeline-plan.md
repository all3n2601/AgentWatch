# AgentWatch data pipeline plan

Status: 2026-10-18. Owner: data pipeline workstream. The column-level contract is
[the data card](data-card.md); model gates and split settings are in the proposed evaluation
contract ([`docs/evaluation-contract.md`](evaluation-contract.md), #11). This plan says what is built, what comes
next, and what the pipeline needs from other workstreams.

## What the pipeline does

```text
recorded runs (results.json + samples.jsonl)
  -> evidence  curated, immutable store: data/evidence/<kind>-<run_id> (or GET /v1/runs)
  -> extract   validate strictly, deduplicate across sources, record telemetry status
  -> flatten   one row per step, resource readings per step window, run lineage
  -> label     experiment-derived labels against a frozen, fingerprinted rule file
  -> build     schema validation, content-addressed version, steps.parquet + manifest.json
  -> gate      TFDV statistics against the frozen schema (pipelines/validation)
  -> split     70/15/15 grouped by task_id, measured-only test, leakage check
  -> version   integrity check, then DVC snapshots of the evidence and dataset stores
```

Every stage rejects untrustworthy runs individually with a reason, including runs whose rows
break the dataset schema. Only an unreadable source, invalid label rules, an empty result, or a
defect in the builder itself stops a build, and a failed build
writes nothing. Each stage is in `pipelines/data/src/agentwatch_data/`, tested at 100% line and
branch coverage, and exercised against real recorded runs and the live ingestion API.

## Roadmap

| Step | Delivers | Status |
|---|---|---|
| Schema and data card | `steps-v1.2` columns, roles, leakage guards, row validation | Done (#1, #12, #13) |
| Extract | Strict validation, retries, cursor and row checks, duplicate detection | Done (#2, #10, #13) |
| Flatten | Step rows, telemetry windows, coverage against the declared interval | Done (#3, #12, #13) |
| Label | `labels-v1.1`, `injection_method`, per-run rejection | Done (#5, #12, #13) |
| 5. Build | `make dataset`: versioned Parquet release, manifest, and per-build report | Done (#14) |
| 6. Quality gate | `make validate-dataset`: TFDV statistics against a frozen, reviewed schema; anomalies fail, drift is reported. Isolated in `pipelines/validation` because TFDV brings TensorFlow (~1.7 GB) | Done (#16) |
| 7. Splits | `make split-dataset`: 70/15/15 grouped by `task_id` from the evaluation contract, measured-only test split, a deterministic split file with per-split row hashes, and the contract's leakage check | Done (#17) |
| 8. Versioning | `make add-evidence` curates runs into an immutable evidence store; `make version-dataset` verifies both stores and snapshots them with DVC (isolated in `pipelines/versioning`); a fresh clone rebuilds a versioned release byte for byte. The GCS remote is connected when F provides the bucket | Done (#28); the shared remote waits on F |
| Hardening pass | Request batching for large API histories; sweeping staging left by killed builds; fuzzing the per-run rejection guarantee with generated bad evidence; a scale run with thousands of runs; real multi-process concurrency of build, gate, split, and versioning; a cross-stage audit of the merged pipeline; anything left from reviews | Before Airflow |
| 9. More resources | Prometheus reader; `steps-v2` inference and retrieval features; `labels-v2` with mixed contention. Split follow-ups: force at least one injection method wholly into test, and check that mixed-contention cases reach test | Waits on B |
| 10. Real collection | Scheduled runs over a task suite on a Linux GCP VM, where PSI and cgroup readings exist. Split follow-up: rebuild and re-split once step events carry a real `task_id`, so the test split holds unseen tasks and training is no longer empty. Before any collection, run the extractor end to end against the live ingestion API and Postgres (deferred from the audit), and add an export of API-saved runs into the evidence store that keeps their saved time, so releases built from them stay reproducible | Waits on A, B, F |
| 11. Airflow DAG | collect → extract → validate → build → split → version, with retries and alerts. A version is only findable through committed pointer files, so the DAG needs an identity allowed to commit `data/*.dvc` (or a data branch), the DVC merge driver for versions committed alongside people's, and a service account for the bucket | DAG done (2026-10-24, `pipelines/orchestration`, `make pipeline-run`); Slack alerts done (2026-10-25); committing pointer files and the bucket account are open |
| 12. Reports | Quality, slice, label-sanity, and drift statistics per release | Slices done (2026-10-26, `make slice-dataset`); the rest after 11 |
| 13. Shared features | `window_readings` moved to `agentwatch_features.legacy`; live `step-features-v1` rows in `agentwatch_features.live` ([shared step features](step-features.md)). Serving them is B03 | B02 done 2026-10-24 |
| 14. Demo | Corrupted evidence rejected live; lineage from model to dataset to raw run | Last |

Split items owned elsewhere: the minimum test counts are judged by evaluation from the counts
each split file records, and freezing a new test split after 5 promotion evaluations is the
promotion owner's call (it means a new `SPLIT_METHOD`).

## What the pipeline needs from other workstreams

| From | Need | Why |
|---|---|---|
| A, agent | A real `task_id` (and `step_type`) in step events. A02 (#27) delivers both: `task_id` is `<name>-r<revision>`, and overhead runs keep their task's own `task_id`, so the step 10 extractor must exclude them from training datasets by `service_id` (`agentwatch-agent-overhead`), not by a suffix. The pipeline does not read live step events or `service_id` yet | Today's `task_id` is a placeholder (`kind:workload:iterations`), so grouped splits cannot guarantee unseen tasks |
| B, contention | Prometheus metric names for inference and retrieval; at least a second injection method; Qdrant in its own CPU-limited container | `steps-v2` features, the contract's held-out injection method, and separable retrieval labels |
| Worker owner | Faster or per-step sampling (250 ms leaves short steps without coverage); the `CpuSampler.summarize` reset fix; monotonic durations; strict contract bounds; `demo.py --output` defaulting by workload; agent evidence written when the LLM loop fails | Measurement quality at the source |
| F, platform | A pinned CI Postgres image (`public.ecr.aws/docker/library/postgres:17-alpine@sha256:b0f9560a…`, verified identical to Docker Hub's) | Docker Hub's anonymous pull limit has failed CI before tests ran |
| F, platform | A GCS bucket for DVC (for example `gs://<bucket>/agentwatch/dvc`), with object read/write for the team, and service accounts for CI and Airflow | Step 8's shared remote: until it exists, versions stay in each person's DVC cache and pointer files are not committed |
| D, model | Consume releases by `dataset_version`; record `label_version` and the contract hash in reports | Reproducible evaluation |

## Known limits

The data card's [known limitations](data-card.md#known-limitations) apply to every release. The
most important today: on CPU-only data the label equals the rule's prediction on every row, so
no model evaluation is informative until inference, retrieval, and mixed contention exist.

## Course requirements

The DADS 7305 data pipeline submission is graded on the criteria below. This section maps each
one to the pipeline and plans what is missing; the roadmap steps above own the work.

| Criterion | How the pipeline meets it | Status |
|---|---|---|
| Data acquisition | Extraction from local recorded runs and the ingestion API, with strict validation and per-run rejection | Done (#2, #10, #13, #21) |
| Reproducible environment | Dependencies pinned in `uv.lock` for the shared workspace and the quality gate (`pipelines/validation`), and the versioning environment (`pipelines/versioning`) | Done (#14, #16, #28) |
| Preprocessing and feature engineering, modular | One module per stage: extract, flatten (step windows, coverage, queue fraction), label, schema validation, build | Done |
| Unit tests (edge cases, missing values, anomalies) | pytest at 100% line and branch coverage, including failure paths, two audits' regressions, Python 3.13 and 3.14 | Done |
| Schema and statistics generation, validated over time | TFDV quality gate against a frozen, reviewed schema; statistics kept per attempt; drift against the previous passing release | Done (#16, #20) |
| Anomaly detection | The gate's anomalies (missing values, out-of-range values, unknown categories, type changes) and per-run rejections with reasons | Done; alerts are planned below |
| Data versioning with DVC, `.dvc` files in git | Evidence and dataset stores versioned, append-only, rebuildable byte for byte | Partial: the local DVC mechanism is in place (#28); open until the `.dvc` files are committed, which needs the shared remote |
| Reproducibility on another machine | A fresh clone pulls and rebuilds the same version byte for byte | Partial: a fresh clone rebuilds byte for byte from a local DVC remote in the versioning tests (#28); open until shown on another machine through the shared remote |
| Error handling and logging | Every stage rejects untrustworthy input with a reason; build reports and gate results are durable records; Python logging; each Airflow task logs its command's output and fails with the exit code and stderr tail | Done |
| Code style (PEP 8) | Ruff on every PR | Done |
| Airflow DAG for the whole workflow | `agentwatch_dataset`: acquire → build → validate → split → version → push → summarize, run end to end on the shared 50-run release and with new recorded runs | Done (2026-10-24) |
| Alerts (email or Slack) for errors and anomalies | Slack incoming webhook: every task's final failure, and each successful run's summary with its drift, rejected-run share, leakage, and unpushed-version findings (`agentwatch_orchestration.alerts`) | Done (2026-10-25); delivered to a real channel 2026-10-26 |
| Pipeline flow optimization with Airflow's Gantt chart | Gantt-chart timings before and after: prepared environments (`make pipeline-setup`) and the slice report beside versioning; measurements under "Pipeline flow optimization" | Done (2026-10-26) on the shared release; remeasure at step 10 volume |
| Bias detection and mitigation through data slicing | `make slice-dataset`: data-level slices with declared thresholds, per-flag mitigations, and balancing weights; findings and mitigation in the data card's "Slices and bias" | Data level done (2026-10-26); model-level slicing waits on a model |
| README: setup, run, structure, DVC | [`pipelines/README.md`](../pipelines/README.md): setup (uv or pip), getting the data, running it, the course-layout mapping, components, DVC and why there is no `dvc.yaml`, tests, troubleshooting; `requirements.txt` files exported from the lockfiles (`make requirements`, checked in CI) | Done (2026-10-26); DVC pull steps work once the remote is committed |

### Airflow DAG (step 11)

Built 2026-10-24 in `pipelines/orchestration` (own environment, Airflow 3.3.2). The DAG is
triggered by hand with the run directories to add; each task passes the release it built to the
next, so a release someone else publishes between tasks is never validated or split in its
place. Measured on the shared release (50 runs, 350 rows) on a MacBook: about 25 s per run, of
which the TFDV gate takes about 19 s. Not built yet: alerts, committing `data/*.dvc`, the API
export, and the report task.

- **Tasks, in order:** acquire (add recorded runs to the evidence store; export API-saved runs
  once step 10 adds it) → build → validate (the TFDV gate) → split → integrity → version →
  push, with a report task (step 12) alongside versioning.
- **Environments:** the gate and DVC keep their isolated environments, so their tasks run in
  those environments (an external Python interpreter or a container per task), not in
  Airflow's own.
- **Retries:** only where a retry can succeed, with bounded backoff (acquisition from the API,
  pushing to the remote). Deterministic stages fail fast; a retry would repeat the same failure.
- **Idempotency:** already built in. Rebuilding the same evidence reuses its release, gate
  attempts append, and versioning is append-only, so a rerun never corrupts earlier output.
- **Committing versions:** the DAG needs an identity allowed to commit `data/*.dvc`, the DVC
  merge driver, and a bucket service account (see the step 11 row).

### Alerts and tracking (step 11)

- **When:**
  - any task fails;
  - and outcomes that end a run without a task failure:
    - the gate fails (its anomalies listed);
    - drift is reported;
    - rejected runs exceed a declared share;
    - the integrity check refuses to version;
    - the leakage check is blocked or inconclusive.
- **Thresholds:** declared before data is collected, never tuned to silence an alert.
- **Where:** a Slack incoming webhook, read from `AGENTWATCH_ALERT_SLACK_WEBHOOK_URL` in
  Airflow's environment or the checkout's git-ignored `.env` (built 2026-10-25). An environment variable rather than an Airflow
  connection, because a connection would need the Slack provider package and the stages
  already read their settings from the environment. The URL stays out of the repository and
  the logs. Gate anomalies and integrity refusals fail their task, so they arrive as failure
  alerts with the error tail; drift, rejections, and leakage arrive from the `alert` task.
- **What:** the release, the stage, the reason, and the path of the build report or gate result
  that holds the evidence.
- **Tracking:** Airflow keeps each task's log. The DAG's last task records a run summary
  (dataset version, rows, rejections by stage, gate status, leakage status, split id, and
  whether a version was pushed), so every run is traceable without reading logs.

### Pipeline flow optimization (step 11)

Once the DAG runs on realistic volume (step 10 collections, or the hardening pass's scale run),
read Airflow's Gantt chart to find the slowest tasks. TFDV's import and statistics and the
reading of evidence are the likely candidates. Then:

- run independent tasks in parallel, for example the report beside versioning, and extraction
  per source;
- reuse the task environments between runs;
- avoid reading the same Parquet file twice.

Record the durations before and after each change in this plan, as evidence of the optimization.

#### Measured 2026-10-26

Workload: the shared release `steps-v1.2+5954f97ef8b2` (50 runs, 350 rows), no new evidence,
on a MacBook (Apple silicon, macOS), each from a fresh checkout. Durations are task start to
end in Airflow's metadata database, which is what the Gantt chart draws. "Cold" is the first
run in a new checkout; "warm" is the mean of the next two.

| Task | Before, cold | Before, warm | After, cold (setup first) | After, warm |
|---|---|---|---|---|
| acquire | 0.99 s | 1.00 s | 0.97 s | 0.97 s |
| build | 2.31 s | 0.25 s | 1.89 s | 0.25 s |
| validate (TFDV gate) | **26.68 s** | 3.09 s | 4.18 s | 2.54 s |
| split | 1.25 s | 0.22 s | 0.71 s | 0.22 s |
| version | 2.99 s | 0.94 s | 1.74 s | 0.90 s |
| slices (new) | n/a | n/a | 0.21 s | 0.21 s |
| **Whole run** | **35.1 s** | 6.3 s | **10.5 s** | 5.9 s |

What the Gantt chart showed: one serial chain, with the gate taking 76% of a cold run and about
half of a warm one. Its time is TensorFlow's import, not statistics: importing TFDV alone takes
2.3 s warm and 18–25 s the first time in a new environment, while the statistics over 350 rows
take well under a second. Skipping uv's lock check per stage (`--frozen`) saved 4 ms per command,
so it was not changed.

Changes:

1. **Prepared environments** (`make pipeline-setup`): syncs the four environments with
   precompiled bytecode and imports TFDV once, before the first run. The first run drops from
   35.1 s to 10.5 s; the setup itself takes about 26 s, once per checkout or worker image.
   Precompiling alone brought the first import from 24.5 s to 18.0 s; the warm import does the
   rest.
2. **The bias slice report runs beside versioning.** It reads only the release and its split,
   so it depends on `split` and feeds `summarize`, not `version`. Under the scheduler (`airflow
   standalone`, LocalExecutor) it ran from 44.85 s to 45.74 s while `version` ran from 44.83 s
   to 46.71 s, so the report adds no time to the critical path. `make pipeline-run` (`airflow
   dags test`) runs tasks one at a time and does not show this overlap.

Limits: 350 rows is small, so stage work is a minor share; scheduler runs add about 0.6 s of
process start per task (16.2 s for the same run under `airflow standalone`). Remeasure at step
10 volume, where reading evidence and computing statistics will matter.

### Bias detection and mitigation (step 12)

The dataset has no demographic attributes. Its slices are the attributes along which the data or a
model could be skewed:

- task (`task_id`);
- experiment kind;
- resource class (`label_resource`);
- condition;
- injection method;
- agent model;
- source;
- telemetry status;
- availability of PSI and cgroup readings, which also stands in for the host platform: macOS
  has neither.

- **Data level, from the first release:** for each slice, its share of rows, its label balance,
  its missing readings and telemetry coverage, and its counts in each split. Slices below
  declared minimum counts are flagged. Today's data, one macOS coding task with two classes,
  will be reported as exactly that imbalance.
- **Model level, with D:** per-slice metrics on the frozen test split (macro F1, the clean-step
  `none` rate, abstention), using Fairlearn's `MetricFrame` or TensorFlow Model Analysis. The
  largest acceptable gap between slices is declared before evaluation, under the evaluation
  contract's change policy.
- **Mitigation, in order of preference:**
  1. Collect runs for under-represented slices (step 10). This is the only fix that adds
     evidence.
  2. Stratify or re-weight training (D).
  3. Adjust a per-slice threshold, only if it was declared before evaluation.

  Each trade-off against overall performance is documented.
- **Documentation:** a "Slices and bias" section in the data card records what was found, what
  was done, and what it cost.

### README and submission details

- **README:** written 2026-10-26 as [`pipelines/README.md`](../pipelines/README.md), covering:
  - setup for the three environments;
  - running it through `make` and Airflow;
  - how the layout maps to the course's example structure: `dags/` is the Airflow DAG folder,
    `data/` is `data/`, `scripts/` is `pipelines/*/src`, `tests/` is `pipelines/*/tests`, and
    `logs/` is Airflow's task logs;
  - pulling and rebuilding with DVC.
- **Committed `.dvc` files:** these need the shared remote, F's bucket. If it is not ready
  before the deadline, the team chooses a fallback remote it controls (another GCS bucket, or
  DVC's Google Drive remote) so a grader's `dvc pull` works.
- **`requirements.txt`:** the `uv.lock` files are the pinned record. `make requirements`
  exports `requirements.txt` (shared workspace) and one per isolated environment, and CI fails
  if they drift from the lockfiles (done 2026-10-26).
- **`dvc.yaml`:** not used. Airflow orchestrates the stages and DVC versions their outputs.
  DVC pipeline stages remove their outputs before rerunning unless marked persistent, which
  conflicts with append-only, immutable releases, and they would duplicate Airflow.

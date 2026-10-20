# AgentWatch data pipeline plan

Status: 2026-10-13. Owner: data pipeline workstream. The column-level contract is
[the data card](data-card.md); model gates and split settings are in the proposed evaluation
contract ([`docs/evaluation-contract.md`](evaluation-contract.md), #11). This plan says what is built, what comes
next, and what the pipeline needs from other workstreams.

## What the pipeline does

```text
recorded runs (local results.json + samples.jsonl, or GET /v1/runs)
  -> extract   validate strictly, deduplicate across sources, record telemetry status
  -> flatten   one row per step, resource readings per step window, run lineage
  -> label     experiment-derived labels against a frozen, fingerprinted rule file
  -> build     schema validation, content-addressed version, steps.parquet + manifest.json
  -> gate      TFDV statistics against the frozen schema (pipelines/validation)
  -> split     70/15/15 grouped by task_id, measured-only test, leakage check
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
| 8. Versioning | DVC with a GCS remote; a rebuild from the same evidence reproduces the same version. #28 builds the evidence store, the integrity check, and DVC snapshots; the shared GCS remote and committed `.dvc` files remain | In review (#28) |
| Hardening pass | Request batching for large API histories; sweeping staging left by killed builds; anything left from reviews | Before Airflow |
| 9. More resources | Prometheus reader; `steps-v2` inference and retrieval features; `labels-v2` with mixed contention. Split follow-ups: force at least one injection method wholly into test, and check that mixed-contention cases reach test | Waits on B |
| 10. Real collection | Scheduled runs over a task suite on a Linux GCP VM, where PSI and cgroup readings exist. Split follow-up: rebuild and re-split once step events carry a real `task_id`, so the test split holds unseen tasks and training is no longer empty | Waits on A, B, F |
| 11. Airflow DAG | collect → extract → validate → build → split → version, with retries and alerts | After 6–8 |
| 12. Reports | Quality, slice, label-sanity, and drift statistics per release | After 11 |
| 13. Shared features | The live attribution service reuses `window_readings` and related code | With E |
| 14. Demo | Corrupted evidence rejected live; lineage from model to dataset to raw run | Last |

Split items owned elsewhere: the minimum test counts are judged by evaluation from the counts
each split file records, and freezing a new test split after 5 promotion evaluations is the
promotion owner's call (it means a new `SPLIT_METHOD`).

## What the pipeline needs from other workstreams

| From | Need | Why |
|---|---|---|
| A, agent | A real `task_id` (and `step_type`) in step events | Today's `task_id` is a placeholder (`kind:workload:iterations`), so grouped splits cannot guarantee unseen tasks |
| B, contention | Prometheus metric names for inference and retrieval; at least a second injection method; Qdrant in its own CPU-limited container | `steps-v2` features, the contract's held-out injection method, and separable retrieval labels |
| Worker owner | Faster or per-step sampling (250 ms leaves short steps without coverage); the `CpuSampler.summarize` reset fix; monotonic durations; strict contract bounds; `demo.py --output` defaulting by workload; agent evidence written when the LLM loop fails | Measurement quality at the source |
| F, platform | A pinned CI Postgres image (`public.ecr.aws/docker/library/postgres:17-alpine@sha256:b0f9560a…`, verified identical to Docker Hub's) | Docker Hub's anonymous pull limit has failed CI before tests ran |
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
| Reproducible environment | Dependencies pinned in `uv.lock` for the shared workspace and the quality gate (`pipelines/validation`); the versioning environment (`pipelines/versioning`) and the curated evidence store come with #28 | Done for the pipeline and gate (#14, #16); versioning in review (#28) |
| Preprocessing and feature engineering, modular | One module per stage: extract, flatten (step windows, coverage, queue fraction), label, schema validation, build | Done |
| Unit tests (edge cases, missing values, anomalies) | pytest at 100% line and branch coverage, including failure paths, two audits' regressions, Python 3.13 and 3.14 | Done |
| Schema and statistics generation, validated over time | TFDV quality gate against a frozen, reviewed schema; statistics kept per attempt; drift against the previous passing release | Done (#16, #20) |
| Anomaly detection | The gate's anomalies (missing values, out-of-range values, unknown categories, type changes) and per-run rejections with reasons | Done; alerts are planned below |
| Data versioning with DVC, `.dvc` files in git | Evidence and dataset stores versioned, append-only, rebuildable byte for byte | In review (#28); committing the `.dvc` files needs the shared remote |
| Reproducibility on another machine | A fresh clone pulls and rebuilds the same version byte for byte | In review (#28) |
| Error handling and logging | Every stage rejects untrustworthy input with a reason; build reports and gate results are durable records; Python logging | Done for the commands; Airflow task logs with step 11 |
| Code style (PEP 8) | Ruff on every PR | Done |
| Airflow DAG for the whole workflow | Not built | Step 11 |
| Alerts (email or Slack) for errors and anomalies | Not built | Step 11 |
| Pipeline flow optimization with Airflow's Gantt chart | Not built | Step 11, once the DAG runs |
| Bias detection and mitigation through data slicing | Not built | Step 12 |
| README: setup, run, structure, DVC | The data card and this plan exist; the README has no pipeline section | With step 11 |

### Airflow DAG (step 11)

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
- **Where:** a Slack incoming webhook or SMTP email, configured as an Airflow connection. The
  webhook URL and credentials stay in Airflow's secret store, never in the repository.
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

- **README:** a data pipeline section covering:
  - setup for the three environments;
  - running it through `make` and Airflow;
  - how the layout maps to the course's example structure: `dags/` is the Airflow DAG folder,
    `data/` is `data/`, `scripts/` is `pipelines/*/src`, `tests/` is `pipelines/*/tests`, and
    `logs/` is Airflow's task logs;
  - pulling and rebuilding with DVC.
- **Committed `.dvc` files:** these need the shared remote, F's bucket. If it is not ready
  before the deadline, the team chooses a fallback remote it controls (another GCS bucket, or
  DVC's Google Drive remote) so a grader's `dvc pull` works.
- **`requirements.txt`:** the `uv.lock` files are the pinned record. If a `requirements.txt` is
  required, export one with `uv export --frozen` and have CI check it stays in sync.
- **`dvc.yaml`:** not used. Airflow orchestrates the stages and DVC versions their outputs.
  DVC pipeline stages remove their outputs before rerunning unless marked persistent, which
  conflicts with append-only, immutable releases, and they would duplicate Airflow.

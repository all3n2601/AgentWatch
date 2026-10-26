# AgentWatch data pipeline

This pipeline turns recorded AgentWatch experiment runs into a versioned, validated dataset of
agent steps. Each row is one step, with resource readings from its time window and a label
naming the resource that slowed it down (or `none`). Airflow orchestrates it, TFDV validates it,
DVC versions it, and Slack receives its alerts.

```text
recorded runs (results.json + samples.jsonl)
  acquire   copy validated runs into the immutable evidence store        data/evidence/
  build     extract → flatten → label → schema check → Parquet release   data/datasets/<release>/
  validate  TFDV statistics against a frozen schema; anomalies fail      data/datasets/validations/
  split     70/15/15 grouped by task, leakage check                      data/datasets/splits/
  slices    bias report per slice (runs beside version)                  .data/reports/<release>/
  version   integrity check, then a DVC snapshot of both stores          data/evidence.dvc, data/datasets.dvc
  push      upload to the DVC remote (skipped when none is configured)
  summarize run summary
  alert     run summary and findings to Slack, with the slice report's flag count
  finish    fails the run if the slice report failed
```

Column meanings, label rules and known limitations are in [the data card](../docs/data-card.md).
What is built and what is planned is in [the data pipeline plan](../docs/data-pipeline-plan.md).

## Contents

- [Folder structure](#folder-structure)
- [Set up the environment](#set-up-the-environment)
- [Get the data](#get-the-data)
- [Run the pipeline](#run-the-pipeline)
- [Pipeline components](#pipeline-components)
- [Data versioning with DVC](#data-versioning-with-dvc)
- [Tests](#tests)
- [Reproducibility notes and troubleshooting](#reproducibility-notes-and-troubleshooting)

## Folder structure

The pipeline follows the repository's layout rather than a single top-level folder: shared
stages live with the other Python packages, and the heavy tools (TensorFlow, DVC, Airflow) each
get an isolated environment so their pins cannot conflict. The course's example structure maps
onto it like this:

| Course example | Here | What it holds |
|---|---|---|
| `Data-Pipeline/` | [`pipelines/`](.) | Everything below |
| `dags/` | [`pipelines/orchestration/dags/`](orchestration/dags/) | The Airflow DAG `agentwatch_dataset` |
| `scripts/` | [`pipelines/data/src/agentwatch_data/`](data/src/agentwatch_data/), [`pipelines/validation/src/`](validation/src/), [`pipelines/orchestration/src/`](orchestration/src/) | Stage modules, the TFDV gate, the stage runner and Slack alerts |
| `tests/` | `pipelines/*/tests/` | pytest suites, one per environment |
| `data/` | [`data/`](../data/) at the repository root | `evidence/` and `datasets/` (DVC-tracked, git-ignored), `shared/` (an interim archive) |
| `logs/` | `.data/airflow/logs/` | Airflow task logs, one file per task try |
| `dvc.yaml` | not used | See [why there is no `dvc.yaml`](#why-there-is-no-dvcyaml) |
| `requirements.txt` | [`requirements.txt`](../requirements.txt) and `pipelines/{validation,versioning,orchestration}/requirements.txt` | Pinned exports of the uv lockfiles |

```text
pipelines/
├── data/                  shared workspace package agentwatch-data (Python 3.13–3.14)
│   ├── src/agentwatch_data/
│   │   ├── evidence.py    acquire: validate runs and copy them into the evidence store
│   │   ├── extract.py     strict reading of results.json and samples.jsonl, deduplication
│   │   ├── flatten.py     one row per step; readings inside each step's time window
│   │   ├── labels.py      experiment-derived labels from a frozen rule file (labels.toml)
│   │   ├── schema.py      column contract (contracts/steps-v1.2.json), row validation
│   │   ├── build.py       content-addressed Parquet release, manifest, build report
│   │   ├── splits.py      grouped splits and the leakage check
│   │   ├── slices.py      bias slicing, flags, mitigations, balancing weights
│   │   ├── integrity.py   store verification before DVC versions anything
│   │   └── retire.py      remove a release that was never versioned
│   └── tests/
├── validation/            TFDV quality gate (own environment, Python 3.13)
│   ├── schema/steps.pbtxt the frozen, reviewed schema
│   ├── src/agentwatch_validation/
│   └── tests/
├── versioning/            DVC 3.67.1 with GCS support (own environment)
│   └── tests/             end-to-end versioning tests
├── orchestration/         Airflow 3.3.2 (own environment)
│   ├── dags/agentwatch_dataset.py
│   ├── src/agentwatch_orchestration/
│   │   ├── stages.py      runs each stage's command in its environment, with timeouts
│   │   └── alerts.py      Slack alerts
│   └── tests/
└── model/evaluation-contract.toml   split fractions, group key, leakage bound
```

## Set up the environment

Prerequisites:

- macOS or Linux (the TFDV gate supports Linux x86_64 and macOS arm64);
- [uv](https://docs.astral.sh/uv/) (tested with 0.13), which installs the right Python versions itself;
- `git` and `make`.

```bash
git clone https://github.com/all3n2601/AgentWatch.git
cd AgentWatch
make pipeline-setup        # all four environments from their lockfiles (about 30 s with uv's cache warm)
```

`make pipeline-setup` runs `uv sync --frozen` for the shared workspace, the TFDV gate, DVC, and
Airflow, precompiles their bytecode, and imports TensorFlow once, so the first pipeline run
does not pay for it. Every version is pinned in a `uv.lock` file, so the environments are the
same on every machine.

**Installing stages without uv.** The pinned `requirements.txt` files are exported from the same lockfiles
(`make requirements`; CI fails if they drift). Use one virtual environment per file, because
the four environments' pins conflict:

```bash
python3.13 -m venv .venv && .venv/bin/pip install -r requirements.txt     # from the repository root
cd pipelines/validation && python3.13 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

The DAG and the `make` targets call each stage through `uv run`, so running the full pipeline
still needs uv; the pip route suits running and testing individual stages.

**Slack alerts (optional).** Put a Slack incoming-webhook URL in the repository's git-ignored
`.env` (copy [`.env.example`](../.env.example)):

```bash
AGENTWATCH_ALERT_SLACK_WEBHOOK_URL=https://hooks.slack.com/services/…
AGENTWATCH_ALERT_MAX_REJECTED_SHARE=0.10
```

Without it, the pipeline runs normally and alerts appear only in the task logs. Creating the
webhook is described in the [root README](../README.md#slack-alerts).

## Get the data

The pipeline's inputs are recorded experiment runs. There are three ways to get them.

**1. From the DVC remote.** Once the team's Google Cloud Storage remote is configured, this
command downloads exactly the evidence and dataset versions recorded in the commit you have
checked out:

```bash
make pull-dataset
```

The remote is not committed yet: until it is, `make pull-dataset` reports that no remote is
configured. Use option 2 meanwhile.

**2. From the shared archive (works today).** A gated release of 50 real runs (25 CPU and 25
coding runs, recorded 2026-10-21) is committed as a zip archive:

```bash
unzip -q data/shared/agentwatch-dataset-steps-v1.2-5954f97ef8b2.zip
cp -R agentwatch-dataset-2026-10-20/data/* data/
rm -r agentwatch-dataset-2026-10-20
```

This fills `data/evidence/` (the raw runs) and `data/datasets/` (release
`steps-v1.2+5954f97ef8b2` with its gate result and split). The archive's own `README.md` and
`SHA256SUMS` describe and verify it.

**3. Record new runs.** `make demo-cpu` and `make demo-coding` run the CPU and coding
experiments locally and write runs under `.data/`. Add them to the pipeline with
`make pipeline-run PIPELINE_RUN_DIRS=".data/coding-demo"`.

## Run the pipeline

```bash
make pipeline-run                                         # the whole DAG once, from the stored evidence
make pipeline-run PIPELINE_RUN_DIRS=".data/coding-demo"   # add recorded runs first
```

`make pipeline-run` runs the DAG with `airflow dags test`: every task runs in order, in one
process, with no scheduler to start. Its metadata and logs go to `.data/airflow/`.

To use Airflow's UI (grid, graph, task logs and the Gantt chart) and run independent tasks in
parallel, start Airflow locally:

```bash
make pipeline-ui           # scheduler, triggerer and UI; the login is printed and saved under .data/airflow
```

Open <http://localhost:8080>, unpause `agentwatch_dataset` and trigger it, optionally with the
`run_dirs` parameter. A run's **Gantt** view is the second button above its task list.

Each stage also has its own `make` target, which runs the newest release:

| Target | Stage |
|---|---|
| `make add-evidence RUN=".data/coding-demo"` | acquire |
| `make dataset` | build |
| `make validate-dataset` | TFDV gate (exits 1 on anomalies) |
| `make split-dataset` | splits and leakage check |
| `make slice-dataset` | bias slices |
| `make version-dataset` | integrity check, then DVC snapshot |
| `make push-dataset` / `make pull-dataset` | DVC remote |
| `make dataset-status` | what is not versioned or uploaded yet |

**Expected result on the shared archive.** The run succeeds in about 6 s once warm (10 s on the
first run after `make pipeline-setup`). It reuses release `steps-v1.2+5954f97ef8b2`, the gate
passes, the leakage check is `blocked` (expected for this data; see the data card), the slice
report raises 17 flags, `push` is skipped because no DVC remote is configured, and the summary
lists both findings. With a webhook set, the same summary arrives in Slack.

## Pipeline components

| Course requirement | Where | Details |
|---|---|---|
| Data acquisition | `evidence.py`, `extract.py` | Recorded runs (local directories or the ingestion API) are validated strictly and copied into an immutable evidence store. Each untrustworthy run is rejected with a reason; duplicates are detected across sources. |
| Preprocessing and feature engineering | `flatten.py`, `labels.py`, `schema.py`, `build.py` | One row per step; telemetry readings joined to each step's time window, with coverage; queue fraction and other features; labels from a frozen, fingerprinted rule file; every row validated against the column contract. Missing readings stay `null`, never zero. |
| Schema and statistics | `pipelines/validation` | TFDV statistics for every release, checked against the reviewed schema in `schema/steps.pbtxt`. Each attempt's statistics are kept; drift is measured against the previous passing release. |
| Anomaly detection | TFDV gate, per-run rejections | Missing values, out-of-range values, unknown categories and type changes fail the gate and stop the run. Drift is reported without failing. |
| Alerts | `alerts.py`, the DAG | Slack message for every task's final failure (error tail and log link) and every run's summary, with findings: drift, rejected-run share over the declared 10%, leakage check not passed, version not pushed. |
| Bias detection and mitigation | `slices.py`, data card "Slices and bias" | Slices by task, kind, label, condition, injection method, model, source, telemetry status and PSI/cgroup availability; flags against thresholds declared before evaluation; a mitigation per flag; balancing weights. |
| Orchestration | `orchestration/dags/agentwatch_dataset.py` | Each task runs one stage in its own environment and passes its release to the next. Only network steps retry (DVC push, Slack); deterministic stages fail fast. One run at a time. |
| Flow optimization | data pipeline plan, "Pipeline flow optimization" | Gantt-chart timings before and after: prepared environments (first run 35.1 s → 10.5 s) and the slice report beside versioning. |
| Tracking and logging | Airflow task logs, build reports, gate results, run summary | Every stage logs its command and output; failures carry the exit code and the end of stderr; the last task records a run summary. |
| Error handling | every stage | Timeouts on every command; per-run rejection with reasons; a failed build writes nothing; refused inputs never fall back to older data. |
| Data versioning | `integrity.py`, `pipelines/versioning`, `data/*.dvc` | See below. |

## Data versioning with DVC

Two stores are versioned: `data/evidence/` (raw runs, append-only) and `data/datasets/`
(releases, gate attempts, splits and build reports, append-only). Git ignores their contents
and tracks the pointer files `data/evidence.dvc` and `data/datasets.dvc`, which record exactly
which files make up each version.

```bash
make version-dataset       # verify both stores, then snapshot them; commit the updated data/*.dvc
make push-dataset          # upload the snapshot to the remote
git add data/evidence.dvc data/datasets.dvc && git commit -m "chore: version dataset <release>"
```

Before snapshotting, the integrity check refuses to version a store that was edited, is
missing files, holds a release its evidence no longer rebuilds, or would drop anything already
versioned. A fresh clone that pulls a version can rebuild its release byte for byte; the
versioning tests check this. Once per clone, `make dataset-merge-driver` lets git merge pointer
files that two people updated at the same time. The full versioning rules are in the data
card's [Versioning](../docs/data-card.md#versioning) section.

**Status.** The DVC mechanism and its tests are in place. The shared Google Cloud Storage
remote and the first committed pointer files are pending; until then versions stay in each
person's DVC cache and the shared archive stands in.

### Why there is no `dvc.yaml`

A `dvc.yaml` defines pipeline stages for `dvc repro` to run. Here Airflow orchestrates the
stages and DVC only versions their outputs, for two reasons:

- **The stores are append-only and immutable.** DVC removes a stage's outputs before
  rerunning it (unless each output is marked persistent), which would delete earlier releases,
  gate attempts and splits that later work depends on.
- **One orchestrator.** A second stage graph in `dvc.yaml` would duplicate the DAG, and the two
  could drift apart. Airflow already provides ordering, retries, logs, alerts and the Gantt
  chart; DVC provides content-addressed versions, the remote and reproducible pulls.

## Tests

```bash
uv run pytest pipelines/data/tests                                            # stages
uv run --project pipelines/validation --frozen pytest pipelines/validation/tests      # TFDV gate
uv run --project pipelines/versioning --frozen pytest pipelines/versioning/tests      # DVC, end to end
uv run --project pipelines/orchestration --frozen pytest pipelines/orchestration/tests  # DAG and alerts
uv run ruff check .                                                           # PEP 8 and lint
```

The tests cover invalid and malformed input, missing readings, duplicate and conflicting runs,
anomalies, leakage, rejected Slack messages and failed tasks. Fixtures are synthetic and marked
as such, apart from a small recorded run and release under the `tests/fixtures/` folders. No
test needs the network (the API and Slack are faked), and the tests write only to temporary
directories, so none touches your data or Airflow home. CI runs all of these on every pull request (`.github/workflows/ci.yaml`).

## Reproducibility notes and troubleshooting

- **A checkout path with spaces** breaks the `make` data targets, which derive
  `AGENTWATCH_ROOT` from the Makefile's location. Clone into a path without spaces, or run
  them with `AGENTWATCH_ROOT=.`, for example `make slice-dataset AGENTWATCH_ROOT=.`.
- **Port 8080 already in use** (by Docker, for example): start the UI on another port, and
  point tasks at it too:
  `AIRFLOW__API__PORT=8095 AIRFLOW__CORE__EXECUTION_API_SERVER_URL=http://localhost:8095/execution/ make pipeline-ui`.
- **The first run is slow** without `make pipeline-setup`: TensorFlow's first import in a new
  environment takes 18–25 s on macOS.
- **The leakage check is `blocked`** on the shared archive. This is expected: two tasks and two
  classes separate perfectly. The split is kept as evidence and refused for evaluation.
- **No Slack message**: check `.env` for `AGENTWATCH_ALERT_SLACK_WEBHOOK_URL`. Without it, the
  `alert` task is marked skipped and lists its findings in its log.

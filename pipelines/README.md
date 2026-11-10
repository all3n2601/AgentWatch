# AgentWatch data pipeline

This pipeline turns recorded AgentWatch experiment runs into a versioned, validated dataset of
agent steps. Each row is one step, with resource readings from its time window and a label
naming the resource that slowed it down (or `none`). Airflow orchestrates it, TFDV validates it,
DVC versions it, and Slack receives its alerts.

```text
recorded runs (results.json + samples.jsonl) and replay experiments (experiment.json + replays/)
  acquire   copy validated runs and replay experiments into the evidence store   data/evidence/
  build     extract → flatten or join features → label → schema check → Parquet  data/datasets/<release>/
  validate  TFDV statistics against a frozen schema; anomalies fail              data/datasets/validations/
  split     split-v2 by task group; class balance and leakage checks             data/datasets/splits/
  slices    bias report per slice (runs beside version)                          .data/reports/<release>/
  version   integrity check, then a DVC snapshot of both stores                  data/evidence.dvc, data/datasets.dvc
  push      upload to the team bucket (only when AGENTWATCH_DVC_PUSH_REMOTE is set)
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
│   │   ├── replay_labels.py  labels-v2 for agent replay experiments (labels-v2.toml)
│   │   ├── replay_rows.py replay release rows: labels joined to step-features-v1
│   │   ├── schema.py      column contract (contracts/steps-v1.3.json), row validation
│   │   ├── build.py       content-addressed Parquet release, manifest, build report
│   │   ├── splits.py      grouped splits (split-v1, split-v2), class balance and leakage checks
│   │   ├── slices.py      bias slicing, flags, mitigations, balancing weights
│   │   ├── integrity.py   store verification before DVC versions anything
│   │   └── retire.py      remove a release that was never versioned
│   └── tests/
├── validation/            TFDV quality gate (own environment, Python 3.13)
│   ├── schema/            the frozen, reviewed schemas: steps-v1.3.pbtxt (default), steps.pbtxt (steps-v1.2)
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
└── model/
    ├── evaluation-contract.toml      eval-contract-v2: split method, group key, minimums, leakage bound
    └── evaluation-contract-v1.toml   eval-contract-v1, kept so steps-v1.2 splits still rebuild
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

**1. From the DVC remote (recommended).** This downloads exactly the evidence and dataset
versions recorded in the commit you have checked out, from the team's Google Cloud Storage
bucket `agentwatch-dvc`. It needs no Google account or login:

```bash
make pull-dataset          # about 5 s: 50 runs in data/evidence/, release steps-v1.2+5954f97ef8b2 in data/datasets/
```

**Dataset v1 is publicly available.** `make pull-dataset` downloads the 50 recorded runs,
54 replay experiments and current format 3 release `steps-v1.3+aafae73eb211` with split
`split-v2+febe942a2b59`. The immutable format 2 predecessor
`steps-v1.3+a1b96c4ff2c8` remains available. An empty-cache independent clone fetched
2,012 objects and reproduced the current release and split byte for byte.

Pulls go through the bucket's public HTTPS address (the `public` remote in `.dvc/config`).
Plain `dvc pull` uses the default `gs://` remote instead, which works without a login but
spends a long time probing for Google credentials first; use `make pull-dataset` or
`dvc pull -r public`.

**2. From the shared archive.** The same gated release (50 real runs: 25 CPU and 25 coding,
recorded 2026-10-21) is also committed as a zip archive:

```bash
unzip -q data/shared/agentwatch-dataset-steps-v1.2-5954f97ef8b2.zip
cp -R agentwatch-dataset-2026-10-20/data/* data/
rm -r agentwatch-dataset-2026-10-20
```

This fills `data/evidence/` (the raw runs) and `data/datasets/` (release
`steps-v1.2+5954f97ef8b2` with its gate result and split). The archive's own `README.md` and
`SHA256SUMS` describe and verify it. `SHA256SUMS` has Windows line endings, so check it with
`tr -d '\r' < SHA256SUMS | shasum -a 256 -c -` from inside the unzipped folder.

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
| `make add-evidence RUN=".data/coding-demo"` | acquire (recorded run directories or replay experiment directories) |
| `make dataset` | build |
| `make validate-dataset` | TFDV gate (exits 1 on anomalies) |
| `make split-dataset` | splits and leakage check |
| `make slice-dataset` | bias slices |
| `make version-dataset` | integrity check, then DVC snapshot |
| `make push-dataset` / `make pull-dataset` | DVC remote |
| `make dataset-status` | what is not versioned or uploaded yet |

**Expected result on the shared archive.** The archive holds the 50 CPU and coding runs and the
`steps-v1.2+5954f97ef8b2` release built from them. The current code builds `steps-v1.3`
releases and splits under `eval-contract-v2`, so the stages do not reuse that release.
Reproduced with the underlying stage commands on 2026-11-06 (not with `make pipeline-run`, so
the version, push and summary stages were not run): the build creates `steps-v1.3+cad82758d117`
(350 rows, 0 rejections); the gate passes against `steps-v1.3.pbtxt` with drift not compared
(no earlier release passed against that schema); the split `split-v2+9d83685148b1` sends both
CPU and coding task groups to test, because their method `worker_queue_blocker` is held out, so
train and validation are empty, the leakage check is `inconclusive` (the training split is
empty), class balance is `not_met` (no `inference`, `retrieval` or `mixed` steps, 50 `clean`),
the held-out check is `not_met` and mixed in test is `not_met`; the slice report (`slices-v2`)
raises 12 flags. The split is kept as evidence and must not be used for evaluation. In a full
run, `push` is skipped because pushing is off (see
[Data versioning with DVC](#data-versioning-with-dvc)), and the leakage check that did not pass
is reported as a finding. With a webhook set, the summary arrives in Slack. The archive's own
`steps-v1.2` split (leakage `blocked`) is still in its datasets store and still loads with the
v1 files (below).

**Dataset v1 (`steps-v1.3`).** With the replay evidence in the store, `make dataset` builds
`steps-v1.3+aafae73eb211` (5,953 rows: 5,603 replay, 175 CPU, 175 coding; 411 recorded
rejections), the gate passes with drift not compared, and `make split-dataset` writes
`split-v2+febe942a2b59` (train 2,028, validation 0, test 3,925; class balance met; leakage
passed at 0.974242 against the 0.98 bound). `make rule-baseline` and `make train-candidates` on
that split stop because `mixed` has no scoring rule until M02. To rerun the `steps-v1.2` work,
pass the v1 contract and schema. The split targets always take the newest release, so split
`steps-v1.2` directly:

```bash
uv run --package agentwatch-data python -m agentwatch_data.splits \
  --release data/datasets/steps-v1.2+5954f97ef8b2 \
  --contract pipelines/model/evaluation-contract-v1.toml \
  --frozen-schema pipelines/validation/schema/steps.pbtxt
make rule-baseline SPLIT=data/datasets/splits/steps-v1.2+5954f97ef8b2/split-v1+58e921fa846a.json \
  EVALUATION_CONTRACT=pipelines/model/evaluation-contract-v1.toml \
  VALIDATION_SCHEMA=pipelines/validation/schema/steps.pbtxt
```

`make train-candidates` takes the same two variables. Details are in the data card's
[Dataset v1](../docs/data-card.md#dataset-v1-steps-v13aafae73eb211) section.

## Pipeline components

| Course requirement | Where | Details |
|---|---|---|
| Data acquisition | `evidence.py`, `extract.py` | Recorded runs (local directories or the ingestion API) and agent replay experiment directories are validated strictly and copied into an immutable evidence store. Each untrustworthy run is rejected with a reason; duplicates are detected across sources. |
| Preprocessing and feature engineering | `flatten.py`, `labels.py`, `schema.py`, `build.py` | One row per step; telemetry readings joined to each step's time window, with coverage; queue fraction and other features; labels from a frozen, fingerprinted rule file; every row validated against the column contract. Missing readings stay `null`, never zero. |
| Schema and statistics | `pipelines/validation`, manifest, split file | TFDV statistics for every release, checked against the reviewed schema in `schema/steps-v1.3.pbtxt` (`schema/steps.pbtxt` for `steps-v1.2`). Each attempt's statistics are kept; drift is measured against the previous release that passed against the same frozen schema. The manifest records null counts per column, and the split file missingness per feature version, class balance and leakage. |
| Anomaly detection | TFDV gate, per-run rejections | Missing values, out-of-range values, unknown categories and type changes fail the gate and stop the run. Drift is reported without failing. |
| Alerts | `alerts.py`, the DAG | Slack message for every task's final failure (error tail and log link) and every run's summary, with findings: drift, rejected-run share over the declared 10%, leakage check not passed, version not pushed. |
| Bias detection and mitigation | `slices.py`, data card "Slices and bias" | Slices by task, kind, label, condition, injection method, model, source, telemetry status and PSI/cgroup availability, plus feature version and replay condition on `steps-v1.3`; flags against thresholds declared before evaluation; a mitigation per flag; balancing weights. |
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
make push-dataset          # upload the snapshot to the team bucket (needs write access)
git add data/evidence.dvc data/datasets.dvc && git commit -m "chore: version dataset <release>"
```

**Remotes** (`.dvc/config`, no credentials committed):

| Remote | Address | Used for |
|---|---|---|
| `gcs` (default) | `gs://agentwatch-dvc/agentwatch/dvc` | Pushing; needs a teammate's write access and `gcloud auth application-default login` |
| `public` | `https://storage.googleapis.com/agentwatch-dvc/agentwatch/dvc` | Pulling (`make pull-dataset`, `make dataset-status`); no login |

The bucket is publicly readable because it holds only data that is already public in this
repository. Writing is limited to the team.

**Pushing from the DAG is opt-in.** Every run appends a quality-gate attempt to the datasets
store, so every run has something new to upload, and a clone without write access would fail
there. The DAG's `push` task therefore uploads only when `AGENTWATCH_DVC_PUSH_REMOTE` names a
remote you can write to; otherwise it is skipped and the run summary reports the version as
not pushed:

```bash
AGENTWATCH_DVC_PUSH_REMOTE=gcs make pipeline-run      # team members with write access
```

Before snapshotting, the integrity check refuses to version a store that was edited, is
missing files, holds a release its evidence no longer rebuilds, or would drop anything already
versioned. A fresh clone that pulls a version can rebuild its release byte for byte; the
versioning tests check this. Once per clone, `make dataset-merge-driver` lets git merge pointer
files that two people updated at the same time. The full versioning rules are in the data
card's [Versioning](../docs/data-card.md#versioning) section.

**Status (2026-10-27).** The shared release is versioned and pushed to the bucket, and its
pointer files are committed. A fresh clone with no Google login ran `make pull-dataset` (109
files in 4 s) and then `make pipeline-run`, which succeeded with `push` skipped.

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

- **A checkout path with spaces** breaks the `make` data targets (`add-evidence`, `dataset`,
  `validate-dataset`, `split-dataset`, `slice-dataset`, `freeze-dataset-schema`,
  `version-dataset` and the DVC targets), which derive `AGENTWATCH_ROOT` from the Makefile's
  location with `$(dir ...)`, and that splits the path at the space (uv then reports
  `error: Project directory ... does not exist` for a fragment of the path). This predates D03;
  the failing commands write nothing, and the Makefile-driven tests in `pipelines/versioning`
  fail the same way from such a checkout.
  Clone into a path without spaces, or run them with `AGENTWATCH_ROOT=.` from the repository
  root, for example `make slice-dataset AGENTWATCH_ROOT=.`; `make -n <target> AGENTWATCH_ROOT=.`
  prints the commands. The underlying commands, run from the repository root:

  ```bash
  uv run --package agentwatch-data python -m agentwatch_data.evidence --store data/evidence <dirs>   # add-evidence
  uv run --package agentwatch-data python -m agentwatch_data.build --source local:data/evidence --output data/datasets   # dataset
  release=$(uv run --package agentwatch-data python -m agentwatch_data.releases --output data/datasets)   # newest release
  uv run --project pipelines/validation python -m agentwatch_validation validate --release "$release" --schema pipelines/validation/schema/steps-v1.3.pbtxt   # validate-dataset
  uv run --package agentwatch-data python -m agentwatch_data.splits --release "$release" --contract pipelines/model/evaluation-contract.toml --frozen-schema pipelines/validation/schema/steps-v1.3.pbtxt   # split-dataset
  mkdir -p .dvc/tmp && uv run --project pipelines/versioning --frozen dvc data status --granular --unchanged --json > .dvc/tmp/version-dataset-status.json   # version-dataset, step 1
  uv run --package agentwatch-data python -m agentwatch_data.integrity --evidence data/evidence --datasets data/datasets --dvc-status .dvc/tmp/version-dataset-status.json   # step 2; stop unless it exits 0
  uv run --project pipelines/versioning --frozen dvc add data/evidence data/datasets   # step 3
  ```
- **Port 8080 already in use** (by Docker, for example): start the UI on another port, and
  point tasks at it too:
  `AIRFLOW__API__PORT=8095 AIRFLOW__CORE__EXECUTION_API_SERVER_URL=http://localhost:8095/execution/ make pipeline-ui`.
- **The first run is slow** without `make pipeline-setup`: TensorFlow's first import in a new
  environment takes 18–25 s on macOS.
- **The leakage check is `inconclusive`** on the shared archive alone. This is expected:
  `split-v2` holds out `worker_queue_blocker`, the method of every CPU and coding run, so the
  training split is empty. The split is kept as evidence and refused for evaluation. (Its
  `steps-v1.2` split under `eval-contract-v1` is `blocked`: two tasks and two classes separate
  perfectly.)
- **No Slack message**: check `.env` for `AGENTWATCH_ALERT_SLACK_WEBHOOK_URL`. Without it, the
  `alert` task is marked skipped and lists its findings in its log.

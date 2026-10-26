# AgentWatch

AgentWatch is a resource-class bottleneck profiler for agentic systems. It estimates how much
of each agent step was working versus waiting, attributes contention to GPU inference, CPU
sandboxes, or retrieval, and displays confidence and model-quality measurements.

An eight-week DADS 7305 MLOps final project.

## Contributing

See [the contributor guide](CONTRIBUTING.md) for setup, branch naming, verification,
and PR review requirements. Coding agents must also follow [AGENTS.md](AGENTS.md).

## Current project plans

- [Project completion and testing checklist](docs/project-checklist.md)
- [Problem statement and project plan](docs/problem-statement-and-plan.md)
- [Feature plan](docs/feature-plan.md)
- [Implementation plan and migration backlog](docs/implementation-plan.md)

Planned stack: an OpenTelemetry-instrumented Python agent, resource sampling, Prometheus,
LightGBM, MLflow, FastAPI, Postgres, Grafana, Airflow, and GKE. Evaluation compares learned
attribution against threshold rules using reproducible contention experiments.

## Implementation status

The repository contains a runnable CPU worker queue experiment, a dashboard, and a PostgreSQL-backed
ingestion API. Experiments export OpenTelemetry spans, apply a rule baseline, and produce a local report.
Hardware saturation sampling, learned attribution, and production dashboards remain planned work.

The dashboard has seven functional pages for workspace activity, recorded runs, trace inspection,
agents, resource evidence, comparisons, and settings. See [the dashboard guide](docs/dashboard.md)
for its filters, saved preferences, and measurement definitions.

### Local LLM with tool support

Run `make llm-up` to start Docker Ollama and download Qwen3 1.7B. With PostgreSQL and the
ingestion API running, `make agent-run` executes one fixed coding task with the instrumented
agent and reports its model calls, tool calls and retrieval as live steps; see
[the agent runner guide](services/agent/README.md). The dashboard's **Agent** tab shows agent
evidence saved by the earlier tool-calling demo. See [the Docker LLM guide](docs/llm-demo.md)
for model setup and limits.
`AGENTWATCH_LLM_GPU=1 make llm-up` adds optional NVIDIA GPU access; the measured signals on the
team's GPU laptop are in [the GPU measurement access record](docs/gpu-measurement-access.md).

## First working experiment

With `uv` installed, run:

```sh
make demo-cpu
```

Open `.data/cpu-demo/report.html` to compare the same CPU task with an idle and busy worker.
The command runs five clean baseline repetitions, then submits a competing CPU task ahead of
the measured task to a single-worker process pool. It uses real enqueue/start timestamps to
attribute sandbox queue delays, without giving the attribution rule injection metadata.

The report includes directly measured queue time, execution wall time, and resource attribution.
`results.json` preserves measurements and experiment checks; `spans.jsonl` contains the
OpenTelemetry spans. Each rerun replaces these local artifacts.

The experiment passes when baseline execution variability is below 15%, the task outputs match,
the clean step is classified `none`, and the busy-worker step has an increased queue delay
classified `cpu_sandbox`. A noisy run exits unsuccessfully and keeps its evidence for inspection.

This first proof measures CPU **worker capacity contention**, not host CPU saturation. Execution
wall time is not pure working time, and the rule does not provide learned or calibrated confidence.
No Docker, cloud account, GPU, or model download is required. See
[the CPU demo guide](docs/cpu-demo.md) for details and the next milestone.

### Coding test-tool experiment

```sh
make demo-coding
```

Open `.data/coding-demo/report.html`. This replaces the hash-only step with an actual test-tool
invocation: a separate Python interpreter executes three checks in a fixed small project.
It uses the same clean/busy-worker replay and exports `sandbox.run_tests` spans.
This is a deterministic tool fixture, not yet a model-driven coding agent.

Both experiments sample Linux host CPU pressure and cgroup v2 CPU counters every 250 ms into
`samples.jsonl` and attach summaries for each evaluation step's time window. On macOS or systems
without those files, metrics are explicitly unavailable. Attribution still uses directly measured
worker queue time; missing hardware telemetry is never replaced with synthetic measurements.

## Dashboard

```sh
make dev-web
```

Open `http://localhost:3000` to explore the recorded CPU and coding experiments. The dashboard
uses Tailwind CSS, shadcn/ui, and Recharts. Switch experiments, compare execution and queue time,
inspect the trace waterfall, filter steps, and open the evidence panel for each attribution.
The telemetry view distinguishes missing CPU pressure readings from measured zero values.

For persistent history, start PostgreSQL and the ingestion API:

```sh
make infra-up
make dev-worker
```

In another terminal, import existing evidence or record a new run:

```sh
make import-runs
make record-coding
```

Use **Refresh**, choose a saved run, and optionally compare it with another run of the same kind.
Each run retains its own results, OpenTelemetry spans, and CPU samples. Downloads follow the selected
run even after new experiments replace the local files. Duplicate imports do not duplicate records.

The dashboard reads history through the FastAPI service at `http://127.0.0.1:8090`.
`AGENTWATCH_API_URL` overrides this address on the Next.js server and publishing client.
If the API is unavailable, the dashboard explicitly falls back to the latest local artifacts;
`AGENTWATCH_DATA_DIR` overrides their directory. Without recorded experiments, setup instructions
appear instead of synthetic charts. Historical run storage is implemented; live agent ingestion
and running-step updates remain future work. See [the ingestion guide](docs/ingestion-api.md).

## Repository layout

- `apps/web` — Next.js product UI and presentation BFF
- `apps/control-plane` — Go health API scaffold
- `apps/cli` — Go command-line client
- `services/worker` — FastAPI ingestion service, profiling, and experiment tooling
- `services/sandbox` — Python sandbox runtime scaffold
- `api` — OpenAPI and Protobuf contracts
- `packages` — shared UI and generated SDK packages
- `db` — database documentation; packaged migrations live with the ingestion service
- `deploy` — local Compose, Helm, and Terraform assets
- `pipelines` — dataset pipeline: stages (`data`), TFDV gate (`validation`), DVC (`versioning`),
  and the Airflow DAG (`orchestration`)

Each language uses its native toolchain. The root `Makefile` only coordinates common tasks.

## Data pipeline in Airflow

The Airflow DAG `agentwatch_dataset` in `pipelines/orchestration/dags/` runs the dataset commands
in order: acquire (add recorded runs to the evidence store) → build → validate (TFDV gate) →
split → version (integrity check, then DVC) → push → summarize → alert. Each task runs the same command
as its `make` target, in that stage's own environment. Refused run directories, gate anomalies,
or a failed integrity check fail the task and stop the run. Runs the build rejects do not: each is
recorded with its reason in the build report, and the run continues with the rest. Only the push
to the DVC remote is retried, and it is skipped only when DVC reports no default remote. The last
task logs a run summary (release, rows, runs and rejections, build report, split, leakage status,
and whether the version was pushed).

```bash
uv sync --project pipelines/orchestration --frozen        # Airflow's environment, once
make pipeline-run                                         # run the DAG once from the stored evidence
make pipeline-run PIPELINE_RUN_DIRS=".data/coding-demo"   # add recorded runs first
make pipeline-ui                                          # scheduler and UI on http://localhost:8080
```

Airflow's home (metadata database and task logs) is `.data/airflow`. Tasks read
`AGENTWATCH_ROOT` (default: this checkout), `AGENTWATCH_DATASET_API_URL` (also build from runs
saved through the ingestion API), and `AGENTWATCH_STAGE_TIMEOUT_SECONDS` (default 1800). The DAG
does not commit the updated `data/*.dvc` files; commit them yourself after a run.

### Slack alerts

The DAG posts to a Slack incoming webhook when any task fails for good (with its error tail and
log link), and posts every successful run's summary (release, rows, runs, rejections, split,
versioned and pushed). The summary is marked ✅ when clean, or ⚠️ with its findings: drift
against the previous passing release, more than `AGENTWATCH_ALERT_MAX_REJECTED_SHARE` of runs
rejected by the build (default 0.10, declared before collection), a leakage check that did not
pass, or a version not pushed because no DVC remote is configured. Without a webhook, a failure
only logs a warning, and the `alert` task lists its findings in its log and is marked skipped,
so nothing looks delivered when it was not.

1. In Slack, create an app at <https://api.slack.com/apps> → **Incoming Webhooks** → turn on →
   **Add New Webhook to Workspace** → pick the alerts channel, and copy the URL.
2. Put it in the repository's `.env` (git-ignored; copy `.env.example` if you have none). The
   DAG reads only the two `AGENTWATCH_ALERT_*` keys from it, and a variable exported in the
   shell or Airflow's environment overrides the file:

   ```bash
   AGENTWATCH_ALERT_SLACK_WEBHOOK_URL=https://hooks.slack.com/services/…
   AGENTWATCH_ALERT_MAX_REJECTED_SHARE=0.10
   ```

3. Run `make pipeline-run`.

Stage details
are in [the data card](docs/data-card.md) and [the data pipeline plan](docs/data-pipeline-plan.md).

## Prerequisites

- Node.js 22 or later and pnpm 10
- Python 3.13 or later and `uv`
- Go 1.27 or later
- Docker with Compose

## Run the development scaffold

```sh
cp .env.example .env
make bootstrap
make infra-up
```

Then start the components you are working on in separate terminals:

```sh
make dev-web
make dev-api
make dev-worker
```

The web app runs at `http://localhost:3000`, the control plane at `http://localhost:8080`,
and the worker administration API at `http://localhost:8090`.

Run all available checks with `make check`. See [the AgentWatch implementation plan](docs/implementation-plan.md)
for the replacement architecture and delivery sequence.

# AgentWatch

AgentWatch is a resource-class bottleneck profiler for agentic systems. It estimates how much
of each agent step was working versus waiting, attributes contention to GPU inference, CPU
sandboxes, or retrieval, and displays confidence and model-quality measurements.

An eight-week DADS 7305 MLOps final project.

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

Run `make llm-up` to start Docker Ollama and download Qwen3 1.7B, then `make demo-llm`.
With the ingestion API running, `make record-llm` saves the model-invoked coding experiment
to dashboard history. The **Agent** tab shows its actual conversation, inference timings,
token counts, and model/tool timeline, with evidence retained both locally and in the saved run.
See [the Docker LLM test guide](docs/llm-demo.md) for setup, evidence, and CPU-only limits.

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

Each language uses its native toolchain. The root `Makefile` only coordinates common tasks.

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

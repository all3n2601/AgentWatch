.DEFAULT_GOAL := help

.PHONY: help bootstrap dev dev-web dev-api dev-worker demo-cpu demo-coding record-coding record-cpu import-runs add-evidence dataset validate-dataset split-dataset slice-dataset rule-baseline train-candidates freeze-dataset-schema version-dataset retire-release push-dataset pull-dataset fetch-dataset dataset-status dataset-merge-driver requirements pipeline-setup pipeline-run pipeline-ui llm-up llm-down agent-tasks agent-run agent-resend agent-overhead agent-experiment sandbox-image sandbox-verify sandbox-sweep infra-up infra-down format lint typecheck test check

help: ## Show available commands
	@awk 'BEGIN {FS = ":.*## "; printf "AgentWatch commands:\n"} /^[a-zA-Z_-]+:.*## / {printf "  %-14s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

bootstrap: ## Install JavaScript and Python dependencies
	pnpm install
	uv sync --all-packages

dev: ## Start infrastructure and print app development commands
	$(MAKE) infra-up
	@echo "Run in separate terminals: make dev-web, make dev-api, make dev-worker"

dev-web: ## Start the Next.js app
	pnpm --filter @agentwatch/web dev

dev-api: ## Start the Go control plane
	go run ./apps/control-plane/cmd/server

dev-worker: ## Start the ingestion API and initialize PostgreSQL schema
	uv run --package agentwatch-worker uvicorn agentwatch_worker.app:app --reload --port 8090

demo-cpu: ## Run the local CPU queue experiment and generate a report
	uv run --package agentwatch-worker python -m agentwatch_worker.demo

demo-coding: ## Profile a real Python test tool with clean and busy-worker runs
	uv run --package agentwatch-worker python -m agentwatch_worker.demo --workload coding --output .data/coding-demo

record-coding: ## Run the coding experiment and save it in PostgreSQL through the API
	uv run --package agentwatch-worker python -m agentwatch_worker.demo --workload coding --output .data/coding-demo --publish

record-cpu: ## Run the CPU experiment and save it through the API
	uv run --package agentwatch-worker python -m agentwatch_worker.demo --publish

import-runs: ## Import existing local experiment results, spans, and samples
	uv run --package agentwatch-worker python -m agentwatch_worker.publish --output .data/coding-demo --kind coding
	uv run --package agentwatch-worker python -m agentwatch_worker.publish --output .data/cpu-demo --kind cpu

# Curated evidence and dataset releases are DVC-tracked under data/; .data/ stays scratch output.
DATASET_EVIDENCE ?= data/evidence
DATASET_SOURCES ?= local:$(DATASET_EVIDENCE)
DATASET_OUTPUT ?= data/datasets
# Tools resolve from this checkout, so the dataset targets also work with make -f from another
# directory (the versioning tests run them in throwaway repositories).
AGENTWATCH_ROOT := $(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))
DATA = uv run --project "$(AGENTWATCH_ROOT)" --package agentwatch-data python -m agentwatch_data
DVC = uv run --project "$(AGENTWATCH_ROOT)/pipelines/versioning" --frozen dvc

add-evidence: ## Validate recorded runs and copy them into the versioned evidence store (RUN=".data/coding-demo ...")
	$(if $(RUN),,$(error Set RUN to one or more recorded run directories, for example RUN=.data/coding-demo))
	$(DATA).evidence --store $(DATASET_EVIDENCE) $(RUN)

dataset: ## Build a versioned step dataset release from the evidence store (DATASET_SOURCES="local:data/evidence api:http://127.0.0.1:8090")
	$(DATA).build $(foreach source,$(DATASET_SOURCES),--source $(source)) --output $(DATASET_OUTPUT)

VALIDATION_SCHEMA ?= pipelines/validation/schema/steps.pbtxt
LATEST_RELEASE = $(DATA).releases --output $(DATASET_OUTPUT)

validate-dataset: ## Check the newest published release with the TFDV quality gate (Python 3.13; exits 1 on anomalies)
	release=$$($(LATEST_RELEASE)) && uv run --project pipelines/validation python -m agentwatch_validation validate --release "$$release" --schema $(VALIDATION_SCHEMA)

EVALUATION_CONTRACT ?= pipelines/model/evaluation-contract.toml

split-dataset: ## Freeze train/validation/test splits of the newest release (refused unless its latest gate verdict passed), with the leakage check
	release=$$($(LATEST_RELEASE)) && $(DATA).splits --release "$$release" --contract $(EVALUATION_CONTRACT) --frozen-schema $(VALIDATION_SCHEMA)

slice-dataset: ## Slice the newest release for bias (with its newest split, if any): flags, mitigations, and balancing weights under .data/reports/<release>
	release=$$($(LATEST_RELEASE)) && $(DATA).slices --release "$$release" --newest-split

RULE_BASELINE_OUTPUT ?= .data/rule-baseline

rule-baseline: ## Evaluate rules-v1 on a frozen split against the evaluation contract (SPLIT=split file; a blocked split exits 1)
	@test -n "$(SPLIT)" || { echo "Set SPLIT to a frozen split file: make rule-baseline SPLIT=<split.json>" >&2; exit 2; }
	uv run --package agentwatch-model python -m agentwatch_model.rule_baseline --split "$(SPLIT)" --contract $(EVALUATION_CONTRACT) --frozen-schema $(VALIDATION_SCHEMA) --output $(RULE_BASELINE_OUTPUT)

CANDIDATES_OUTPUT ?= .data/candidates

train-candidates: ## Train LightGBM candidates on a frozen split, compare with rules-v1, and track the run in MLflow (SPLIT=split file; blocked or failed runs exit 1)
	@test -n "$(SPLIT)" || { echo "Set SPLIT to a frozen split file: make train-candidates SPLIT=<split.json>" >&2; exit 2; }
	uv run --package agentwatch-model python -m agentwatch_model.train_candidates --split "$(SPLIT)" --contract $(EVALUATION_CONTRACT) --frozen-schema $(VALIDATION_SCHEMA) --output $(CANDIDATES_OUTPUT)
freeze-dataset-schema: ## Regenerate the frozen TFDV schema from the newest release; review the diff in a PR
	release=$$($(LATEST_RELEASE)) && uv run --project pipelines/validation python -m agentwatch_validation freeze-schema --release "$$release" --schema $(VALIDATION_SCHEMA)

DVC_STATUS = .dvc/tmp/version-dataset-status.json

version-dataset: ## Verify the stores (intact, reproducible, append-only), then snapshot both with DVC; commit the updated data/*.dvc files
	mkdir -p .dvc/tmp && $(DVC) data status --granular --unchanged --json > $(DVC_STATUS)
	$(DATA).integrity --evidence $(DATASET_EVIDENCE) --datasets $(DATASET_OUTPUT) --dvc-status $(DVC_STATUS)
	$(DVC) add $(DATASET_EVIDENCE) $(DATASET_OUTPUT)

retire-release: ## Remove a release that was never versioned, with its reports, gate attempts, and splits (RELEASE=<name>; refused if DVC holds any of it)
	$(if $(RELEASE),,$(error Set RELEASE to the release to remove, for example RELEASE=steps-v1.2+0123456789ab))
	mkdir -p .dvc/tmp && $(DVC) data status --granular --unchanged --json > $(DVC_STATUS)
	$(DATA).retire --datasets $(DATASET_OUTPUT) --dvc-status $(DVC_STATUS) $(RELEASE)

# Pulls use the bucket's public HTTPS address, so they need no Google login; pushes go to the
# default gs:// remote and need write access (gcloud auth application-default login).
DVC_PULL_REMOTE ?= public

push-dataset: ## Upload versioned evidence and datasets to the team bucket (needs write access; see docs/data-card.md)
	$(DVC) push

pull-dataset: ## Download the evidence and datasets versioned at this commit, no login needed (TARGET="<paths>" for part of it)
	$(DVC) pull --remote $(DVC_PULL_REMOTE) $(TARGET)

fetch-dataset: ## Fetch every committed version's data into the DVC cache (before merging branches that both versioned data)
	$(DVC) fetch --all-commits

dataset-merge-driver: ## Once per clone: let git merge concurrent data/*.dvc versions as the union of their files
	git config merge.dvc.name "DVC merge driver for the append-only data stores"
	git config merge.dvc.driver "$(DVC) git-hook merge-driver --ancestor %O --our %A --their %B"

dataset-status: ## Show changes not yet versioned, and (with a remote configured) versioned data not yet uploaded
	$(DVC) status
	$(DVC) status --cloud --remote $(DVC_PULL_REMOTE)

# The Airflow DAG runs the dataset targets above in order. Airflow has its own environment, and
# its home (metadata database, task logs) is local scratch output under .data/. Run from the
# repository root; $(CURDIR) keeps paths that contain spaces intact.
AIRFLOW_HOME ?= $(CURDIR)/.data/airflow
AIRFLOW = AIRFLOW_HOME="$(AIRFLOW_HOME)" AIRFLOW__CORE__DAGS_FOLDER="$(CURDIR)/pipelines/orchestration/dags" AIRFLOW__CORE__LOAD_EXAMPLES=False uv run --project pipelines/orchestration --frozen airflow
PIPELINE_RUN_DIRS ?=
comma := ,
space := $(subst ,, )
PIPELINE_RUN_DIRS_JSON = [$(subst $(space),$(comma),$(foreach dir,$(PIPELINE_RUN_DIRS),"$(dir)"))]

# Run from the repository root (as make does by default): the cd paths are relative to it.
requirements: ## Export pinned requirements.txt files from the uv lockfiles, for pip users (CI checks they match)
	uv export --frozen --package agentwatch-data --no-dev --no-hashes --quiet -o requirements.txt
	cd pipelines/validation && uv export --frozen --no-dev --no-hashes --quiet -o requirements.txt
	cd pipelines/versioning && uv export --frozen --no-dev --no-hashes --quiet -o requirements.txt
	cd pipelines/orchestration && uv export --frozen --no-dev --no-hashes --quiet -o requirements.txt

pipeline-setup: ## Prepare every stage environment once and warm TensorFlow, so the first DAG run does not pay for it
	uv sync --frozen --all-packages --compile-bytecode
	uv sync --project pipelines/validation --frozen --compile-bytecode
	uv sync --project pipelines/versioning --frozen --compile-bytecode
	uv sync --project pipelines/orchestration --frozen --compile-bytecode
	uv run --project pipelines/validation --frozen python -c "import tensorflow_data_validation"

pipeline-run: ## Run the whole Airflow dataset DAG once, without a scheduler (PIPELINE_RUN_DIRS=".data/coding-demo ..." to add runs first)
	$(if $(findstring ",$(PIPELINE_RUN_DIRS))$(findstring ',$(PIPELINE_RUN_DIRS)),$(error PIPELINE_RUN_DIRS paths cannot contain quotes; trigger the DAG with its run_dirs parameter instead))
	$(AIRFLOW) db migrate
	$(AIRFLOW) dags test agentwatch_dataset -c '{"run_dirs": $(PIPELINE_RUN_DIRS_JSON)}'

pipeline-ui: ## Start Airflow locally (UI on http://localhost:8080; the login is printed and saved under AIRFLOW_HOME)
	$(AIRFLOW) standalone

# Make variables rather than shell defaults: Windows make may run recipes through cmd.exe.
AGENTWATCH_LLM_MODEL ?= qwen3:1.7b
OLLAMA_HOST_PORT ?= 11434
OLLAMA_URL ?= http://127.0.0.1:$(OLLAMA_HOST_PORT)
export AGENTWATCH_LLM_MODEL OLLAMA_HOST_PORT OLLAMA_URL
LLM_COMPOSE = docker compose -f deploy/compose/compose.yaml $(if $(filter 1,$(AGENTWATCH_LLM_GPU)),-f deploy/compose/compose.gpu.yaml)

llm-up: ## Start Docker Ollama (AGENTWATCH_LLM_GPU=1 for NVIDIA, OLLAMA_HOST_PORT to move it) and pull the model
	$(if $(filter-out 0 1,$(AGENTWATCH_LLM_GPU)),$(error AGENTWATCH_LLM_GPU must be 0 or 1 (got '$(AGENTWATCH_LLM_GPU)')))
	$(LLM_COMPOSE) --profile llm up -d --wait ollama
	$(LLM_COMPOSE) exec -T ollama ollama pull $(AGENTWATCH_LLM_MODEL)

llm-down: ## Stop the local LLM while retaining downloaded models
	docker compose -f deploy/compose/compose.yaml stop ollama

AGENT_TASK ?= fix-mean-empty

agent-tasks: ## List the instrumented agent's fixed tasks
	uv run --package agentwatch-agent python -m agentwatch_agent tasks

agent-run: ## Run one fixed task with the real model, reporting live steps (AGENT_TASK=name)
	uv run --package agentwatch-agent python -m agentwatch_agent run --task $(AGENT_TASK)

agent-resend: ## Resend a run journal after a live API outage (JOURNAL=path)
	uv run --package agentwatch-agent python -m agentwatch_agent resend $(JOURNAL)

agent-overhead: ## Measure agent instrumentation overhead (live API must be running)
	uv run --package agentwatch-agent python -m agentwatch_agent overhead

agent-experiment: ## Replay a task cleanly and under bounded CPU/inference/retrieval/mixed contention (AGENT_TASK=name, SOURCE=run dir to replay)
	uv run --package agentwatch-agent python -m agentwatch_agent experiment --task $(AGENT_TASK) $(if $(SOURCE),--source "$(SOURCE)")

SANDBOX_IMAGE = $$(uv run --package agentwatch-sandbox python -c "from agentwatch_sandbox.container import SANDBOX_IMAGE; print(SANDBOX_IMAGE)")

sandbox-image: ## Pull the digest-pinned image the isolated test runner uses (3 attempts)
	@image="$(SANDBOX_IMAGE)"; for attempt in 1 2 3; do \
		docker pull "$$image" && exit 0; \
		echo "docker pull failed (attempt $$attempt of 3)" >&2; \
		[ "$$attempt" -lt 3 ] && sleep 10; \
	done; exit 1

sandbox-verify: ## Verify container isolation boundaries and cleanup (needs Docker)
	AGENTWATCH_TEST_DOCKER=1 uv run pytest services/sandbox/tests/test_container.py

sandbox-sweep: ## Remove sandbox containers left by a killed runner (past their deadline label)
	uv run --package agentwatch-sandbox python -c "from agentwatch_sandbox.container import remove_orphaned_sandboxes as r; print('\n'.join(r()) or 'none')"

infra-up: ## Start the AgentWatch PostgreSQL service
	docker compose -f deploy/compose/compose.yaml up -d

infra-down: ## Stop local infrastructure
	docker compose -f deploy/compose/compose.yaml down

format: ## Format all supported languages
	pnpm -r exec prettier --write .
	uv run ruff format .
	@if command -v go >/dev/null; then gofmt -w apps/control-plane apps/cli; else echo "go not installed; skipped gofmt"; fi

lint: ## Run linters
	pnpm -r lint
	uv run ruff check .
	@if command -v go >/dev/null; then go vet ./apps/control-plane/... ./apps/cli/...; else echo "go not installed; skipped go vet"; fi

typecheck: ## Run static type checks
	pnpm -r typecheck

test: ## Run unit tests
	pnpm -r test
	uv run pytest
	@if command -v go >/dev/null; then go test ./apps/control-plane/... ./apps/cli/...; else echo "go not installed; skipped Go tests"; fi

check: lint typecheck test ## Run all repository checks

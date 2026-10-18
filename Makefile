.DEFAULT_GOAL := help

.PHONY: help bootstrap dev dev-web dev-api dev-worker demo-cpu demo-coding record-coding record-cpu import-runs add-evidence dataset validate-dataset split-dataset freeze-dataset-schema version-dataset push-dataset pull-dataset fetch-dataset dataset-status dataset-merge-driver llm-up llm-down demo-llm record-llm infra-up infra-down format lint typecheck test check

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
DVC = uv run --project pipelines/versioning --frozen dvc

add-evidence: ## Validate recorded runs and copy them into the versioned evidence store (RUN=".data/coding-demo ...")
	$(if $(RUN),,$(error Set RUN to one or more recorded run directories, for example RUN=.data/coding-demo))
	uv run --package agentwatch-data python -m agentwatch_data.evidence --store $(DATASET_EVIDENCE) $(RUN)

dataset: ## Build a versioned step dataset release from the evidence store (DATASET_SOURCES="local:data/evidence api:http://127.0.0.1:8090")
	uv run --package agentwatch-data python -m agentwatch_data.build $(foreach source,$(DATASET_SOURCES),--source $(source)) --output $(DATASET_OUTPUT)

VALIDATION_SCHEMA ?= pipelines/validation/schema/steps.pbtxt
LATEST_RELEASE = uv run --package agentwatch-data python -m agentwatch_data.releases --output $(DATASET_OUTPUT)

validate-dataset: ## Check the newest published release with the TFDV quality gate (Python 3.13; exits 1 on anomalies)
	release=$$($(LATEST_RELEASE)) && uv run --project pipelines/validation python -m agentwatch_validation validate --release "$$release" --schema $(VALIDATION_SCHEMA)

EVALUATION_CONTRACT ?= pipelines/model/evaluation-contract.toml

split-dataset: ## Freeze train/validation/test splits of the newest release (refused unless its latest gate verdict passed), with the leakage check
	release=$$($(LATEST_RELEASE)) && uv run --package agentwatch-data python -m agentwatch_data.splits --release "$$release" --contract $(EVALUATION_CONTRACT) --frozen-schema $(VALIDATION_SCHEMA)

freeze-dataset-schema: ## Regenerate the frozen TFDV schema from the newest release; review the diff in a PR
	release=$$($(LATEST_RELEASE)) && uv run --project pipelines/validation python -m agentwatch_validation freeze-schema --release "$$release" --schema $(VALIDATION_SCHEMA)

DVC_STATUS = .dvc/tmp/version-dataset-status.json

version-dataset: ## Verify the stores (intact, reproducible, append-only), then snapshot both with DVC; commit the updated data/*.dvc files
	mkdir -p .dvc/tmp && $(DVC) data status --granular --json > $(DVC_STATUS)
	uv run --package agentwatch-data python -m agentwatch_data.integrity --evidence $(DATASET_EVIDENCE) --datasets $(DATASET_OUTPUT) --dvc-status $(DVC_STATUS)
	$(DVC) add $(DATASET_EVIDENCE) $(DATASET_OUTPUT)

push-dataset: ## Upload versioned evidence and datasets to the DVC remote (configure one first; see docs/data-card.md)
	$(DVC) push

pull-dataset: ## Download the evidence and datasets versioned at this commit (TARGET="<paths>" for part of it; see docs/data-card.md)
	$(DVC) pull $(TARGET)

fetch-dataset: ## Fetch every committed version's data into the DVC cache (before merging branches that both versioned data)
	$(DVC) fetch --all-commits

dataset-merge-driver: ## Once per clone: let git merge concurrent data/*.dvc versions as the union of their files
	git config merge.dvc.name "DVC merge driver for the append-only data stores"
	git config merge.dvc.driver "$(DVC) git-hook merge-driver --ancestor %O --our %A --their %B"

dataset-status: ## Show changes not yet versioned, and (with a remote configured) versioned data not yet uploaded
	$(DVC) status
	@if $(DVC) remote default >/dev/null 2>&1; then $(DVC) status --cloud; else echo "No DVC remote configured, so uploads are not checked; see docs/data-card.md"; fi

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

demo-llm: ## Run a real local LLM tool-call loop and retain its evidence
	uv run --package agentwatch-worker python -m agentwatch_worker.llm_demo

record-llm: ## Run the LLM loop and save its coding measurements in the dashboard
	uv run --package agentwatch-worker python -m agentwatch_worker.llm_demo --publish

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

.DEFAULT_GOAL := help

.PHONY: help bootstrap dev dev-web dev-api dev-worker demo-cpu demo-coding record-coding record-cpu import-runs dataset validate-dataset split-dataset freeze-dataset-schema llm-up llm-down agent-tasks agent-run agent-resend agent-overhead infra-up infra-down format lint typecheck test check

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

DATASET_SOURCES ?= local:.data
DATASET_OUTPUT ?= .data/datasets

dataset: ## Build a versioned step dataset release (DATASET_SOURCES="local:.data api:http://127.0.0.1:8090")
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

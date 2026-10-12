.DEFAULT_GOAL := help

.PHONY: help bootstrap dev dev-web dev-api dev-worker demo-cpu demo-coding record-coding record-cpu import-runs dataset llm-up llm-down demo-llm record-llm infra-up infra-down format lint typecheck test check

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

llm-up: ## Start Docker Ollama and download the small tool-capable model
	docker compose -f deploy/compose/compose.yaml --profile llm up -d --wait ollama
	docker compose -f deploy/compose/compose.yaml exec -T ollama ollama pull $${AGENTWATCH_LLM_MODEL:-qwen3:1.7b}

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

.DEFAULT_GOAL := help

.PHONY: help bootstrap dev dev-web dev-api dev-worker infra-up infra-down format lint typecheck test policy-test check

help: ## Show available commands
	@awk 'BEGIN {FS = ":.*## "; printf "Model Passport commands:\n"} /^[a-zA-Z_-]+:.*## / {printf "  %-14s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

bootstrap: ## Install JavaScript and Python dependencies
	pnpm install
	uv sync --all-packages

dev: ## Start infrastructure and print app development commands
	$(MAKE) infra-up
	@echo "Run in separate terminals: make dev-web, make dev-api, make dev-worker"

dev-web: ## Start the Next.js app
	pnpm --filter @model-passport/web dev

dev-api: ## Start the Go control plane
	go run ./apps/control-plane/cmd/server

dev-worker: ## Start the Python worker administration API
	uv run --package model-passport-worker uvicorn model_passport_worker.app:app --reload --port 8090

infra-up: ## Start PostgreSQL, MinIO, Temporal, and OPA
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

policy-test: ## Run OPA policy tests in a container
	docker run --rm -v "$(CURDIR)/policy:/policy:ro" openpolicyagent/opa:1.4.2-static test /policy

check: lint typecheck test policy-test ## Run all repository checks

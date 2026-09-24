# Model Passport

Model Passport is a modular control plane for registering models, running isolated assurance
checks, storing evidence, and enforcing deployment policy.

## Repository layout

- `apps/web` — Next.js product UI and presentation BFF
- `apps/control-plane` — authoritative Go API
- `apps/cli` — Go command-line client
- `services/worker` — customer-resident Python worker
- `services/sandbox` — isolated Python assessment runtime
- `api` — OpenAPI and Protobuf contracts
- `packages` — shared UI and generated SDK packages
- `policy` — OPA/Rego policy bundles
- `db` — database migrations and queries
- `deploy` — local Compose, Helm, and Terraform assets

Each language uses its native toolchain. The root `Makefile` only coordinates common tasks.

## Prerequisites

- Node.js 22 or later and pnpm 10
- Python 3.13 or later and `uv`
- Go 1.27 or later
- Docker with Compose

## First run

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

Run all available checks with `make check`. See [the implementation plan](docs/implementation-plan.md)
for the architecture and delivery sequence.

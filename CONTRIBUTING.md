# Contributing to AgentWatch

Read [AGENTS.md](AGENTS.md) for repository-wide engineering and collaboration
rules. Web contributors should also read [apps/web/AGENTS.md](apps/web/AGENTS.md).
Use the [project checklist](docs/project-checklist.md) to distinguish working
features from planned work.

## Setup

Install the prerequisites listed in [README.md](README.md): Node.js 22+, pnpm 10,
Python 3.13+, uv, Go 1.27+, and Docker Compose. Copy `.env.example` to `.env` only
if you do not already have a local configuration, then run `make bootstrap`.
Use the README's service instructions for your component. Never commit `.env`,
credentials, or local `.data` evidence.

## Coordinate and create a branch

1. Agree on one primary contributor and a focused task. Inspect existing issues
   and open PRs before implementing overlapping work. Document dependencies and
   intended merge order where necessary.
2. Preserve existing local work, fetch `origin`, and start from the latest
   `origin/main`. Do not push directly to `main` or work in another person's branch.
3. Use lowercase, hyphen-separated names: `feature/run-history`,
   `fix/session-timeout`, `docs/setup-guide`, `refactor/worker-queue`,
   `test/api-validation`, or `chore/update-dependencies`.
4. Make focused commits with titles such as `fix: handle expired sessions`.
   Avoid unrelated formatting, broad cleanup, empty commits, and pushes after
   every edit.

## Verify before pushing

Run checks for the components you changed. The exact CI baseline is in
[ci.yaml](.github/workflows/ci.yaml).

| Area | Local verification |
| --- | --- |
| JavaScript/TypeScript | `pnpm lint`, `pnpm typecheck`, `pnpm test`, `pnpm build` |
| Python | `uv run ruff check .`, `uv run pytest` |
| Go | Inspect `gofmt -d apps/control-plane apps/cli`; run `go vet ./apps/control-plane/... ./apps/cli/...` and `go test ./apps/control-plane/... ./apps/cli/...` |
| Infrastructure | `docker compose -f deploy/compose/compose.yaml config --quiet` with the required local environment |
| Documentation only | Check accuracy, referenced paths, and Markdown formatting |

For ingestion, storage, or migration changes, set
`AGENTWATCH_TEST_DATABASE_URL` to a dedicated PostgreSQL test database and run the
integration tests. A suite that skips database tests is incomplete verification.
Preserve the tests' isolated schemas and never clear development history.

For UI changes, check the affected flow in a browser at mobile (360–390 px),
tablet (768 px), desktop (1280–1440 px), and 200% zoom. Verify keyboard navigation,
loading/empty/error states, and contained table/chart overflow. Report which
checks actually ran.

Sync with `origin/main` before final verification. Resolve conflicts carefully,
preserving both contributors' intended behavior, and rerun affected checks.
Push only coherent, verified changes. If verification is blocked, report it and
keep the work local unless explicitly asked to share an unverified draft.

## Open and review a pull request

Use the PR template to describe the problem, behavior, testing evidence, and
dependencies. Include UI screenshots and API, migration, configuration, or
measurement impact where relevant. Request a reviewer familiar with the affected
component; component owner assignments will be documented once the team agrees
on primary and backup reviewers.

`main` requires a PR, one approval from another contributor, resolved review
conversations, and the `javascript`, `python`, `go`, and `infrastructure` checks.
The branch must be up to date with `main`. New reviewable changes dismiss prior
approvals. Force pushes and deletion of `main` are blocked for contributors.
Repository administrators have an explicit bypass, including direct pushes;
`all3n2601` is currently the only administrator. Anyone granted administrator
access in the future will also receive this bypass. Prefer PRs for ordinary work.
These requirements also apply to documentation PRs; they still run all CI jobs.

Review correctness, edge cases, test evidence, shared contracts, and AgentWatch's
measurement invariants. Do not accept fabricated telemetry or silent historical
artifact substitutions. Address feedback and obtain approval for the final code.
Coding agents must have user authorization to merge and must not use the admin
bypass unless the user explicitly authorizes that action.
After merge, the task branch may be deleted when it is no longer needed by others.

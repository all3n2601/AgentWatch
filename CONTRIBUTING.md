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

## Keep history tidy

Use one active branch and PR per task. Put related audit findings and review fixes
in the existing PR while it is open. Finish the planned audit and local checks
before requesting review, and push at verified milestones rather than after each
edit. Split only for independently testable and reversible changes or an
impractically large review; link the related PRs and explain the split and merge
order. Different pipeline stages can justify separate PRs, but a new agent
session or another finding in the same task does not.

Before pushing, review `git diff origin/main...HEAD` and
`git log --oneline origin/main..HEAD`. Use a specific behavior-focused PR title,
for example `fix: reject mismatched step checksums`, instead of `fix: audit gaps`.

GitHub permits **squash merges only**, giving each merged PR one commit on `main`.
The existing main ruleset also requires linear history and permits only squash.
Use the PR title (`<type>: <specific change>`) and a concise PR description for
the squash message; review it before merging. Task branches can contain useful
checkpoint commits and merges from main without adding them to main's history.
Start the next task from the latest `origin/main`, not the merged task branch.

These settings apply to future merges; existing commits are preserved. Do not
rewrite published history or close/consolidate another contributor's PRs without
explicit authorization. PR scope is a contributor and reviewer responsibility;
squash settings cannot determine whether two tasks should be combined. Existing
review, CI, and administrator-bypass requirements still apply.

## Verify before pushing

Run checks for the components you changed. The exact CI baseline is in
[ci.yaml](.github/workflows/ci.yaml).

| Area | Local verification |
| --- | --- |
| JavaScript/TypeScript | `pnpm lint`, `pnpm typecheck`, `pnpm test`, `pnpm build` |
| Python | `uv run ruff check .`, `uv run pytest` |
| Dataset quality gate (`pipelines/validation`, Python 3.13, macOS arm64 or Linux x86_64) | `uv run --project pipelines/validation pytest pipelines/validation/tests` |
| Dataset versioning (`pipelines/versioning`, DVC; needs `git` and `uv`) | `uv run --project pipelines/versioning pytest pipelines/versioning/tests` |
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
component and the repository owner `all3n2601`. The owner is the sole CODEOWNER
for every file, including `.github/CODEOWNERS`; do not add alternate owners or
path overrides that remove the mandatory owner approval.

`main` requires a PR, two approvals (owner `all3n2601` plus another contributor),
resolved review conversations, and the `javascript`, `python`, `go`, and `infrastructure` checks.
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

Authors cannot approve their own PRs. Owner-authored PRs require two other
reviewers or the owner's explicit admin bypass. For everyone else, another
reviewer's approval alone does not permit merging without owner approval.

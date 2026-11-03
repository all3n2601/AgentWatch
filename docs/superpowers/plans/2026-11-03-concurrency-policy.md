# O02 Versioned Manual Concurrency Policy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** An immutable, audited, versioned manual concurrency policy for the coding agent, with two optional limits (`inference_requests`, `test_jobs`). It is exposed through the worker API, the Python/TypeScript SDKs and a plain dashboard screen, and linked to the live runs it was active for.

**Architecture:** Pydantic models live in a new SDK module, following the existing `agentwatch_sdk.live` pattern. A `PolicyStore` writes append-only version rows and run links in PostgreSQL (migration 0004) under a transaction-scoped advisory lock. `LiveStore.create` takes the same lock to link each new agent run to the active version. A FastAPI router serves it, and its OpenAPI/TypeScript contracts are exported like the live API's. Next.js server routes proxy it with zod validation to a minimal client component.

**Tech Stack:** Python 3 / FastAPI / Pydantic v2 / psycopg 3 / PostgreSQL; Next.js App Router / zod / Vitest.

**Spec:** `docs/superpowers/specs/2026-11-03-concurrency-policy-design.md`

## Global Constraints

- Controls are exactly `inference_requests` and `test_jobs`; a present limit is a strict integer `>= 1`; "no limit" = key absent. `null`, `0`, negatives, floats, booleans, strings → 422.
- The only target is `coding-agent`. Any other path target → 404 `"Unsupported policy target"`.
- Version 0 = synthesized baseline (`action: "baseline"`, empty limits, `operator`/`reason`/`request_id`/`created_at` null). Stored versions start at 1. Never update or delete a version row.
- `controlled_services` for `coding-agent`: `{"inference_requests": {"service_id": "ollama", "resource_class": "inference"}, "test_jobs": {"service_id": "agentwatch-agent-runner", "resource_class": "cpu_sandbox"}}`; `enforcement_scope: "runner_process"`.
- A run is a `coding-agent` run iff a registered resource has `service_id == "agentwatch-agent-runner"`.
- Policy advisory lock key: `730503` (live run creation uses `730502`, migrations `730501`). Lock order is always 730502 → 730503.
- Reuse `Identifier` (operator, max 128) and `Reason` (reason, max 512) from `agentwatch_sdk.live`.
- Error shapes match B01: 409 conflict, 422 validation (no echoed body), 404 missing, 503 storage unavailable.
- PostgreSQL tests use the existing `client` fixture (isolated schema via `AGENTWATCH_TEST_DATABASE_URL`). A skipped database test counts as unverified.
- Commit titles: `feature: ...` / `test: ...` / `docs: ...`. No Claude co-author trailer (repo preference).

## Review Focus

1. A run registered while a version is being created must still be linked to the new version. `LiveStore.create` takes lock 730503 after 730502. Task 2 tests concurrent writers; Task 3 tests registration ordering.
2. Concurrent dashboard tabs submitting the same `expected_version` → exactly one 201 and one 409 (Task 2, threaded test).
3. Restore of a version that doesn't exist or is `>= ` the next version → 422, nothing written (Task 2).
4. Non-agent runs (overhead, CPU demo, synthetic fixtures) → `applicable: false`, `versions: []`, never a fabricated baseline (Task 3).
5. Policy API unavailable → the dashboard shows "Policy service unavailable", never "No limits" (Task 6, Vitest).

---

### Task 1: SDK policy contract models

**Files:**
- Create: `packages/sdk-python/src/agentwatch_sdk/policy.py`
- Modify: `packages/sdk-python/src/agentwatch_sdk/__init__.py` (export `PolicyChange`, `PolicyLimits`)
- Test: `services/worker/tests/test_policy_contracts.py`

**Interfaces:**
- Produces:
  - `PolicyTarget = Literal["coding-agent"]`; `SUPPORTED_TARGETS: dict[str, dict]` maps target → `{"controlled_services": ..., "enforcement_scope": "runner_process", "runner_service_id": "agentwatch-agent-runner"}`
  - `Limit = Annotated[int, Field(strict=True, ge=1, le=1024)]`
  - `class PolicyLimits(LiveModel)`: `inference_requests: Limit | None = None`, `test_jobs: Limit | None = None`; a `model_validator(mode="before")` rejects any key whose value is `None` (`"Omit a limit for no limit; null is not accepted"`). Serialize with `model_dump(exclude_none=True)`.
  - `PolicyAction = Literal["set", "disable", "restore"]`
  - `class PolicyChange(LiveModel)`: `request_id: UUID`, `expected_version: Annotated[int, Field(strict=True, ge=0)]`, `action: PolicyAction`, `limits: PolicyLimits | None = None`, `restored_from: Annotated[int, Field(strict=True, ge=0)] | None = None`, `operator: Identifier`, `reason: Reason`. Validator: `set` needs `limits` with at least one limit and no `restored_from`; `disable` has neither; `restore` needs `restored_from` and no `limits`.
  - `class PolicyVersion(LiveModel)`: `target: str`, `version: int`, `action: Literal["baseline","set","disable","restore"]`, `limits: PolicyLimits`, `restored_from: int | None`, `operator: str | None`, `reason: str | None`, `request_id: UUID | None`, `created_at: datetime | None`, `controlled_services: dict[str, ControlledService]`, `enforcement_scope: Literal["runner_process"]`
  - `class ControlledService(LiveModel)`: `service_id: str`, `resource_class: ResourceClass`
  - `class PolicyChangeResult(LiveModel)`: `created: bool`, `policy: PolicyVersion`
  - `class PolicyPage(LiveModel)`: `versions: list[PolicyVersion]`, `next_before: int | None`
  - `class RunPolicy(LiveModel)`: `run_id: UUID`, `target: str`, `applicable: bool`, `versions: list[PolicyVersion]`

- [ ] **Step 1: Write failing tests** in `test_policy_contracts.py` (no database):
  - `test_limits_accept_positive_integers_and_absence`: `PolicyLimits.model_validate({"test_jobs": 2}).model_dump(exclude_none=True) == {"test_jobs": 2}`; `{}` → `{}`.
  - `test_limits_reject_invalid` parametrized over `{"test_jobs": 0}`, `-1`, `None`, `2.0`, `True`, `"2"`, `1025`, `{"gpu": 1}` → `ValidationError`.
  - `test_change_action_shapes`: `set` without limits / with `{}` / with `restored_from`, `disable` with limits, `restore` without `restored_from` → `ValidationError`; one valid example of each passes.
  - `test_operator_and_reason_reject_blank`: `operator=" "` and `reason=""` → `ValidationError`.
- [ ] **Step 2:** `uv run pytest services/worker/tests/test_policy_contracts.py -q` → FAIL (import error).
- [ ] **Step 3:** Implement `policy.py` with the interfaces above.
- [ ] **Step 4:** Same command → PASS; `uv run ruff check packages services` clean.
- [ ] **Step 5:** Commit `feature: add concurrency policy contract models`.

### Task 2: Migration 0004 and PolicyStore

**Files:**
- Create: `services/worker/src/agentwatch_worker/migrations/0004_concurrency_policy.sql`
- Create: `services/worker/src/agentwatch_worker/policy_store.py`
- Test: `services/worker/tests/test_policy_store.py`

**Interfaces:**
- Consumes: Task 1 models; `RunStore.connect()`, `ConflictError` from `agentwatch_worker.store`.
- Produces:
  - Tables: `concurrency_policy_versions(target TEXT, version INTEGER CHECK (version > 0), action TEXT CHECK (action IN ('set','disable','restore')), limits JSONB NOT NULL, restored_from INTEGER CHECK (restored_from >= 0 AND restored_from < version), operator TEXT NOT NULL, reason TEXT NOT NULL, request_id UUID NOT NULL, request JSONB NOT NULL, controlled_services JSONB NOT NULL, enforcement_scope TEXT NOT NULL, created_at TIMESTAMPTZ NOT NULL DEFAULT now(), PRIMARY KEY (target, version), UNIQUE (target, request_id))` and `live_run_policy_links(run_id UUID REFERENCES live_runs(run_id), target TEXT, version INTEGER NOT NULL CHECK (version >= 0), linked_at TIMESTAMPTZ NOT NULL DEFAULT now(), PRIMARY KEY (run_id, target, version))`. Version 0 has no row, so links have no FK on `(target, version)`; the CHECK and store logic guard it.
  - `class UnsupportedTargetError(Exception)`, `class InvalidRestoreError(ValueError)`
  - `class PolicyStore`: `__init__(self, store: RunStore)`; `active(target: str) -> dict`; `history(target: str, before: int | None, limit: int) -> dict` (newest first, includes synthesized version 0 as the last entry); `change(target: str, change: PolicyChange) -> dict` (returns `{"created": bool, "policy": {...}}`); `@staticmethod link_new_run(conn, run_id: UUID, registration: dict) -> None`; `run_policy(run_id: UUID) -> dict` (raises `MissingRunError` from `live_store`).

- [ ] **Step 1: Write failing tests** (all use the `client` fixture so the schema is migrated; build `PolicyStore(client.app.state.store)`):
  - `test_active_defaults_to_baseline`: `active("coding-agent")["version"] == 0`, `["limits"] == {}`, `["action"] == "baseline"`, `["controlled_services"]["test_jobs"]["service_id"] == "agentwatch-agent-runner"`.
  - `test_set_creates_next_version`: `set {"inference_requests": 2}` with `expected_version=0` → `created is True`, `version == 1`; `active` returns it.
  - `test_stale_expected_version_conflicts`: second `set` with `expected_version=0` → `ConflictError`; history still has exactly versions 1 and 0.
  - `test_retry_is_idempotent`: same `PolicyChange` twice → second `created is False`, same version; same `request_id` with different reason → `ConflictError`.
  - `test_disable_and_restore`: v1 set `{test_jobs: 1}`, v2 disable → limits `{}`, v3 `restore restored_from=1` → limits `{test_jobs: 1}`, `restored_from == 1`; `restore restored_from=0` → limits `{}`.
  - `test_restore_missing_version_rejected`: `restored_from=7` with `expected_version` current → `InvalidRestoreError`; no row written.
  - `test_unsupported_target`: `active("cpu-demo")` and `change("cpu-demo", ...)` → `UnsupportedTargetError`.
  - `test_concurrent_writers_one_wins`: two threads, each with its own `PolicyChange(expected_version=0)` and a distinct `request_id` → results sorted are exactly one success and one `ConflictError`.
  - `test_failed_change_rolls_back`: monkeypatch a statement after the insert to raise → no version row and no link rows remain.
  - `test_migration_upgrades_populated_0003_schema`: mirror `test_live_api.py::test_migration_upgrades_existing_schema`. Apply 0001–0003, insert one live run, run `migrate()`, assert the run still reads back and `active()` is version 0.
- [ ] **Step 2:** `uv run pytest services/worker/tests/test_policy_store.py -q` with `AGENTWATCH_TEST_DATABASE_URL` set → FAIL.
- [ ] **Step 3:** Write the migration and `PolicyStore`. In `change`, take `pg_advisory_xact_lock(730503)` and set `statement_timeout '5s'` / `lock_timeout '3s'` as `LiveStore` does. Check `request_id` (idempotency) before `expected_version`. Insert the version, then link it to every `live_runs` row with `state IN ('queued','running')` whose `registration->'resources'` contains `service_id = runner_service_id`. Use parameterized SQL only.
- [ ] **Step 4:** Same command → PASS, 0 skipped.
- [ ] **Step 5:** Commit `feature: store versioned concurrency policy with audit history`.

### Task 3: Link live runs at registration

**Files:**
- Modify: `services/worker/src/agentwatch_worker/live_store.py` (`create`, ~line 65)
- Test: `services/worker/tests/test_policy_store.py`

**Interfaces:**
- Consumes: `PolicyStore.link_new_run`, `PolicyStore.run_policy`.
- Produces: every newly created agent run has a link row for the version active at registration.

- [ ] **Step 1: Write failing tests** (agent registration = conftest `registration` with resource `service_id` replaced by `agentwatch-agent-runner`):
  - `test_runs_record_version_active_at_registration`: run A registered at v0, then set v1, run B registered → `run_policy(A)` versions `[0, 1]` (A was open), `run_policy(B)` versions `[1]`.
  - `test_terminal_runs_not_linked_to_later_versions`: register agent run, move it to `running` then `completed` through `LiveStore.change`, set v1 → `run_policy` versions `[0]`.
  - `test_non_agent_runs_not_applicable`: conftest `registration` (service `fixture-service`) → `{"applicable": False, "versions": []}` even after a version is set.
  - `test_retried_registration_does_not_duplicate_links`: post the same registration twice → one link row.
  - `test_run_policy_unknown_run`: random UUID → `MissingRunError`.
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3:** In `LiveStore.create`, after inserting a new run (not on the idempotent-return path), call `PolicyStore.link_new_run(conn, run_id, payload)`. That method takes lock 730503 before reading the active version, and links only agent runs.
- [ ] **Step 4:** `uv run pytest services/worker/tests -q` → all PASS, 0 skipped (existing live tests unchanged).
- [ ] **Step 5:** Commit `feature: link live agent runs to the active concurrency policy`.

### Task 4: Policy HTTP API

**Files:**
- Create: `services/worker/src/agentwatch_worker/policy_api.py`
- Modify: `services/worker/src/agentwatch_worker/app.py` (include router; extend the `/v1/live/` path checks in the validation and psycopg handlers to `/v1/policies/` too)
- Modify: `services/worker/src/agentwatch_worker/live_api.py` (add `GET /{run_id}/policy`)
- Modify: `services/worker/src/agentwatch_worker/live_limits.py:19` (apply the body limit to `/v1/policies/`)
- Test: `services/worker/tests/test_policy_api.py`

**Interfaces:**
- Produces: `router = APIRouter(prefix="/v1/policies/concurrency", tags=["concurrency policy"])` with
  - `GET /{target}` → `PolicyVersion`
  - `GET /{target}/versions?before=&limit=` (`limit` 1–100, default 50) → `PolicyPage`
  - `POST /{target}/versions` body `PolicyChange` → `PolicyChangeResult`, 201 created / 200 retried
  - Live router: `GET /v1/live/runs/{run_id}/policy` → `RunPolicy`
  - Mapping: `UnsupportedTargetError` → 404 `"Unsupported policy target"`, `ConflictError` → 409, `InvalidRestoreError` → 422, `MissingRunError` → 404.

- [ ] **Step 1: Write failing tests** with `client`:
  - `test_get_baseline`: `GET /v1/policies/concurrency/coding-agent` → 200, `version == 0`.
  - `test_post_set_then_retry`: POST → 201; identical POST → 200 with same version.
  - `test_post_invalid_limits_422_without_echo`: `{"limits": {"test_jobs": 0}}` → 422, and response text does not contain the operator string.
  - `test_post_stale_409` and `test_unsupported_target_404` (`/v1/policies/concurrency/cpu-demo`).
  - `test_restore_missing_version_422`.
  - `test_history_pagination`: three sets → `limit=2` returns versions `[3, 2]` and a `next_before` that yields `[1, 0]`.
  - `test_run_policy_endpoint`: agent run → 200 `applicable True`; unknown UUID → 404.
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3:** Implement the router and wiring.
- [ ] **Step 4:** `uv run pytest services/worker/tests -q` → PASS, 0 skipped.
- [ ] **Step 5:** Commit `feature: serve concurrency policy versions over HTTP`.

### Task 5: Published contracts and SDK clients

**Files:**
- Modify: `services/worker/src/agentwatch_worker/export_live_contracts.py` (also write `api/openapi/policy.json` and `packages/sdk-typescript/src/policy.ts` from the policy router)
- Create (generated): `api/openapi/policy.json`, `packages/sdk-typescript/src/policy.ts`
- Modify: `packages/sdk-typescript/src/index.ts` (`export type * from "./policy";`)
- Modify: `packages/sdk-python/src/agentwatch_sdk/client.py` (methods on `LiveClient`)
- Test: `services/worker/tests/test_policy_contracts.py`

**Interfaces:**
- Produces on `LiveClient`: `get_policy(target: str = "coding-agent") -> PolicyVersion`, `list_policy_versions(target: str = "coding-agent", before: int | None = None, limit: int = 50) -> PolicyPage`, `change_policy(change: PolicyChange, target: str = "coding-agent") -> PolicyChangeResult`, `get_run_policy(run_id: UUID) -> RunPolicy`. Reuse `_request`, so timeouts and response validation match the live methods.

- [ ] **Step 1: Write failing tests**: `test_published_policy_contracts_match_router` (same shape as `test_published_live_contracts_match_router`), and `test_sdk_policy_methods_validate_responses` using the existing monkeypatched-transport style from `test_live_contracts.py`.
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3:** Extend the exporter and run `uv run python -m agentwatch_worker.export_live_contracts`. Add the client methods.
- [ ] **Step 4:** `uv run pytest services/worker/tests -q` → PASS; `pnpm typecheck` → PASS.
- [ ] **Step 5:** Commit `feature: publish concurrency policy contracts and SDK methods`.

### Task 6: Minimal dashboard controls (Meghana refines in O05)

**Files:**
- Create: `apps/web/lib/policy-api-schema.ts` (zod schemas mirroring `PolicyVersion`, `PolicyPage`, `PolicyChangeResult`, `RunPolicy`)
- Create: `apps/web/lib/policy-api.ts` (`createPolicyApi(baseUrl, fetchImpl)` with `active()`, `history(before?)`, `change(body)`, `runPolicy(runId)` returning the existing `LiveApiResult<T>` shape. Add `"conflict"` → 409 to the failure kinds locally. Timeout `LIVE_API_TIMEOUT_MS`.)
- Modify: `apps/web/lib/experiments.ts` (add `policyUrl = () => \`${apiUrl()}/v1/policies/concurrency/coding-agent\``)
- Create: `apps/web/app/api/policies/concurrency/route.ts` (GET active + first history page; POST validated change)
- Create: `apps/web/app/api/live/runs/[runId]/policy/route.ts`
- Create: `apps/web/components/concurrency-controls.tsx` (`"use client"`)
- Modify: `apps/web/lib/dashboard-data.ts` (add `"controls"` to `pages`), `apps/web/components/dashboard.tsx` (nav item "Controls" and render branch), `apps/web/components/live-run.tsx` (one line: "Policy while open: v0 → v1", or "Not an agent run")
- Test: `apps/web/tests/policy-api.test.ts`

**Interfaces:**
- Consumes: Task 4 endpoints; TS types from `packages/sdk-typescript/src/policy.ts` only as reference (the zod schema is the runtime check).

- [ ] **Step 1: Write failing Vitest tests**:
  - `parses baseline and versioned policies` (fixture JSON for v0 and v2).
  - `maps 409 to conflict and 422 to invalid_request` with the service `detail` as the message.
  - `maps 503 and network failure to unavailable` (Review Focus 5). The component's status helper `policyStatusText(result)` returns `"Policy service unavailable"`, never `"No limits"`.
  - `limitsText({})` → `"No limits (baseline)"`; `limitsText({test_jobs: 2})` → `"Inference: no limit · Tests: 2"`.
- [ ] **Step 2:** `pnpm --filter web test` → FAIL.
- [ ] **Step 3:** Implement. The UI uses only the existing `Card`, `Input`, `Button`, `Badge` primitives. It includes: active limits, a form (two optional number inputs, where empty = no limit; operator; reason), Disable, Restore on each history row, and an error text region (`role="alert"`). Submit sends `expected_version` = displayed version and a fresh `crypto.randomUUID()` request id per form submission, and reuses it on retry. Disable the button while pending. Refetch after success or 409.
- [ ] **Step 4:** `pnpm lint && pnpm typecheck && pnpm test && pnpm build` → all PASS. Then check in a browser at 390 px and 1280 px with the worker running: set, retry, stale tab → conflict text, disable, restore.
- [ ] **Step 5:** Commit `feature: add minimal concurrency controls screen`.

### Task 7: Docs, checklist and full verification

**Files:**
- Create: `docs/concurrency-policy.md` (API, semantics, links mean "active while open" not "enforced", idempotency/conflict rules, migration 0004 and recovery: additive tables, drop both to roll back before any O03 data depends on them, handoff to Akshayaa/Meghana/Akshay)
- Modify: `docs/optimization-contract.md:3-5` (replace "No concurrency control exists" with: O02 stores policy, no enforcement until O03)
- Modify: `docs/project-checklist.md` (O02 entry dated 2026-11-03 with test file references; keep enforcement/evaluation open)
- Modify: `.env.example` only if a new variable was introduced (none expected)

- [ ] **Step 1:** Write the docs.
- [ ] **Step 2:** `git fetch origin && git merge origin/main`. Then run the CI baseline: `uv run ruff check .`; `uv run pytest` with `AGENTWATCH_TEST_DATABASE_URL` (expect 0 skipped); `pnpm lint && pnpm typecheck && pnpm test && pnpm build`; `gofmt -d apps/control-plane apps/cli`; `go vet ./apps/control-plane/... ./apps/cli/...`; `go test ./apps/control-plane/... ./apps/cli/...`; `docker compose -f deploy/compose/compose.yaml config --quiet`.
- [ ] **Step 3:** Inspect `git diff origin/main...HEAD` and `git log --oneline origin/main..HEAD`; nothing outside this task's scope.
- [ ] **Step 4:** Commit `docs: document versioned concurrency policy and record O02 status`.

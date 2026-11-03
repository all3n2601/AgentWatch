# Versioned manual concurrency policy (O02)

The worker serves manual concurrency policy at `http://127.0.0.1:8090/v1/policies/concurrency`
and a run's policy history at `/v1/live/runs/{UUID}/policy`. [Policy OpenAPI](../api/openapi/policy.json)
and [live OpenAPI](../api/openapi/live.json) define the shapes. Python models/client methods live
in `agentwatch_sdk` (`policy.py`, `LiveClient.get_policy`, `list_policy_versions`, `change_policy`,
`get_run_policy`); TypeScript types are exported by `@agentwatch/sdk`. The semantics come from the
[optimization contract](optimization-contract.md), which is still *proposed* (`opt-contract-v1`);
O02 implements its policy rules without claiming acceptance.

**O02 stores and serves policy. Nothing enforces it.** Runner admission and cleanup are O03
(Akshayaa). A stored limit, and a link between a run and a version, are not evidence that any
request or test job was limited.

## Model

- Target: only `coding-agent`. Any other path target returns `404 "Unsupported policy target"`.
- Controls: `inference_requests` and `test_jobs`. A present limit is a strict integer from 1 to
  1024. "No limit" is the key being **absent**. `null`, `0`, negatives, floats, booleans, strings
  and sentinels such as `-1` return `422`.
- Version 0 is the synthesized baseline: `action: "baseline"`, empty limits, and null
  `operator`, `reason`, `request_id` and `created_at`. It has no database row. Stored versions
  start at 1 and are never updated or deleted; the versions table is the append-only audit log.
- Every version records `controlled_services` (`inference_requests` → `ollama`, class `inference`;
  `test_jobs` → `agentwatch-agent-runner`, class `cpu_sandbox`) and
  `enforcement_scope: "runner_process"`. These are stored per row, so a later change to the target
  definition never rewrites history. Per-process resource IDs are unknown when a policy is set;
  O03 must record them per admission.

## API

| Request | Result |
| --- | --- |
| `GET /v1/policies/concurrency/{target}` | Active version (`PolicyVersion`) |
| `GET /v1/policies/concurrency/{target}/versions?before=&limit=` | Newest first, `limit` 1–100 (default 50); `next_before` is the cursor for the next older page; version 0 closes the final page |
| `POST /v1/policies/concurrency/{target}/versions` | `PolicyChangeResult`: `201` new version, `200` exact retry (`created: false`) |
| `GET /v1/live/runs/{UUID}/policy` | `RunPolicy`: `{run_id, target, applicable, versions}`; `404` for an unknown run |

The change body is `{request_id, expected_version, action, limits?, restored_from?, operator,
reason}`. `operator` is bounded free text (max 128) and `reason` is required (max 512); no login
system exists, so `operator` is self-reported.

| Action | Body shape | Resulting limits |
| --- | --- | --- |
| `set` | At least one limit; no `restored_from` | The supplied limits |
| `disable` | Neither `limits` nor `restored_from` | Empty (baseline limits) |
| `restore` | `restored_from`; no `limits` | Copied from that earlier version; version 0 is allowed and equals disable |

Restoring creates a new version that references `restored_from`; the earlier row is untouched.
A nonexistent `restored_from` returns `422`.

## Idempotency and conflicts

- `request_id` is unique per target. The same id with an identical body returns the original
  version (`200`, `created: false`) even after later versions exist: the idempotency check runs
  before the revision check, so a faithful retry after an uncertain outcome succeeds. The same id
  with a different body returns `409`.
- `expected_version` must equal the active version, otherwise `409` and nothing is written.
  Refresh the active policy before issuing a new command with a new `request_id`.
- Writers are serialized by a transaction-scoped advisory lock, key `730503`. Live run creation
  uses `730502` and migrations `730501`; the lock order is always 730502 then 730503.
- Errors follow B01: `400` unparseable body, `408`/`413` body time/size bounds (shared with live
  writes), `409` conflict, `422` validation (without echoing the body), `503` database
  unavailable or lock/statement timeout, `500` unexpected storage failure.

## Run links mean "active while open"

When a coding-agent live run is registered, the active version is linked to it in the same
transaction. When a new version is created it is linked to every run still `queued` or `running`.
Terminal runs are never linked to later versions. A run's policy history is its link rows in
version order. This records which policy was *active while the run was open*; it does **not** say
the runner applied it. Per-step application is O03's evidence.

- A run is a coding-agent run if and only if a registered resource has `service_id`
  `agentwatch-agent-runner`. Other runs (overhead, CPU demo, other producers) get no links and
  report `applicable: false` with an empty list, never a fabricated baseline.
- Runs registered before migration `0004` report `applicable: true` with an **empty** versions
  list. Their pre-policy history is unknown, not baseline. The dashboard shows
  "No recorded policy (registered before policy tracking)".
- A missing policy service must never be displayed as "no limits"; the dashboard shows an error
  instead.

## Dashboard

The Controls page (`apps/web/components/concurrency-controls.tsx`) is a deliberately plain
working screen: active limits, a form (two optional limits, operator, reason), disable,
restore-per-row, history, and conflict/validation errors as text. Server-side proxy routes under
`apps/web/app/api/policies/concurrency` and `apps/web/app/api/live/runs/[runId]/policy` validate
with zod; the live run view shows the selected run's policy history for its UUID. The finished
controls UI and comparisons are O05 (Meghana).

## Migration, recovery and verification

Startup applies ordered `0004_concurrency_policy.sql` under the existing migration locking and
transactionality. It is additive: two new tables, `concurrency_policy_versions` and
`live_run_policy_links`. No existing table is rewritten and no seed row is inserted (version 0 is
synthesized). Links carry no foreign key on `(target, version)` for that reason. Back up before
deployment. A failed migration transaction rolls back; repair its cause and restart.

Application rollback: the old application ignores the new tables, so retain them and the migration
record. Dropping them is only appropriate before any consumer depends on them: drop
`live_run_policy_links`, then `concurrency_policy_versions`, and delete the
`0004_concurrency_policy.sql` row from `schema_migrations`. Do not do this once O03 has recorded
admissions that reference policy versions, and never rewrite the applied migration file.

Run `AGENTWATCH_TEST_DATABASE_URL=<dedicated test database> uv run pytest` and `pnpm test`.
Coverage lives in `services/worker/tests/test_policy_contracts.py`, `test_policy_store.py` and
`test_policy_api.py` (isolated schemas: invalid limits, unsupported target, conflicts, idempotent
retries, disable/restore, concurrent writers, per-run links, terminal runs, populated-0003
upgrade, rollback) and `apps/web/tests/policy-api.test.ts`. Regenerate contracts and types with
`uv run python -m agentwatch_worker.export_live_contracts`; drift is regression-tested.

## Handoff

- **Akshayaa (O03):** consume the active policy (`GET .../{target}` or `LiveClient.get_policy`)
  for runner-process admission limits; record the applied version and per-process resource IDs on
  your own step evidence. Lowering a limit affects future admissions only.
- **Meghana (O05):** restyle the Controls screen and build comparisons on `PolicyVersion` /
  `RunPolicy`; keep the "active while open, not enforced" wording and the unknown-history state.
- **Akshay (O04):** versions are immutable, carry `request_id`, `created_at` and
  `controlled_services`, and link to runs by UUID; use them for manifest policy lineage.

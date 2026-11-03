# O02 Versioned manual concurrency controls — design

Status: approved in chat 2026-11-03 (Allen). Branch `feature/concurrency-policy`.

## Intent

An operator sets the two manual concurrency limits from the dashboard before an
optimization experiment, and can return to the baseline or re-apply an earlier setting.
Every change is an immutable, audited version, and each live run's UUID shows which
policy versions were active while it ran. O03 (Akshayaa) enforces the limits in the runner;
O05 (Meghana) owns the finished controls UI. O02 supplies the API, storage, SDK types and a
plain working dashboard screen she will restyle.

Semantics come from the merged `opt-contract-v1`
([optimization-contract.md](../../optimization-contract.md#policy-semantics) and
`pipelines/model/optimization-contract.toml` `[policy]`). That contract is still
*proposed*; O02 implements its policy rules without claiming acceptance.

## Goals

- Two independent limits, `inference_requests` and `test_jobs` (TOML `controls`). A present
  limit is a strict integer `>= 1` (TOML `min_limit`); "no limit" is the key being absent.
  Zero, negative, null, booleans, floats, strings and sentinels such as `-1` are rejected (422).
- One supported target: `coding-agent`, the only workload with model requests and test jobs.
  Any other target is rejected as unsupported (404 with a "unsupported target" message).
- Baseline default: before any change the active policy is version 0, both limits absent,
  `action = "baseline"`. Version 0 is synthesized, not stored, so existing databases need no
  seed row.
- Every write creates the next version with `expected_version` optimistic concurrency:
  stale → 409 and nothing written.
- Actions: `set` (new limits), `disable` (back to baseline limits), `restore` (copy the limits
  of an earlier version `restored_from`; restoring version 0 is allowed and equals disable).
- Each version records: target, version, action, limits, operator, reason, `restored_from`,
  `request_id`, server `created_at` (timestamptz). Operator and reason are required, bounded
  free text (no login system exists).
- Idempotent retries: `request_id` (UUID) is unique per target. Same request id with an
  identical body returns the original version (200, `created: false`); with a different body
  → 409. New version → 201.
- Audit: the versions table is the append-only audit log; no row is ever updated or deleted.
- Run linkage (server-recorded, chosen 2026-11-03): when a live run is registered, the active
  version is linked to that run in the same transaction. When a new version is created, it
  is linked to every live run not yet in a terminal state (`queued`/`running`). A run's
  policy history is its link rows in version order. Links mean "active while the run was
  open", **not** "enforced by the runner"; docs and API descriptions say so.
- Thin dashboard screen: active limits, a form (two optional limits, operator, reason),
  disable, restore-per-row, history table, conflict/validation errors as text, and the
  selected live run's policy history on the existing live run view.

## Builder's calls (settled 2026-11-03 after the builder check)

- **Which runs belong to `coding-agent`:** a live run whose registration includes a resource
  with `service_id` `agentwatch-agent-runner`. That is how the agent runner registers its tool
  host (`services/agent/src/agentwatch_agent/runner.py` `_resources`). It follows the same
  precedent as overhead runs, which are recognised by `service_id` alone. Runs without that resource
  (overhead, CPU demo, other producers) get no policy links, and their policy endpoint returns
  `applicable: false` with an empty list, never a fabricated baseline.
- **Service identity per version:** each stored version records `controlled_services`, fixed by
  the target definition: `inference_requests` → `ollama` (resource class `inference`),
  `test_jobs` → `agentwatch-agent-runner` (resource class `cpu_sandbox`), plus
  `enforcement_scope = "runner_process"` from the TOML. These are stored in the row, so a later
  change to the target definition never rewrites history. Resource IDs are per-runner-process
  and unknown when a policy is set, so they are not recorded; O03 records them per admission.
- **Step correlation:** stays out of scope. O02 links versions to runs. O03 records which
  version each admission applied, on its own step evidence. The run-level link says "active
  while open", which is all O02 can know.

## Non-goals

- Enforcement, admission queues, queue bounds, permits (O03).
- Polished controls UI, comparisons (O05). Experiment manifests and policy hashes (O04).
- Cross-runner limits, adaptive control (O07), authentication.
- Changing the merged B01 run-registration contract (`RunCreate`).

## Anti-goals

- A limit of 0 or a sentinel silently meaning "unlimited".
- Overwriting or deleting history to "restore".
- A run's history implying the runner applied a limit.
- Showing a missing/unavailable policy service as "no limits".

## Constraints

- Follow the B01 patterns: Pydantic models in `packages/sdk-python/src/agentwatch_sdk/`,
  FastAPI router in `services/worker`, store class over `RunStore` connections, exported
  OpenAPI (`api/openapi/`), generated TypeScript types in `packages/sdk-typescript`,
  zod-validated server-side proxy routes in `apps/web/app/api/`.
- Ordered migration `0004_concurrency_policy.sql`, additive only.
- Serialize version creation per target with a transaction-scoped advisory lock (new key).
- PostgreSQL integration tests with isolated schemas, `AGENTWATCH_TEST_DATABASE_URL`, no skips.
- Errors keep the B01 shapes: 409 conflict, 422 validation (without echoing bodies), 503
  storage unavailable.

## API (builder may adjust paths if the pattern suggests better)

- `GET  /v1/policies/concurrency/{target}` → active version.
- `GET  /v1/policies/concurrency/{target}/versions?before&limit` → history, newest first.
- `POST /v1/policies/concurrency/{target}/versions` → body
  `{request_id, expected_version, action, limits?, restored_from?, operator, reason}`.
- `GET  /v1/live/runs/{run_id}/policy` → `{run_id, versions: [...]}`; 404 for unknown run.

## Acceptance tests (from the board card)

Invalid limits, unsupported target, revision conflict, idempotent retry (same and different
body), disable, restore (including version 0 and nonexistent version → 422), concurrent
writers, per-run history for two runs registered under different versions, terminal runs not
linked to later versions, migration upgrade over a populated 0003 schema, rollback on failure.

## Left to the builder

File/module names, pagination details, exact field length bounds, dashboard layout,
where the new section sits in navigation, docs page name (`docs/concurrency-policy.md`).

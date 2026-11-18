# B03 design: serve versioned rules and model predictions

Task B03 (Allen), milestone M4. Handoff: prediction API plus model and feature versions to
Meghana (F03, verdict evidence and rule comparison). Status: approved by Allen, 2026-11-17 (sampling interval: option A).

## Intent

When a live step reaches a terminal state, AgentWatch stores a versioned **rules-v1 verdict** and,
when a model is servable, a **calibrated model prediction** with the evidence that produced it,
and lets the dashboard read both by polling. A missing, broken or unapproved model never removes
the rules verdict and is reported explicitly, never as a quiet absence or as `none`.

Why: the M4 gate is "model and rules evaluated on identical held-out evidence; uncertainty is
explicit". F03 cannot show a rule/model comparison until a server produces both from the same
`step-features-v1` row.

## Goals

- One `step-features-v1` row per terminal step feeds both scorers, so they compare on identical
  inputs.
- Serve the pair described in [model serving contract](../../model-serving-contract.md): 14
  ordered inputs (NaN for missing, never 0), calibrated probabilities, confidence, abstention,
  and the clipped excess-latency estimate.
- Refuse to load on any contract mismatch (artifact hashes, versions, `status`, `fitted`,
  probability-column agreement across `preprocessing.json` and `calibration.json`, regressor
  hashes). Report the reason through the API.
- Serve a pair with `blocks_promotion: true` only behind an explicit not-promoted flag, and label
  every prediction with it.
- Measure step completion to visible verdict latency against the declared
  `max_attribution_latency_p95_seconds = 15.0` (evaluation contract v3) and record the result.
- Tests cover missing features, load failure, tampered artifacts, abstention, rule fallback,
  retries/duplicates and the latency measurement itself.

## Non-goals

- Promotion, rollback or registry loading (M05). The server reads local artifact folders only.
- Clearing the four `not_evaluated` gates, or any claim that the pair is good: the latency
  measurement reports one gate input and nothing else.
- SSE or WebSocket delivery. The live transport is cursor polling and stays so.
- Retraining, recalibration, new features, GPU or live-sampling changes.
- Dashboard UI (F03).

## Anti-goals (failure even if tests pass)

- A missing reading becoming 0, or an abstention folded into `none`.
- The model's failure rolling back or delaying ingestion of step evidence.
- A second implementation of rules, calibration or feature encoding that can drift from the
  pipeline's.
- Silently serving an unpromoted pair, or silently swapping models.

## Constraints

- `agentwatch-data` depends on `agentwatch-worker` (`extract.py` imports worker contracts), so the
  worker cannot depend on `agentwatch-model` or `agentwatch-data`: the dependency would be
  circular. Reusing the pipeline code requires moving the serving-relevant parts out of it.
- Ingestion is atomic and idempotent; terminal steps are immutable; one `live_records` table is
  the evidence. Add an ordered migration (`0005_...`), never edit applied ones.
- Run PostgreSQL integration tests with `AGENTWATCH_TEST_DATABASE_URL` pointing at a dedicated
  database; skipped database tests are incomplete verification.
- API changes update `api/openapi/live.json`, the SDK types, TypeScript contracts, tests and docs
  together (`export_live_contracts`).
- Preserve existing identities: `rules-v1+<hash>` and `calibration-v1+<hash>` must not change
  because code moved.

## Approaches considered

How the worker gets the model code:

| | Approach | Trade-off |
|---|---|---|
| A | Worker depends on `agentwatch-model` | Impossible: circular through `agentwatch-data`. |
| C | Make training dependencies optional extras | Same circularity; does not help. |
| **B** | **Extract a lean `packages/attribution`** | A refactor of pipeline code, but the only structure without duplication or cycles. **Chosen** (the user asked for the best long-term structure and accepts the refactor cost). |

When predictions are computed:

| | Approach | Trade-off |
|---|---|---|
| 1 | Inside the ingest transaction | Atomic, but a model failure or slow predict would block or roll back evidence. Rejected. |
| **2** | **After the ingest commit, in its own transaction, plus reconcile on read** | Evidence is never put at risk; reconcile covers a crash between the two. **Chosen.** |
| 3 | A separate background worker | More moving parts and more latency for no current need. |

## Design

### 1. `packages/attribution` (`agentwatch-attribution`, import `agentwatch_attribution`)

Depends on `numpy` and `lightgbm` only (no data, worker, mlflow, matplotlib).

- `hashing`: `canonical_sha256`, `file_sha256`. `agentwatch_data.extract` and `.files` import them
  from here so there is one implementation.
- `classes`: `RESOURCE_CLASSES`, `NONE`, `is_number`. `agentwatch_model.metrics` imports them.
- `rules` (+ `rules.toml`): moved from `agentwatch_model`. Same content, so the same
  `rules-v1+hash`.
- `inputs`: `MODEL_INPUTS` (the 14 ordered names), `encode`, `feature_matrix`. A test in
  `pipelines/model` asserts `MODEL_INPUTS == agentwatch_data.schema.feature_columns()` so schema
  drift fails loudly.
- `calibration`: `scale` and `PROBABILITY_FLOOR`. Fitting stays in `agentwatch_model`, which
  imports `scale`.
- `served_model`: loads a candidates folder and calibration folder under the contract's checks
  (regressor hash and cross-artifact probability-column check included), exposes the identity
  (versions, file hashes, temperature, threshold, `blocks_promotion`), and `predict(rows)`
  returning raw and calibrated probabilities, confidence, abstained flag, top class (only when not
  abstaining) and the clipped excess fraction. Typed errors name the failed check.

`agentwatch_model` keeps its public names by importing from the new package; the pipeline's
existing tests must pass unchanged, which proves the move changed no behavior. The held-out
loader (`held_out_model`) is not rewritten.

### 2. Worker

- Config (env, documented in `.env.example`): model candidates dir, calibration dir, and an
  explicit allow-unpromoted flag. Unset means model status `not_configured`.
- Model status is one of `ready_not_promoted`, `ready_promoted`, `not_configured`,
  `not_promoted_refused`, `load_failed` (with the failed check), and is part of every response.
- A step's inputs are the `step-features-v1` row from `live_feature_rows`. The three inputs that
  are not live features (`workload_iterations`, `tests_per_step`, `worker_capacity`) are passed as
  NaN and listed per prediction as missing inputs, as on the replay rows the pair was trained on.
- Migration `0005`: append-only table of per-step attributions: cursor sequence, run, step,
  feature version, the stored feature row and its lineage ids, the rules verdict, the model
  outcome (prediction, or the unavailability reason), serving identity, the terminal step's
  applied time and the time the row was committed. Idempotent on retry; a conflicting row for the
  same key is rejected.
- Computation runs after the ingest commit for newly terminal steps, in its own transaction. A
  model error is recorded as that step's model outcome; the rules verdict is still stored. Reading
  reconciles terminal steps that lack a row.
- API under `/v1/live/runs/{run_id}`: cursor-polled attributions (same `after`/`limit` convention
  as the run snapshot) and a model-status endpoint. UUID-specific; a missing run is 404, a run
  with no terminal steps is an empty page, distinct from model unavailability.

### 3. Prediction semantics returned to the dashboard

Per step: feature version and row, missing inputs, telemetry status; `rules` (version, verdict,
no confidence, never abstains); `model` (status, training and calibration versions,
not-promoted label, calibrated probabilities, confidence, threshold, abstained, class or null,
excess-latency estimate labelled an estimate). No causal wording; the wording rules in the serving
contract apply.

### 4. Latency evidence

Record per step the terminal step's applied time and the row's commit time. A measurement
command drives a declared synthetic workload through the real API, polls as the dashboard would,
and reports step completion to visible verdict for n steps: p50, p95, max, polling interval, the
model status used, and machine. It reports against the declared 15 s p95 but does not change any
promotion state. The result and its limits (synthetic steps, one machine) go in the checklist and
a docs page with the date.

## Verification

Python: `uv run ruff check .`, `uv run pytest` with PostgreSQL via
`AGENTWATCH_TEST_DATABASE_URL`, none skipped. The pipeline's existing tests pass unchanged after
the extraction. New tests: missing features, NaN versus zero, load failure and each tampered
artifact, unpromoted refusal and flag, abstention, rule fallback, duplicate delivery and rollback,
UUID isolation, the OpenAPI export staying in sync. JS checks if contracts or types change.

## Calls made after the builder check (author's calls, open to correction)

- **Terminal states:** every terminal step gets a stored row with its rules verdict and step
  state. The model scores only `completed` steps, the only kind it trained on; `failed` and
  `cancelled` steps carry model outcome `not_scored_step_state`.
- **Frozen rows:** a step's feature row is computed once, after its terminal event is applied,
  and is not overwritten. The sources (event and sample ids) are stored with it. A later
  reconcile that would produce a different row (a late sample inside the window) is reported as
  a flag on the step; it never rewrites the stored row. No grace wait, so latency is not padded.
- **Latency clock:** the measurement command uses one monotonic clock: from just before it sends
  the terminal step's records to the poll response that contains the verdict. That figure,
  including the poll interval, is what is compared with the 15 s p95. Server-side compute time
  (both timestamps from the database clock) is reported separately. Client wall clock is never
  compared with server time.
- **Contract:** "eval-contract-v3" is `pipelines/model/evaluation-contract.toml`; the latency
  statistic is unchanged since v1.
- **Rules move:** `rules.py` and `rules.toml` are moved (git mv), not copied; the pipeline imports
  them from `agentwatch_attribution`. The hash covers parsed content, not the path, so
  `rules-v1+<hash>` is unchanged; a test pins it.
- **Test artifacts:** tests build a tiny synthetic pair in a temp directory and label it
  synthetic. A parity test in `pipelines/model` checks that the served model reproduces
  `held_out_model.frozen_scores` on the same synthetic pair. The latency run uses the real local
  pair if present and says which it used.
- **LightGBM import** is lazy, so a missing OpenMP library becomes model status `load_failed`
  with the reason, never a worker that fails to start.
- **Sampling interval (decided by Allen, 2026-11-17: option A):** run registration gains an
  optional `sampling_interval_seconds` (positive, finite), a backward-compatible change to the
  SDK models, `api/openapi/live.json`, the TypeScript contracts, docs/live-ingestion.md and the
  agent runner, which sends its configured value. Runs registered without it keep
  `telemetry_coverage` null (NaN to the model, listed as a missing input); the worker never
  guesses a constant. Existing runs need no migration of their registration JSON.

## Left to the builder

Table and index details, endpoint names and response field names within the rules above,
reconcile mechanics, the measurement command's CLI, and exactly which tests move.

## Coordination

No open PR overlaps `services/worker` or `api/`. PRs #92 to #95 touch `services/agent`, the
Makefile and docs. Rohit owns `pipelines/model` (M03/M04): the extraction changes only imports
there and will be called out in the PR.

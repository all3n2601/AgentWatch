# 0014: Serve attributions from a shared `packages/attribution` after the ingest commit

Date: November 18, 2026. Status: proposed in the `feature/serve-attribution-predictions` PR (B03).
Serves the pair described in the [model serving contract](../model-serving-contract.md);
operations and the latency measurement are in [attribution serving](../attribution-serving.md).

B03 must serve, for each terminal live step, the `rules-v1` verdict and the M03/M04 model's
calibrated prediction, computed exactly as in training: the same rules file and identity
(`rules-v1+<hash>`), the same 14 ordered inputs with NaN for missing, and the same temperature
scaling. That code lived in `agentwatch_model`, which depends on `agentwatch-data`, which depends
on `agentwatch-worker` (dataset building reads the worker's live API and history). The worker
could not import it without a dependency cycle, and copying the code into the worker would let
serving drift from training without any test noticing.

**Decision.**

- A new workspace package, `packages/attribution` (`agentwatch-attribution`, import
  `agentwatch_attribution`), depending only on `numpy` and `lightgbm`. It holds the shared
  hashing helpers, class constants, `rules-v1` (`rules.py` and `rules.toml`, moved with
  `git mv`; the identity hashes parsed content, so `rules-v1+<hash>` is unchanged and a test pins
  it), model input encoding (`MODEL_INPUTS`, `encode`, `feature_matrix`), temperature `scale`,
  and a loader (`served_model`) that refuses a pair on any serving-contract check (artifact
  hashes including the regressor's, versions, `status`, `fitted`, probability columns across
  both files) and names the failed check. `agentwatch_model` and `agentwatch_data` import these
  names; their existing tests pass unchanged, and a pipeline test asserts `MODEL_INPUTS` equals
  the dataset schema's feature columns.
- The worker computes attributions after the ingest commit, never inside it: a batch holding a
  terminal step event, and a run closing, schedule a background pass after the response; each
  attributions read first attributes terminal steps that lack a row. Rows go to an append-only
  table (migration `0005_step_attributions.sql`), keyed by run, step, feature version and serving
  key (rules version plus model state, reason and identity); a stored record is never rewritten.
- Transport stays cursor polling (`GET /v1/live/runs/{UUID}/attributions`, and
  `GET /v1/live/attribution/model`), like the run snapshot.
- A pair whose calibration has `blocks_promotion: true` is served only when
  `AGENTWATCH_ALLOW_UNPROMOTED_MODEL` is `1` or `true`, and every prediction is labelled not
  promoted; otherwise it is refused and reported as `not_promoted_refused`.

**Alternatives rejected.**

| Alternative | Why not |
| --- | --- |
| Worker depends on `agentwatch-model` | Circular: `agentwatch-model` → `agentwatch-data` → `agentwatch-worker` |
| Make training dependencies optional extras | The package dependency is still circular; extras change what is installed, not the graph |
| Score inside the ingest transaction | A model error or slow prediction would block or roll back acknowledged evidence |
| A separate background worker process | More moving parts and more latency, for no current need |

**Consequences.** Training and serving share one implementation of rules, encoding and scaling,
so a change to either is a change to both and is caught by both test suites. The worker's
environment now needs LightGBM's OpenMP runtime (libomp on macOS, libgomp on Linux) to serve a
pair; LightGBM is imported lazily, so a missing library becomes model state `load_failed`
(`lightgbm_unavailable`) rather than a worker that fails to start. A
failed background pass delays a row until the next read; it never fails the ingest. Each pass
recomputes the run's feature rows, so its cost grows with run length (measured only up to 50
steps). Promotion, rollback and registry loading stay with M05; the worker reads local folders
only. The worker runs on the host (`make dev-worker`); Compose has no worker service, so no
deployment passes the model variables yet (B04).

**Recorded exception: `canonical_sha256` exists twice.** `agentwatch_data.extract.canonical_sha256`
is intentionally not replaced by an import from `agentwatch_attribution.hashing`.
`replay_labels.logic_source()` fingerprints the source text of `extract.py` (with
`replay_labels.py`, `replay_evidence.py` and `labels.py`), so any edit to that file changes the
pinned `labels-v2+d803a9c6` identity recorded in the evaluation contracts, the
[data card](../data-card.md) and [ADR 0009](0009-combined-dataset-release-and-split-v2.md), and
with it every release and split built on it. `agentwatch_attribution.hashing.canonical_sha256`
duplicates the same three lines, and `packages/attribution/tests/test_hashing.py` pins that both
return equal output. Removing the duplicate is a deliberate labels-v2 version bump, not a
refactor.

**Migration and recovery.** `0005_step_attributions.sql` is additive: it creates
`live_step_attributions` and its indexes under the existing migration lock and transaction, and
rewrites no existing run, record or sample. A failed migration rolls back; fix the cause and
restart. To roll the application back, leave the table and its migration record in place: the
older worker ignores it. Never drop the table or edit the applied migration to recover.

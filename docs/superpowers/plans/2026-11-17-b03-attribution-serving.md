# B03 Attribution Serving Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Store a versioned rules-v1 verdict and a calibrated M03/M04 model prediction for every terminal live step, serve both by cursor polling, and measure step completion to visible verdict latency.

**Architecture:** Extract the serving-relevant pipeline code (rules, input encoding, temperature scaling, hashing, a contract-checking model loader) into a new dependency-light workspace package `packages/attribution`, which `agentwatch-model`, `agentwatch-data` and `agentwatch-worker` all import. The worker computes attributions after the ingest commit (FastAPI background task) from the existing `live_feature_rows`, stores them append-only in a new table, and reconciles missing rows on read. Model problems become an explicit status, never a failed ingest.

**Tech Stack:** Python 3.13, FastAPI, psycopg 3, PostgreSQL, LightGBM core API, numpy, pydantic, uv workspace, pytest, Ruff.

**Spec:** `docs/superpowers/specs/2026-11-17-b03-attribution-serving-design.md` (read it first; it also holds the author's calls). Serving contract: `docs/model-serving-contract.md`.

## Global Constraints

- Work only in the worktree `/Users/all3n2601/agentwatch-b03`, branch `feature/serve-attribution-predictions` (the main checkout has unrelated uncommitted work; never touch it). Its path has no spaces, which the Makefile data targets need.
- Python `>=3.13,<3.15`; Ruff `line-length = 100`; no `any`-style shortcuts: type annotations on every new function, explicit data contracts at boundaries.
- Do not add dependencies beyond `numpy`, `lightgbm` (the same version ranges `pipelines/model` pins) for the new package; regenerate `uv.lock` with `uv lock`, never hand-edit it.
- A missing reading is NaN to the model and never 0; abstention is its own state and never `none`; the excess fraction is an estimate, clipped to `preprocessing.json`'s `excess_fraction_clip` (`[0, 1]`).
- Abstain when `confidence < abstention_threshold`. Calibration: clip each probability to `[1e-12, 1]`, log, divide by `temperature`, softmax. Predict with all trees.
- Identities must not change because code moved: `rules-v1+<hash>` and `calibration-v1+<hash>`.
- `max_attribution_latency_p95_seconds = 15.0` (`pipelines/model/evaluation-contract.toml`). Report against it; never wire it to promotion (M05).
- Serve a pair with `blocks_promotion: true` only when `AGENTWATCH_ALLOW_UNPROMOTED_MODEL` is `1` or `true`, and label every prediction not promoted.
- Existing tests in `pipelines/model/tests` must pass unchanged except import-path edits in the tests that import moved modules directly.
- PostgreSQL tests need `AGENTWATCH_TEST_DATABASE_URL` pointing at a dedicated database (`postgresql://agentwatch:agentwatch_test@localhost:5432/agentwatch_test` in CI); a skipped database test is incomplete verification. Never reset the dev database.
- Commit titles `<type>: <specific change>`; no Claude co-author trailers or other attribution lines in commits or the PR (the repository owner asked for this); push and open the PR only in Task 12.

## Review Focus

Failure modes the spec implies that no requirement lists; each has a test in the named task.

1. A pre-upgrade run registration (stored without `sampling_interval_seconds`) must stay an idempotent retry, not a 409 (Task 6).
2. A step that never executed (null window, `missing` telemetry) still gets a rules verdict and a model outcome, with every absent input listed (Task 8).
3. A confidence exactly equal to the threshold is not an abstention (Task 5).
4. Two concurrent attribute passes for one step leave exactly one row (Task 9).
5. Changing or deleting model files after startup does not change the served identity or status until restart; no silent model swap (Task 7).

---

## File Structure

| Path | Responsibility |
|---|---|
| `packages/attribution/` (new) | `hashing.py`, `classes.py`, `rules.py`+`rules.toml`, `inputs.py`, `calibration.py`, `served_model.py`; tests in `packages/attribution/tests/` |
| `pipelines/model/src/agentwatch_model/{metrics,features,calibration,rules}.py` | import from the new package; training and fitting code stays |
| `pipelines/data/src/agentwatch_data/{extract,files}.py` | re-export hashing from the new package |
| `packages/sdk-python/src/agentwatch_sdk/live.py` | `RunCreate.sampling_interval_seconds` |
| `packages/sdk-python/src/agentwatch_sdk/attribution.py` (new) | pydantic response models for the attribution API |
| `services/worker/src/agentwatch_worker/attribution.py` (new) | pure `build_attribution` |
| `.../attribution_model.py` (new) | env config and `ModelService` (status plus loaded model) |
| `.../attribution_store.py` (new) | `AttributionStore`: compute, persist, reconcile, page |
| `.../attribution_api.py` (new) | routes, background trigger |
| `.../attribution_latency.py` (new) | latency measurement command |
| `.../migrations/0005_step_attributions.sql` (new) | table |
| `docs/attribution-serving.md`, `docs/adr/0014-attribution-serving-package.md` (new) | operation, results, decision |

---

### Task 1: Package scaffold, hashing, classes

**Files:**
- Create: `packages/attribution/pyproject.toml`, `packages/attribution/src/agentwatch_attribution/{__init__,hashing,classes}.py`, `packages/attribution/tests/test_hashing.py`
- Modify: root `pyproject.toml` (workspace member and `testpaths`), `pipelines/data/pyproject.toml` and `pipelines/model/pyproject.toml` (workspace dependency; the worker gains it in Task 7), `pipelines/data/src/agentwatch_data/extract.py` (`canonical_sha256`), `pipelines/data/src/agentwatch_data/files.py` (`file_sha256`), `pipelines/data/src/agentwatch_data/schema.py` (`RESOURCE_CLASSES`), `pipelines/model/src/agentwatch_model/metrics.py` (`CLASSES`, `NONE`, `is_number`), `uv.lock`

**Interfaces:**
- Produces: `agentwatch_attribution.hashing.canonical_sha256(value: object) -> str` (identical to today's `agentwatch_data.extract.canonical_sha256`: sorted keys, `(",", ":")` separators, `allow_nan=False`, SHA-256 hex); `file_sha256(path: Path) -> str` (1 MiB blocks); `agentwatch_attribution.classes.RESOURCE_CLASSES: tuple[str, ...] == ("none", "cpu_sandbox", "inference", "retrieval")`, `NONE: str == "none"`, `is_number(value: object) -> TypeGuard[int | float]` (finite, not bool).
- The data package keeps its public names by importing these; `agentwatch_data.schema.RESOURCE_CLASSES` is defined as the imported tuple (one definition).

- [ ] **Step 1: Write the failing tests** in `test_hashing.py`: `test_canonical_hash_is_key_order_independent`, `test_canonical_hash_refuses_nan` (raises `ValueError`), `test_file_hash_matches_hashlib` (tmp file larger than 1 MiB), `test_is_number_rejects_bool_nan_inf_and_strings`, and `test_data_package_reexports_the_same_functions` asserting `agentwatch_data.extract.canonical_sha256 is agentwatch_attribution.hashing.canonical_sha256` (and the same for `file_sha256`, `RESOURCE_CLASSES`).
- [ ] **Step 2: Run** `uv run pytest packages/attribution/tests -v`. Expected: collection error, package missing.
- [ ] **Step 3: Create the package** (hatchling build like `packages/features/pyproject.toml`; dependencies `numpy>=2,<3` and `lightgbm>=4.6,<5` with the same OpenMP comment as `pipelines/model/pyproject.toml`; no workspace dependencies). Move the three definitions in, replace the originals by imports, add the workspace member, `testpaths` entry and `[tool.uv.sources]` entries, then `uv lock`.
- [ ] **Step 4: Run** `uv run pytest packages/attribution/tests pipelines/data/tests pipelines/model/tests -q` and `uv run ruff check .`. Expected: all pass, ruff clean.
- [ ] **Step 5: Commit** `refactor: add the agentwatch-attribution package with shared hashing and class constants`.

### Task 2: Move rules-v1

**Files:**
- Move (git mv): `pipelines/model/src/agentwatch_model/rules.py` and `rules.toml` to `packages/attribution/src/agentwatch_attribution/`; `pipelines/model/tests/test_rules.py` and `rule_fixtures.py` usages to `packages/attribution/tests/` where they only exercise rules
- Modify: every `agentwatch_model.rules` importer (`grep -rn "agentwatch_model.rules\|from agentwatch_model import rules" pipelines services`)
- Test: `packages/attribution/tests/test_rules.py`

**Interfaces:**
- Consumes: Task 1 hashing and classes.
- Produces: `agentwatch_attribution.rules` with the same public names as today (`Rules`, `RulesError`, `load_rules(path: Path = DEFAULT_RULES) -> Rules`, `DEFAULT_RULES`, `VERSION`, `throttled_fraction`). `agentwatch_model.rules` is deleted (no shim); `agentwatch_model` code imports from `agentwatch_attribution.rules`.

- [ ] **Step 1: Capture the current identity:** run `uv run python -c "from agentwatch_model.rules import load_rules; print(load_rules().version)"` and record the value `rules-v1+<hash>`.
- [ ] **Step 2: Write the failing test** `test_packaged_rules_keep_their_published_identity` asserting `load_rules().version == "<recorded value>"`, importing from `agentwatch_attribution.rules`. Run it: FAIL (module missing).
- [ ] **Step 3: `git mv` the two files and the rules-only tests; fix imports.** `Rules.predict` and hashing behavior must not change.
- [ ] **Step 4: Run** `uv run pytest packages/attribution pipelines/model/tests pipelines/data/tests -q` (the model tests that score rules, such as `test_rule_baseline.py`, are the proof the move changed nothing). Expected: all pass.
- [ ] **Step 5: Commit** `refactor: move rules-v1 into the attribution package`.

### Task 3: Model inputs and temperature scaling

**Files:**
- Create: `packages/attribution/src/agentwatch_attribution/{inputs,calibration}.py`, `packages/attribution/tests/test_inputs.py`, `packages/attribution/tests/test_calibration.py`
- Modify: `pipelines/model/src/agentwatch_model/features.py` (import `FeatureError`, `encode`, `feature_matrix`; `FEATURES = MODEL_INPUTS`), `pipelines/model/src/agentwatch_model/calibration.py` (import `scale`, `PROBABILITY_FLOOR`)
- Test: `pipelines/model/tests/test_features.py` (add drift test)

**Interfaces:**
- Produces: `inputs.MODEL_INPUTS: tuple[str, ...]` (the 14 names in the order of the serving contract's table); `inputs.FeatureError(ValueError)`; `inputs.encode(value: object, name: str, index: int) -> float` (None to NaN, bool to 1.0/0.0, finite number to float, else `FeatureError`); `inputs.feature_matrix(rows: Sequence[Mapping[str, object]]) -> np.ndarray` (float64, columns in `MODEL_INPUTS` order, `FeatureError` naming a missing column); `calibration.scale(probabilities: np.ndarray, temperature: float) -> np.ndarray` and `PROBABILITY_FLOOR = 1e-12`.

- [ ] **Step 1: Write the failing tests:** `test_none_is_nan_and_zero_stays_zero` (row with `cpu_seconds=None` and `queue_seconds=0.0`: NaN vs 0.0), `test_booleans_encode_as_one_and_zero`, `test_missing_column_is_refused_by_name`, `test_non_finite_value_is_refused`, `test_column_order_is_the_contract_order` (compare with the literal 14-name tuple), `test_scale_keeps_the_top_class_and_sums_to_one`, `test_scale_floor_survives_zero_probability`. In `pipelines/model/tests/test_features.py` add `test_model_inputs_equal_the_dataset_schema_feature_columns` asserting `MODEL_INPUTS == tuple(agentwatch_data.schema.feature_columns())`.
- [ ] **Step 2: Run** them. Expected: FAIL.
- [ ] **Step 3: Move `encode`, `feature_matrix`, `FeatureError` and `scale` verbatim** (behavior unchanged) and re-point the model package.
- [ ] **Step 4: Run** `uv run pytest packages/attribution pipelines/model/tests -q`. Expected: pass; the existing candidate and calibration tests are the proof that training numbers did not move.
- [ ] **Step 5: Commit** `refactor: share model input encoding and temperature scaling with serving`.

### Task 4: Verify and load a model pair

**Files:**
- Create: `packages/attribution/src/agentwatch_attribution/served_model.py` (loading half), `packages/attribution/tests/pair_fixtures.py`, `packages/attribution/tests/test_served_model_load.py`

**Interfaces:**
- Consumes: `hashing.file_sha256`, `inputs.MODEL_INPUTS`.
- Produces:
  - `CHECK_CODES: tuple[str, ...] = ("artifact_unreadable", "hash_mismatch", "version_mismatch", "not_evaluated", "not_fitted", "columns_mismatch", "features_mismatch", "lightgbm_unavailable", "model_load_failed")`
  - `class ServedModelError(ValueError)` with attribute `check: str` (one of `CHECK_CODES`) and a message naming the file and field
  - `@dataclass(frozen=True) class ModelIdentity` with `training_version: str`, `calibration_version: str`, `classifier_sha256: str`, `regressor_sha256: str`, `preprocessing_sha256: str`, `calibration_sha256: str`, `temperature: float`, `abstention_threshold: float`, `class_order: tuple[str, ...]`, `blocks_promotion: bool`
  - `class ServedModel` with `identity: ModelIdentity`
  - `load_served_model(candidates_dir: Path, calibration_dir: Path) -> ServedModel`
- `pair_fixtures.py` provides `write_synthetic_pair(root: Path) -> tuple[Path, Path]`: trains a tiny LightGBM classifier (4 classes) and regressor on made-up rows over `MODEL_INPUTS`, saves `classifier.txt`, `regressor.txt`, writes `preprocessing.json` and `calibration.json` in the real field layout (`model_files.<name>.sha256`, `probability_columns`, `class_order`, `training_version`, `status: "evaluated"`, `fitted: true`, `temperature`, `abstention_threshold`, `version`, `blocks_promotion`, `excess_fraction_clip`). Its module docstring states it is synthetic, as `pipelines/model/tests/frozen_model_fixtures.py` does. Read that file and `train_candidates.py` for the exact layout; do not import `agentwatch_model`.

The checks are exactly the serving contract's: both model files hash to `model_files` SHA-256 in both JSON files; `training_version` equal across the two; `status` is `evaluated` in both; `fitted` is true; `preprocessing.json` `features` equals `MODEL_INPUTS`; `probability_columns` equals `class_order` and equals `calibration.json`'s. Import `lightgbm` inside the function; an `ImportError` or `OSError` becomes `lightgbm_unavailable`.

- [ ] **Step 1: Write the failing tests** (one per check code, each mutating the synthetic pair): `test_good_pair_loads_and_reports_identity`, `test_missing_file_is_artifact_unreadable`, `test_tampered_classifier_is_hash_mismatch`, `test_tampered_regressor_is_hash_mismatch`, `test_regressor_hash_must_match_in_calibration_json_too`, `test_training_version_disagreement_is_version_mismatch`, `test_status_not_evaluated_is_refused`, `test_unfitted_calibration_is_refused`, `test_calibration_columns_disagreeing_with_preprocessing_is_columns_mismatch`, `test_reordered_feature_list_is_features_mismatch`, `test_corrupt_model_text_is_model_load_failed`, `test_missing_lightgbm_is_lightgbm_unavailable` (monkeypatch the import to raise `ImportError`). Every refusal asserts `error.check` equals the code.
- [ ] **Step 2: Run** `uv run pytest packages/attribution/tests/test_served_model_load.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement** `load_served_model` and the types; the loader reads files only here (identity is fixed at load).
- [ ] **Step 4: Run** the file again. Expected: pass.
- [ ] **Step 5: Commit** `feature: load a model pair only after the serving contract checks pass`.

### Task 5: Predict

**Files:**
- Modify: `packages/attribution/src/agentwatch_attribution/served_model.py`
- Create: `packages/attribution/tests/test_served_model_predict.py`, `pipelines/model/tests/test_served_model_parity.py`

**Interfaces:**
- Consumes: Task 4 loader and fixtures, Task 3 `feature_matrix` and `scale`.
- Produces: `@dataclass(frozen=True) class ModelPrediction` with `raw_probabilities: dict[str, float]`, `calibrated_probabilities: dict[str, float]` (both keyed in `class_order`), `confidence: float`, `abstained: bool`, `predicted_class: str | None` (None when abstained), `excess_fraction: float`; and `ServedModel.predict(rows: Sequence[Mapping[str, object]]) -> list[ModelPrediction]` (empty input gives `[]`).

- [ ] **Step 1: Write the failing tests:** `test_predict_matches_manual_booster_and_scale` (recompute with `lightgbm.Booster` and `scale`), `test_abstains_below_threshold_and_has_no_class`, `test_exactly_at_threshold_does_not_abstain` (build a model view with `abstention_threshold` set to the computed confidence), `test_missing_inputs_pass_as_nan_not_zero` (a row with `cpu_pressure_max_avg10=None` predicts the same as NaN and differs from 0.0 on a pair trained so the feature matters), `test_excess_fraction_is_clipped_to_the_declared_range` (a pair whose regressor overshoots), `test_row_missing_a_column_raises_feature_error`. In `pipelines/model/tests/test_served_model_parity.py` build a pair with `frozen_model_fixtures` and assert `ServedModel.predict` class, confidence and abstention equal `held_out_model.frozen_scores` on the same rows (tolerance `1e-9`).
- [ ] **Step 2: Run** them. Expected: FAIL.
- [ ] **Step 3: Implement `predict`:** all trees; temperature scaling via `scale`; abstain when `confidence < threshold`; clip the regressor output to `excess_fraction_clip` from `preprocessing.json` (store the clip on the instance at load).
- [ ] **Step 4: Run** `uv run pytest packages/attribution pipelines/model/tests/test_served_model_parity.py -q`. Expected: pass.
- [ ] **Step 5: Commit** `feature: predict calibrated attribution and the excess-latency estimate`.

### Task 6: Optional `sampling_interval_seconds` on run registration

**Files:**
- Modify: `packages/sdk-python/src/agentwatch_sdk/live.py` (`RunCreate`), `services/worker/src/agentwatch_worker/live_store.py` (`create`), `services/agent/src/agentwatch_agent/runner.py` (the `RunCreate(...)` call near line 382; pass `self.sampling_interval_seconds`), `docs/live-ingestion.md`
- Test: `services/worker/tests/test_live_api.py`, `services/agent/tests/` (the runner registration test; locate with `grep -rn "RunCreate\|configuration_digest" services/agent/tests`), `packages/sdk-python` tests if present

**Interfaces:**
- Produces: `RunCreate.sampling_interval_seconds: PositiveSeconds | None = None` (finite, strictly greater than 0; reuse the pydantic annotation style of `agentwatch_features.live.PositiveSeconds`, or define it in `live.py` next to `Seconds`). `LiveStore.create` stores and compares the registration payload with this key omitted when `None`, so registrations stored before this change stay identical on retry.

- [ ] **Step 1: Write the failing tests:** SDK: `test_sampling_interval_rejects_zero_negative_nan_and_inf`; worker: `test_registration_with_interval_round_trips_in_the_run_summary`, `test_retry_of_a_registration_stored_without_the_interval_is_idempotent` (insert a `live_runs` row with the old JSON directly, then POST the same registration with the field absent: 200, `created: false`), `test_same_request_id_with_a_different_interval_conflicts` (409); agent: `test_runner_registers_its_sampling_interval`.
- [ ] **Step 2: Run** the four. Expected: FAIL.
- [ ] **Step 3: Implement** field, payload handling and runner change; add the field to the registration section of `docs/live-ingestion.md` (optional, seconds, what null means for `telemetry_coverage`).
- [ ] **Step 4: Regenerate contracts:** `uv run python -m agentwatch_worker.export_live_contracts`; run `uv run pytest services/worker/tests services/agent/tests packages -q`, then `pnpm typecheck` (TypeScript types changed). Expected: pass; diff in `api/openapi/live.json` and `packages/sdk-typescript/src/live.ts` shows only the new optional field.
- [ ] **Step 5: Commit** `feature: let producers declare their sampling interval at run registration`.

### Task 7: Model configuration and status

**Files:**
- Create: `services/worker/src/agentwatch_worker/attribution_model.py`, `services/worker/tests/test_attribution_model.py`
- Modify: `services/worker/pyproject.toml` (`agentwatch-attribution` dependency and `[tool.uv.sources]`; `uv lock`), `.env.example`

**Interfaces:**
- Consumes: Task 4 loader.
- Produces: `ModelState = Literal["not_configured", "not_promoted_refused", "ready_not_promoted", "ready_promoted", "load_failed"]`; `@dataclass(frozen=True) class ModelStatus` with `state: ModelState`, `reason: str | None` (a `CHECK_CODES` value for `load_failed`), `identity: ModelIdentity | None`; `class ModelService` with `status: ModelStatus`, `served: ServedModel | None`, `@classmethod from_environ(cls, environ: Mapping[str, str]) -> ModelService`. Env: `AGENTWATCH_MODEL_CANDIDATES_DIR`, `AGENTWATCH_MODEL_CALIBRATION_DIR` (both unset or empty: `not_configured`; only one set: `load_failed` with `artifact_unreadable`), `AGENTWATCH_ALLOW_UNPROMOTED_MODEL` (`1` or `true`, case-insensitive).
- The model is loaded once in `from_environ`; the instance never re-reads files.

- [ ] **Step 1: Write the failing tests:** `test_unconfigured_is_not_configured`, `test_one_directory_only_is_load_failed`, `test_blocked_pair_without_the_flag_is_refused_and_not_loaded` (`served is None`, identity still reported), `test_blocked_pair_with_the_flag_is_ready_not_promoted`, `test_pair_with_blocks_promotion_false_is_ready_promoted`, `test_tampered_pair_is_load_failed_with_the_check_code`, `test_files_changed_after_startup_do_not_change_status_or_identity` (Review Focus 5), `test_flag_values_other_than_one_or_true_do_not_enable` (`"yes"`, `"0"`, empty).
- [ ] **Step 2: Run** them. Expected: FAIL.
- [ ] **Step 3: Implement**, and document the three variables in `.env.example` with empty placeholders and a one-line warning that the current pair is not promoted.
- [ ] **Step 4: Run** the file. Expected: pass.
- [ ] **Step 5: Commit** `feature: configure which model pair the worker serves and report why it cannot`.

### Task 8: Build one attribution record (pure)

**Files:**
- Create: `services/worker/src/agentwatch_worker/attribution.py`, `services/worker/tests/test_attribution_record.py`

**Interfaces:**
- Consumes: `agentwatch_features.FeatureRow`, `agentwatch_attribution.rules.Rules`, Task 4/5 `ServedModel`, Task 7 `ModelStatus` (the function takes `status: ModelStatus` and `model: ServedModel | None`).
- Produces: `NOT_LIVE_INPUTS: tuple[str, ...] = ("workload_iterations", "tests_per_step", "worker_capacity")`; `model_input_row(row: FeatureRow) -> dict[str, object]` (the 11 `features` plus the three above as `None`); `build_attribution(row: FeatureRow, step_state: str, rules: Rules, status: ModelStatus, model: ServedModel | None) -> dict` returning the JSON-ready record: `run_id`, `step_id`, `step_state`, `feature_version`, `feature_row` (`row.model_dump(mode="json")`), `missing_inputs: list[str]` (inputs that are `None`; `False` and `0.0` are values), `telemetry_status`, `rules: {version, verdict}`, `model: {state, reason, training_version, calibration_version, not_promoted, outcome, predicted_class, confidence, abstention_threshold, abstained, calibrated_probabilities, excess_fraction_estimate}` where `outcome` is one of `predicted`, `abstained`, `not_scored_step_state`, `unavailable`.

- [ ] **Step 1: Write the failing tests:** `test_completed_step_gets_rules_and_model` (synthetic pair; fixture `FeatureRow` built through `compute_feature_rows` as `services/worker/tests/test_live_feature_join.py` does or constructed directly), `test_failed_and_cancelled_steps_are_not_scored_by_the_model_but_keep_rules` (outcome `not_scored_step_state`, rules verdict present), `test_step_that_never_executed_still_gets_both_and_lists_every_missing_input` (null window, `missing` telemetry; no exception), `test_measured_zero_is_not_listed_missing` (`execution_seconds=0.0`, `cgroup_available=False`), `test_abstention_is_not_none` (`outcome == "abstained"`, `predicted_class is None`, confidence and threshold present), `test_model_unavailable_keeps_the_rules_verdict_and_names_the_reason`, `test_unpromoted_model_is_labelled`, `test_excess_fraction_is_named_an_estimate_and_kept_apart_from_queue_seconds`.
- [ ] **Step 2: Run** them. Expected: FAIL.
- [ ] **Step 3: Implement** the two functions; rules receive `row.features.model_dump()`; no wording that claims cause.
- [ ] **Step 4: Run** `uv run pytest services/worker/tests/test_attribution_record.py -v`. Expected: pass.
- [ ] **Step 5: Commit** `feature: build a step's rules verdict and model prediction from one feature row`.

### Task 9: Persist attributions

**Files:**
- Create: `services/worker/src/agentwatch_worker/migrations/0005_step_attributions.sql`, `services/worker/src/agentwatch_worker/attribution_store.py`, `services/worker/tests/test_attribution_store.py`

**Interfaces:**
- Consumes: `live_feature_rows`, `LiveStore.steps`, `LiveStore.lock`, Task 8 `build_attribution`, Task 7 `ModelService`.
- Table `live_step_attributions`: `sequence BIGSERIAL UNIQUE NOT NULL` (cursor), `run_id UUID NOT NULL REFERENCES live_runs(run_id)`, `step_id TEXT NOT NULL`, `feature_version TEXT NOT NULL`, `serving_key TEXT NOT NULL` (`canonical_sha256` of rules version plus model state, reason and identity), `record JSONB NOT NULL`, `step_terminal_received_at TIMESTAMPTZ NOT NULL` (`live_records.received_at` of the step's terminal revision), `committed_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp()`, `evidence_changed BOOLEAN NOT NULL DEFAULT false`, `UNIQUE (run_id, step_id, feature_version, serving_key)`, index on `(run_id, sequence)`. Rows are append-only except `evidence_changed`.
- Produces: `class AttributionStore(store: RunStore, rules: Rules, model: ModelService)` with `attribute_run(run_id: UUID) -> int` (rows inserted; locks the run with `LiveStore.lock`; scores every terminal applied step lacking a row for the current serving key; for steps that already have one, recomputes the feature row and sets `evidence_changed` when its `source_sample_ids` differ, never rewriting `record`; raises `MissingRunError` for an unknown run), `pending_terminal_steps(run_id: UUID) -> list[str]` (cheap SQL, no feature compute), `page(run_id: UUID, after: int, limit: int) -> dict` (`{run_id, model: <status dict>, attributions: [{sequence, serving_key, committed_at, step_terminal_received_at, evidence_changed, record}], next_cursor}`; unknown run raises `MissingRunError`; a run without terminal steps returns an empty list).

- [ ] **Step 1: Write the failing tests** (PostgreSQL, reuse `client`, `live`, `post`, `event` helpers from `test_live_api`): `test_terminal_step_gets_one_row_with_rules_and_unavailable_model`, `test_attribute_run_is_idempotent`, `test_incomplete_steps_get_no_row`, `test_two_threads_attributing_one_step_leave_one_row` (Review Focus 4), `test_late_sample_sets_evidence_changed_and_keeps_the_stored_row`, `test_new_serving_configuration_appends_a_new_row_and_keeps_the_old`, `test_page_cursor_and_limit` (`after` beyond the end gives an empty page; `limit` honored; `next_cursor` set only when more rows exist), `test_unknown_run_raises_missing_run_error_and_empty_run_is_an_empty_page`, `test_attribution_never_mixes_runs` (two runs, same step ids), `test_migration_is_additive_and_applies_after_0004` (existing tables untouched, `schema_migrations` lists 0005).
- [ ] **Step 2: Run** them with `AGENTWATCH_TEST_DATABASE_URL` set. Expected: FAIL.
- [ ] **Step 3: Implement** migration and store with parameterized SQL only; insert with `ON CONFLICT DO NOTHING`; a feature or serving error for one step must not discard the other steps' rows.
- [ ] **Step 4: Run** the file, then `uv run pytest services/worker/tests -q`. Expected: pass, none skipped.
- [ ] **Step 5: Commit** `feature: store append-only step attributions with their serving identity`.

### Task 10: API, background trigger, contracts

**Files:**
- Create: `services/worker/src/agentwatch_worker/attribution_api.py`, `packages/sdk-python/src/agentwatch_sdk/attribution.py`, `services/worker/tests/test_attribution_api.py`
- Modify: `services/worker/src/agentwatch_worker/app.py` (lifespan builds `app.state.attribution`; include router), `services/worker/src/agentwatch_worker/live_api.py` (`ingest_records` and `change_run`), `services/worker/src/agentwatch_worker/export_live_contracts.py` (add `api/openapi/attribution.json` and `packages/sdk-typescript/src/attribution.ts`, following the policy pattern), `docs/live-ingestion.md`

**Interfaces:**
- Produces: `GET /v1/live/runs/{run_id}/attributions?after=&limit=` (`after` 0..`MAX_CURSOR`, `limit` 1..256, same conventions as the run snapshot; 404 for an unknown run; first runs `attribute_run` when `pending_terminal_steps` is non-empty) and `GET /v1/live/attribution/model` (the status dict: `state`, `reason`, identity with versions and hashes, `blocks_promotion`, `not_promoted`). Response models `AttributionPage`, `AttributionItem`, `ModelStatusView` in `agentwatch_sdk.attribution`, built on `ResponseModel`. After a successful `POST .../records` whose batch holds a terminal `step_event`, and after `POST .../state` moving a run to a terminal state, a FastAPI `BackgroundTasks` call runs `attribute_after_ingest(store: AttributionStore, run_id: UUID) -> None`, which catches `psycopg.Error` and `MissingRunError`, logs type and sqlstate only (no payloads), and returns: the evidence is already committed and the next read reconciles.

- [ ] **Step 1: Write the failing tests:** `test_terminal_event_makes_its_attribution_visible_on_the_next_poll`, `test_poll_with_cursor_returns_only_new_rows`, `test_unknown_run_is_404_and_run_without_terminal_steps_is_an_empty_page` (distinct bodies), `test_model_status_endpoint_for_each_state` (monkeypatch `app.state.attribution` services), `test_ingest_succeeds_and_next_read_reconciles_when_attribution_fails_after_commit` (monkeypatch `attribute_run` to raise `psycopg.OperationalError` once), `test_non_terminal_batches_do_not_trigger_attribution`, `test_invalid_cursor_and_limit_are_422`, `test_attribution_error_bodies_do_not_echo_producer_values`. A contract test asserts the committed `api/openapi/attribution.json` equals `export_live_contracts` output (follow `test_live_contracts.py`).
- [ ] **Step 2: Run** them. Expected: FAIL.
- [ ] **Step 3: Implement** the router, models, wiring and exporter; `LIVE_API_PREFIXES` already covers `/v1/live/`. Add an attribution section to `docs/live-ingestion.md` (endpoints, cursor, empty page versus 404, model states, polling not SSE, late-sample limitation: samples arriving after the last audit are flagged only at the next audit).
- [ ] **Step 4: Regenerate contracts** with `uv run python -m agentwatch_worker.export_live_contracts`, then run `uv run pytest services/worker/tests packages -q` and `pnpm lint && pnpm typecheck && pnpm test && pnpm build`. Expected: all pass; none skipped.
- [ ] **Step 5: Commit** `feature: serve step attributions and model status through the live API`.

### Task 11: Latency measurement, docs, ADR

**Files:**
- Create: `services/worker/src/agentwatch_worker/attribution_latency.py`, `services/worker/tests/test_attribution_latency.py`, `docs/attribution-serving.md`, `docs/adr/0014-attribution-serving-package.md`
- Modify: `docs/model-serving-contract.md` (add a "Served live" pointer), `docs/project-checklist.md` (B03 status with date and evidence), `Makefile` (a `measure-attribution-latency` target, space-safe), `docs/runbooks/README.md` only if it indexes runbooks

**Interfaces:**
- Produces: `summarize(latencies_seconds: list[float]) -> dict` (`count`, `p50`, `p95` by nearest rank, `max`; an empty list gives `None` statistics, never 0); `measure(base_url: str, steps: int, poll_interval_seconds: float, step_duration_seconds: float) -> dict` and a CLI `python -m agentwatch_worker.attribution_latency --base-url URL --steps N --poll-interval-seconds S --output PATH`. Per step, one `time.monotonic()` clock runs from just before the request that delivers its terminal record to the poll response containing its attribution. The report records: `steps`, poll interval, the model status and identity seen, `synthetic_workload: true`, machine description, the 15.0 s p95 limit, `p95_within_limit` (true, false, or `null` when `count` is 0), server-side `committed_at - step_terminal_received_at` statistics reported separately, and every step that never became visible within a stated timeout, counted as failed, not dropped.

- [ ] **Step 1: Write the failing tests:** `test_summarize_uses_nearest_rank_percentiles`, `test_summarize_of_nothing_is_none_not_zero`, `test_unseen_steps_are_counted_and_make_the_gate_inconclusive` (use a stub poller; no network, no sleeps beyond zero-length intervals), `test_report_names_workload_as_synthetic_and_never_claims_promotion`.
- [ ] **Step 2: Run** them. Expected: FAIL.
- [ ] **Step 3: Implement** with the standard library HTTP client and the live contracts' record models to build step events and samples; the timeout and workload are declared before the run (state them in the doc first).
- [ ] **Step 4: Measure.** Start PostgreSQL and the worker locally against a throwaway database; if `.data/candidates` and `.data/calibration` exist in the main checkout, point the two model variables at them with `AGENTWATCH_ALLOW_UNPROMOTED_MODEL=1` and say so in the report, otherwise use a synthetic pair and write that in the doc. Declare the workload (steps, poll interval, timeout) in `docs/attribution-serving.md` before running, run once for the evidence and once more to show variability, and record both results, including any failures. Do not rerun to get a better number.
- [ ] **Step 5: Write `docs/attribution-serving.md`** (configuration, model states, API summary, results with date, machine and limitations: synthetic steps, one machine, polling interval included, three inputs always missing, current pair not promoted, M05 owns promotion) and ADR 0014 (extraction of `packages/attribution`, the circular dependency, alternatives rejected); update the checklist with dates and test and artifact references while leaving unverified items open.
- [ ] **Step 6: Commit** `feature: measure step completion to visible attribution latency and document serving`.

### Task 12: Verify, sync, open the PR

- [ ] **Step 1: Sync:** `git fetch origin && git merge origin/main`; resolve conflicts by understanding both sides (expect `uv.lock`, `Makefile`, `docs/project-checklist.md`); regenerate `uv.lock` with `uv lock` if it conflicts.
- [ ] **Step 2: Run the CI baseline** from `.github/workflows/ci.yaml`: `uv sync --all-packages --frozen`, `uv run ruff check .`, `uv run pytest` with `AGENTWATCH_TEST_DATABASE_URL` set to a dedicated test database, `pnpm lint`, `pnpm typecheck`, `pnpm test`, `pnpm build`, and the Go and Compose checks if the sync touched them. Expected: all pass, zero skipped database tests. Report any failure with its output; do not push on failure.
- [ ] **Step 3: Review the branch:** `git diff origin/main...HEAD --stat` and `git log --oneline origin/main..HEAD`; only B03 changes; no secrets, no stray artifacts.
- [ ] **Step 4: Ask the user** before pushing; on approval push `feature/serve-attribution-predictions` and open a PR titled `feature: serve versioned rules and model predictions`. Describe shared-interface changes (new package, registration field, new API, migration 0005), the circular-dependency reason, checks run, the latency result and its limits, and that the current pair is not promoted. Partner reviewer Akshay; owner approval required; Meghana receives the API and versions. Link the PR, and do not merge.

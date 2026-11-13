# A06 Held-out Scoring Implementation Plan

> Superseded 2026-11-12 by docs/superpowers/plans/2026-11-12-a06-scoring-m03-m04.md.

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development (recommended) or executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Score `rules-v1` and the learned classifier on the frozen A06 sweep evidence, as a
baseline of missed inference contention and clean-step false alarms.

**Architecture:** Two stages along existing package lines. `services/agent` writes one truth
record per sweep step (`capacity-sweep-truth`), reusing the sweep's own baseline and threshold
logic. `pipelines/model` joins those records to `steps-v1.3` feature rows built from each run's
journal, trains the classifier on dataset v1, and reports the declared measures
(`score_held_out`).

**Tech Stack:** Python 3.13, uv workspace, pytest, Ruff, LightGBM 4.6, MLflow (SQLite).

**Spec:** [docs/superpowers/specs/2026-11-11-a06-held-out-scoring-design.md](../specs/2026-11-11-a06-held-out-scoring-design.md)

## Global Constraints

- Branch `feature/a06-held-out-scoring`; merge after #69 (evidence at `data/held-out/a06-20261111/`).
- Accepted sweeps: protocol version `natural-overload-sweep-v1`, `held_out: true`, and
  `protocol_sha256` exactly one of
  `22e26dc7536fe5e88e3d9ea9f01445a754788ce587b1626d6a4eb055ec4524b9` (cfg-default) or
  `1608deee2a3bcb5245b07d73e13e73dcd1e82702ef14aa683ecd94144e0f18cd` (cfg-hardware).
- Truth method `sweep-truth-v1`; truth names `queue`, `slower`, `protocol_clean`; null reasons
  `span_not_measured`, `no_baseline`, `not_completed`, `malformed`, `not_applicable`.
- Training data: release `steps-v1.3+aafae73eb211`, split
  `data/datasets/splits/steps-v1.3+aafae73eb211/split-v2+febe942a2b59.json`; train and
  validation only; the test split is never loaded.
- Default outputs `.data/a06-scoring/truth/<sweep folder name>/` and
  `.data/a06-scoring/scores/<attempt id>/`; nothing is written under `data/held-out/`.
- MLflow experiment `a06-held-out-scoring`; the existing `agentwatch-attribution` is untouched.
- Bootstrap: `BOOTSTRAP_RESAMPLES = 2000`, `BOOTSTRAP_SEED = 0`, level 0.95, groups are whole
  batches, `MIN_BOOTSTRAP_GROUPS = 2`.
- No pass/fail gate, no promotion, no threshold change. Missing readings stay null.
- Commit titles `<type>: <change>`; no Claude co-author trailer or "Generated with" footer.
- Tests run in WSL through the repository's uv setup; PostgreSQL tests need
  `AGENTWATCH_TEST_DATABASE_URL`.

## Review Focus

- A sweep folder copied elsewhere (relative `output_dir` in `sweep.json`): run folders must
  resolve under the batch (`_resolve_run_dir(..., confine_to_batch=True)`), never the current
  directory. Test in Task 2.
- A step with several model calls where one call is unmeasured: the step's `queue` truth is
  null with `span_not_measured`, never a partial sum. Test in Task 1.
- An excluded ramp batch (`metrics_exclusion`, or no `server`): its runs are not scored, and are
  counted under `excluded_batches`. Test in Task 2.
- A truth step whose run journal is missing or names another run: that run's steps are refused
  with `feature_evidence_unreadable` / `feature_row_missing`, the others still score. Test in
  Task 4.
- A candidate class absent from dataset v1's train split: it can never be predicted; the report
  lists `classes_absent_from_train`. Test in Task 5.

---

### Task 1: Per-step truths in `sweep_labels`

**Files:**
- Modify: `services/agent/src/agentwatch_agent/sweep_labels.py` (`_tally_run`, new `StepTruth`, `step_truths`)
- Test: `services/agent/tests/test_sweep_labels.py`

**Interfaces:**
- Produces:
  - `@dataclass(frozen=True) class StepTruth`: `step_id: str`, `step_type: str`,
    `state: str`, `duration_seconds: float | None`, `queue_seconds: float | None`
    (runner dispatch), `execution_seconds: float | None`, `baseline_mean_seconds: float | None`,
    `noise_threshold_seconds: float | None`, `excess_seconds: float | None`,
    `vllm_queue_seconds: float | None`, `vllm_service_seconds: float | None`,
    `slower: bool | None`, `queued: bool | None`, `reason: str | None` (one of the null reasons).
  - `step_truths(run_dir: Path, reference: Baselines | dict) -> tuple[str | None, list[StepTruth]]`
    returning the run's plan SHA-256 and one record per journaled step. `slower`/`queued` are
    None for non-model steps (`reason="not_applicable"` only when the step is otherwise valid),
    and `queued` is None with `reason="span_not_measured"` when the span status is not
    `measured`.
  - `_tally_run` consumes `step_truths` and produces exactly the tallies it produces today.

- [ ] **Step 1: Write the failing tests**

```python
def test_step_truths_mark_a_model_call_queued_and_slower(tmp_path): ...
    # baseline mean 1.0, stdev 0 -> threshold 0.01; step 1.5 s, vLLM queue 0.4 s
    # assert truth.slower is True and truth.queued is True and truth.reason is None
def test_step_truths_leave_a_partly_unmeasured_step_without_queue_truth(tmp_path): ...
    # two calls on one step, second span not recorded -> queued is None, reason "span_not_measured"
def test_step_truths_report_missing_baseline_not_completed_and_malformed(tmp_path): ...
    # reasons "no_baseline", "not_completed", "malformed"; slower and queued both None
def test_step_truths_give_tool_steps_no_model_truth(tmp_path): ...
    # step_type "tool": slower and queued None, reason "not_applicable", excess_seconds set
```

Build run folders with the helpers `test_sweep_labels.py` already uses for `_tally_run`.

- [ ] **Step 2: Run them and see them fail**

Run: `uv run pytest services/agent/tests/test_sweep_labels.py -k step_truths -v`
Expected: FAIL, `ImportError: cannot import name 'step_truths'`.

- [ ] **Step 3: Implement `StepTruth` and `step_truths`, then rewrite `_tally_run` on top of them**

Move the per-step branch of `_tally_run` into `step_truths` unchanged in its comparisons
(`excess > threshold`, `queue > threshold`). `_tally_run` keeps its counters: a `not_completed`
or `malformed` model step goes to `excluded`, a tool step to `tool`, a model step with an
unmeasured span to `excluded[status]` (keep the span status, not the reason name, for that
counter). Add `StepTruth` and `step_truths` to `__all__`.

- [ ] **Step 4: Run the whole label suite**

Run: `uv run pytest services/agent/tests/test_sweep_labels.py services/agent/tests/test_sweep_offline.py -v`
Expected: PASS, with no existing test edited.

- [ ] **Step 5: Add the frozen-evidence regression and run it**

`test_sweep_analysis_still_reproduces_the_frozen_label_validation`, skipped with reason "DVC
data not pulled" when `data/held-out/a06-20261111/capacity-sweeps` is absent: copy each sweep to
`tmp_path`, run `analyse_sweep_folder`, assert its `label_validation` equals the one in the
frozen `sweep.json`.

Run: `uv run pytest services/agent/tests/test_sweep_labels.py -k frozen -v`
Expected: PASS on the laptop (2 sweeps), SKIP in CI.

- [ ] **Step 6: Commit**

```bash
git add services/agent/src/agentwatch_agent/sweep_labels.py services/agent/tests/test_sweep_labels.py
git commit -m "refactor: expose the sweep's per-step label and queue truths"
```

### Task 2: `capacity-sweep-truth` command

**Files:**
- Create: `services/agent/src/agentwatch_agent/sweep_truth.py`
- Modify: `services/agent/src/agentwatch_agent/cli.py` (parser beside `sweep-record`, dispatch), `Makefile` (target `agent-capacity-sweep-truth`), `services/agent/README.md`
- Test: `services/agent/tests/test_sweep_truth.py`, `services/agent/tests/test_cli.py`

**Interfaces:**
- Consumes: `step_truths`, `baselines`, `ramp_run_dirs` (Task 1, existing);
  `sweep_offline._read_record`, `SweepFolderError`; `sweep_analysis._resolve_run_dir`,
  `split_ramp_entries`.
- Produces:
  - `TRUTH_METHOD = "sweep-truth-v1"`, `ACCEPTED_PROTOCOL_SHA256: dict[str, str]`
    (configuration name -> full hash from Global Constraints), `TRUTH_FILE = "truth.jsonl"`,
    `SUMMARY_FILE = "truth.json"`.
  - `sweep_truths(sweep_dir: Path) -> tuple[dict, list[dict]]`: the summary and one record per
    step. Raises `SweepFolderError` for a refused sweep.
  - `write_truths(sweep_dir: Path, output_dir: Path) -> Path` writing both files (LF, sorted
    keys, `allow_nan=False`) and refusing an `output_dir` inside any `data/held-out` folder.
  - Record keys: `run_id`, `step_id`, `step_type`, `plan_sha256`, `task_id`, `configuration`,
    `n`, `phase`, `batch_sequence`, `batch_id`, `state`, `duration_seconds`, `queue_seconds`,
    `execution_seconds`, `baseline_mean_seconds`, `noise_threshold_seconds`, `excess_seconds`,
    `vllm_queue_seconds`, `truths` = `{"queue": cls|None, "slower": cls|None,
    "protocol_clean": "none"|None}`, `truth_reasons` = same keys, reason or None.
  - Summary keys: `method`, `sweep_id`, `configuration`, `protocol_version`, `protocol_sha256`,
    `sweep_json_sha256`, `held_out: true`, `steps`, `counts` (per truth: class -> count, reason
    -> count), `excluded_batches`, `runs_without_evidence`.

Truth classes: `queue`/`slower` map True -> `"inference"`, False -> `"none"`, None -> null.
`protocol_clean` is `"none"` when the batch's level is 1 and its phase is `ramp`, or `control`
with level 1; otherwise null with `not_applicable`. Phases read: `ramp` (usable entries only),
`live`, `injected`, `control`; `warmup` is skipped and not counted as excluded.

- [ ] **Step 1: Write the failing tests**

```python
def test_truths_cover_every_scored_phase_and_skip_warmup(tmp_path): ...
    # synthetic sweep: warmup n1, ramp n1 x3, ramp n2 x3, live n2, injected n1, control n1
    # assert {r["phase"] for r in records} == {"ramp", "live", "injected", "control"}
def test_protocol_clean_is_none_only_for_one_agent_ramp_and_control(tmp_path): ...
    # injected n1 -> protocol_clean None ("not_applicable"); control n1 -> "none"; control n4 -> None
def test_a_refused_sweep_writes_nothing(tmp_path): ...
    # parametrize: protocol version "other", unknown hash, held_out False -> SweepFolderError, no files
def test_runs_resolve_inside_a_copied_sweep(tmp_path): ...
    # output_dir recorded relative to another cwd; records still read from batch/runs/<name>
def test_excluded_ramp_batches_are_counted_not_scored(tmp_path): ...
def test_output_inside_data_held_out_is_refused(tmp_path): ...
def test_cli_capacity_sweep_truth_writes_both_files_and_exits_0(tmp_path, capsys): ...  # in test_cli.py
```

Build sweeps with the synthetic helpers `test_sweep_offline.py` uses; mark them synthetic.

- [ ] **Step 2: Run and see them fail**

Run: `uv run pytest services/agent/tests/test_sweep_truth.py -v`
Expected: FAIL, `ModuleNotFoundError: agentwatch_agent.sweep_truth`.

- [ ] **Step 3: Implement `sweep_truth.py`, the CLI command and the Make target**

Baselines from `ramp_run_dirs(..., confine_to_batch=True)` level 1, as `ramp_label_results`
builds them. `run_id` and `task_id` come from the batch's run entry. CLI:
`capacity-sweep-truth SWEEP_DIR [--output DIR]`, default output
`.data/a06-scoring/truth/<SWEEP_DIR.name>`; `SweepFolderError` or `OSError` -> stderr and
`EXIT_USAGE` (2). Make target `agent-capacity-sweep-truth` requires `SWEEP`.

- [ ] **Step 4: Run the tests**

Run: `uv run pytest services/agent/tests/test_sweep_truth.py services/agent/tests/test_cli.py -k "truth" -v`
Expected: PASS.

- [ ] **Step 5: Document and commit**

README: one paragraph under the capacity-sweep section: the command, the three truths, the
output location, the refusals.

```bash
git add services/agent Makefile
git commit -m "feature: write per-step queue, slower and protocol-clean truths for a capacity sweep"
```

### Task 3: Batch-grouped bootstrap and named MLflow experiment

**Files:**
- Modify: `pipelines/model/src/agentwatch_model/metrics.py` (`bootstrap_interval`, `bootstrap_macro_f1`)
- Modify: `pipelines/model/src/agentwatch_model/tracking.py` (`experiment_id`, `log_run`)
- Test: `pipelines/model/tests/test_metrics.py`, `pipelines/model/tests/test_tracking.py` (create if absent; otherwise the file that tests `log_run`)

**Interfaces:**
- Produces:
  - `bootstrap_interval(rows: list[dict], statistic: Callable[[list[int]], float | None], *, group_key: str, resamples: int, seed: int, level: float = 0.95, min_groups: int = MIN_BOOTSTRAP_GROUPS) -> dict`
    with the same result keys as today's `bootstrap_macro_f1`; `statistic` receives the drawn
    row indices.
  - `bootstrap_macro_f1(...)` keeps its signature and delegates with `group_key="task_id"`.
  - `experiment_id(artifact_root: Path, experiment: str = EXPERIMENT) -> str` and
    `log_run(..., experiment: str = EXPERIMENT)`.

- [ ] **Step 1: Write the failing tests**

```python
def test_bootstrap_macro_f1_is_unchanged_by_the_generalisation(): ...
    # fixed rows/predictions; assert the interval dict equals a literal captured from the
    # current implementation BEFORE editing it (capture it first, paste the literal)
def test_bootstrap_interval_resamples_whole_groups(): ...
    # 2 groups with all-correct vs all-wrong rows: every resample's value is in {0, 0.5, 1}
def test_bootstrap_interval_needs_min_groups(): ...  # one group -> low/high None
def test_log_run_uses_the_named_experiment(tmp_path): ...
```

- [ ] **Step 2: Run and see the new ones fail**

Run: `uv run pytest pipelines/model/tests/test_metrics.py -k bootstrap -v`
Expected: the unchanged-result test PASSES (captured literal), the others FAIL on the missing name.

- [ ] **Step 3: Implement**

Keep the generator draw order of today's loop, so the macro F1 result is identical.

- [ ] **Step 4: Run the model suite**

Run: `uv run pytest pipelines/model/tests -q`
Expected: PASS (LightGBM tests skip only without OpenMP).

- [ ] **Step 5: Commit**

```bash
git add pipelines/model
git commit -m "refactor: bootstrap any statistic by group and log runs to a named experiment"
```

### Task 4: Held-out feature rows from truth files

**Files:**
- Modify: `pipelines/data/src/agentwatch_data/replay_rows.py` (extract `feature_values`)
- Create: `pipelines/model/src/agentwatch_model/held_out_rows.py`
- Test: `pipelines/data/tests/test_replay_rows.py`, `pipelines/model/tests/test_held_out_rows.py`

**Interfaces:**
- Consumes: Task 2's record and summary keys; `agentwatch_data.replay_rows`
  (`declared_interval_seconds`, `same_run`, `joined_feature`, `ReplayFeatures`,
  `FeatureJoinError`, codes), `agentwatch_data.replay_journal.read_journal`,
  `agentwatch_features.compute_feature_rows`, `agentwatch_data.schema.feature_columns`.
- Produces:
  - `feature_values(feature: FeatureRow) -> dict[str, object]`: the 14 `steps-v1.3` feature
    columns exactly as `release_row` sets them (`workload_iterations`, `tests_per_step`,
    `worker_capacity` None; `counter_delta` on the throttling delta). `release_row` uses it.
  - `class HeldOutError(ValueError)`.
  - `load_truths(directory: Path) -> tuple[dict, list[dict]]`: refuses a summary without
    `held_out: true`, another `method`, or a protocol hash outside the accepted pair.
  - `held_out_rows(truth_dirs: list[Path], sweep_root: Path) -> tuple[list[dict], dict[str, int], list[dict]]`:
    joined rows (truth record + `feature_values` + `telemetry_status`), refusal counts by code,
    and the summaries. `sweep_root` is the folder holding the sweep folders; each summary's
    sweep folder is found by name under it.

- [ ] **Step 1: Write the failing tests**

```python
def test_release_rows_are_unchanged_by_feature_values(): ...  # existing replay_rows tests pass; add
    # assert feature_values(row) keys == set(feature_columns())
def test_held_out_rows_join_truth_and_features(tmp_path): ...
def test_a_timing_mismatch_refuses_only_that_step(tmp_path): ...  # code "feature_timing_mismatch"
def test_a_journal_of_another_run_refuses_that_run(tmp_path): ...  # code "feature_row_missing"
def test_a_missing_journal_refuses_that_run(tmp_path): ...  # code "feature_evidence_unreadable"
def test_truths_without_held_out_are_refused(tmp_path): ...
```

Synthetic journals come from the fixtures `pipelines/data/tests` already uses for replay rows.

- [ ] **Step 2: Run and see them fail**

Run: `uv run pytest pipelines/model/tests/test_held_out_rows.py -v`
Expected: FAIL, module not found.

- [ ] **Step 3: Implement**

Per run: `read_journal` -> `same_run` -> `compute_feature_rows(events, samples,
declared_interval_seconds(run_dir))` -> `ReplayFeatures(by_step=..., source_sha256=<canonical
SHA-256 of the run's two files>)`; join each truth with `joined_feature` (it reads
`duration_seconds`, `queue_seconds`, `execution_seconds` from the truth record).

- [ ] **Step 4: Run both suites**

Run: `uv run pytest pipelines/data/tests/test_replay_rows.py pipelines/model/tests/test_held_out_rows.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add pipelines/data pipelines/model
git commit -m "feature: build steps-v1.3 feature rows for held-out sweep truths"
```

### Task 5: Scorers: `rules-v1` and the calibrated classifier

**Files:**
- Create: `pipelines/model/src/agentwatch_model/held_out_scorers.py`
- Test: `pipelines/model/tests/test_held_out_scorers.py`

**Interfaces:**
- Consumes: `load_split`, `load_contract`, `load_rules`, `load_training_config`,
  `fit_candidate`, `predict`, `check_lightgbm`, `calibrate.fit_on_validation`,
  `calibrate.score_test`, `load_calibration_config`, `save_models` (train_candidates).
- Produces:
  - `@dataclass(frozen=True) class ScoredStep`: `scorer: str`, `label: str`,
    `confidence: float | None`, `abstained: bool`.
  - `rule_scores(rules: Rules, rows: list[dict]) -> list[ScoredStep]` (`scorer="rules-v1"`,
    confidence None, never abstains).
  - `train_classifier(split: Path, contract_path: Path, frozen_schema: Path, config: TrainingConfig, loader=load_split) -> tuple[Candidate, list[dict], frozenset[str]]`
    loading only `"train"` and `"validation"`; returns the candidate and the validation rows.
  - `seen_task_groups(train: list[dict]) -> frozenset[str]`: the `task_group` values of the
    train rows; `train_classifier` also returns it (third tuple element).
  - `classifier_scores(candidate, validation: list[dict], rows: list[dict], contract, calibration) -> tuple[list[ScoredStep], dict]`
    (`scorer="lightgbm-classifier-training-v1"`); the dict records the fit (`fitted`,
    `temperature`, `threshold`, `reason`), or `abstention: "unavailable"` with the reason.

- [ ] **Step 1: Write the failing tests**

```python
def test_only_train_and_validation_are_loaded(tmp_path): ...
    # fake loader records names; assert names == ["train", "validation"]
def test_rules_v1_scores_never_abstain(): ...
def test_classifier_abstention_applies_the_validation_threshold(): ...
    # synthetic rows (training_fixtures); confidences below fit.threshold abstain
def test_unfittable_calibration_reports_abstention_unavailable(): ...
def test_classes_absent_from_train_are_reported(): ...
def test_seen_task_groups_come_from_train_rows_only(): ...
    # train rows of "fix-mean-empty", validation rows of "other": result == {"fix-mean-empty"}
```

- [ ] **Step 2: Run and see them fail**

Run: `uv run pytest pipelines/model/tests/test_held_out_scorers.py -v`
Expected: FAIL, module not found.

- [ ] **Step 3: Implement**

Validation prediction records for `fit_on_validation` are `{"prediction": p.label,
"probabilities": p.probabilities}`; columns are `candidate.class_order`; `early_stopping` is
`candidate.early_stopping`.

- [ ] **Step 4: Run the tests**

Run: `uv run pytest pipelines/model/tests/test_held_out_scorers.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add pipelines/model
git commit -m "feature: score held-out rows with rules-v1 and the calibrated classifier"
```

### Task 6: Report measures, outputs and `make score-held-out`

**Files:**
- Create: `pipelines/model/src/agentwatch_model/held_out_report.py`, `pipelines/model/src/agentwatch_model/score_held_out.py`
- Modify: `Makefile` (target `score-held-out`, `HELD_OUT_SCORES_OUTPUT ?= .data/a06-scoring/scores`)
- Test: `pipelines/model/tests/test_held_out_report.py`, `pipelines/model/tests/test_score_held_out.py`

**Interfaces:**
- Consumes: Tasks 3–5.
- Produces:
  - `measures(rows: list[dict], scored: list[ScoredStep], truth: str, *, resamples: int, seed: int) -> dict`
    over rows whose `truths[truth]` is not null: `steps`, `confusion` (truth class -> predicted
    class -> count, with `"abstained"` as a predicted column), `inference_steps`, `miss_rate`
    (inference-truth steps not predicted `inference`, abstentions counted as missed),
    `answered_miss_rate`, `none_steps`, `false_alarm_rate` (none-truth steps predicted other
    than `none` and not abstained), `answered_false_alarm_rate`, `abstention_rate`, and
    `intervals` for `miss_rate` and `false_alarm_rate` via `bootstrap_interval(group_key="batch_id")`.
    A rate with a zero denominator is None.
  - `build_report(rows, scores: dict[str, list[ScoredStep]], context: dict) -> dict` with
    `cells`: one entry per populated (scorer, truth, configuration, n, phase, step_type,
    task_seen_in_training), plus rollups per (scorer, truth), (scorer, truth, configuration),
    (scorer, truth, task_seen_in_training) and (scorer, truth, configuration,
    task_seen_in_training). `task_seen_in_training` is
    `task_group(row["task_id"]) in seen_task_groups`; `context["seen_task_groups"]` is listed
    in the report; `refused` by code;
    `unscored` (truth null) by truth and reason; `limitations` (the spec's three, verbatim);
    `held_out: true`; `protocol_sha256` per configuration; `gate: "none (descriptive baseline)"`;
    `model` = training version and config hash, LightGBM version, release, split file SHA-256,
    best iterations, `classes_absent_from_train`, and each saved model file's SHA-256.
  - `run(truth_dirs, sweep_root, split, output, *, contract_path, frozen_schema, rules_path, training_path, calibration_path, tracking_uri, artifact_root) -> str`
    returning `"completed"` or `"failed"`; writes `report.json`, `predictions.jsonl` (one line
    per step and scorer: run_id, step_id, scorer, label, confidence, abstained, and the three
    truths) and `models/`; input SHA-256s for every truth file, the split file and each run's
    two files; logs to MLflow experiment `a06-held-out-scoring`, failed attempts included.
  - CLI `python -m agentwatch_model.score_held_out --truth DIR [DIR ...] --sweep-root DIR --split FILE --output DIR`.

- [ ] **Step 1: Write the failing tests**

```python
def test_measures_match_a_hand_counted_confusion(): ...
    # 6 rows: 3 inference (1 caught, 1 missed, 1 abstained), 3 none (1 false alarm)
    # assert miss_rate == 2/3, answered_miss_rate == 1/2, false_alarm_rate == 1/3,
    #        abstention_rate == 1/6
def test_zero_denominators_are_none_not_zero(): ...
def test_intervals_group_by_batch(): ...  # rows of one batch move together
def test_unscored_steps_are_counted_by_reason(): ...
def test_cells_split_seen_and_unseen_tasks(): ...
    # rows of fix-mean-empty-r1 and fix-median-even-r1, seen {"fix-mean-empty"}:
    # the (scorer, truth, task_seen_in_training) rollups have keys True and False with their counts
def test_run_writes_report_predictions_and_models_and_logs_the_named_experiment(tmp_path): ...
def test_a_failed_attempt_is_reported_and_tracked(tmp_path): ...
def test_outputs_inside_data_held_out_are_refused(tmp_path): ...
def test_scores_output_is_not_accepted_as_evidence(tmp_path): ...
    # agentwatch_data.evidence refuses the scores and truth folders (non-zero exit)
```

- [ ] **Step 2: Run and see them fail**

Run: `uv run pytest pipelines/model/tests/test_held_out_report.py pipelines/model/tests/test_score_held_out.py -v`
Expected: FAIL, modules not found.

- [ ] **Step 3: Implement both modules and the Make target**

Attempt id: UTC `YYYYMMDDTHHMMSSZ` plus the first 6 hex digits of the canonical SHA-256 of the
input hashes. Make: `score-held-out` requires `TRUTH`; defaults `SWEEP_ROOT ?=
data/held-out/a06-20261111/capacity-sweeps` and the dataset v1 split path.

- [ ] **Step 4: Run the model and data suites, Ruff**

Run: `uv run ruff check . && uv run ruff format --check pipelines services && uv run pytest pipelines services/agent -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add pipelines/model Makefile
git commit -m "feature: report held-out miss and false-alarm rates with batch-grouped intervals"
```

### Task 7: Real run, record and handoff

**Files:**
- Modify: `docs/capacity-record.md` (new "Attribution baseline" section), `docs/project-checklist.md`, `docs/data-card.md`, `docs/model-training.md` (pointer), the spec's status line if anything changed during implementation

- [ ] **Step 1: Run on the frozen evidence from a clean tree**

```bash
git status --short   # empty
make agent-capacity-sweep-truth SWEEP=data/held-out/a06-20261111/capacity-sweeps/cfg-default-20261111T211237Z-2d75ea
make agent-capacity-sweep-truth SWEEP=data/held-out/a06-20261111/capacity-sweeps/cfg-hardware-20261111T213039Z-b4ad4a
make score-held-out TRUTH=".data/a06-scoring/truth/cfg-default-20261111T211237Z-2d75ea .data/a06-scoring/truth/cfg-hardware-20261111T213039Z-b4ad4a"
```

Expected: both truth commands exit 0; scoring prints `completed` and the attempt folder. Keep a
failed attempt if one happens and record it.

- [ ] **Step 2: Write the record from `report.json` only**

Per configuration and overall, and separately for the seen (`fix-mean-empty`) and unseen
(`fix-median-even`) task: miss rate and false-alarm rate with intervals for both scorers and
each truth, the abstention rate, the predicted-class breakdown on inference-truth steps,
refused and unscored counts, the commit, the attempt id and the input hashes, and the
limitations. No number is typed by hand that the report does not contain.

- [ ] **Step 3: Update the checklist, data card and model-training doc (dated 2026-11-11 or the run date)**

Close "Evaluate natural overload in addition to injector-driven experiments" citing the record.
Add the evidence to "Compare models and rules on the same held-out tasks and contention cases",
which stays open for M2 PR 4's rule. Data card: the scores stay local in `.data/a06-scoring/`,
in no release.

- [ ] **Step 4: Final verification**

Run the CI baseline from AGENTS.md (Python with the PostgreSQL test database, JavaScript, Go,
Compose config). Expected: all pass, or each failure shown as pre-existing on `main`.

- [ ] **Step 5: Commit, push and open the PR**

```bash
git add docs
git commit -m "docs: record the A06 attribution baseline on the held-out sweeps"
git push -u origin feature/a06-held-out-scoring
```

PR title `feature: score rules-v1 and the learned classifier on the held-out A06 sweeps`; body:
what, why (the dated plan change), the results table, checks, limitations, merge after #69.

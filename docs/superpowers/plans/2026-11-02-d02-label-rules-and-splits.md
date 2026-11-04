# D02 Mixed-Contention Labels and Frozen Splits Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Label replay-experiment steps under frozen `labels-v2` rules (including mixed
contention shown as every confirmed resource) and provide the frozen `split-v2` task/method
assignment with independence checks, without changing the current build, split or model code.

**Architecture:** Three new modules in `pipelines/data/src/agentwatch_data/`:
`replay_evidence.py` reads and validates one experiment directory into typed records;
`replay_labels.py` applies `labels-v2.toml` to those records and returns label rows plus
rejections; `split_assignment.py` loads `split-assignment.toml` and assigns/checks labeled rows.
`schema.py` gains a separate `REPLAY_LABEL_COLUMNS` tuple (not added to `COLUMNS`, so releases and
the TFDV schema are unchanged). Docs record the rules, an ADR, and the contract proposal.

**Tech Stack:** Python 3.13+, stdlib (`tomllib`, `statistics`, `json`), pytest, Ruff. No new
dependencies.

**Spec:** `docs/superpowers/specs/2026-11-02-d02-label-rules-and-splits-design.md`

## Global Constraints

- Do not edit `labels.py`, `labels.toml`, `splits.py`, `build.py`, `COLUMNS`, `RESOURCE_CLASSES`,
  anything under `pipelines/model/` or `pipelines/validation/`. `PINNED_LABEL_VERSION =
  "labels-v1.1+fd085fa4"` in `test_dataset_labels.py` must still pass unchanged.
- Noise values: `stdev_multiplier = 3.0`, `min_seconds = 0.01`; "exceeded" means
  `T - B > threshold` (strict).
- Condition → class: `cpu → cpu_sandbox`, `inference → inference`, `retrieval → retrieval`.
- `label_resources` is a comma-separated, sorted string; empty string when no resource.
- `label_resource` values: `RESOURCE_CLASSES + ("mixed",)`.
- Held-out method `worker_queue_blocker`; declared test group `fix-window-count`;
  hash prefix for other groups stays `split-v1`.
- Synthetic fixtures live only under `pipelines/data/tests/` and are named as synthetic.
- Every rejection reason is a distinct code from `REJECTION_CODES`; reason strings start with
  the code followed by `: `.
- Commit titles `<type>: <specific change>`, no co-author trailers (owner preference).

## Review Focus

- A replay directory path in `experiment.json` like `../../etc` or an absolute path must be
  rejected, never read. (Task 1 test `test_paths_outside_experiment_are_rejected`.)
- A mixed replay whose injection log lists one injector only (mixed set misconfigured) must be
  rejected as `resource_class_mismatch`, not labeled as solo. (Task 3 test
  `test_mixed_with_single_injector_is_rejected`.)
- An interrupted experiment (`status: "cancelled"`) yields zero rows and one rejection, not
  partial labels. (Task 3 test `test_incomplete_experiment_yields_no_rows`.)
- Baseline step with `matches_source: false` must shrink that step's reference, and fewer than
  two usable values must reject the step everywhere. (Task 3 test
  `test_differing_baseline_outputs_shrink_reference`.)
- Two revisions of one task (`fix-a-r1`, `fix-a-r2`) must land in one split. (Task 4 test
  `test_revisions_share_a_split`.)

---

### Task 1: Read and validate one replay experiment

**Files:**
- Create: `pipelines/data/src/agentwatch_data/replay_evidence.py`
- Create: `pipelines/data/tests/replay_fixtures.py` (synthetic experiment writer)
- Test: `pipelines/data/tests/test_replay_evidence.py`

**Interfaces:**
- Consumes: `agentwatch_data.extract.parse_json`, `parse_jsonl`.
- Produces:
  ```python
  MAX_EVIDENCE_BYTES = 64 * 1024 * 1024
  class ReplayEvidenceError(ValueError): ...
  @dataclass(frozen=True)
  class StepTiming:
      step_id: str; step_type: str; state: str
      total_seconds: float | None; queue_seconds: float | None; execution_seconds: float | None
      is_leaf: bool; matches_source: bool | None   # None when the replay evidence has no entry
  @dataclass(frozen=True)
  class Injector:
      method: str; resource_class: str
  @dataclass(frozen=True)
  class Replay:
      index: int; role: str; condition: str; state: str; task_passed: bool | None
      run_id: str | None; injection_active: bool | None
      injectors: tuple[Injector, ...]          # empty for clean replays
      steps: dict[str, StepTiming]             # empty when unreadable
  @dataclass(frozen=True)
  class Experiment:
      name: str                                 # directory name
      status: str; checks: dict[str, bool]
      task_id: str; plan_sha256: str; model: str
      replays: tuple[Replay, ...]
  def read_experiment(directory: Path) -> Experiment
  ```
- Fixture helper (tests only):
  `write_experiment(root: Path, *, name="synthetic-exp", task_id="fix-a-r1", baseline=..., contended=..., controls=..., status="completed", checks=None, mismatched=(), parents=None) -> Path`
  where `baseline: list[dict[str, float]]` is per-baseline-replay step seconds and
  `contended: dict[str, dict[str, float]]` maps condition → step seconds. It writes
  `experiment.json`, `replays/NN/agent-evidence.json`, `replays/NN/live-journal.jsonl`
  (`{"kind": "record", "payload": {"record_type": "step_event", "step_id", "parent_step_id"}}`
  lines, root `agent-run` parent of every other step unless `parents` overrides) and
  `injections/NN.json` (`{"injectors": [{"method": ..., "contended_resource_class": ...}]}`
  using the real method names from `services/injectors/README.md`). Order: one warm-up, the
  baselines, then each contended condition followed by a clean control, as in
  `ExperimentSettings.schedule()`.

Read the real shapes from `services/agent/src/agentwatch_agent/experiment.py` (`_replay`,
`step_timings`) and `services/injectors/src/agentwatch_injectors/injection_set.py` (`log()`)
before writing the fixture; match key names exactly.

- [ ] **Step 1: Write failing tests**
  - `test_reads_synthetic_experiment`: fixture with 3 baselines, `cpu` and `mixed` conditions;
    `read_experiment` returns `task_id == "fix-a-r1"`, 1 warm-up + 3 baselines + 2 contended +
    2 controls replays, `replays[-2].injectors` has classes `{"cpu_sandbox", "inference"}`,
    and `steps["agent-run"].is_leaf is False`, `steps["model-call-1"].is_leaf is True`.
  - `test_matches_source_comes_from_agent_evidence`: `mismatched=[(4, "model-call-1")]` gives
    `matches_source is False` for that replay/step, `True` elsewhere.
  - `test_paths_outside_experiment_are_rejected`: `evidence_dir = "../outside"` and an absolute
    path each raise `ReplayEvidenceError` matching `"outside the experiment"`.
  - `test_duplicate_keys_and_oversize_are_rejected`: duplicate key in `experiment.json` raises;
    monkeypatch `MAX_EVIDENCE_BYTES = 10` raises matching `"larger than"`.
  - `test_unknown_role_or_condition_is_rejected`: role `"warmup2"` or condition `"gpu"` raises.
  - `test_nonfinite_durations_are_kept_as_none_or_rejected`: `total_seconds` of `NaN`
    (written as a JSON string is invalid JSON; write `1e999`) raises matching `"finite"`;
    `null` is read as `None`.
- [ ] **Step 2: Run** `uv run pytest pipelines/data/tests/test_replay_evidence.py -q` — expect
  import errors.
- [ ] **Step 3: Implement `read_experiment` in `replay_evidence.py`.** Known roles:
  `("warmup", "baseline", "contended", "clean_control")`; known conditions: `("clean", "cpu",
  "inference", "retrieval", "mixed")`. Resolve every relative path against
  `directory.resolve()` and require `is_relative_to`; reject absolute paths before resolving.
  Read files with a size check before decoding. A replay whose `steps` is `null` in the manifest
  keeps `steps = {}`. Leaf = step id that is no other step's `parent_step_id` in that replay's
  journal; a journal line that is not valid JSON raises (do not skip). Module docstring states
  the input is untrusted and condition/injection data are labeling metadata.
- [ ] **Step 4: Run** the test file — expect PASS. Run `uv run ruff check pipelines/data`.
- [ ] **Step 5: Commit** `feature: read and validate replay experiment evidence for labeling`

### Task 2: Frozen `labels-v2` rules

**Files:**
- Create: `pipelines/data/src/agentwatch_data/labels-v2.toml`
- Create: `pipelines/data/src/agentwatch_data/replay_labels.py` (config part only)
- Test: `pipelines/data/tests/test_replay_labels.py`

**Interfaces:**
- Consumes: `labels.require_keys`, `labels.require_number`, `labels.require_text`,
  `labels.LabelingError`, `extract.canonical_sha256`, `schema.RESOURCE_CLASSES`.
- Produces:
  ```python
  @dataclass(frozen=True)
  class ReplayLabelConfig:
      version: str; stdev_multiplier: float; min_seconds: float
      condition_resources: dict[str, str]; fingerprint: str
      @property
      def label_version(self) -> str   # f"{version}+{fingerprint[:8]}"
  def load_replay_config(path: Path | None = None) -> ReplayLabelConfig
  ```
  `labels-v2.toml` keys: `version = "labels-v2"`, `[noise] stdev_multiplier = 3.0`,
  `min_seconds = 0.01`, `[condition_resources] cpu = "cpu_sandbox"`, `inference = "inference"`,
  `retrieval = "retrieval"`. Comments state the rules are frozen before any evaluation and
  condition data is never a feature.

- [ ] **Step 1: Write failing tests**
  - `test_packaged_replay_rules_load`: values above, `label_version.startswith("labels-v2+")`.
  - `test_replay_label_version_is_pinned`: `PINNED_REPLAY_LABEL_VERSION = "labels-v2+<fill after
    first green run>"` with the same comment as `PINNED_LABEL_VERSION` in
    `test_dataset_labels.py`.
  - `test_formatting_does_not_change_fingerprint` (comments, `3` vs `3.0`, CRLF) and
    `test_changed_values_change_fingerprint`.
  - `test_invalid_rules_are_rejected`: unknown key, missing table, `nan`, unknown condition
    (`gpu`), resource not in `RESOURCE_CLASSES - {"none"}` each raise `LabelingError`.
- [ ] **Step 2: Run** — expect FAIL.
- [ ] **Step 3: Implement `load_replay_config`**, mirroring `labels.load_config`; fingerprint the
  validated values plus `replay_labels.py` source (normalized line endings).
- [ ] **Step 4: Run** — PASS; then fill the pin with the printed value and rerun.
  `uv run pytest pipelines/data/tests/test_dataset_labels.py -q` must still pass unchanged.
- [ ] **Step 5: Commit** `feature: freeze labels-v2 rules for replay experiments`

### Task 3: Label replay steps

**Files:**
- Modify: `pipelines/data/src/agentwatch_data/replay_labels.py`
- Modify: `pipelines/data/src/agentwatch_data/schema.py` (add constants only)
- Test: `pipelines/data/tests/test_replay_labels.py`

**Interfaces:**
- Consumes: Task 1 `Experiment`, `Replay`, `StepTiming`, `read_experiment`; Task 2
  `ReplayLabelConfig`; `labels.excess_fraction`, `extract.Rejection`.
- Produces:
  ```python
  # schema.py
  MIXED = "mixed"
  LABEL_RESOURCE_VALUES = (*RESOURCE_CLASSES, MIXED)
  REPLAY_CONDITIONS = ("clean", "cpu", "inference", "retrieval", "mixed")
  REPLAY_ROLES = ("baseline", "contended", "clean_control")
  REPLAY_LABEL_BASES = (
      "baseline_reference", "clean_within_noise", "clean_exceeded_noise",
      "contention_below_threshold", "contention_exceeded_threshold",
      "mixed_below_threshold", "mixed_confirmed", "mixed_partial", "mixed_joint_only",
  )
  # replay_labels.py
  REJECTION_CODES = (
      "experiment_not_completed", "protocol_checks_failed", "replay_not_succeeded",
      "injection_inactive", "resource_class_mismatch", "step_not_completed",
      "output_differs_from_source", "baseline_reference_missing",
      "mixed_solo_evidence_missing",
  )
  BLOCKING_CHECKS = ("baseline_replays_complete", "stable_baseline",
                     "same_steps_every_replay", "injectors_cleaned_up")
  def task_group(task_id: str) -> str          # strips a trailing "-r<digits>"
  def label_experiment(experiment: Experiment, config: ReplayLabelConfig
                       ) -> tuple[list[dict], list[Rejection]]
  def label_directory(directory: Path, config: ReplayLabelConfig | None = None
                      ) -> tuple[list[dict], list[Rejection]]
  def main(argv: list[str] | None = None) -> int   # prints {"rows": [...], "rejections": [...]}
  ```
  Each row has exactly these keys: `experiment`, `task_id`, `task_group`, `plan_sha256`,
  `agent_model`, `run_id`, `replay_index`, `role`, `condition`, `step_id`, `step_type`,
  `duration_seconds`, `queue_seconds`, `execution_seconds`, `label_resource`,
  `label_resources`, `label_excess_fraction`, `label_basis`, `injection_method`,
  `baseline_mean_seconds`, `noise_threshold_seconds`, `control_exceeded_noise`,
  `failed_checks` (comma-joined failed non-blocking checks or `None`), `label_version`.
  Rows are ordered by `(replay_index, step_id)`.

Rule summary for the implementer (spec "Labels" table is authoritative): only leaf steps of
baseline / contended / clean_control replays; a step is usable when `state == "completed"`,
`total_seconds` is finite positive, and `matches_source is not False`. Per step, B and threshold
come from usable baseline copies (≥ 2 required). Solo condition X: class from
`condition_resources`; each injector's `resource_class` must equal it and there must be exactly
one injector. Mixed: ≥ 2 injectors with distinct classes, none `none`; members = their classes.
Member evidence = the solo contended replay whose class is that member, same `step_id`, usable;
otherwise reject the mixed step `mixed_solo_evidence_missing`. `control_exceeded_noise` uses the
clean_control at `index + 1`; `None` if that replay or step is unusable; `None` for non-contended
rows. `injection_method`: `"none"`, the solo method, or sorted methods joined by `+`.

- [ ] **Step 1: Write failing tests** (fixture from Task 1; baseline step seconds
  `[1.00, 1.01, 0.99]` → B = 1.00, threshold = max(0.01, 3 × 0.01) = 0.03):
  - `test_solo_exceeded_and_below` — cpu `model-call-1` 1.50 → `cpu_sandbox`,
    `contention_exceeded_threshold`, `label_excess_fraction == pytest.approx(1/3)`;
    1.02 → `none`, `contention_below_threshold`.
  - `test_threshold_tie_is_not_exceeded` — contended value exactly `B + threshold` (choose
    baseline values so this is exact in float) → `none`.
  - `test_mixed_confirmed` — cpu 1.5, inference 1.4, mixed 1.9 → `label_resources ==
    "cpu_sandbox,inference"`, `label_resource == "mixed"`, basis `mixed_confirmed`,
    `injection_method == "competing-ollama-generate-v1+cpu-spin-processes-v1"`.
  - `test_mixed_partial_single_member` — cpu 1.5, inference 1.01, mixed 1.9 →
    `"cpu_sandbox"`, `label_resource == "cpu_sandbox"`, `mixed_partial`.
  - `test_mixed_joint_only` — cpu 1.01, inference 1.01, mixed 1.9 → both, `mixed`,
    `mixed_joint_only`.
  - `test_mixed_below_threshold` — mixed 1.01 → `none`, `""`, `mixed_below_threshold`.
  - `test_mixed_missing_solo_member` — no `inference` condition scheduled → mixed rows rejected
    with code `mixed_solo_evidence_missing`; cpu rows still labeled.
  - `test_mixed_with_single_injector_is_rejected` — `resource_class_mismatch`.
  - `test_clean_rows_and_noisy_control` — baselines `baseline_reference`; control 1.5 →
    `clean_exceeded_noise`, and the preceding contended row has `control_exceeded_noise is True`
    with its label unchanged.
  - `test_only_leaf_steps_are_labeled` — no row has `step_id == "agent-run"`.
  - `test_differing_baseline_outputs_shrink_reference` — two of three baselines mismatched on
    `model-call-1` → every row for that step rejected `baseline_reference_missing`; other steps
    labeled; a mismatched contended step rejected `output_differs_from_source`.
  - `test_incomplete_experiment_yields_no_rows` — `status="cancelled"` → `([], [one
    experiment_not_completed])`; failed `stable_baseline` → `protocol_checks_failed`.
  - `test_unsucceeded_or_inactive_replay_is_rejected` — `task_passed=False` →
    `replay_not_succeeded`; `injection_active=False` → `injection_inactive`.
  - `test_non_blocking_failed_checks_are_recorded` — `evidence_delivered: False` → rows carry
    `failed_checks == "evidence_delivered"`.
  - `test_labels_reproduce` — label the same directory twice and a `shutil.copytree` copy;
    rows equal (ignoring `experiment` name for the copy) and `label_version` equal.
  - `test_task_group` — `"fix-mean-empty-r1" → "fix-mean-empty"`, `"a-r2x" → "a-r2x"`.
  - `test_cli_prints_rows` — `main([str(dir)])` returns 0 and stdout JSON has `rows`;
    unreadable directory returns 1 with the error on stderr.
- [ ] **Step 2: Run** — expect FAIL.
- [ ] **Step 3: Implement** `task_group`, `label_experiment`, `label_directory`, `main`
  (`python -m agentwatch_data.replay_labels <experiment dir>`; exit 0 labeled, 1 unreadable or
  `ReplayEvidenceError`, 2 bad arguments). Rejection origin:
  `f"experiment {name}"`, `f"experiment {name} replay {index:02d}"`, or with `step {step_id}`.
  Log rejections at warning, like `labels.label_rows`.
- [ ] **Step 4: Run** the replay test files and `uv run pytest pipelines/data -q` — PASS; ruff.
  Update `PINNED_REPLAY_LABEL_VERSION` (this task changes `replay_labels.py`) and rerun.
- [ ] **Step 5: Commit** `feature: label replay steps with every confirmed contended resource`

### Task 4: Replay label columns stay out of features

**Files:**
- Modify: `pipelines/data/src/agentwatch_data/schema.py`
- Test: `pipelines/data/tests/test_replay_labels.py`

**Interfaces:**
- Produces: `REPLAY_LABEL_COLUMNS: tuple[Column, ...]` in `schema.py`, one `Column` per row key
  of Task 3 except those already in `COLUMNS` with the same meaning (`run_id`, `step_id`,
  `task_id`, `queue_seconds`, `execution_seconds`, `duration_seconds`, `label_excess_fraction`,
  `injection_method`, `failed_checks`, `label_version`, `baseline_mean_seconds`,
  `noise_threshold_seconds`, `agent_model`). Roles: `task_group`, `experiment`,
  `plan_sha256`, `replay_index` → `ID`; `label_resource` (allowed `LABEL_RESOURCE_VALUES`),
  `label_resources` → `LABEL`; `role` (allowed `REPLAY_ROLES`), `condition` (allowed
  `REPLAY_CONDITIONS`), `label_basis` (allowed `REPLAY_LABEL_BASES`),
  `control_exceeded_noise` (bool, nullable) → `LINEAGE`; `step_type` → `ID`. Not added to
  `COLUMNS`; module docstring says D03 adds them with `steps-v1.3`.

- [ ] **Step 1: Write failing tests**
  - `test_replay_columns_describe_every_row_key` — for a labeled fixture row, every key is in
    `COLUMNS` or `REPLAY_LABEL_COLUMNS` names, and each row value satisfies that column's
    `allowed`.
  - `test_replay_label_metadata_is_never_a_feature` — none of `task_group`, `experiment`,
    `plan_sha256`, `replay_index`, `role`, `condition`, `label_resource`, `label_resources`,
    `label_basis`, `control_exceeded_noise`, `injection_method` is in `feature_columns()`, and
    no `REPLAY_LABEL_COLUMNS` entry has `Role.FEATURE`.
  - `test_release_schema_is_unchanged` — `SCHEMA_VERSION == "steps-v1.2"` and no
    `REPLAY_LABEL_COLUMNS` name is in `COLUMNS` except shared ones listed above.
  - `test_replay_step_ids_carry_no_condition` — fixture step ids and `replays/NN` names contain
    none of `REPLAY_CONDITIONS` except `clean`-free tokens (assert no `cpu`, `inference`,
    `retrieval`, `mixed` substring).
- [ ] **Step 2: Run** — FAIL. **Step 3:** add `REPLAY_LABEL_COLUMNS`. **Step 4:** run
  `uv run pytest pipelines/data -q` — PASS (`test_dataset_schema.py` data-card column test must
  still pass because `COLUMNS` is unchanged).
- [ ] **Step 5: Commit** `feature: declare replay label columns as non-features`

### Task 5: Frozen `split-v2` assignment and independence checks

**Files:**
- Create: `pipelines/data/src/agentwatch_data/split-assignment.toml`
- Create: `pipelines/data/src/agentwatch_data/split_assignment.py`
- Test: `pipelines/data/tests/test_split_assignment.py`

**Interfaces:**
- Consumes: `splits.assign(group, fractions)` and `splits.SPLITS` (unchanged), Task 3
  `task_group`, `labels.NO_INJECTION`.
- Produces:
  ```python
  class AssignmentError(ValueError): ...
  @dataclass(frozen=True)
  class Assignment:
      version: str                       # "split-v2"
      held_out_injection_methods: tuple[str, ...]
      test_task_groups: tuple[str, ...]
      sha256: str                        # canonical_sha256 of parsed file, first 8 hex
  def load_assignment(path: Path | None = None) -> Assignment
  def row_methods(row: dict) -> set[str]   # injection_method split on "+", minus "none"
  def assign_groups(rows: list[dict], assignment: Assignment, fractions: dict[str, float]
                    ) -> dict[str, str]    # task_group -> split
  def independence(members: dict[str, list[dict]]) -> dict
      # {"status": "passed"|"failed", "shared": {"task_group": [...], "run_id": [...],
      #  "plan_sha256": [...]}}
  def held_out_status(members: dict[str, list[dict]], assignment: Assignment) -> dict
      # {"status": "met"|"not_met", "reason": str | None, "only_in_test": [...]}
  def mixed_in_test(members: dict[str, list[dict]]) -> dict   # {"status", "rows": int}
  def partition_v2(rows: list[dict], assignment: Assignment, fractions: dict[str, float]
                   ) -> tuple[dict[str, str], dict[str, list[dict]], dict]
      # (groups, members, {"independence": ..., "held_out": ..., "mixed_in_test": ...});
      # raises AssignmentError when independence fails
  ```
  `split-assignment.toml`: `version = "split-v2"`, `held_out_injection_methods =
  ["worker_queue_blocker"]`, `test_task_groups = ["fix-window-count"]`, comments recording
  the selection rule (highest `split-v1:<name>` hash position: 0.502 vs 0.476) and that a
  change requires `split-v3`. Rows use `task_group` when present, else `task_id` (CPU/coding
  rows). Held-out: training must contain another method whose rows have the same
  `label_resource` as a held-out-method row; otherwise `not_met`.

- [ ] **Step 1: Write failing tests** (rows built by a small synthetic helper in the test file;
  fractions `{"train": 0.7, "validation": 0.15, "test": 0.15}`):
  - `test_packaged_assignment_loads` — exact values above.
  - `test_invalid_assignment_is_rejected` — unknown key, empty method name, `"none"` as method,
    duplicate entries, wrong version prefix each raise `AssignmentError`.
  - `test_declared_test_group_goes_to_test` — `fix-window-count` rows → test even though
    `splits.assign("fix-window-count", fractions) == "train"` (assert both).
  - `test_held_out_method_moves_whole_group` — a group with one `worker_queue_blocker` row and
    clean rows → every row in test.
  - `test_mixed_method_containing_held_out_moves_group` — method
    `"cpu-spin-processes-v1+worker_queue_blocker"` → test.
  - `test_other_groups_use_split_v1_hash` — an undeclared group's split equals
    `splits.assign(group, fractions)`.
  - `test_revisions_share_a_split` — `fix-a-r1` and `fix-a-r2` rows have one split.
  - `test_shared_plan_or_run_fails_independence` — same `plan_sha256` in two groups landing in
    different splits → `AssignmentError` naming `plan_sha256`; same for `run_id`.
  - `test_held_out_needs_contrasting_training_method` — no `cpu-spin-processes-v1` rows in
    train → `not_met` with a reason; with them → `met`.
  - `test_mixed_in_test_reported` — no mixed rows in test → `not_met`.
  - `test_partition_is_deterministic` — row order shuffled → identical groups and member
    `(run_id, step_id)` lists.
- [ ] **Step 2: Run** — FAIL. **Step 3: Implement.** **Step 4: Run** the file and the existing
  `uv run pytest pipelines/data/tests/test_dataset_splits.py -q` unchanged — PASS; ruff.
- [ ] **Step 5: Commit** `feature: freeze split-v2 task and injection-method hold-out`

### Task 6: Documentation, ADR and contract proposal

**Files:**
- Modify: `docs/data-card.md` (new "Replay labels (`labels-v2`)" and "Frozen split assignment
  (`split-v2`)" sections; update the "Mixed contention is not defined" sentence and Known
  limitations: step-type spillover, two tasks, validation empty, no replay evidence stored)
- Create: `docs/adr/0007-multi-resource-labels-and-frozen-hold-out.md` (follow `docs/adr/README.md`
  format)
- Create: `docs/evaluation-contract-v2-proposal.md` (the four bullets from the spec's "Evaluation
  contract proposal"; marked proposed, owner Rohit)
- Modify: `docs/project-checklist.md` (D02: implemented 2026-11-02 with test file references;
  open items: real replay evidence, contract v2 acceptance, D03 wiring)
- Modify: `pipelines/data/README.md` if it lists modules or commands (check first; add
  `python -m agentwatch_data.replay_labels`)

- [ ] **Step 1:** Write the docs. Every rule value quoted must match `labels-v2.toml` and
  `split-assignment.toml`.
- [ ] **Step 2:** Run `uv run pytest pipelines/data/tests/test_dataset_schema.py -q` (data-card
  column test) and check every relative link resolves:
  `grep -o '](\([^)#]*\)' docs/data-card.md docs/adr/0007-*.md docs/evaluation-contract-v2-proposal.md`
  and `ls` each target.
- [ ] **Step 3: Commit** `docs: record labels-v2, split-v2 and the eval-contract-v2 proposal`

### Task 7: Final verification

- [ ] **Step 1:** `git fetch origin && git merge origin/main` if main moved; resolve by reading
  both sides.
- [ ] **Step 2:** `uv run ruff check .` and `uv run ruff format --check pipelines/data` — clean.
- [ ] **Step 3:** `AGENTWATCH_TEST_DATABASE_URL=<dedicated disposable test DB> uv run pytest -q`
  — all pass, no skipped database tests. If no test database is available, report that as
  incomplete verification.
- [ ] **Step 4:** Data-validation tests as in `.github/workflows/ci.yaml` (`data-validation` job
  command).
- [ ] **Step 5:** Real-data check: `ls .data/agent-experiments data/evidence` for any replay
  experiment; if one exists, run `uv run python -m agentwatch_data.replay_labels <dir>` twice and
  diff the output. Otherwise record "no recorded replay experiment available" in the checklist.
- [ ] **Step 6:** `git diff origin/main...HEAD --stat` and `git log --oneline origin/main..HEAD`
  contain only this task's files.

# D02 — Mixed-contention label rules and frozen grouped splits

Status: approved by the owner, 2026-11-02. Board task D02 (owner Allen, partner reviewer
Akshay; Rohit reviews the frozen evaluation sets before training).

## Intent

The model must learn which resource made an agent step wait. Today's labels (`labels-v1.1`)
cover one injected resource per run, from the CPU and coding experiments. The bounded
injectors and replay runner (A05, PRs #41/#43) now record real agent tasks replayed under
`cpu`, `inference` and `retrieval` contention alone and under `mixed` (`cpu` + `inference`)
contention together. D02 freezes, before any model is trained or evaluated:

1. how those replayed steps are labeled, including mixed contention, noisy cases and ties;
2. the regression target;
3. which tasks and which injection method are held out for the test split.

Acceptance (board): label reproduction and split independence verified; injection and label
metadata excluded from features.

## Owner decisions

- **A step slowed by two contentions shows both resources.** No single "winning" resource is
  chosen for a mixed step.
- **Each resource in a mixed label needs its own evidence**: the same step, replayed in the
  same experiment under that resource's contention alone, must also have slowed beyond noise.
- **Today's data keeps `split-v1`.** The new split (`split-v2`) applies only when the
  evaluation contract selects it, so the current release and Rohit's training on it (#45,
  #46) are unchanged.
- **Data side plus a proposal.** D02 changes only `pipelines/data` and docs. It writes an
  `eval-contract-v2` proposal for Rohit and does not edit `pipelines/model` or the evaluation
  contract. `make dataset` keeps building `labels-v1.1` releases until that contract and the
  D03 ingestion land.
- The owner delegated the remaining rule choices to the designer and may revise them in
  review. Every such choice below is marked *(designer's call)*.

## Goals

- `labels-v2`: deterministic step labels for one replay experiment, computed from its
  `experiment.json`, injection logs and per-replay `agent-evidence.json`, with an explicit
  multi-resource label.
- `split-v2`: a frozen, declared assignment of task groups and one held-out injection method
  to the test split, with split independence checked on every write and read.
- Every new label and injection column is declared non-feature, enforced by tests.
- Reproduction: labeling and splitting twice from the same evidence gives identical output and
  hashes.

## Non-goals

- Ingesting replay experiments into dataset releases (evidence-store entries, sample joins,
  step features, Parquet rows, `steps-v1.3`, the frozen TFDV schema). That is D03, which
  depends on D02.
- Changing `RESOURCE_CLASSES`, `pipelines/model`, or `pipelines/model/evaluation-contract.toml`.
- Collecting experiments or adding tasks (Benoy's serialized GPU sessions, A05 notes).
- Training, calibration, or any model evaluation.
- Changing `labels.toml` (`labels-v1.1`) or its pinned fingerprint.

## Anti-goals

- Inventing per-resource shares of a mixed step's slowdown. The evidence has one total
  duration per step; splitting it would fabricate attribution.
- Tuning the noise threshold, the held-out method or the test tasks after looking at labels
  or model output.
- Dropping steps because the rule's answer is inconvenient. Steps are rejected only for
  missing or invalid evidence, with a distinct recorded reason.
- Breaking the current build, split or model pipeline while the contract still pins
  `labels-v1.1`.

## Constraints

- `AGENTS.md`: missing evidence stays distinct from zero; queue, execution and total time stay
  distinct; synthetic fixtures live only in tests; label rules are fingerprinted and pinned.
- `eval-contract-v1` pins `labels-v1.1`, four classes (checked against `RESOURCE_CLASSES` by both
  `splits.load_contract` and `agentwatch_model.contract`), and `group_key = "task_id"`.
- Real evidence: `data/evidence` holds 25 CPU and 25 coding runs and no replay experiments.
  Two agent tasks exist: `fix-mean-empty-r1` and `fix-window-count-r1`.

## Design

### Inputs

One replay experiment directory (protocol `agent-replay-experiment-v1`):

- `experiment.json`: `status`, `checks`, `plan.task_id` (`<name>-r<revision>`),
  `plan.plan_sha256`, `plan.model`, and `replays[]` with `index`, `role` (`warmup`,
  `baseline`, `contended`, `clean_control`), `condition` (`clean`, `cpu`, `inference`,
  `retrieval`, `mixed`), `state`, `task_passed`, `run_id`, `injection_active`,
  `evidence_dir`, `injection_log`, and `steps` (per `step_id`: `step_type`, `state`,
  `total_seconds`, `queue_seconds`, `execution_seconds`, from the replay's own monotonic
  ticks).
- `injections/<NN>.json`: each injector's `method` and `contended_resource_class`.
- `replays/<NN>/agent-evidence.json`: `replay.model_outputs[]` and `replay.tool_results[]`
  (`step_id`, `matches_source`).
- `replays/<NN>/live-journal.jsonl`: each step's `parent_step_id`, to find leaf steps.

All of it is untrusted input, validated at the boundary the way `extract.py` validates
`results.json`: unique keys, bounded depth and file size, known roles and conditions, finite
positive durations, and paths confined to the experiment directory.

### Which evidence may be labeled

- **Experiment:** labelable only when `status` is `completed` and the protocol checks
  `baseline_replays_complete`, `stable_baseline`, `same_steps_every_replay` and
  `injectors_cleaned_up` pass. Otherwise every row is rejected with the failed checks. These are
  protocol-validity checks, not rule outcomes, so rejecting on them does not select data by its
  label. Other failed checks go to `failed_checks` and do not block labels.
- **Replay:** warm-ups are never labeled. A replay is labeled only if `state == "completed"`
  and `task_passed is True` (the experiment's own `_succeeded`), and, when contended, its
  `injection_active` is true. Otherwise its rows are rejected with that reason.
- **Step:** only **leaf** steps are labeled: model calls, tool executions and retrievals with
  no child steps. The root `agent-run` and parent steps cover their children's time, so
  labeling them would count one slowdown several times. *(designer's call)* A leaf step is
  labeled only if its `state` is `completed` and its `total_seconds` is finite and positive.
- **Same work:** a step whose `matches_source` is false (a different model reply, or a
  different tool outcome) did different work, so its duration explains itself. It is rejected
  (`output_differs_from_source`) in every role, and a baseline step that differs is left out of
  that step's reference. This is the per-step form of `labels-v1.1`'s `same_workload_output`
  rejection. *(designer's call)*

### Per-step baseline and noise

For each `step_id`, B is the mean `total_seconds` over the baseline replays where that step is
usable, and the threshold is `max(min_seconds, stdev_multiplier × sample stdev)` with the
`labels-v1.1` values (`0.01 s`, `3.0`). Fewer than two usable baseline values means no
reference, and the step is rejected in every replay (`baseline_reference_missing`). A step
"exceeded noise" when `T − B` is strictly greater than the threshold.

Total time (submission to finish) is the labeled quantity, as in `labels-v1.1`, because
contention can show up as queueing or as slower execution. Queue and execution time stay
separate.

### Labels (`labels-v2`)

Condition to class: `cpu → cpu_sandbox`, `inference → inference`, `retrieval → retrieval`.
Each injector's `contended_resource_class` must agree, or the replay is rejected.

| Replay role / condition | Step outcome | `label_resources` | `label_resource` | `label_basis` |
|---|---|---|---|---|
| `baseline` | any | `` (empty) | `none` | `baseline_reference` |
| `clean_control` | within noise | `` | `none` | `clean_within_noise` |
| `clean_control` | exceeded noise | `` | `none` | `clean_exceeded_noise` |
| solo `X` | within noise | `` | `none` | `contention_below_threshold` |
| solo `X` | exceeded noise | `X` | `X` | `contention_exceeded_threshold` |
| `mixed` | within noise | `` | `none` | `mixed_below_threshold` |
| `mixed` | exceeded; every member's solo step exceeded | all members | `mixed` | `mixed_confirmed` |
| `mixed` | exceeded; some members' solo steps exceeded | those members | one member: that class; else `mixed` | `mixed_partial` |
| `mixed` | exceeded; no member's solo step exceeded | all members | `mixed` | `mixed_joint_only` |

- `label_resources` is a comma-separated, sorted list of classes (like `failed_checks`), so
  rows stay flat. It is the multi-resource target.
- `label_resource` is one value per step: the class when `label_resources` has one entry,
  `mixed` when it has two or more, `none` when empty. `mixed` is a label value only. It is not
  added to `RESOURCE_CLASSES`, which `rule_prediction` and the model package use.
- **Ties.** Every confirmed resource is listed, so no resource is ranked above another and
  there is no tie to break. A slowdown exactly at the threshold is "not exceeded".
  *(designer's call)*
- **Joint-only slowdowns:** the step slowed only when both contentions ran. Each injected
  resource was necessary for it, so both are listed, and the distinct basis lets evaluation
  report these steps separately. *(designer's call; the alternative was rejecting them.)*
- **Missing members:** if a member's solo replay was not scheduled, was rejected, or lacks a
  usable copy of the step, the mixed step is rejected (`mixed_solo_evidence_missing`).
  *(designer's call)*
- **Step type is not filtered.** A model call slowed by the CPU injector is `cpu_sandbox`: the
  label names the contention that was applied, an operational proxy. The injectors' documented
  spillover (CPU load slows the model server; the retrieval flood slows every runner thread) is
  recorded in the data card. `step_type` stays on each row so evaluation can report each class
  by step type. *(designer's call)*
- **Noisy controls:** `control_exceeded_noise` records whether the clean control that follows a
  contended replay exceeded noise on the same step (missing when that control is unusable). It
  never changes the label: drift is reported, not tuned away. *(designer's call)*
- `injection_method`: the solo injector's `method`, the sorted members' methods joined by `+`
  for mixed, or `none`.
- `label_version`: `labels-v2` plus an 8-hex fingerprint of the validated rule values and the
  labeling code, pinned in a test, the same mechanism as `labels-v1.1`. The rules live in a new
  `labels-v2.toml`; `labels.toml` and its pin are untouched. *(designer's call)*

### Regression target

`label_excess_fraction = max(0, min(1, (T − B) / T))`, the `labels-v1.1` formula against the
per-step baseline. For a mixed step it is the whole step's excess, never split by resource.

### Task identity and groups

Each row carries `task_id` (the real `plan.task_id`) and `task_group`, the task name with the
`-r<revision>` suffix removed, so revisions of a task never cross splits. `task_group` is an
identity column, not a feature. *(designer's call)*

### Frozen split assignment (`split-v2`)

`split-v2` is used only when the evaluation contract's `[splits]` names it
(`method = "split-v2"`, `group_key = "task_group"`, `assignment = "<file>"`). `eval-contract-v1`
names neither, so `split-v1` runs exactly as today. The assignment is the data-owned file
`pipelines/data/src/agentwatch_data/split-assignment.toml`, versioned and hashed into every
split:

- `held_out_injection_methods = ["worker_queue_blocker"]`. Every group with a row using a
  held-out method goes entirely to test, and training learns `cpu_sandbox` from
  `cpu-spin-processes-v1`. Holding out `cpu-spin-processes-v1` instead would send every mixed row
  to test too, and holding out the only inference or retrieval method would remove that class
  from training.
- `test_task_groups = ["fix-window-count"]`. With two agent tasks, 70/15/15 hashing cannot
  place one in test reliably, so the test task is declared. It was chosen by a rule fixed before
  any label was seen: the highest `split-v1:<name>` hash position (`fix-window-count` 0.502,
  `fix-mean-empty` 0.476).
- Other groups use the `split-v1` hash and the contract's fractions, so new tasks never move
  frozen ones. A declared group cannot move within `split-v2`; changing the file requires
  `split-v3` and a new test split.

Checks, recorded in the split and recomputed by `load_split`:

- **Independence (fails the split):** no `task_group`, `run_id` or `plan_sha256` appears in
  more than one split.
- **Held-out method:** each declared method appears only in test, and training has another
  method for the same class. Otherwise `not_met`.
- **Mixed in test:** the test split has mixed-condition rows. Otherwise `not_met`.
- The existing gates stay: quality-gate recheck, reconstruct-and-compare on read, measured-only
  test sources, training leakage AUC, and the contract's count minimums (below them the
  dependent gate is `inconclusive`). With two tasks, validation is empty, and that is recorded.

### Feature exclusion

`label_resources`, `label_basis`, `control_exceeded_noise`, `injection_method`, the replay role
and condition, `task_group`, and every injection-log field are declared in `schema.py` with
`LABEL`, `LINEAGE` or `ID` roles in a replay-label column set that D03 adds to the release
schema. A test asserts none of them is in `feature_columns()`, and that replay `step_id`s and
evidence directory names carry no condition.

### Evaluation contract proposal

`docs/evaluation-contract-v2-proposal.md`, for Rohit to accept, change or reject:

- accept `labels-v2`; add `mixed` to the label values evaluated;
- add a multi-resource metric (per-resource F1 on `label_resources`) next to macro F1, and
  define how a four-class prediction is scored on a `mixed` step;
- select `split-v2` with `group_key = "task_group"` and the assignment file;
- leave every threshold unchanged.

### Verification

- Unit tests with synthetic, clearly labeled `experiment.json` fixtures: every row of the label
  table, each rejection reason, exact-threshold ties, missing solo members, differing outputs,
  non-leaf steps, unknown roles or conditions, disagreeing resource classes, NaN, infinite and
  zero durations, duplicate keys, and paths escaping the directory.
- Reproduction: label one experiment twice and from a copied directory; identical rows and
  `label_version`. Build a `split-v2` split twice; identical split id and `rows_sha256`.
- Independence: regression tests where two revisions of one task, or a reused `plan_sha256`,
  would straddle splits.
- `split-v1` unchanged: the existing split tests pass unmodified, and today's release produces
  the same split id as before.
- Real data: label one recorded replay experiment per agent task if one exists. None is in the
  evidence store today, so D02 acceptance on real replays stays open and is reported as such.
- CI baseline: `uv run ruff check .`, `uv run pytest` with the PostgreSQL test database, and
  the data-validation tests.

### Docs

`docs/data-card.md` (labels-v2, split-v2, limitations), `docs/project-checklist.md` (D02 status
with date and evidence), the contract proposal, and an ADR in `docs/adr/` (0007) for the
multi-resource label and held-out choices.

## Left to the builder

- Module layout (for example `replay_labels.py` next to `labels.py`) and function names.
- Rejection message text, as long as each reason is distinct and tested.
- How the label function reports rejections, following `labels.label_rows`.

## Open items after D02

- D03: ingest replay experiments into releases with these labels and `steps-v1.3`.
- Rohit: accept or revise `eval-contract-v2`.
- Benoy: record at least one replay experiment per task and add it to the evidence store.

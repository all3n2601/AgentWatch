# 0012: split-v3 holds out the A06 evidence and fills validation

Date: November 11, 2026. Status: accepted in PR #70 (November 12, 2026), with
`eval-contract-v3` set to `accepted` before merge so its identifier (`1cab0cbe`) does not move. Adds `split-v3`
(`pipelines/data/src/agentwatch_data/split-assignment-v3.toml`) and `eval-contract-v3`, and
keeps [split-v2](0009-combined-dataset-release-and-split-v2.md) and `eval-contract-v2`
unchanged so their splits still rebuild.

Under split-v2, dataset v1 (`steps-v1.3+aafae73eb211`) could not be evaluated or calibrated:

- Its test split mixed two feature definitions. The 350 CPU and coding rows use `steps-v1.2`
  features and the replay rows use `step-features-v1`, and evaluation refuses that mix.
- Its validation split was empty. Only one non-test task group (`fix-mean-empty`) exists, and
  whole-task grouping hashed it into train.

The A06 capacity sweeps (PR #65) also need a frozen place before any of their rows reach a
release: the prefill-flood runs, the natural-overload replays, and the new task
`fix-median-even`.

**Decision.** split-v3 keeps split-v2's declared test task groups, `split-v1` hash positions
for everything else and refusal of shared runs and plans, changes how held-out methods are
handled, and adds the rest below. Rohit proposed these with Benoy on 2026-11-11
(`fix-median-even` staying unseen is Benoy's decision). Every rule is label-independent by
construction: it reads feature versions, method names, task groups, protocol hashes and
experiment names, never labels or outcomes. Labels of `fix-mean-empty` were already seen under
`split-v2`; no rule was chosen from them.

- **One feature definition.** Only rows with `feature_version` in `feature_versions`
  (`step-features-v1`) are split. Other rows are in no split; the split file counts them under
  `split_v3_checks.excluded_feature_versions`.
- **Held-out methods.** `worker_queue_blocker` and `prefill-flood-openai-chat-v1`
  ([ADR 0011](0011-prefill-flood-held-out-injection-method.md)). Unlike `split-v2`, a row of a
  held-out method on a task that is not a declared test task refuses the split instead of
  moving the task to test: moving `fix-mean-empty`, the only training task, would empty
  training. Prefill-flood rows of `fix-mean-empty` need a decision (for example, the held-out
  set) before they reach a release.
- **Test task groups.** `fix-window-count` and `fix-median-even`. `fix-median-even` is new
  (revision 1) and stays unseen; its `split-v1` position (0.622) would otherwise put it in
  train.
- **Natural overload, a separate held-out set.** A row whose `protocol_sha256` is listed under
  `[held_out_protocol]` is held out of train, validation and test, whatever its protocol
  version. The list holds the two A06 protocol sweeps (cfg-default, sweep
  `c5a40c4c-e4b3-4dc1-b660-21ca4c88bc93`; cfg-hardware, sweep
  `068e27a9-5933-4f99-b465-75502f7e523b`) and the preflight dry run (sweep
  `3a8dd5b2-ac98-4095-b58c-3d90e0950976`), which uses the same protocol version with a 300 s
  run limit and so its own hash. A row of `natural-overload-sweep-v1` whose hash is not listed is refused,
  so an unknown sweep can never reach training. The held-out set may share a task with any
  split but never a run or a plan; sharing refuses the split. The split file records the set
  under `split_v3_checks.held_out_evaluation`, including its task groups, so a report can tell
  an unseen condition on a seen task (`fix-mean-empty`) from both unseen (`fix-median-even`).
- **Every split row names its experiment.** A kept row with no `experiment` refuses the
  split. Capacity-sweep runs have no experiment, so an untagged natural-overload row of a
  training task is refused, not trained on. Every `step-features-v1` row of dataset v1 has one.
- **Validation by experiment.** Within a task group assigned to train, a whole experiment goes
  to validation when the position of `split-v3:<experiment>` (SHA-256, first 8 bytes over
  2^64) is below validation / (train + validation) = 0.15 / 0.85. Train and validation may
  share a task group; they may never share a run or a plan, and test shares no task group with
  either. The split file lists the chosen experiments under
  `split_v3_checks.validation_experiments`; `groups` stays task-level, so a training task maps
  to train there. The salt and the share formula live in `split_assignment.py`, outside the
  assignment file's hash; the recorded experiment list and per-split `rows_sha256` detect any
  change to them, and `test_split_v3.py` pins two positions.

**Why by experiment, not by run.** Rohit first proposed splitting `fix-mean-empty` by run. A
replay experiment's labels and noise threshold come from its own clean baselines, so its runs
share baseline-derived values. Splitting an experiment's runs across train and validation would
put those values on both sides. Whole experiments keep baselines, contended replays, controls
and plan together. Each of `fix-mean-empty`'s 21 experiments has its own plan.

**Why a hash threshold.** Like `split-v1`, a threshold on a hash position never moves an
existing experiment when a new one is added. On dataset v1 it sends 3 of 21 experiments (273
steps) to validation, above the 100 steps `calibration-v1` needs.

**Consequences.**

- On dataset v1 the split is `split-v3+932b2df410a7`: train 1,755 steps, validation 273, test
  3,575 (`fix-window-count` only). Leakage, class balance, independence and mixed-in-test pass.
- The held-out-method check is `not_met`: neither held-out method has a `step-features-v1` row
  in dataset v1, so the contract's `held_out_injection_methods = 1` is unmet. The
  `worker_queue_blocker` rows are `steps-v1.2` and left out, so that listing has no effect on
  dataset v1; it stays so a later `step-features-v1` row of it is placed correctly. The rules do not evaluate held-out generalization, and the check stays honest
  until prefill-flood rows are in a release.
- The CPU and coding rows are no longer evaluated. Evaluating them needs a model on their own
  features, which is not planned.
- No row carries `protocol_sha256` yet, so the natural-overload rule runs on synthetic
  fixtures only. An untagged sweep row is refused because it names no experiment, but a sweep
  row that somehow carried an experiment and no protocol fields would be split like any other
  row. The PR that ingests sweep replays must therefore write both fields on every row, and
  must not merge without a test that its rows are held out. Scoring the held-out set also needs a
  natural-overload label rule, because `labels-v2` labels every such step `none`.
- Validation is small and clustered: 273 steps from 3 experiments, 234 of them `none` (8
  `cpu_sandbox`, 14 `inference`, 12 `retrieval`, 5 `mixed`). All three use the 7-step plan
  shape recorded on 2026-11-05, interleaved in time with training experiments on the same
  laptop. A fitted temperature and abstention threshold from it are weak evidence; one
  abstention moves a class's validation rate by 7–20 points. Validation measures unseen
  experiments of the training task, not unseen tasks or drift, so the test-split calibration
  and abstention gates are the real check.
- The `slices` report places split-v3 rows from `validation_experiments`, the left-out feature
  definitions and the held-out hashes the split file records.

# AgentWatch evaluation contract

Status: `eval-contract-v2`, accepted (identifier `9b55506c`). Rohit, the M02 owner, accepted it
and told the owner directly on 2026-11-05; his approval of the D03 pull request is the written
confirmation. It adopts items 1, 3 and 4 of [the v2 proposal](evaluation-contract-v2-proposal.md);
item 2, the multi-resource metric and the rule for scoring a `mixed` step, is not encoded and
remains Rohit's M02 decision. Thresholds live in `pipelines/model/evaluation-contract.toml`.
`eval-contract-v1` (identifier `3022770b`) is kept unchanged as
`pipelines/model/evaluation-contract-v1.toml`, so splits and results made under it still rebuild.
No model or rule baseline has been evaluated yet.

These targets are fixed before any evaluation so that no threshold is tuned to held-out results.
Each value's basis is **Plan** (from the planning documents), **Plan, metric chosen here** (the
plan sets an aim and this contract picks the exact metric), or **Proposed** (no supporting data;
must be accepted or changed before first use).

## Dataset and splits

Uses `steps-v1.3` or a later compatible minor version (`steps-v1.x`, which only adds columns),
the label rules `labels-v1.1` and `labels-v2` (a release is accepted when every label version it
holds is listed), the classes `none`, `cpu_sandbox`, `inference`, and `retrieval`, and the
`split-v2` split from [the data card](data-card.md): grouped by `task_group`, with the frozen
assignment file `split-assignment.toml` declaring the test task group and the held-out injection
method; other groups keep their `split-v1` hash position under the 70/15/15 fractions. `mixed`
is a `labels-v2` label value outside the four prediction classes. The test split contains
`measured` runs only and is frozen, with its hash recorded, before any tuning. Label versions do
not follow the schema's compatible-minor rule: any other label version needs a contract change.
Tuning, calibration, and the abstention threshold use the validation split only.

`eval-contract-v1` evaluated `steps-v1.1` or later with exactly `labels-v1.1` and `split-v1`
(70/15/15 grouped by `task_id`). To rebuild its splits and results, pass
`pipelines/model/evaluation-contract-v1.toml` and `pipelines/validation/schema/steps.pbtxt`
(see [the data card](data-card.md#splits)).

| Requirement | Basis |
|---|---|
| At least one injection method and unseen tasks appear only in the test split | Plan |
| Mixed-contention cases appear in the test split | Plan |
| At least 30 test steps per class, 100 `clean`, and 30 mixed-contention; otherwise the dependent gate is inconclusive. Splits record this as `class_balance`, counting steps labeled `mixed` for the mixed minimum | Proposed |
| No single feature has a one-vs-rest AUC above 0.98 for `label_resource` on the training split; otherwise evaluation is blocked. A step is a positive for every class in its `label_resources` | Proposed |

The leakage check exists because on the CPU and coding data `label_resource` equals
`rule_prediction` on every row and `queue_fraction` nearly separates the classes. On the first
`split-v2` split (`split-v2+2efb6e737ba2`, dataset v1) the check passed, with an observed
maximum of 0.974242 (`queue_seconds`, `retrieval`) against the 0.98 bound.

## Gates

| Gate | Threshold | Basis |
|---|---|---|
| Primary metric | Resource macro F1 over all four classes; an abstention counts as a miss | Plan |
| Gain over rules, mixed contention | At least +0.10 absolute macro F1 over `rule_prediction` | Plan, metric chosen here |
| Gain over rules, overall | Macro F1 not lower than the rule baseline | Proposed |
| Clean-step correctness | At least 95% of `clean` steps predicted `none`, and at least 95% of `baseline` and `clean` steps together; an abstention counts as a miss | Plan |
| Regression error | MAE of `label_excess_fraction` at most 0.10, and lower than constant training-mean and constant-zero predictors | Proposed |
| Calibration | Top-label ECE at most 0.05, 10 equal-width bins, all test steps before abstention | Proposed |
| Abstention | At most 20% of steps, overall, per true class, and on the mixed-contention subset | Proposed |
| Instrumentation overhead | Median step wall-time increase at most 5% versus uninstrumented runs | Proposed |
| Attribution latency | p95 at most 15 s from step completion to visible attribution | Plan, metric chosen here |

The clean-step gate is applied to `clean` steps alone because `baseline` steps are the reference
the label noise threshold is computed from, so they pass almost by construction.

Some gates cannot be evaluated yet and are reported as not evaluated until their evidence exists:

| Gate | Needs |
|---|---|
| Unseen tasks, and task-grouped bootstrap intervals | More than one task per split: dataset v1 has one training task group, an empty validation split, and an interim `task_id` on CPU and coding rows |
| Held-out injection method, as a model gate | A model evaluated on a split where it is held out; `split-v2` holds out `worker_queue_blocker` and records the check, but no model or rule evaluation has run |
| Mixed-contention gain, and any score on `mixed` steps | The `mixed` scoring rule (proposal item 2, Rohit's M02 decision); until then `make rule-baseline` and `make train-candidates` refuse a split with `mixed` labels |
| Calibration, abstention, and validation-dependent gates | A non-empty validation split; dataset v1 has none, so these are inconclusive |

## Promotion

A candidate is promoted only when every gate passes. An inconclusive or not-evaluated gate blocks
promotion. On the same frozen test split and dataset version, the candidate must also not regress
against the incumbent or the best model accepted on that split:

| Comparison | Limit | Basis |
|---|---|---|
| Macro F1 | Drops by at most 0.01 | Proposed |
| Clean-step `none` rate | Does not drop | Proposed |
| MAE | Increases by at most 0.01 | Proposed |

The rule baseline is the first classification incumbent. After 5 promotion evaluations on one
test split (**Proposed**), a new test split is frozen from newly measured runs.

Each evaluation report records the contract version and contract hash, the dataset version, the
full `label_version` including its fingerprint, the test-split hash, every gate's outcome (pass,
fail, inconclusive, or not evaluated), per-class precision, recall, and abstention, and 95%
bootstrap intervals for macro F1 and MAE resampled by `task_id`. Failed and inconclusive results
are kept. The contract hash is the first 8 hex digits of the SHA-256 of the parsed TOML serialized
as canonical JSON (sorted keys, no whitespace), so line endings and comment-only edits do not
change it.

## Change policy

Change thresholds only before an evaluation uses them, and bump `version` with every change.
Moving `status` to `accepted` requires review by the dataset owner and the promotion-check owner.
Keep the previous contract file when a new version changes how splits are made, so earlier
splits and results still rebuild (`evaluation-contract-v1.toml`).

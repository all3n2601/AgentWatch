# AgentWatch evaluation contract

Status: proposed `eval-contract-v1`, pending review. Thresholds live in
`pipelines/model/evaluation-contract.toml`. No model or rule baseline has been evaluated yet.

These targets are fixed before any evaluation so that no threshold is tuned to held-out results.
Each value's basis is **Plan** (from the planning documents), **Plan, metric chosen here** (the
plan sets an aim and this contract picks the exact metric), or **Proposed** (no supporting data;
must be accepted or changed before first use).

## Dataset and splits

Uses `steps-v1` or a compatible minor version (`steps-v1.x`, which only adds columns), `labels-v1`,
the classes `none`, `cpu_sandbox`, `inference`, and `retrieval`, and the 70/15/15 split grouped by
`task_id` from [the data card](data-card.md). The test split contains `measured` runs only and is
frozen, with its hash recorded, before any tuning. Tuning, calibration, and the abstention
threshold use the validation split only.

| Requirement | Basis |
|---|---|
| At least one injection method and unseen tasks appear only in the test split | Plan |
| Mixed-contention cases appear in the test split | Plan |
| At least 30 test steps per class, 100 `clean`, and 30 mixed-contention; otherwise the dependent gate is inconclusive | Proposed |
| No single feature has a one-vs-rest AUC above 0.98 for `label_resource` on the training split; otherwise evaluation is blocked | Proposed |

The leakage check exists because on current data `label_resource` equals `rule_prediction` on
every row and `queue_fraction` nearly separates the classes.

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
| Unseen tasks, and `task_id`-grouped bootstrap intervals | Real task identity in step events; today's `task_id` is an interim placeholder that can place one project in several splits |
| Held-out injection method | The `injection_method` lineage column and at least two injection methods |
| Mixed-contention gain | Label rules that define mixed contention and ties; `labels-v1` does not |

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

Each evaluation report records the contract version and contract hash, the dataset, label, and
test-split versions, every gate's outcome (pass, fail, inconclusive, or not evaluated), per-class
precision, recall, and abstention, and 95% bootstrap intervals for macro F1 and MAE resampled by
`task_id`. Failed and inconclusive results are kept. The contract hash is the first 8 hex digits
of the SHA-256 of the parsed TOML serialized as canonical JSON (sorted keys, no whitespace), so
line endings and comment-only edits do not change it.

## Change policy

Change thresholds only before an evaluation uses them, and bump `version` with every change.
Moving `status` to `accepted` requires review by the dataset owner and the promotion-check owner.

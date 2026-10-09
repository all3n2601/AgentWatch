# AgentWatch evaluation contract

Status: proposed `eval-contract-v1`, pending review. Thresholds live in
`pipelines/model/evaluation-contract.toml`. No model or rule baseline has been evaluated yet.

These targets are fixed before any evaluation so that no threshold is tuned to held-out results.
Each value's basis is **Plan** (from the planning documents), **Plan, metric chosen here** (the
plan sets an aim and this contract picks the exact metric), or **Proposed** (no supporting data;
must be accepted or changed before first use).

## Dataset and splits

Uses `steps-v1`, `labels-v1`, the classes `none`, `cpu_sandbox`, `inference`, and `retrieval`, and
the 70/15/15 split grouped by `task_id` from [the data card](data-card.md). The test split
contains `measured` runs only and is frozen, with its hash recorded, before any tuning. Tuning,
calibration, and the abstention threshold use the validation split only.

| Requirement | Basis |
|---|---|
| At least one injection method and unseen tasks appear only in the test split | Plan |
| Mixed-contention cases appear in the test split | Plan |
| At least 30 test steps per class, 100 `baseline`/`clean`, and 30 mixed-contention; otherwise the dependent gate is inconclusive | Proposed |

## Gates

| Gate | Threshold | Basis |
|---|---|---|
| Primary metric | Resource macro F1 over all four classes; an abstention counts as a miss | Plan |
| Gain over rules, mixed contention | At least +0.10 absolute macro F1 over `rule_prediction` | Plan, metric chosen here |
| Gain over rules, overall | Macro F1 not lower than the rule baseline | Proposed |
| Clean-run false positives | At most 5% of `baseline`/`clean` steps predicted as a resource | Plan, metric chosen here |
| Regression error | MAE of `label_excess_fraction` at most 0.10, and lower than constant training-mean and constant-zero predictors | Proposed |
| Calibration | Top-label ECE at most 0.05, 10 equal-width bins, all test steps before abstention | Proposed |
| Abstention | At most 20% of steps, overall, per true class, and on the mixed-contention subset | Proposed |
| Instrumentation overhead | Median step wall-time increase at most 5% versus uninstrumented runs | Proposed |
| Attribution latency | p95 at most 15 s from step completion to visible attribution | Plan, metric chosen here |

The mixed-contention gate and the held-out injection method cannot be evaluated until labels
define mixed contention and ties, and the dataset records each step's injection method. Neither
exists in `labels-v1` or `steps-v1`. Until then those gates are reported as not evaluated.

## Promotion

A candidate is promoted only when every gate passes. An inconclusive or not-evaluated gate blocks
promotion. On the same frozen test split and dataset version, the candidate must also not regress
against the incumbent or the best model accepted on that split:

| Comparison | Limit | Basis |
|---|---|---|
| Macro F1 | Drops by at most 0.01 | Proposed |
| Clean-run false-positive rate | Does not increase | Proposed |
| MAE | Increases by at most 0.01 | Proposed |

The rule baseline is the first classification incumbent. After 5 promotion evaluations on one
test split (**Proposed**), a new test split is frozen from newly measured runs.

Each evaluation report records the contract version and the first 8 hex digits of the contract
file's SHA-256, the dataset, label, and test-split versions, every gate's outcome (pass, fail,
inconclusive, or not evaluated), per-class precision, recall, and abstention, and 95% bootstrap
intervals for macro F1 and MAE resampled by `task_id`. Failed and inconclusive results are kept.

## Change policy

Change thresholds only before an evaluation uses them, and bump `version` with every change.
Moving `status` to `accepted` requires review by the dataset owner and the promotion-check owner.

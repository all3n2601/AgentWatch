# AgentWatch rule baseline

Status: `rules-v1`, implemented and tested; not yet evaluated on a dataset that can be evaluated.
On the interim `steps-v1.2+5954f97ef8b2` split (2026-10-26) the report is **blocked** by the
leakage check, with no metrics.

The rule baseline is the transparent reference a learned model must beat. It names the resource
a step waited on from measured features, and is scored against
[the evaluation contract](evaluation-contract.md) on the same frozen splits the model will use.

## Rules

Thresholds live in `pipelines/model/src/agentwatch_model/rules.toml`. They are proposed values,
declared before any evaluation and never tuned on the test split. The rules version is
`rules-v1` plus the first 8 hex digits of the SHA-256 of the parsed file as canonical JSON, so
any retuned threshold gets a new version.

| Prediction | When | Basis |
|---|---|---|
| `cpu_sandbox` | `queue_fraction` at least 0.5 | Proposed |
| `cpu_sandbox` | `cpu_pressure_max_avg10` at least 20%, when `cpu_pressure_available` | Proposed |
| `cpu_sandbox` | `cgroup_throttled_usec_delta` at least 10% of `duration_seconds`, when `cgroup_available` | Proposed |
| `none` | No rule fires | Proposed |

A missing or unreadable reading never fires a rule. `steps-v1.2` has no feature that measures
inference or retrieval waiting, so `rules-v1` never predicts `inference` or `retrieval`; the
report lists them as uncovered rather than guessing.

## Running it

```sh
make rule-baseline SPLIT=data/datasets/splits/<release>/<split>.json
```

Splits are read only through `load_split`, which refuses a split whose quality verdict or
leakage check does not pass. The command writes to `.data/rule-baseline/`:

- `report.json`: status, rules and contract identity, split file and hash, metrics and gates;
- `predictions.jsonl`: one line per scored step (`run_id`, `step_id`, `task_id`, `condition`,
  `split`, `dataset_version`, `label_version`, `label_resource`, `prediction`, `rules_version`,
  `evaluation`), kept for comparison with the learned model on identical rows. The report
  records this file's SHA-256, and `(run_id, step_id)` must be unique.

`--explore` writes the same two files to `.data/rule-baseline/exploratory/`, so it never
replaces an evaluated result. Each run first removes its directory's earlier report and
predictions, so files from different runs never sit side by side.

| Status | Meaning |
|---|---|
| `evaluated` | Metrics and gate outcomes on the test split of a split that passed every check |
| `blocked` | `load_split` refused the split; the reason is recorded, with no metrics or predictions; exit code 1 |
| `exploratory` | Run with `--explore`, which skips only the leakage check and scores the validation split, so test results cannot steer later threshold changes; no gate is evaluated |

Consumers should branch on `status`. `evaluation` is false and no gate passes in both blocked
and exploratory reports, and an exploratory report has metrics while a blocked one does not.

A malformed rules or contract file (including contract classes that differ from the dataset's),
or a split row that is missing a required field, repeats a step, has an unknown label or an
excess fraction outside 0 to 1, or comes from a different dataset version than the other rows,
exits with code 2 and an error message instead of a report. `make rule-baseline` without
`SPLIT` also exits with code 2.

## Report contents

- Confusion matrix and per-class support, precision, recall and F1. A value that cannot be
  measured, such as the precision of a class never predicted, or the recall and F1 of a class
  absent from the split even if it was predicted, is null rather than zero.
- Macro F1 two ways: the contract's four-class value, null while any class's F1 is undefined,
  and the mean over the classes with a defined F1. It is marked inconclusive while any class
  has fewer test steps than the contract's minimum. The defined-class value has a 95%
  percentile bootstrap interval that resamples whole tasks (2,000 resamples, recorded seed,
  valid resamples counted); with fewer than 5 tasks (proposed) the interval is null.
- Slices: a confusion matrix per experiment condition, and the mixed-contention slice reported
  as unavailable, with its reason, until labels define mixed contention.
- Clean-step gates: at least 95% of `clean` steps, and of `baseline` and `clean` steps
  together, predicted `none`; inconclusive below the contract's minimum clean steps. Each gate
  records the steps its rate covers (`steps`) and, separately, the clean steps sufficiency is
  judged on (`clean_steps`, `min_clean_steps`); baseline steps never count toward the minimum.
- MAE of the contract's constant references (training mean and zero) for the excess-latency
  fraction, beside the contract's MAE limit. `rules-v1` predicts no fraction, so it has no
  regression error of its own.
- A summary counting each gate outcome, with all four counts present (zeros on a blocked
  report). Any outcome other than pass blocks promotion.
- Every other contract gate as `not_evaluated`, with the reason: the rules are the reference
  for gain over rules, mixed contention is undefined in `labels-v1.1`, the rules give no
  confidence or abstention, task identity is interim, only one injection method exists, and
  overhead and latency are measured on live runs.

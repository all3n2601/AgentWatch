# AgentWatch rule baseline

Status: `rules-v1+1bf54c19`, evaluated on 2026-11-11 on dataset v1
(`steps-v1.3+aafae73eb211`, split `split-v3+932b2df410a7`, `eval-contract-v3` `1cab0cbe`,
accepted). It **fails** both clean-step gates and leaves nine gates `not_evaluated`, so it blocks
promotion; see [Results on dataset v1](#results-on-dataset-v1). Earlier runs had no metrics: on
the interim `steps-v1.2+5954f97ef8b2` split (2026-10-26) the report is **blocked** by the
leakage check, and on `split-v2+febe942a2b59` it stops before scoring because that test split
mixes two feature definitions. Steps labeled `mixed` are scored by `mixed-scoring-v1` (see
below).

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
leakage check does not pass. The defaults are the current contract (`eval-contract-v3`, which
makes `split-v3` splits) and frozen schema (`steps-v1.3.pbtxt`). A split made under another
contract is refused with the contract to pass. To rerun on a `split-v2` split, pass
`EVALUATION_CONTRACT=pipelines/model/evaluation-contract-v2.toml`; on a `steps-v1.2` split, pass
the files it was made with:

```sh
make rule-baseline SPLIT=data/datasets/splits/steps-v1.2+5954f97ef8b2/split-v1+58e921fa846a.json \
  EVALUATION_CONTRACT=pipelines/model/evaluation-contract-v1.toml \
  VALIDATION_SCHEMA=pipelines/validation/schema/steps.pbtxt
```

**Mixed steps.** `mixed-scoring-v1` (`pipelines/model/src/agentwatch_model/mixed.py`) is Rohit's rule for
item 2 of [the v2 proposal](evaluation-contract-v2-proposal.md), fixed before any evaluation: a
four-class prediction on a step labeled `mixed` is correct (a hit) when it names any resource
in its `label_resources`. Steps with basis `mixed_joint_only` are reported apart (`joint_only`:
steps, and how many predictions named a member) and never scored, for classification or
regression. Every score uses the recorded labels as fixed truth, never the prediction, so the
model and the rules are measured against the same answer:

- the four-class confusion matrix, per-class scores, macro F1 and their minimums cover
  single-class steps only, as a mixed step has no single class;
- mixed steps are scored by their hit rate;
- per-resource scores (`multi_resource`) are one-vs-rest on `label_resources` over every scored
  step: a step is a positive for every confirmed resource, so a single prediction on a mixed
  step leaves its other members as misses and per-resource recall on mixed steps stays below 1;
- the mixed-contention macro F1 is the mean per-resource F1 on scored mixed steps over the
  resources confirmed on at least one of them, a set fixed by the labels. Naming one resource on
  every step can reach a high hit rate but scores zero for the other resources.

A `mixed` row must name two or more known resources and a known mixed basis
(`mixed_confirmed`, `mixed_partial`, `mixed_joint_only`); otherwise it is an unusable row.
Its `steps-v1.2` behavior is unchanged.

The command writes to `.data/rule-baseline/`:

- `report.json`: status, rules and contract identity, split file and hash, metrics and gates;
- `predictions.jsonl`: one line per scored step (`run_id`, `step_id`, `task_id`, `condition`,
  `split`, `dataset_version`, `label_version`, `label_resource`, `prediction`, `rules_version`,
  `evaluation`), kept for comparison with the learned model on identical rows. The report
  records this file's SHA-256, and `(run_id, step_id)` must be unique.

`--explore` writes the same two files to `.data/rule-baseline/exploratory/`, so it never
replaces an evaluated result. A run that produces a report, blocked or not, first removes its
directory's earlier report and predictions, so files from different runs never sit side by
side. A run that stops with an error before that (exit 2) leaves the earlier result in place.
A blocked `--explore` run also exits 1; its report is under `exploratory/` and says nothing about
the evaluated split.

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

## Results on dataset v1

Run on 2026-11-12 at commit `f229a4c` (branch `feature/split-v3`, PR #70, after
`eval-contract-v3` was set to `accepted`). To reproduce, with the release pulled (see
[the data card](data-card.md)):

```sh
uv run --package agentwatch-data python -m agentwatch_data.splits --release data/datasets/steps-v1.3+aafae73eb211
make rule-baseline SPLIT=data/datasets/splits/steps-v1.3+aafae73eb211/split-v3+932b2df410a7.json
```

The first command rebuilds (or reuses) the split; the second writes `.data/rule-baseline/`,
which is not committed. Two runs gave byte-identical files:

- `report.json`: SHA-256 `5d1b25e23feaa06783a3f5b2addfd021bf9e378f184da79c69b5f23957192ed8`
- `predictions.jsonl` (3,575 rows, one per test step):
  SHA-256 `3dda66a896c282fc8a0e6e94db42ef3e14bb6cb21587eefc05b5518c8d1f2083`

The test split is one task, `fix-window-count`, with 3,575 steps (3,482 single-class, 82 scored
`mixed`, 11 joint-only). The rules were declared on `steps-v1.2` features before any
evaluation and were not changed for this run. The first runs, on 2026-11-11 at `6ade73d`
under the then-proposed contract (`b9f20820`, `split-v3+95309c38a53e`, same rows), wrote the
same `predictions.jsonl`; their report differed only in the contract status and hash and the
split file and hash.

| Gate or measure | Result | Contract |
|---|---|---|
| Macro F1 (four classes, single-class steps) | 0.200; no interval (1 task group, needs 5) | Reported |
| Clean steps predicted `none` | 0.731 (804 of 1,100) | **fail**, needs 0.95 |
| Baseline and clean steps predicted `none` | 0.749 (1,853 of 2,475) | **fail**, needs 0.95 |
| Mixed-contention slice | 82 steps; hit rate 0.049; macro F1 0.047 over `cpu_sandbox` and `inference` | Reported |
| Excess-latency fraction | Not predicted by `rules-v1`; constant references: MAE 0.194 (zero), 0.233 (training mean) | Limit 0.10 |
| Other gates | 9 `not_evaluated` (reasons in the report) | — |

Per class (single-class steps):

| Class | Support | Predicted | Precision | Recall | F1 |
|---|---|---|---|---|---|
| `none` | 2,923 | 2,606 | 0.801 | 0.714 | 0.755 |
| `cpu_sandbox` | 127 | 876 | 0.026 | 0.181 | 0.046 |
| `inference` | 226 | 0 | n/a | 0.000 | 0.000 |
| `retrieval` | 206 | 0 | n/a | 0.000 | 0.000 |

Confusion matrix (rows: label; columns: prediction):

| Label | `none` | `cpu_sandbox` | `inference` | `retrieval` |
|---|---|---|---|---|
| `none` | 2,088 | 835 | 0 | 0 |
| `cpu_sandbox` | 104 | 23 | 0 | 0 |
| `inference` | 226 | 0 | 0 | 0 |
| `retrieval` | 188 | 18 | 0 | 0 |

What this shows, without claiming a cause:

- **False positives.** 835 steps labeled `none` are predicted `cpu_sandbox`: 326 of 1,375
  baseline steps, 296 of 1,100 clean steps and 213 of 448 contended steps whose label is
  `none`. The rules were declared on `steps-v1.2` features before any evaluation (see
  `rules.toml`); on these replay steps they fire far more often than the labels do.
- **Missed contention.** `rules-v1` has no rule for `inference` or `retrieval`, so all 432
  such steps are predicted `none` or `cpu_sandbox`. Of the 127 `cpu_sandbox` steps, 23 are found.
- **Mixed steps.** 4 of 82 scored mixed steps name a confirmed resource (all `cpu_sandbox`);
  the rest are predicted `none`. 9 of the 11 joint-only steps would name a member, but they are
  not scored.
- **Uncertainty.** With one test task, the task-grouped bootstrap gives no interval, and every
  result describes `fix-window-count` on one laptop and one model only.

This is the reference the learned model is compared against (M03). The handoff to Allen and
Meghana is the report and predictions above.

## Report contents

- Confusion matrix and per-class support, precision, recall and F1. A value that cannot be
  measured, such as the precision of a class never predicted, or the recall and F1 of a class
  absent from the split even if it was predicted, is null rather than zero.
- Macro F1 two ways: the contract's four-class value, null while any class's F1 is undefined,
  and the mean over the classes with a defined F1. It is marked inconclusive while any class
  has fewer test steps than the contract's minimum. The defined-class value has a 95%
  percentile bootstrap interval that resamples whole tasks (2,000 resamples, recorded seed,
  valid resamples counted); with fewer than 5 tasks (proposed) the interval is null.
- Per-resource scores on `label_resources` (`multi_resource`) beside macro F1, as described
  under Mixed steps.
- Slices: a confusion matrix per experiment condition over single-class steps, and the
  mixed-contention slice: scored mixed steps, their `hit_rate`, per-resource scores and macro F1
  with the resources it covers, and the joint-only steps apart; inconclusive below the
  contract's 30 mixed steps.
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
  for gain over rules and mixed-contention gain, the rules give no
  confidence or abstention, task identity is interim, only one injection method exists, and
  overhead and latency are measured on live runs.

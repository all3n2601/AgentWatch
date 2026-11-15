# AgentWatch model training

Status: the packaged default (`training.toml`, used by `make train-candidates` and `make calibrate`)
is `training-v3+71c4dd58`: `training-v2+fd027e8a`, chosen on validation by the declared tuning run
(on test it does not improve on `training-v1`; see [Tuning and
training-v2](#tuning-and-training-v2)), trained with condition-balancing weights that the [model
bias check](model-bias.md) adopted on validation. The frozen evaluated pair, which held-out scoring
and the serving contract pin, is `training-v1+a34ea0be`, trained and evaluated on dataset v1
(`steps-v1.3+aafae73eb211`, split `split-v3+932b2df410a7`, `eval-contract-v3` `1cab0cbe`, accepted);
see [Results on dataset v1](#results-on-dataset-v1). It passes every gate it evaluates and blocks
promotion only because six gates are `not_evaluated`. Earlier: the interim `steps-v1.2+5954f97ef8b2`
split is blocked by the leakage check, and on `split-v2+febe942a2b59` a run is a `failed` attempt
because that test split mixes two feature definitions. Steps labeled `mixed` are scored by
`mixed-scoring-v1` (see Running it). What serving must reproduce, and what each prediction means, is
in the [serving contract](model-serving-contract.md).

M03 trains two LightGBM candidates on CPU: a resource classifier and an excess-latency
regressor. It scores them and [the rule baseline](rule-baseline.md) on identical rows against
[the evaluation contract](evaluation-contract.md), and tracks every attempt in MLflow.

## Training

Settings live in `pipelines/model/src/agentwatch_model/training.toml`, fixed before training
and never tuned on the test split. The training version is `training-v1` plus a hash of the
parsed file. One thread, a fixed seed and LightGBM's deterministic mode make retraining
reproducible on the same LightGBM version and platform; both versions are recorded.

| Model | Target | Objective |
|---|---|---|
| Classifier | `label_resource`, over the classes present in the training split | Multiclass log loss |
| Regressor | `label_excess_fraction`, clipped to 0–1 at prediction | L1 (MAE) |

- Both train on the training split only. The validation split is used only for early stopping,
  recorded per model: `early_stopping` for the classifier (single-class validation rows of
  trained classes, with the rest counted in `validation_rows_dropped`) and
  `regressor_early_stopping` for the regressor (every validation row). A model without
  validation rows to stop on runs the fixed number of rounds, recorded as `not_used`.
- Inputs are exactly the schema's feature columns. Missing readings (null in the dataset) go to
  LightGBM as NaN and are never imputed. A NaN or infinite value already stored in a feature is
  not treated as missing: it is a `FeatureError` and the attempt fails, since recorded data keeps
  missing readings null; booleans become 1.0 and 0.0. The only learned preprocessing is the class order
  from the training split.
- A training split with fewer than two classes fails the run rather than producing a model.

## Running it

```sh
make train-candidates SPLIT=data/datasets/splits/<release>/<split>.json
```

The defaults are the current contract (`eval-contract-v3`, which makes `split-v3` splits) and
frozen schema (`steps-v1.3.pbtxt`). A split made under another contract is refused with the
contract to pass. To rerun on a `split-v2` split, pass
`EVALUATION_CONTRACT=pipelines/model/evaluation-contract-v2.toml`; on a `steps-v1.2` split, pass
the files it was made with: `EVALUATION_CONTRACT=pipelines/model/evaluation-contract-v1.toml
VALIDATION_SCHEMA=pipelines/validation/schema/steps.pbtxt`.

**Mixed steps.** The model and the rules are scored on them alike, by the rule in
[the rule baseline](rule-baseline.md) (`mixed-scoring-v1`). A mixed step has no single class to
learn, so the classifier trains, and stops early, on single-class steps only, and the report
counts the mixed training steps it skipped (`train_rows_not_classified`). The regressor trains
and stops early on every step's excess fraction. A `mixed` row without two or more known
`label_resources` or a known mixed basis makes the attempt `failed`. Behavior on `steps-v1.2` splits is unchanged.

LightGBM needs OpenMP: `libgomp` on Linux (present on CI runners) and `libomp` on macOS. Without
it the command exits with code 2 and says so, and the LightGBM tests skip with that reason.

Outputs go to `.data/candidates/`:

| File | Contents |
|---|---|
| `classifier.txt`, `regressor.txt` | Trained LightGBM models at their best iteration |
| `preprocessing.json` | What serving must reproduce: feature order, missing-value and boolean handling, class order (the probability columns), objectives, best iterations, the excess-fraction clip range, model file hashes, LightGBM version, dataset and label versions, and the attempt's status and evaluation flag |
| `predictions.jsonl` | Per scored step: identity, condition, label, model class and per-class probabilities, excess fraction, and the rule prediction |
| `validation_predictions.jsonl` | The same for the validation split, read by [calibration](model-calibration.md). Validation also drove early stopping, so calibration fitted on it is slightly optimistic |
| `report.json` | Model and rule metrics on the same rows, regression MAE, coverage gaps, gates and summary, and the SHA-256 of `predictions.jsonl`, `validation_predictions.jsonl` and `preprocessing.json`; written last, so its presence marks a complete attempt |
| `tracking.json` | Whether the attempt reached MLflow, with the run id or the tracking error |

`--explore` writes to `.data/candidates/exploratory/`, so an exploratory model never replaces an
evaluated one. Each attempt first removes the previous attempt's files in its directory.

| Status | Meaning | Exit code |
|---|---|---|
| `evaluated` | Trained and scored on the test split | 0 |
| `exploratory` | `--explore`: leakage check skipped, validation scored, no gate evaluated | 0 |
| `blocked` | `load_split` refused the split; no model is trained | 1 |
| `failed` | The attempt could not finish: one training class, a row with an unusable label or feature, rows from mixed releases, or a LightGBM error; the reason is recorded | 1 |
| Any status, not tracked | The report is kept but MLflow logging failed; `tracking.json` holds the error | 3 |

A malformed contract, rules or training file exits with code 2 before any attempt starts.

## Results on dataset v1

Run on 2026-11-12 on the code of PR #70 at commit `f229a4c` (after `eval-contract-v3` was
set to `accepted`), in a Linux container (`ghcr.io/astral-sh/uv:python3.13-bookworm` with `libgomp1`, since LightGBM needs
OpenMP; LightGBM 4.7.0, NumPy 2.5.4). To reproduce, after rebuilding the split as in
[the rule baseline](rule-baseline.md#results-on-dataset-v1). PR #70 was squash-merged as
`386d494` on `main`; its data and model implementation is identical to `f229a4c`, so that
main commit provides the merged source for reproduction:

```sh
make train-candidates SPLIT=data/datasets/splits/steps-v1.3+aafae73eb211/split-v3+932b2df410a7.json
```

A second run reproduced every file byte for byte:

| File | SHA-256 |
|---|---|
| `report.json` | `f98da5478d8295288c716c4997c7a6eff7570f985b6a3ce8485980a150f4a793` |
| `predictions.jsonl` | `8c20640d65638b302f3cea24516fc42ff32c8142d96b479426d114750237c825` |
| `classifier.txt` | `b35152b71b4c2504e4066ca3aae334cf65bb1b4dac941f647c7741e9f4f85271` |
| `regressor.txt` | `f19c93359f9484c7604fe1a4edbe45e2d596f4e891b0ca498341337036fe7bb5` |
| `preprocessing.json` | `7abf667d18b147394e9fd51153c8d11a961831d1d4e8aa463ef7cc1235c4fa05` |
| `validation_predictions.jsonl` | `c1bf9555c597282c75b8c9fbbe3f8ecaa15ef67fc607eb62cc0e92312d5ad6ff` |

The files and the MLflow run (`b7afc8c3c2c04ec39ff9accba103e255`, in a local SQLite store) are
in `.data/` on the machine that ran them, not committed. The first runs, on 2026-11-11 at
`6ade73d` under the then-proposed contract (`b9f20820`, `split-v3+95309c38a53e`, same rows),
wrote the same predictions, preprocessing record and model files; their report differed only
in the contract status and hash and the split file and hash. Training used the 1,755 training
steps (48 `mixed` steps train the regressor only), with early stopping on validation for both
models (best iterations: classifier 43, regressor 137). The test split is 3,575
`fix-window-count` steps; 3,564 are scored and 11 joint-only steps are not.

| Gate or measure | Model | Rules (same rows) | Contract |
|---|---|---|---|
| Macro F1 (four classes) | 0.790 | 0.200 | Gain over rules ≥ 0: **pass** |
| Clean steps predicted `none` | 0.991 | 0.731 | ≥ 0.95: **pass** |
| Baseline and clean steps predicted `none` | 0.992 | 0.749 | ≥ 0.95: **pass** |
| Mixed-contention macro F1 (82 steps) | 0.534 | 0.047 | Gain ≥ 0.10: **pass** |
| Excess-latency MAE | 0.076 | — | ≤ 0.10: **pass** (zero predicts 0.194) |
| Calibration, abstention | — | — | `not_evaluated` here; M04 |
| Held-out method, unseen tasks, overhead, latency | — | — | `not_evaluated` |

Per class (single-class steps; no bootstrap interval with one test task):

| Class | Support | Precision | Recall | F1 |
|---|---|---|---|---|
| `none` | 2,923 | 0.954 | 0.983 | 0.968 |
| `cpu_sandbox` | 127 | 0.643 | 0.354 | 0.457 |
| `inference` | 226 | 0.969 | 0.969 | 0.969 |
| `retrieval` | 206 | 0.834 | 0.709 | 0.766 |

Confusion matrix (rows: label; columns: prediction):

| Label | `none` | `cpu_sandbox` | `inference` | `retrieval` |
|---|---|---|---|---|
| `none` | 2,873 | 20 | 2 | 28 |
| `cpu_sandbox` | 76 | 45 | 5 | 1 |
| `inference` | 4 | 3 | 219 | 0 |
| `retrieval` | 58 | 2 | 0 | 146 |

How far these results reach:

- **One test task, uncontrolled.** Every number describes `fix-window-count` replays on one
  laptop and one model. The model and rules are compared on identical rows, but the gap is an
  observed difference, not a measured cause.
- **Timing features carry the labels.** The training leakage check passed, with a highest
  single-feature AUC of 0.974 (`queue_seconds` for `retrieval`) against the 0.98 bound, and
  0.949 for `duration_seconds` on `inference`. `step-features-v1` reads no GPU, inference-server
  or retrieval metric, so `inference` and `retrieval` are recognized through timings alone.
- **`cpu_sandbox` is the weak class.** 76 of its 127 steps are predicted `none`.
- **Mixed steps.** All 82 name a confirmed resource (hit rate 1.0), almost always `inference`;
  the mixed macro F1 is low because `cpu_sandbox` is rarely named (4 of 82).
- **Validation is shared.** The same 273 validation steps chose both early-stopping iterations
  and, in M04, the calibration; see [ADR 0012](adr/0012-split-v3-held-out-a06-evidence-and-validation.md).

## Tuning and training-v2

```sh
make tune-candidates SPLIT=$SPLIT   # validation only; writes .data/tuning/{report.json,training-v2.toml}
make train-candidates SPLIT=$SPLIT  # training.toml (training-v2), scored once on test
make calibrate SPLIT=$SPLIT
```

**Declared before any run** (`tuning.toml`, `tuning-v1+f729092b`, committed in `aed9f8c`):

- **Search space:** 162 grid points, `learning_rate` 0.03 / 0.05 / 0.1, `num_leaves` 7 / 15 / 31,
  `min_data_in_leaf` 5 / 10 / 20, `lambda_l2` 0 / 1 / 5 and `feature_fraction` 0.8 / 1.0.
  `training-v1`'s settings are a grid point.
- **Selection, validation only:** lowest multiclass log loss on the 268 single-class validation
  steps for the classifier, lowest excess-fraction MAE on all 273 validation steps for the
  regressor. Log loss, not macro F1, because validation holds 8 `cpu_sandbox`, 12 `retrieval` and
  14 `inference` steps, so macro F1 moves in large steps. The classifier and regressor train
  independently, so each takes its own best point from the same grid.
- **Noise margin:** the best point replaces `training-v1` only when it improves the
  `training-v1` value by at least 2% (relative); ties go to the simpler setting.
- **Test:** `tune-candidates` never reads it (a test asserts only train and validation are
  loaded). The proposal became `training.toml` (`training-v2`, `a6fad7c`) before test was scored.

Every grid point is a nested MLflow run under one parent in the `agentwatch-tuning` experiment
(parent `ed393fcc226e4d3c8303f3628fcc1a0b`, local); failed points would be kept with their reason.
None failed. That run predates `7a3a6f6`, which changed `select()` for an unmeasured base point and
the limitations text; a re-run with the shipped code (parent `b9e3cae389ab43d0bfdd1be82520688f`)
reproduced every point's validation values and the selection, and its proposal is byte-identical
to the committed `training.toml`.

**Validation results** (run of 2026-11-15, CPU in a Linux container):

| Model | Measure | `training-v1` | Selected | Improvement | Settings |
|---|---|---|---|---|---|
| Classifier | log loss | 0.1579 | 0.1462 | 7.4% | 31 leaves, learning rate 0.1, L2 5, feature fraction 0.8, min data 5 |
| Classifier | macro F1 (reported) | 0.796 | 0.816 | | |
| Regressor | MAE | 0.0524 | 0.0476 | 9.3% | 15 leaves, learning rate 0.03, L2 0, feature fraction 1.0, min data 10 |

Log loss over the 162 points ranges from 0.146 to 0.175. **Sensitivity** (median validation value
over every point with each setting; the full min / median / max is in the report's
`sensitivity`): for the classifier, `lambda_l2` (0.161 at 0, 0.153 at 5) and `num_leaves` (0.153
at 7, 0.161 at 31) matter most, and `learning_rate`, `min_data_in_leaf` and `feature_fraction`
move the median by under 0.003. The selected point has 31 leaves although 7 leaves has the best
median: the minimum is one point on a small split. For the regressor no setting moves the median
MAE by more than 0.0014.

**Test results, scored once after `training-v2` was committed** (same 3,564 scored test steps,
`eval-contract-v3`):

| Measure | `training-v1+a34ea0be` | `training-v2+fd027e8a` |
|---|---|---|
| Macro F1 (rules-v1: 0.200) | 0.790 | 0.787 |
| Clean steps predicted `none` | 0.991 | 0.989 |
| Excess-latency MAE (limit 0.10) | 0.076 | 0.077 |
| Gates | 5 pass, 6 `not_evaluated` | 5 pass, 6 `not_evaluated` |
| Calibration (M04) | T 0.897, threshold 0.647, ECE 0.0178 (uncalibrated 0.0074), 2.4% abstain | T 0.807, threshold 0, ECE 0.0207 (uncalibrated 0.0053), none abstain |

The validation gains did not carry over to test: `training-v2` is slightly worse on every
measure, by amounts that one test task cannot separate from noise (no interval is defined with
one task group). Selection stays the declared validation one; the test result is reported, not
used to switch back, since choosing between configurations on test would be tuning on it. On
`training-v2`, M04's validation search found no abstention threshold that improved its
objective, so it abstains on nothing. The frozen held-out pair (`score-held-out`) stays
`training-v1+a34ea0be`. MLflow runs (local): candidates `3b6b8b327de74cc8a2a35e048ba9986e`,
calibration `3b4f8fdcc94e4dcc8860d21064dcf8d6`.

Limitations: validation is 273 steps from 3 experiments of one task and also stops training
early, so every point's validation value is optimistic; test is one unseen task; training has
one task group. A second task group in the next release is needed before tuning results are
read as more than descriptive.

## Result figures and SHAP

```sh
make model-analysis SPLIT=$SPLIT   # default: selected=.data/candidates; add others as name=folder
```

`model-analysis` reads evaluated `train-candidates` folders and, when present, the tuning report,
checks that every report was scored on the `--split` file's rows with the files it recorded (classifier and predictions hashes, step count), and writes to `.data/analysis/`: a confusion matrix per model and for `rules-v1`
(`confusion-*.png`), a bar chart of macro F1 and the clean-step `none` rate on the same test rows
(`comparison.png`), the explained model's SHAP importance (`shap-<name>.png`), each setting's
validation sensitivity (`sensitivity.png`, median with min to max over the grid), and
`analysis.json` with every number and each figure's SHA-256. It trains, fits and selects nothing.
SHAP values are LightGBM's exact TreeSHAP contributions (`predict(pred_contrib=True)`), which add
up to each class's raw score (a test checks this); importance is the mean absolute value per
feature over the test rows, averaged over classes. Equal inputs give byte-identical figures on the same platform and Matplotlib version (checked within one run, not across platforms).

The same steps run in Colab from a fresh clone with
[`pipelines/model/notebooks/model_pipeline_colab.ipynb`](../pipelines/model/notebooks/model_pipeline_colab.ipynb):
it pulls dataset v1 from the public DVC remote, rebuilds `split-v3` and checks its SHA-256, then
runs (optionally) tuning, training, calibration and the figures with the repository's commands.

Run of 2026-11-15 on the 3,564 scored test steps:

| | `rules-v1` | `training-v1` | `training-v2` |
|---|---|---|---|
| Macro F1 | 0.200 | 0.790 | 0.787 |
| Clean steps predicted `none` | 0.731 | 0.991 | 0.989 |

SHAP importance of `training-v2`'s classifier (mean |SHAP|, raw score): `duration_seconds` 0.233,
`execution_seconds` 0.219, `queue_seconds` 0.209, `queue_fraction` 0.123, then
`cpu_pressure_max_avg10` 0.033 and `cpu_seconds` 0.032; `telemetry_coverage` and
`cpu_pressure_available` are below 0.01, and the always-missing workload features contribute 0.
Its confusion matrix shows where it fails: 60% of `cpu_sandbox` and 29% of `retrieval` test steps
are predicted `none`, while 97% of `inference` steps are found.

**Why `training-v2` is the selected model.** It is the declared validation choice: both its
boosters beat `training-v1` on validation by more than the 2% margin set before the search. On
test it is level with `training-v1` within noise (macro F1 0.787 against 0.790), and both clear
every evaluated gate by the same margins. Switching back to `training-v1` because of the test
result would be selecting on test. For held-out scoring the frozen pair remains `training-v1`,
since that is the pair the A06 sweeps were scored with (#76). Neither is promotable: four gates
are `not_evaluated`.

## Gates

- Clean-step `none` rates, as for the rules.
- Gain over rules: the model's four-class macro F1 point estimate must not be lower than the
  rules' on the same rows; inconclusive while a contract class is absent from training, a class
  lacks test steps, or the four-class value is undefined.
- Regression error: MAE at most the contract limit and below both constant references;
  inconclusive while a class lacks test steps.
- Coverage gaps are reported, not hidden: validation rows of classes unseen in training (which
  cannot drive the classifier's early stopping), mixed training steps the classifier skipped,
  and classes absent from training.
- Mixed-contention gain: the model's mixed-contention macro F1 (per-resource F1 on scored mixed
  steps, over the resources confirmed there) at least 0.10 above the rules' on the same steps;
  inconclusive below the contract's 30 mixed steps or when a confirmed resource is absent from
  training.
- Joint-only mixed steps are left out of every gate, including the regression error and its
  constant references.
- Calibration and abstention are left to [M04](model-calibration.md); raw LightGBM probabilities are not calibrated
  confidence. Unseen tasks, held-out injection methods, overhead and latency are not evaluated,
  with the same reasons as the rule baseline.

## Held-out A06 scoring

```sh
make agent-capacity-sweep-steps SWEEP=data/held-out/a06-20261111/capacity-sweeps/<sweep folder>
make sweep-truth STEPS=.data/a06-scoring/steps/<sweep folder>
uv run --package agentwatch-data python -m agentwatch_data.splits \
  --release data/datasets/steps-v1.3+aafae73eb211
SPLIT=data/datasets/splits/steps-v1.3+aafae73eb211/split-v3+932b2df410a7.json
make train-candidates SPLIT=$SPLIT    # M03, into .data/candidates
make calibrate SPLIT=$SPLIT           # M04, into .data/calibration
make score-held-out TRUTH=".data/a06-scoring/truth/<sweep folder> ..." SPLIT=$SPLIT
```

`score-held-out` scores `rules-v1` and the frozen M03 classifier, with its M04 temperature and
abstention threshold, on the `sweep-truth-v2` truths of the frozen A06 capacity sweeps
([design](superpowers/specs/2026-11-11-a06-held-out-scoring-design.md), "Change after #70, #72
and #74"). It trains and fits nothing. A model pair whose M03 report does not name the
`--split` file and `eval-contract-v3`, or whose files do not hash to the values M03 and M04
recorded, is a setup error (exit 2) and no attempt is written. So is any pair but the pinned
one: M03's preprocessing and report (`model.version`) and M04's `calibration.json` must name
`training-v1+a34ea0be`, and `calibration.json` must be `calibration-v1+08119e78`; there is no
option to accept another. M04's `report.json` must also carry `calibration.json`'s version,
temperature and abstention threshold, name the `--split` file, hash the `predictions.jsonl`
beside it, and record positive `fit.steps` and `scored_steps`, and `calibration.json` must hash
M03's `report.json`. Every M04 file enters the attempt's input hashes, keyed `candidates/<file>`
and `calibration/<file>`. A truth folder whose
`label_version` is not `sweep-truth-v2+c211f3e8`, whose `truth_sha256` is not the hash of the
`truth.jsonl` beside it, whose step and recorder hashes are not 64 lowercase hex digits, or with
a record whose `plan_sha256` is absent or neither null nor 64 lowercase hex digits, a
booster that does not reproduce M03's validation predictions, a pair that does not reproduce
M04's test predictions, and a scored run or plan found in any row of the split each end the
attempt as a recorded `failed` one. Neither reproduction check can pass on an empty or partial
record: M03's `validation_predictions.jsonl` must hold exactly the `fit.steps` M04 fitted on,
and every one must match (label, and each class probability within 1e-9); M04's
`predictions.jsonl` must hold exactly the steps its report counts (`scored_steps` plus
`joint_only_steps_not_scored`, as M04 records a prediction for every test step), each a
test-split step, and every one must match after the same temperature step held-out scoring uses (label, abstention,
calibrated confidence and each calibrated class probability within 1e-9). It builds `steps-v1.3`
feature rows from each run's journal and scores model calls whose truth is a class; root, tool
and retrieval steps, and model calls the rule leaves null, are counted by reason. The split's
rows are read only for those checks (all three splits for the leakage check, validation and
test for the two reproduction checks) and to decide which task groups are seen (train). The
regressor is not scored. The A06 evidence is never used for training, calibration or tuning,
and there is no gate: the report is a descriptive baseline of miss, false-alarm and abstention
rates, with 95% intervals that resample whole batches. Its limitations include one computed
from the rules: `rules-v1` covers only `none` and `cpu_sandbox` and never predicts
`inference`, so its `inference` miss rate is 1.000 by construction, not a measured weakness.

The report's `feature_availability` shows, for every model feature in `FEATURES` order, how
the scored held-out steps (overall and per configuration) compare with the train split: `rows`,
`missing_fraction` (null or NaN, the values the classifier receives as NaN; a measured zero is
present), `distinct_values` among present values, and `most_common_fraction`, the share of
present values equal to the most common one (1.0 for a constant feature). A group with no rows
or no present value has null fractions, never 0. It describes the inputs and changes no score.

Each attempt writes `.data/a06-scoring/scores/<attempt id>/` (`report.json`,
`predictions.jsonl`, `tracking.json`) and is logged to the separate MLflow experiment
`a06-held-out-scoring`, failed attempts included. The attempt id is the UTC start time plus the
first digits of the hash of every input, the model files included. An attempt never reuses a
folder: one started with the same id (identical inputs in the same second, or two runs at once)
gets `<attempt id>-2`, then `-3`, each created with an exclusive `mkdir`, so no attempt
overwrites another's files. Exit codes: 0 completed, 1
failed (the reason is in the report), 3 not tracked, 2 for a malformed configuration, a model
pair that does not match, an environment where LightGBM cannot load, or an output inside
`data/held-out`. Scores stay local and belong in no release. Attempt
`20261113T011157Z-f2e0c7` (2026-11-13), with this classifier (`b35152b7…`) and
`calibration-v1+08119e78`, is recorded in the
[capacity record](capacity-record.md#attribution-baseline). It has the same scores and
predictions as the 2026-11-12 attempt `20261112T222540Z-d3316f`, which ran before these checks. The earlier attempt
`20261112T025531Z-220363`, which retrained a classifier on `split-v2` and used the former
`sweep-truth-v1` truths, is listed there as superseded.

### Feature shift on held-out rows

```sh
make held-out-feature-availability TRUTH=".data/a06-scoring/truth/<sweep folder> ..." SPLIT=$SPLIT
```

`held-out-feature-availability` extends the report's `feature_availability` above, with the same
counting (`held_out_features.availability`), in four ways:

- **Train rows:** the `split-v3` train rows the classifier was fitted on (single-class steps;
  mixed steps are counted as `train_not_classified`), where `feature_availability` uses every
  train-split row.
- **Gain share:** each feature's share of the frozen classifier's split gain.
- **Flags:** `missing_on_held_out_only` (some train value present, every held-out value
  missing) and `constant_on_held_out_only` (train values vary, held-out values are constant);
  no threshold is tuned.
- **Value range:** p05, p50 and p95 of the present values (linear interpolation), the minimum
  and maximum, and the shares of held-out values below the train minimum and above the train
  maximum.

It reads the model through the same checked loader as `score-held-out` and runs the same
validation-reproduction, calibration-reproduction and split-overlap checks; it scores, trains and
fits nothing, and writes `feature-availability.json` and two Markdown tables to
`.data/a06-scoring/feature-availability/` (exit 2, and nothing written, for a mismatched model pair,
a pair that does not reproduce M03's validation predictions or M04's test predictions, a held-out
run or plan found in the split, no judged step, a refused split or an output inside
`data/held-out`). The same inputs give byte-identical output.

Run of 2026-11-12 on the 1,740 scored model calls (825 `cfg-default`, 915 `cfg-hardware`) against
the 1,707 single-class train rows (48 mixed train steps not classified), with classifier
`b35152b7…` and `calibration-v1+08119e78`, truths `92da54bc…` and `069a8303…` (report SHA-256
`6229fc8960a50e14…`, local):

- **Present everywhere:** `execution_seconds`, `queue_seconds`, `duration_seconds` and
  `queue_fraction` hold 0.894 of the gain and are present on every train and held-out row.
- **Missing on every held-out row:** `cpu_seconds` (gain 0.070), `telemetry_coverage` (0.019),
  `cpu_pressure_max_avg10` (0.018) and `cgroup_throttled_usec_delta` (0.000), 0.106 of the gain
  together. They are mostly missing in train as well (84.2%, 98.1%, 89.6% and 99.2%).
- **Constant on held-out:** `cgroup_available`, `cpu_pressure_available` and
  `telemetry_sample_count`, with no gain.
- `workload_iterations`, `tests_per_step` and `worker_capacity` are missing on every row of both.

Value ranges of the four timing features, train against held-out (p05 / p50 / p95, seconds or
fraction):

| Feature | Train | Held-out | Held-out above train max |
|---|---|---|---|
| `execution_seconds` | 0.00065 / 0.153 / 0.952 | 0.248 / 0.706 / 3.51 | 0.8% |
| `duration_seconds` | 0.00159 / 0.154 / 0.954 | 0.251 / 0.712 / 3.52 | 0.8% |
| `queue_seconds` | 0.00045 / 0.00112 / 0.0214 | 0.00063 / 0.00302 / 0.0259 | 0.0% |
| `queue_fraction` | 0.00069 / 0.0114 / 0.770 | 0.00037 / 0.00394 / 0.0382 | 0.0% |

No held-out timing value is below the train minimum. The held-out median `execution_seconds` is
1.71 s on `cfg-default` and 0.43 s on `cfg-hardware`.

So the high-gain timing features are present on every held-out row, and the features missing on
every held-out row were mostly missing in training too. The held-out model calls are slower
(median `execution_seconds` 0.71 s against 0.15 s, p95 3.5 s against 0.95 s) and spend a smaller
fraction of their time queued than the train steps, yet almost all of them lie inside the train
range: they sit where train rows were sparse rather than outside it. No test was made of what
missing inputs or this shift do to the held-out misses (queue 0.288, slower 0.599); LightGBM
routes a missing value down a learned default branch, so the 0.106 of gain on all-missing
features can still matter. Train rows are mostly Ollama runs and held-out rows vLLM runs, so
these are different workloads as well as different servers.

## Tracking

Every attempt is an MLflow run in the `agentwatch-attribution` experiment, tagged with its
status and reason, with the training, rules, contract, dataset and label versions as
parameters, measured metrics (unmeasured values are omitted, not logged as zero), and the
output files as artifacts. Only the named output files are logged, never other files in the
directory or the `exploratory/` subdirectory; `tracking.json` is written after logging, since it
records the outcome, so it is never an artifact. The default backend is local SQLite at `.data/mlflow/mlflow.db`
with artifacts in `.data/mlflow/artifacts`; `--tracking-uri` points elsewhere. Failed,
blocked and exploratory runs are kept.

For a `sqlite:///` URI the database's parent directory is created if missing, and MLflow
applies its own schema migrations to that file. Every file in the output directory is logged as
an artifact, including `predictions.jsonl` and `validation_predictions.jsonl` with per-step
labels and probabilities, so a `--tracking-uri` on a remote server uploads them there. Keep
the default local backend unless that server is meant to hold this data.

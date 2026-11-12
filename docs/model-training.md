# AgentWatch model training

Status: `training-v1+a34ea0be`, trained and evaluated on 2026-11-11 on dataset v1
(`steps-v1.3+aafae73eb211`, split `split-v3+95309c38a53e`, `eval-contract-v3` `b9f20820`,
proposed); see [Results on dataset v1](#results-on-dataset-v1). It passes every gate it
evaluates and blocks promotion only because six gates are `not_evaluated`. Earlier: the interim
`steps-v1.2+5954f97ef8b2` split is blocked by the leakage check, and on `split-v2+febe942a2b59`
a run is a `failed` attempt because that test split mixes two feature definitions. Steps
labeled `mixed` are scored by `mixed-scoring-v1` (see Running it).

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

The defaults are the current contract (`eval-contract-v2`) and frozen schema
(`steps-v1.3.pbtxt`). To rerun on a `steps-v1.2` split, pass the files it was made with:
`EVALUATION_CONTRACT=pipelines/model/evaluation-contract-v1.toml
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

Run on 2026-11-11 in a Linux container (`ghcr.io/astral-sh/uv:python3.13-bookworm` with
`libgomp1`; LightGBM 4.7.0, NumPy 2.5.4), with
`python -m agentwatch_model.train_candidates --split <split-v3+95309c38a53e.json>` and the live
contract. A second run reproduced the report, predictions and both model files byte for byte:

| File | SHA-256 |
|---|---|
| `report.json` | `fa7ea335d108a3b06dfa407a70ec5e6e53ef28f29c9542d18598465ead8e45de` |
| `predictions.jsonl` | `8c20640d65638b302f3cea24516fc42ff32c8142d96b479426d114750237c825` |
| `classifier.txt` | `b35152b71b4c2504e4066ca3aae334cf65bb1b4dac941f647c7741e9f4f85271` |
| `regressor.txt` | `f19c93359f9484c7604fe1a4edbe45e2d596f4e891b0ca498341337036fe7bb5` |

The files and the MLflow run (`5218bafb5cb949d0aebd9989ac9179e8`, in a local SQLite store) are
in `.data/` on the machine that ran them, not committed. Training used the 1,755 training
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

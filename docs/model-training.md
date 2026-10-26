# AgentWatch model training

Status: `training-v1`, implemented and tested on synthetic rows; not yet trained on an evaluable
release. The interim `steps-v1.2+5954f97ef8b2` split is blocked by the leakage check, so a run
on it produces a blocked report and a blocked MLflow run, with no model.

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

- Both train on the training split only. The validation split is used only for early stopping;
  without usable validation rows the fixed number of rounds is used and recorded.
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

LightGBM needs OpenMP: `libgomp` on Linux (present on CI runners) and `libomp` on macOS. Without
it the command exits with code 2 and says so, and the LightGBM tests skip with that reason.

Outputs go to `.data/candidates/`:

| File | Contents |
|---|---|
| `classifier.txt`, `regressor.txt` | Trained LightGBM models at their best iteration |
| `preprocessing.json` | What serving must reproduce: feature order, missing-value and boolean handling, class order (the probability columns), objectives, best iterations, the excess-fraction clip range, model file hashes, LightGBM version, dataset and label versions, and the attempt's status and evaluation flag |
| `predictions.jsonl` | Per scored step: identity, condition, label, model class and per-class probabilities, excess fraction, and the rule prediction |
| `validation_predictions.jsonl` | The same for the validation split, for M04 calibration. Validation also drove early stopping, so calibration fitted on it is slightly optimistic |
| `report.json` | Model and rule metrics on the same rows, regression MAE, coverage gaps, gates and summary; written last, so its presence marks a complete attempt |
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

## Gates

- Clean-step `none` rates, as for the rules.
- Gain over rules: the model's four-class macro F1 point estimate must not be lower than the
  rules' on the same rows; inconclusive while a contract class is absent from training, a class
  lacks test steps, or the four-class value is undefined.
- Regression error: MAE at most the contract limit and below both constant references;
  inconclusive while a class lacks test steps.
- Coverage gaps are reported, not hidden: validation rows of classes unseen in training (which
  cannot drive early stopping) and classes absent from training.
- Calibration and abstention are left to M04; raw LightGBM probabilities are not calibrated
  confidence. Mixed contention, unseen tasks, held-out injection methods, overhead and latency
  are not evaluated, with the same reasons as the rule baseline.

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

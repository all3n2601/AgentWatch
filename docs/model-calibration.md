# AgentWatch confidence calibration and abstention

Status: `calibration-v1`, implemented and tested on synthetic fixtures (M04); not yet run on an
evaluable release. On `split-v2` the validation split is empty, so the calibration is not fitted
and the calibration and abstention gates are inconclusive.

Raw LightGBM class scores are not calibrated confidence. M04 calibrates the M03 classifier on the
validation split only, chooses an abstention threshold there, and scores both once on the test
split against [the evaluation contract](evaluation-contract.md) (`[calibration]`).

## Method

Settings live in `pipelines/model/src/agentwatch_model/calibration.toml`. They were fixed before
any calibration was fitted, and the version is `calibration-v1` plus the first 8 hex digits of
the parsed file's SHA-256.

| Step | Rule | Basis |
|---|---|---|
| Calibration | Temperature scaling: divide the log class scores by one temperature, fitted by minimizing validation log loss within 0.05 to 20. It never changes the predicted class | Proposed |
| Minimum fit size | At least 100 usable validation steps (rows whose class the model was trained on); fewer leaves the model uncalibrated | Proposed |
| Abstention | A step abstains when its calibrated top-label confidence is below the threshold | Contract |
| Threshold | Of zero and every distinct validation confidence, the one with the best validation macro F1 (defined classes, abstentions counted as misses) that keeps abstention at or below 20% overall and for every true class; ties keep the lowest | Proposed |

The validation split also chose M03's early-stopping iteration, so a calibration fitted on it is
slightly optimistic. The report records this as a limitation; the test-split ECE gate is the
check, and an optimistic calibration shows up there as a failure rather than being hidden.

## Running it

```sh
make train-candidates SPLIT=data/datasets/splits/<release>/<split>.json
make calibrate SPLIT=data/datasets/splits/<release>/<split>.json
```

`make calibrate` reads `.data/candidates/` and re-reads the validation and test splits through
`load_split`. It refuses, with exit code 2 and no report, when:

- there is no candidates report, or a candidates file cannot be read;
- the split file is not the one the candidates report names, or the contract has changed since
  training (its hash differs);
- `predictions.jsonl`, `validation_predictions.jsonl` or `preprocessing.json` no longer
  matches the hash in the candidates report, the model file hashes in `preprocessing.json`
  differ from the report's, or the report is not an evaluation of the test split;
- prediction records do not cover exactly the split's rows, repeat a step, record a different
  label than the split, or hold probabilities that are not a distribution over the
  `preprocessing.json` columns;
- a split row fails the rule-baseline row checks (missing field, unknown label, mixed release).

Candidates that were not `evaluated`, or a split `load_split` refuses, give a `blocked` report
(exit 1). Every attempt with a report is logged to MLflow under task `M04` with the calibration
version and the M03 run id; an attempt that could not be logged exits 3 and keeps its report.
Each attempt with a report replaces the directory's earlier outputs, as for the candidates;
MLflow keeps every attempt's report, predictions and calibration file.

## Outputs

`.data/calibration/`, written in this order, so the report's presence marks a complete attempt:

| File | Contents |
|---|---|
| `predictions.jsonl` | Per test step: identity, label, the model's class (`prediction`), `attribution` (null when abstained), `abstained`, calibrated `confidence`, `raw_confidence`, calibrated `probabilities` (both calibrated fields null when unfitted), whether the step was `calibrated`, and the calibration version |
| `calibration.json` | What serving needs: `fitted`, `temperature`, `abstention_threshold`, the abstention rule, probability columns, the M03 model file hashes and training version, the candidates report hash and run id, and the contract identity, calibration and abstention gate outcomes and `blocks_promotion`. Serve it only when `blocks_promotion` is false. When `fitted` is false, temperature and threshold are null |
| `report.json` | Calibration, fit and metrics, every gate, and the summary |
| `tracking.json` | Whether the attempt reached MLflow |

A consumer shows an abstained step as uncertain: `attribution` is null and `prediction` keeps
the model's best guess for inspection only.

## Report and gates

- Calibration: top-label ECE over all test steps before abstention, 10 equal-width bins, before
  and after calibration; at most 0.05 passes.
- Abstention: rates overall and per true class, at most 20% each. The contract's mixed-contention
  scope is unmeasured until M02 scores mixed steps, so the gate is at best inconclusive; any
  measured rate above the limit fails it.
- Model per-class scores and macro F1 with abstentions counted as misses, beside the rules'.
- Clean-step `none` rates and the gain over rules, recomputed with abstentions as misses.
- The regression error and every other gate calibration cannot change, carried from the
  candidates report with their reasons.
- Slices: by experiment condition, and steps with any missing feature against complete steps,
  with a per-feature count of missing readings (accuracy, abstention rate and calibrated ECE).
- Calibration and abstention are inconclusive while the calibration is unfitted. A measured
  failure (ECE over the limit, or abstention over the limit overall or for a class with at
  least the contract's minimum test steps) fails the gate; otherwise a class below that
  minimum leaves it inconclusive. Any gate other than pass blocks promotion.

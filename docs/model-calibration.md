# AgentWatch confidence calibration and abstention

Status: `calibration-v1`, fitted and evaluated on 2026-11-11 on dataset v1
(`split-v3+95309c38a53e`, `eval-contract-v3` `b9f20820`, proposed); see
[Results on dataset v1](#results-on-dataset-v1). Every gate it evaluates passes. Temperature
scaling made the test-split ECE slightly worse, not better. On `split-v2` the validation split
is empty, so no calibration is fitted there.

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
| Threshold | Of zero and every distinct validation confidence, the one with the best validation macro F1 (single-class steps, defined classes, abstentions counted as misses) that keeps abstention at or below 20% overall, for every true class and on scored mixed steps; ties keep the lowest | Proposed |

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
  `preprocessing.json` columns (validation records are checked even when too few to fit);
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

## Results on dataset v1

Run on 2026-11-11 right after the [M03 run](model-training.md#results-on-dataset-v1), in the same
Linux container, with `python -m agentwatch_model.calibrate --split <split-v3+95309c38a53e.json>
--candidates .data/candidates`. A second run reproduced the predictions and every value; only
the recorded MLflow run IDs differ (`4203a3279a254b71825a87d8ff972dc1` for this one, local and
not committed). `predictions.jsonl` SHA-256:
`9f6b03fed9fe23e3263df413a311b73b2129f5aeffe3cd91b80fd2de61a8c70d`.

**Fit on validation** (273 steps, 268 usable): temperature 0.897 (below 1, so it sharpens the
scores), abstention threshold 0.647 on calibrated top-label confidence. On validation the
threshold abstains on 1.8% of steps, none of them mixed, with macro F1 0.809.

**Test split** (3,564 scored steps; 11 joint-only steps not scored):

| Gate or measure | Result | Contract |
|---|---|---|
| Top-label ECE, calibrated | 0.0178 (uncalibrated 0.0074) | ≤ 0.05: **pass** |
| Abstention overall | 2.4% (85 steps) | ≤ 20%: **pass** |
| Abstention per true class | `none` 1.0%, `cpu_sandbox` 6.3%, `inference` 2.7%, `retrieval` 19.9% | ≤ 20% each: **pass** |
| Abstention on mixed steps | 0% (82 steps) | ≤ 20%: **pass** |
| Macro F1, abstentions as misses | 0.780 (rules 0.200) | Gain ≥ 0: **pass** |
| Clean, and baseline-and-clean, `none` rates | 0.987 and 0.989 | ≥ 0.95: **pass** |
| Mixed-contention macro F1 | 0.534 (rules 0.047) | Gain ≥ 0.10: **pass** |
| Excess-latency MAE (from M03) | 0.076 | ≤ 0.10: **pass** |

By condition (ECE calibrated, then uncalibrated): baseline 0.015 vs 0.023, clean 0.015 vs
0.022, contended 0.083 vs 0.066, mixed steps 0.047 vs 0.067. Accuracy before abstention is 0.944
overall and 0.835 on contended steps.

What this shows, without claiming more:

- **Calibration did not help overall.** The raw scores were already within the limit, and the
  fitted temperature, chosen on a validation split that is 86% `none`, made contended steps
  more over-confident. The gate passes because of the raw scores, not the calibration.
- **`retrieval` abstention is at the limit.** 41 of 206 `retrieval` steps abstain (19.9%,
  limit 20%); one more would fail the gate.
- **Weak evidence.** Validation is 273 steps from three experiments of the training task, and it
  also chose M03's early-stopping iterations, so the fit is optimistic (the report says so).
  Test is one unseen task on one laptop and one model. See
  [ADR 0012](adr/0012-split-v3-held-out-a06-evidence-and-validation.md).
- **Not shown yet.** Showing uncertain results to users belongs to serving and the dashboard.

## Report and gates

- Mixed steps follow `mixed-scoring-v1` ([rule baseline](rule-baseline.md)): a prediction on a
  `mixed` step is correct when it names any resource in `label_resources`; steps with basis
  `mixed_joint_only` are never scored, and the report counts them apart
  (`joint_only_steps_not_scored`, `slices.mixed_contention.joint_only`). The four-class metrics
  cover single-class steps only. The abstention threshold's macro F1 is judged on single-class
  validation steps, as the classifier's fit is, but the 20% limit also covers scored mixed
  validation steps (the contract's mixed-contention scope; validation limits the same
  `abstention_rate_scopes` the test gate reads), so a threshold that would abstain
  on more of them is never chosen; `fit.threshold_selection` records the mixed rate. Unlike the
  test gate, validation applies the limit at any step count, as it does per class: with three
  mixed validation steps, one abstention (33%) rules a threshold out. The recorded overall
  validation rate counts scored mixed steps too.
- Calibration: top-label ECE over all scored test steps before abstention, 10 equal-width bins,
  before and after calibration; at most 0.05 passes. A hit on a mixed step counts as correct.
- Abstention: rates overall (every scored step), per true class and for the mixed-contention
  scope (scored mixed steps), at most 20% each. A rate is judged only on at least the
  contract's minimum steps (per class, or the minimum mixed steps); a measured rate above the
  limit fails the gate, and a mixed scope below the minimum leaves it inconclusive.
- Model per-class scores and macro F1 with abstentions counted as misses, beside the rules'.
- Clean-step `none` rates, the gain over rules and the mixed-contention gain, recomputed with
  abstentions as misses.
- The regression error and every other gate calibration cannot change, carried from the
  candidates report with their reasons.
- Slices: by experiment condition, and steps with any missing feature against complete steps,
  with a per-feature count of missing readings (accuracy, abstention rate and calibrated ECE).
- While the calibration is unfitted, every calibrated metric (ECE, mean confidence, slice ECE)
  is null and the raw-score values are reported under `_uncalibrated` names; calibration and
  abstention are inconclusive. A measured
  failure (ECE over the limit, or abstention over the limit overall or for a class with at
  least the contract's minimum test steps) fails the gate; otherwise a class below that
  minimum leaves it inconclusive. Any gate other than pass blocks promotion.

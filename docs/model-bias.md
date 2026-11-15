# AgentWatch model bias check

`make bias-check` checks whether the attribution model serves some kinds of steps much worse
than others, decides a mitigation on the validation split, and reports both models on test. The
thresholds and the mitigation trade-off are declared in
[`bias.toml`](../pipelines/model/src/agentwatch_model/bias.toml) (`bias-v1`), committed before
the check was run on any model.

```sh
make bias-check SPLIT=data/datasets/splits/steps-v1.3+aafae73eb211/split-v3+932b2df410a7.json
```

## Slices and measure

The dataset has no demographic attributes. The slices are the conditions a model could serve
unevenly, the model-level counterpart of the data slices (`make slice-dataset`, #48):

| Attribute | Values |
|---|---|
| `condition` | `baseline`, `clean`, `contended` run |
| `telemetry_readings` | `complete`, or `missing_any` of the data slices' nullable readings (telemetry coverage, CPU PSI, cgroup throttling) |
| `contention` | on contended steps with a resource label: `single_class` or `mixed` |

M04's calibration slices count a step as missing when any of the 14 features is null; on agent
replays the workload features are always null, so that definition puts every replay step in one
slice. The telemetry readings are the ones that are sometimes present.

A step is **correct** when the predicted class is its label; a mixed step when the prediction
names one of its confirmed resources (`mixed-scoring-v1`); joint-only mixed steps are never
scored. A slice's **disparity** is the overall correct rate minus the slice's rate. It is
**flagged** when the disparity exceeds 0.10, and **insufficient** (neither flagged nor cleared)
below 30 steps, the data slices' minimum.

## Mitigation and decision

The mitigation trains the same settings with condition-balancing weights (the data slices'
`balancing_weights`: each condition carries the same total weight). It replaces the unweighted
model only when, **on validation**, it lowers the worst disparity among slices with enough steps
by at least 0.02 and raises the classifier's validation log loss by at most 5% (relative), and
only when the unweighted model has a flagged slice. The same run reports both models on test;
the decision rule reads only validation. Both models stop early on the unweighted validation
split, and each disparity is relative to that model's own overall rate (here the weighted
model's overall rate is not lower, 0.945 against 0.941, so its smaller disparity is not an
artefact of a lower overall rate). The trade-off is explicit: up-weighting the smaller
conditions shifts errors toward the dominant clean and baseline steps.

## Results (2026-11-15, `split-v3+932b2df410a7`, `bias-v1+…`)

Unweighted `training-v2+fd027e8a` against condition-weighted `training-v3+71c4dd58` (correct rate,
steps):

| Slice | Validation unweighted | Validation weighted | Test unweighted | Test weighted |
|---|---|---|---|---|
| Overall | 0.941 (272) | 0.945 | 0.943 (3,564) | 0.944 |
| `condition` baseline | 0.962 (105) | 0.962 | 0.992 (1,375) | 0.992 |
| `condition` clean | 1.000 (84) | 1.000 | 0.989 (1,100) | 0.988 |
| `condition` contended | 0.855 (83) | 0.867 | **0.835 (1,089) flagged** | **0.837 flagged** |
| `contention` single_class | **0.824 (34) flagged** | 0.853 | **0.732 (559) flagged** | **0.741 flagged** |
| `contention` mixed | 1.000 (4) insufficient | 1.000 insufficient | 1.000 (82) | 1.000 |
| `telemetry_readings` complete | none | none | 0.952 (21) insufficient | 0.952 insufficient |
| `telemetry_readings` missing_any | 0.941 (272) | 0.945 | 0.943 (3,543) | 0.944 |

**Decision (validation):** weighting lowered the worst disparity from 0.118 to 0.092 (by 0.026,
past the 0.02 margin) and raised validation log loss by 1.9% (within 5%), so the declared rule
adopted the weighted settings as `training-v3`, the packaged `training.toml`. The rule read only
validation; the same run had already reported both models on test (below), so those test numbers
existed when `training-v3` was committed, though they did not enter the decision. **The evidence
is inconclusive:** the flagged slice is 34 validation steps, where one step moves its rate by
0.029, more than the 0.026 reduction, and on test the weighting changes little. MLflow: `agentwatch-bias` run
`be80cffe7a784ba383cf4e927c49874b` (local).

**Test:** both models fail the check on the same slices. Contended steps are attributed correctly
about 0.84 of the time and single-class contended steps about 0.73 to 0.74, against 0.94 overall;
the weighting moves single-class contention from 0.732 to 0.741 and leaves the overall rate
unchanged. The confusion matrices show why: most of those misses are `cpu_sandbox` and
`retrieval` steps predicted `none`. Re-weighting conditions does not add the evidence those
classes lack; the data slices' mitigation for that is more contended runs per class
(`make slice-dataset`), which is the next release's job.

**`training-v3` evaluated** (`make train-candidates` and `make calibrate` after it was
committed; same 3,564 scored test steps, `eval-contract-v3`):

| Measure | `training-v1` | `training-v2` | `training-v3` (weighted) |
|---|---|---|---|
| Macro F1 (rules-v1: 0.200) | 0.790 | 0.787 | 0.790 |
| Clean steps predicted `none` | 0.991 | 0.989 | 0.988 |
| Excess-latency MAE | 0.076 | 0.077 | 0.076 |
| Gates | 5 pass, 6 `not_evaluated` | same | same |
| Calibration | T 0.897, 2.4% abstain | T 0.807, none abstain | T 0.797, none abstain (ECE 0.020, uncalibrated 0.009) |

MLflow (local): candidates `fbd1f6730a4843198ff349879026154d`, calibration
`fa87e3a89b854ea291eade0bc3200ee1`. The three settings are level on test within what one test task
can separate; `training-v3` is the selected model because both of its choices were made, as
declared, on validation. It is not promotable: four contract gates are `not_evaluated`, and the
bias check fails on test.

## Limitations

- Validation, which decided the mitigation, has 273 steps from 3 experiments of one task; its
  single-class contention slice is 34 steps and its mixed slice 4.
- Test is one unseen task; the rates have no interval, and a disparity is a difference in
  observed rates, not a cause.
- Condition slices differ in difficulty by design: contended steps carry the resource labels,
  clean and baseline steps are almost all `none`.
- The complete-telemetry slice is too small to judge (21 test steps).

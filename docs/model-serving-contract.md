# Model serving and prediction contract (M03/M04 handoff)

This is the handoff from M03 (LightGBM candidates) and M04 (calibration and abstention) to
serving (B03, Allen) and the dashboard (Meghana). It says what a server must reproduce to feed
the trained models the inputs they learned from, and what each prediction means. The training
and calibration commands, gates and results are in [model training](model-training.md) and
[model calibration](model-calibration.md); this page does not repeat them.

Current pair: classifier and regressor `training-v1+a34ea0be`, calibration
`calibration-v1+08119e78`, trained on `split-v3+932b2df410a7` of `steps-v1.3+aafae73eb211`
under `eval-contract-v3` (`1cab0cbe`, accepted). It is **not promotable**: `calibration.json`
records `blocks_promotion: true`, because four contract gates are `not_evaluated`: held-out
injection method, unseen tasks, instrumentation overhead and attribution latency. Serve it only
behind a flag that labels it as not promoted, or not at all, until a pair has
`blocks_promotion: false`.
Promotion checks and rollback are M05.

## Artifacts

`make train-candidates` writes the M03 folder and `make calibrate` the M04 folder (defaults
`.data/candidates/` and `.data/calibration/`). They are local outputs, not versioned in DVC or
registered in MLflow's model registry; the same split and code rebuild them byte for byte (see
the reproduction notes in both documents).

| File | Folder | What serving uses it for |
|---|---|---|
| `classifier.txt` | M03 | LightGBM multiclass booster (`objective` `multiclass`) |
| `regressor.txt` | M03 | LightGBM booster for the excess-latency fraction (`regression_l1`) |
| `preprocessing.json` | M03 | Feature order, encoding, class order, clip range, model file hashes, versions |
| `calibration.json` | M04 | Temperature, abstention threshold and rule, gate outcomes, `blocks_promotion` |

Before serving, check that both files' `model_files` SHA-256 values match the model files, that
`calibration.json`'s `training_version` equals `preprocessing.json`'s, that `status` is
`evaluated` in both and `fitted` is true, and that `probability_columns` agree. Refuse to serve
on any mismatch. PR #76 adds `load_frozen_classifier`
(`pipelines/model/src/agentwatch_model/held_out_model.py`), which performs these checks and
can be reused once it merges.

## Inputs

One row per completed agent step. The 14 features, in `preprocessing.json`'s `features` order:

| Feature | Unit or encoding | Source |
|---|---|---|
| `queue_seconds` | seconds | Measured wait before a worker started |
| `execution_seconds` | seconds | Worker start to finish wall time |
| `duration_seconds` | seconds | Submission to finish wall time |
| `cpu_seconds` | seconds, or missing | Process and child CPU time; missing when none was recorded |
| `queue_fraction` | 0–1 | `queue_seconds / duration_seconds` |
| `workload_iterations` | count, or missing | Fixed work size; missing on agent replay rows |
| `tests_per_step` | count, or missing | Tests executed by the tool step |
| `worker_capacity` | count, or missing | Sandbox workers available; missing on agent replay rows |
| `cpu_pressure_max_avg10` | percent 0–100, or missing | Max host CPU PSI `some avg10` in the step window |
| `cpu_pressure_available` | boolean | Whether PSI was readable in the window |
| `cgroup_throttled_usec_delta` | microseconds, or missing | cgroup v2 throttled time gained in the step |
| `cgroup_available` | boolean | Whether a throttling delta was measured |
| `telemetry_sample_count` | count | Resource samples inside the window |
| `telemetry_coverage` | 0–1, or missing | Samples received / samples expected at the measured interval |

Encoding (`pipelines/model/src/agentwatch_model/features.py`, `encode` and `feature_matrix`):

- A missing reading is passed as NaN and never imputed. LightGBM routes NaN down each split's
  learned default branch. A server must never turn a missing reading into 0: a measured zero
  and a missing reading are different inputs.
- Booleans are 1.0 (true) and 0.0 (false); every other value is a finite float64.
- Columns follow `features` order exactly; a row without one of them is refused.
- The definitions, units and clocks are those of the dataset schema
  (`pipelines/data/src/agentwatch_data/schema.py`); a server computing them from live spans
  must use the same windows and clocks as `step-features-v1` ([data card](data-card.md)).

The current pair was trained only on `step-features-v1` rows (agent replays, mostly on Ollama).
On the held-out A06 vLLM sweeps its timing inputs were present but shifted: median
`execution_seconds` 0.71 s against 0.15 s in training (PR #78, not yet merged).

## Outputs

For each row:

1. **Raw class probabilities.** `classifier.predict(features)` gives one probability per class in
   `probability_columns` order (`none`, `cpu_sandbox`, `inference`, `retrieval`). The saved
   model files contain only the best iterations (`best_iterations` in `preprocessing.json`), so
   predict with all trees. `mixed` is never predicted: the classifier learned single-class steps
   only.
2. **Calibrated probabilities** (`calibration.py`, `scale`): clip each probability to
   [1e-12, 1], take its log, divide by `temperature` (0.897 for the current pair), and apply a
   softmax across the classes. Temperature scaling keeps the top class unchanged.
3. **Confidence** is the largest calibrated probability.
4. **Abstention**: the prediction abstains when confidence is below `abstention_threshold`
   (0.647 for the current pair), as `abstain_when` states. An abstaining prediction has no
   attributed resource; show it as uncertain, not as `none`.
5. **Predicted class**: the top class, reported only when the prediction does not abstain.
   `none` means no contended resource was identified for the step, not that its telemetry was
   complete.
6. **Excess-latency fraction**: `regressor.predict(features)` clipped to
   `excess_fraction_clip` ([0, 1]): the estimated share of the step's duration spent as excess
   waiting. It is an estimate, not a measured queue delay; keep it distinct from
   `queue_seconds` and from measured resource saturation.

## What the dashboard must show

- The model identity: `training_version`, calibration `version`, and that the pair is not
  promoted while `blocks_promotion` is true.
- Abstentions as their own state, with the confidence and threshold, never folded into `none`.
- The step's telemetry status and missing inputs (for example `telemetry_coverage` missing or
  `cpu_pressure_available` false), beside the prediction, since missing readings are inputs the
  model saw as missing.
- The excess-latency fraction labelled as an estimate.
- No causal wording: a prediction attributes likely contention; it does not establish a cause.

## Limits of the current pair

From the M03 and M04 results and the held-out A06 scoring (PR #76, not yet merged): one training
task group, one test task, a small validation split shared with early stopping, test ECE worse after
calibration than before (0.0178 against 0.0074), and on the A06 vLLM sweeps a 0.288 miss rate on
queued model calls with 13% abstention. These bound what the dashboard may claim for it.

# A06 PR 3 Held-out scoring of the natural-overload evidence — design

Status: approved in chat 2026-11-11 (Benoy), section by section; implemented and run
2026-11-11 ([capacity record](../../capacity-record.md#attribution-baseline)). One difference
from the text below: live-confirmation runs are not replays and have no plan, so they get no
truth records (the phase `live` never occurs) and are counted as runs without a plan. Branch
`feature/a06-held-out-scoring`, from `main` at `50f393d`. Merge after #69, which moves the
evidence to `data/held-out/a06-20261111/`. Parent design:
[A06 natural overload](2026-11-10-natural-overload-capacity-sweep-design.md).

**Changed 2026-11-12, before the rerun. Proposed for Benoy's approval.** The section
[Change after #70, #72 and #74](#change-after-70-72-and-74-2026-11-12) replaces Stage 1, the
Candidates part of Stage 2, and parts of the record. The run of 2026-11-11 (attempt
`20261112T025531Z-220363`) stays recorded as superseded.

## Intent

Score today's attribution, `rules-v1` and the learned candidates, on the frozen A06 evidence:
two capacity sweeps on the GPU laptop, natural overload plus the held-out prefill flood. The
result is a **baseline**: how often today's rules and models miss inference contention that
vLLM measured, and how often they raise a false alarm on clean steps. Neither has an
inference feature, so the expected result is a blind spot. This PR measures its size, per
configuration and concurrency level, instead of asserting it.

When M2 PR 4 adds inference wait features and its rule, that rule becomes one more scorer in
the same command, under its own version. Nothing here is tuned, promoted or trained on the
A06 evidence.

## Change to the A06 plan (2026-11-11)

The parent design had PR 3 wait for M2 PR 4. M2 PR 4 follows M2 PRs 2 (Qdrant) and 3
(Prometheus), so waiting would keep A06 open for weeks. Benoy decided on 2026-11-11 to score
now without the M2 rule. The parent design records the change. These obligations from it are
unchanged:

- every row keeps the sweep's `protocol.version` and `protocol_sha256`;
- the evidence and every output stay out of `make add-evidence`, `make dataset` and every
  release until `split-v3` exists;
- the sweep's label verdicts validate the noise threshold only, never `labels-v2`.

The truths, measures and the absence of any pass/fail gate below are declared here, before any
scoring runs.

## Scope

1. `capacity-sweep-truth` in `services/agent` writes one truth record per step of a sweep.
2. `score-held-out` in `pipelines/model` builds feature rows for those steps, trains the
   candidates on dataset v1, and scores the candidates and `rules-v1`.
3. One real run on both frozen sweeps, recorded in `docs/capacity-record.md`, with
   `docs/project-checklist.md` and `docs/data-card.md` updated.

Out of scope: new features or rules, `split-v3` or label changes, threshold retuning, GPU runs
and any claim beyond this laptop and these two configurations.

## Stage 1: truth (`services/agent`)

**Refactor.** The per-step decision in `sweep_labels._tally_run` moves into
`step_truths(run_dir, reference)`, which yields one record per step; `_tally_run` counts the
records. Behaviour is unchanged: the existing `sweep_labels` tests pass unmodified, and
`sweep-analyse` on both frozen sweeps reproduces the recorded `label_validation` exactly.

**Command.** `make agent-capacity-sweep-truth SWEEP=<sweep folder>`, which runs
`python -m agentwatch_agent capacity-sweep-truth <sweep folder> [--output <dir>]`.

- Refuses (exit 2, nothing written) a sweep whose protocol version is not
  `natural-overload-sweep-v1`, whose `protocol_sha256` is not one of the two recorded hashes
  (cfg-default `22e26dc7536f…`, cfg-hardware `1608deee2a3b…`, pinned in full in code), or
  whose `held_out` is not true.
- Reads the batches listed in `sweep.json`. Baselines come from the N = 1 ramp runs, exactly
  as `sweep-analyse` builds them. The warm-up batch is skipped.
- Writes `truth.jsonl` and `truth.json` (method `sweep-truth-v1`, sweep id, configuration,
  protocol version and hash, SHA-256 of `sweep.json`, `held_out: true`, counts by truth and
  reason). Default output `.data/a06-scoring/truth/<sweep folder name>/`, never inside
  `data/held-out/`.

**Record per step.** Identity (run, step, step type, plan SHA-256, task), position
(configuration, N, phase `ramp | live | injected | control`, batch sequence), timing (total
seconds, baseline mean, noise threshold, excess) and three truths. Each truth is a class or
`null` with a reason (`span_not_measured`, `no_baseline`, `not_completed`, `malformed`,
`not_applicable`).

| Truth | Steps | `inference` when | `none` when |
|---|---|---|---|
| `queue` | model calls | vLLM's measured queue for the request is above the step's noise threshold, max(0.01 s, 3σ) | it is not |
| `slower` | model calls | the step is slower than its N = 1 baseline by more than the threshold | it is not |
| `protocol_clean` | every step type | never | the step ran in a 1-agent batch (ramp N = 1, and control batches where N_inj = 1) |

- Tool and retrieval steps at N > 1 have no `queue` or `slower` truth (`not_applicable`). They
  are kept and reported as unscored.
- The N = 1 steps form the baseline, so `slower` is partly circular there; for them only
  `queue` and `protocol_clean` are read.
- Flood batches use the measured `queue` truth too, not `inference` by assumption, so the
  flood's actual effect is what gets checked.
- N_inj = 1 for cfg-default (capacity 3) and 4 for cfg-hardware (capacity 8). cfg-hardware's
  control batches are therefore not protocol-clean.

## Stage 2: scoring (`pipelines/model`)

**Command.** `make score-held-out TRUTH="<truth dir> ..."` runs
`python -m agentwatch_model.score_held_out`.

**Feature rows.** For each run in `truth.jsonl` the command reads that run's
`live-journal.jsonl` through `agentwatch_data` (`replay_features`, then `compute_feature_rows`),
the `step-features-v1` definition dataset v1 was built with. The (run, step) join uses
`joined_feature`'s timing check: a step whose feature timings differ from its truth timings is
refused with that code. A journal of another run, or an unreadable one, refuses that run's
steps. Rows keep only the `steps-v1.3` feature columns plus identity; missing readings stay
null and reach LightGBM as NaN.

**Candidates.** Dataset v1, `steps-v1.3+aafae73eb211`, with its split `split-v2+febe942a2b59`:

- `load_split` reads the train and validation splits with the leakage check on. The release's
  test split is never opened (`make train-candidates` refuses it because it mixes two feature
  definitions; this command does not evaluate it).
- `fit_candidate` trains the classifier with `training-v1` settings: one thread, fixed seed,
  deterministic LightGBM. The regressor is not scored: A06 has no excess-fraction truth that
  matches its target.
- Calibration and abstention reuse #61: the temperature (`fit_temperature`) and threshold
  (`choose_threshold`) are fitted on dataset v1's validation predictions under the current
  contract, then applied unchanged to A06. If either cannot be fitted, abstention is reported
  `unavailable` with the reason, and scoring continues without it.
- Recorded identity: training version and config hash, LightGBM version, release, split file
  SHA-256, and the saved models' SHA-256.

**Scorers.** `rules-v1` (`Rules.predict`) and the candidate classifier. Each scorer has a name
and version, so M2 PR 4 adds `rules-v2` as a third without changing the report format.

**Outputs.** `.data/a06-scoring/scores/<attempt id>/`: `report.json`, `predictions.jsonl` (one
line per step and scorer, with the class, confidence and abstention), and `models/`. Every
input has a SHA-256; the report carries `held_out: true` and both protocol hashes. The command
refuses a truth file without `held_out: true`. Attempts, failed ones included, are tracked in
MLflow under the separate experiment `a06-held-out-scoring`, never the training experiment.

## Report measures (declared before scoring)

Broken down by scorer × truth × configuration × N × phase × step type × task seen in training,
and overall. A step's task is **seen** when dataset v1's train split has rows of its task group
(`agentwatch_data.replay_labels.task_group`): `fix-mean-empty` is in train (recorded on Ollama),
while `fix-median-even` is in no dataset v1 split, so it is unseen. The two are never pooled
without also being shown apart. (Added 2026-11-11, Benoy, before any scoring.)

Measures:

- confusion by predicted class (for example `cpu_sandbox` predicted on GPU contention);
- **inference miss rate**: 1 − recall on steps whose truth is `inference`;
- **false-alarm rate**: share of steps predicted other than `none` among steps whose truth is
  `none`, for `queue` and `protocol_clean` separately;
- **abstention rate**, and the miss and false-alarm rates on the steps the candidate answered;
- **95% intervals** by percentile bootstrap with the existing fixed seed and resample count,
  resampling **whole batches**: runs in one batch share their contention, so they are not
  independent. A cell with fewer than `MIN_BOOTSTRAP_GROUPS` batches has no interval. The
  existing `bootstrap_macro_f1` is generalised to take a grouping key and a statistic, with its
  current results unchanged; no second bootstrap is added;
- every refused or unscored step, counted by reason.

No pass/fail gate and no promotion. The report states its limits: one laptop, one sweep per
configuration; dataset v1's replays were recorded mostly on Ollama, A06 on vLLM; no
`steps-v1.3` feature describes GPU or vLLM state; one of the two A06 tasks (`fix-mean-empty`) was
seen in training, so only `fix-median-even` tests an unseen task.

## Tests

Synthetic fixtures, labelled as such:

- `step_truths`: counts match the current tally; each truth's edge cases (span not measured,
  no baseline, not completed, malformed time, flood batch, N_inj = 1 control).
- `capacity-sweep-truth`: refuses another protocol, an unknown hash, a sweep not held out;
  writes nothing on refusal; never writes under `data/held-out/`.
- `score-held-out`: timing-mismatch and foreign-journal refusals; the dataset v1 test split is
  never opened; metrics against hand-counted confusions; batch-grouped intervals; abstention
  `unavailable` path; refusal of a truth file without `held_out`.
- Existing `bootstrap_macro_f1` callers give identical results after the generalisation.
- `.data/a06-scoring/` cannot reach `make add-evidence` or `make dataset`.
- A regression check, skipped when the DVC data is not pulled, that `sweep-analyse` on the
  frozen sweeps reproduces the recorded `label_validation`.

## Real run and record

On the laptop, CPU only: `capacity-sweep-truth` on both frozen sweeps, then `score-held-out`
on both truth folders, from a clean tree at a recorded commit. The results go into
`docs/capacity-record.md` (a new "Attribution baseline" section), with dated entries in
`docs/project-checklist.md` (the A06 "Evaluate natural overload" item closes when the record
exists; the "Compare models and rules on the same held-out tasks and contention cases" item
gains this evidence and stays open for M2 PR 4's rule) and in `docs/data-card.md`. Outputs
stay local in `.data/a06-scoring/`; versioning them is out of scope.

## Change after #70, #72 and #74 (2026-11-12)

Proposed 2026-11-12 for Benoy's approval, before the rerun and before any result that uses
`sweep-truth-v1` is pushed. Three changes landed after the run of 2026-11-11:

- On #70, Rohit and Akshay accepted `sweep-truth-v1` as the held-out set's label rule. They
  required it to land as a versioned rule in `pipelines/data`, reviewed in its own PR, with
  root and tool steps unscored. That is #74.
- Rohit asked for the scoring to use his `split-v3` M03 model and M04 calibration (#72) instead
  of retraining. The same model is then evaluated on dataset v1's test split and on A06, and its
  calibration is valid: none of it was fitted on `split-v2` training rows.
- `split-v3` holds the A06 sweeps out of every split, by protocol hash.

What follows replaces the corresponding parts above. Everything not named here is unchanged:
the feature rows, the batch-grouped bootstrap, the measures and their breakdowns, the absence of
a gate, the outputs, and the separate MLflow experiment.

Updated 2026-11-12: #74 became `sweep-truth-v2+c211f3e8` after Akshay's review; the scorer accepts only that version and requires the three step and recorder hashes.

### Truths come from `sweep-truth-v2` (replaces Stage 1)

- `capacity-sweep-truth`, `services/agent/src/agentwatch_agent/sweep_truth.py`, its Make target
  and its tests are removed. So is the `step_truths` refactor of `sweep_labels` (83be357, and
  807285c, which only repairs that refactor). `sweep_labels` goes back to `main`'s version, so
  `sweep-analyse` is unchanged.
- The truths are produced by #74's two commands: `make agent-capacity-sweep-steps SWEEP=<sweep>`,
  then `make sweep-truth STEPS=<steps folder>`. They write `.data/a06-scoring/truth/<sweep>/`
  with `truth.jsonl` and `truth.json`.
- `score-held-out` refuses a truth folder, writing nothing, unless all of these hold:
  - `truth.json`'s `label_version` is `sweep-truth-v2+c211f3e8`;
  - its `truth_sha256` is the hash of the `truth.jsonl` beside it;
  - its `steps_sha256`, `steps_summary_sha256` and `recorder_sha256` are each 64 lowercase hex
    characters;
  - `held_out` is true;
  - its protocol hash is one of the two accepted ones.
- The report records `label_version`, `steps_sha256`, `steps_summary_sha256` and `truth_sha256`
  for each sweep. Together they identify the truth set.
- **Scored steps:** model calls whose truth is a class. Steps the rule leaves null are counted by
  reason and not scored. That covers root steps (`root_step`), tool, retrieval and other steps
  (`no_queue_reference`), and model calls with `not_completed`, `no_baseline`,
  `span_not_measured` or `not_applicable`. The false-alarm rate on `protocol_clean` is therefore
  over model calls in 1-agent batches only. The run of 2026-11-11 counted every step type.

### The model is M03 with its M04 calibration (replaces Candidates)

- `score-held-out` trains nothing. `train_classifier` and the `--training` and `--calibration`
  rule-file options are removed.
- New options:
  - `--candidates <dir>`: M03's output, default `.data/candidates`, as written by
    `make train-candidates`.
  - `--calibrated <dir>`: M04's output, default `.data/calibration`, as written by
    `make calibrate`.
- Both folders are rebuilt with #72's two commands on
  `split-v3+932b2df410a7`. On 2026-11-12 that rebuild reproduced every M03 file byte for byte
  (classifier `b35152b7…`). M04's `calibration.json` and `report.json` differ only in M03's
  MLflow run ID.
- Before scoring, the command refuses, writing nothing, unless all of these hold:
  - M03's `report.json` names the split file passed with `--split` (by id and SHA-256) and the
    accepted contract `eval-contract-v3` (`1cab0cbe`);
  - `classifier.txt` and `preprocessing.json` hash to the values that report and
    `calibration.json` record;
  - `calibration.json` is `fitted: true` with status `evaluated`, and names that classifier.
- The classifier is applied with M03's `preprocessing.json`: the same feature columns, order and
  class columns. M04's temperature and abstention threshold are applied unchanged. Nothing is
  refitted on A06 or on dataset v1.
- Recorded identity: training version (`training-v1+a34ea0be`), calibration version, contract,
  split id and SHA-256, and the SHA-256 of each model and calibration file. The capacity record
  compares them with #72's published hashes.
- **Leakage check:** the command refuses if any scored run id, or any scored step's plan
  SHA-256, appears in a row of the split file's train, validation or test splits. `split-v3`
  already excludes these sweeps; this confirms it for the rows actually scored.
- **Seen task:** a task group with rows in `split-v3+932b2df410a7`'s train split.
  `fix-mean-empty` is seen; `fix-median-even` has no rows in dataset v1, so it is unseen. The
  split's train rows are read only for this and for the leakage check.
- Abstention is no longer `unavailable`. The report gives the abstention rate, and the miss and
  false-alarm rates on the steps the classifier answered, as declared above.

### Run and record

- One new attempt, from a clean tree at a recorded commit, CPU only:
  - #74's two commands on both frozen sweeps;
  - #72's two commands on `split-v3+932b2df410a7`;
  - `score-held-out` on both truth folders.
- In `docs/capacity-record.md`, the attribution baseline is rewritten from the new attempt. The
  2026-11-11 attempt is listed as superseded, with the reason: a different truth rule, model
  and set of scored steps. Its numbers are not compared with the new ones as an improvement.
- `docs/project-checklist.md`: as declared above, "Evaluate natural overload" is ticked when the
  new record exists, not before. The merge of `main` on 2026-11-12 left it unchecked.
  "Compare models and rules on the same held-out tasks and contention cases" gains the new entry
  and stays open.
- `docs/data-card.md` and `docs/model-training.md` name the new attempt.

### Order

- The code is built on this branch with #74's branch merged in locally, because it reads #74's
  output format. #74 is Benoy's own PR, and the dependency is stated in this PR.
- Nothing is pushed until #74 has merged. Then `main` is merged in, the full checks rerun, and
  the PR opens for Rohit's review.

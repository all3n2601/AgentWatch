# sweep-truth-v1 as a versioned label rule in pipelines/data — design

Status: approved in chat 2026-11-12 (Benoy), section by section. Branch
`feature/sweep-truth-rule`, from `main` at `50f393d`. Reviewer for
the rule: Akshay. Parent designs: [A06 held-out scoring](2026-11-11-a06-held-out-scoring-design.md)
and [A06 natural overload](2026-11-10-natural-overload-capacity-sweep-design.md).

## Why

A06's held-out scoring (branch `feature/a06-held-out-scoring`, not pushed) judges every step of
the frozen natural-overload sweeps with its own truths, `sweep-truth-v1`, computed in
`services/agent`. On #70 (2026-11-12) Rohit and Akshay accepted `sweep-truth-v1` as the held-out
set's label rule on two conditions:

- it lands first as a versioned rule in `pipelines/data`, in its own pull request that Akshay
  reviews, and no results using it are pushed before that;
- tool steps, which have no queue reference, are unscored rather than `none`, and the root
  `agent-run` step is excluded.

This PR does that and nothing else. It reports no miss rates, false-alarm rates or model results.

## Layout

`pipelines/data` must not import `services/agent`, and reading sweep folders is the agent
service's job. The work is therefore split as replay experiments already are: the agent records,
the data pipeline labels.

| Package | Command | Reads | Writes | Decides |
|---|---|---|---|---|
| `services/agent` | `capacity-sweep-steps <sweep folder> [--output <dir>]` | `sweep.json`, scored run folders | `steps.jsonl`, `steps.json` | nothing |
| `pipelines/data` | `make sweep-truth STEPS=<dir> [OUTPUT=<dir>]` | `steps.jsonl`, `steps.json`, `sweep-truth-v1.toml` | `truth.jsonl`, `truth.json` | every truth and reason |

Default outputs are `.data/a06-scoring/steps/<sweep folder>/` and
`.data/a06-scoring/truth/<sweep folder>/`. Neither command writes inside `data/held-out/`.

**Overlap kept on purpose.** `sweep-analyse` (#65, merged) has its own queue-against-threshold
comparison, which produces the `label_validation` numbers in the capacity record. It validates
the noise threshold during a sweep and labels nothing. It stays. Its two values, the 0.01 s floor
and the 3σ multiplier (`sweep_protocol.py`), must equal `sweep-truth-v1.toml`'s; both files say so
in a comment.

## Recorder: `capacity-sweep-steps` (services/agent)

It is the scoring branch's `capacity-sweep-truth` with the judgement removed. The commits that
expose the per-step reading (`83be357`, `807285c` on the scoring branch) come across unchanged.

**Accepted input.** Unchanged refusals (exit 2, nothing written): a protocol other than
`natural-overload-sweep-v1`; a `protocol_sha256` that is not cfg-default `22e26dc7536f…` or
cfg-hardware `1608deee2a3b…` (pinned in full), or that does not hash the recorded protocol; a
sweep whose `held_out` is not true; a malformed scored batch entry; a batch folder outside the
sweep; an unreadable run; an output folder inside `data/held-out/`. The warm-up batch is skipped.
Phases recorded: the ramp batches `split_ramp_entries` keeps, `live`, `injected`, `control`. Live
runs have no plan and are counted as runs without a plan, as now.

**`steps.jsonl`, one line per journaled step of a scored run:**

- identity: `run_id`, `run_folder` (relative to the sweep folder), `step_id`, `step_type`,
  `parent_step_id` (null for the run's root step), `plan_sha256`, `task_id`;
- position: `configuration`, `n`, `phase`, `batch_sequence`, `batch_id`;
- timing (runner ticks): `state`, `duration_seconds`, `queue_seconds`, `execution_seconds`;
- vLLM span: `span_status`, one of `measured`, `unavailable`, `malformed`, `incomplete`,
  `not_recorded` (the statuses `_step_spans` already produces; a step with no model call is
  `not_recorded`), and `vllm_queue_seconds` and `vllm_service_seconds` (null unless measured);
- baseline for `(plan_sha256, step_id)` from the N = 1 ramp runs: `baseline` is null or
  `{count, mean_seconds, stdev_seconds}` (`stdev_seconds` null when count is 1). The recorder keeps
  `baselines`' existing omission of keys with fewer than `BASELINE_BATCHES_PER_SOURCE` (3) values.

Missing readings stay null; nothing is recorded as a measured zero.

**`steps.json`:** `format = "sweep-steps-v1"`, `sweep_folder`, `sweep_id`, `configuration`,
`protocol_version`, `protocol_sha256`, `sweep_json_sha256`, `held_out: true`, `steps` (line
count), `excluded_batches`, `runs_without_evidence`, `runs_without_plan`. Written last, through a
temporary file, so its presence marks the lines as complete.

## Rule: `sweep-truth-v1` (pipelines/data)

**`sweep-truth-v1.toml`**, frozen before any scoring with it:

| Key | Value | Source |
|---|---|---|
| `version` | `"sweep-truth-v1"` | |
| `noise.stdev_multiplier`, `noise.min_seconds` | 3.0, 0.01 | `labels-v2` and `sweep-analyse` |
| `baseline.min_runs` | 3 | the protocol's 3 batches per source at N = 1 |
| `steps.scored_types` | `["model_call"]` | Akshay's conditions, #70 |
| `protocol_clean.phases`, `protocol_clean.level` | `["ramp", "control"]`, 1 | unchanged |
| `slower.baseline_phase`, `slower.baseline_level` | `"ramp"`, 1 | unchanged |
| `protocols` | cfg-default and cfg-hardware `protocol_sha256`, in full | the accepted held-out sweeps |

**Identifier.** `label_version = "sweep-truth-v1+<first 8 hex of fingerprint>"`, where the
fingerprint is `canonical_sha256` of the validated values and the source of the rule's modules
(the truth decisions and the input validation), as `labels-v2` does. Comments and formatting do not change it; any value or
code change does.

**Truths.** Each is a class or null with a reason, never both.

| Step | `queue` | `slower` | `protocol_clean` |
|---|---|---|---|
| model call | `inference` when `vllm_queue_seconds` > threshold, else `none` | `inference` when `duration_seconds − mean` > threshold, else `none`; null `not_applicable` on ramp N = 1 | `none` in a ramp or control batch at N = 1, else null `not_applicable` |
| root step (`parent_step_id` null) | null `root_step` | null `root_step` | null `root_step` |
| any other step type | null `no_queue_reference` | null `no_queue_reference` | null `no_queue_reference` |

Threshold = max(`min_seconds`, `stdev_multiplier` × `stdev_seconds`); a baseline with null
`stdev_seconds` uses the floor alone. A model call that cannot be judged keeps the current order
and reasons: `not_completed` (state not completed, or completed without a duration), then
`malformed` (duration not finite or negative), then `no_baseline` (no baseline, or `count` below
`baseline.min_runs`), then, for `queue` only, `span_not_measured`. `slower` is still judged when
the span is unmeasured. `protocol_clean` is `none` for a model call in a clean batch whatever its
state, as now; such a step usually has no feature row and scoring reports it as refused.

**Difference from the scoring branch's truths.** Only Akshay's two conditions: tool, retrieval,
other and root steps no longer get `protocol_clean = none`, and their null reason is `root_step`
or `no_queue_reference` instead of `not_applicable`. Every model-call truth is unchanged.

**Output.** `truth.jsonl`: each `steps.jsonl` line's identity and position fields, the timing and
baseline values used, `noise_threshold_seconds`, `excess_seconds`, `truths` and `truth_reasons`.
`truth.json`: `label_version`, the `steps.json` provenance fields, SHA-256 of `steps.jsonl` and
`steps.json`, `held_out: true`, `steps`, and counts per truth by class and by reason. Both are
written with sorted keys and LF line ends, through temporary files, summary last.

## Validation (refuse the sweep, exit 2, write nothing)

- `steps.json`: `format` is not `sweep-steps-v1`; `held_out` is not true; the protocol version or
  hash is not in the rule's `protocols`; `steps` differs from the line count; a field missing or
  of the wrong type.
- Each line: a missing field; `run_id` not a UUID; a repeated `(run_id, step_id)`; `n` not a
  positive integer or `batch_sequence` not a non-negative integer; `phase` not one of the four; a seconds value that is
  not null and not a finite number ≥ 0; an unknown `span_status`, or measured values with a
  status other than `measured`; a `baseline` that is not null and lacks `count`, `mean_seconds`
  or `stdev_seconds`.
- The output folder is inside `data/held-out/`.
- A malformed rule file exits 2 before any input is read.

## Tests

Synthetic fixtures, labelled as such.

- Rule: one test per cell of the truth table; each null reason and its order; the floor-only
  threshold; `baseline.min_runs`; each refusal above writes nothing; the identifier changes with a
  value and not with a comment; identical input gives byte-identical output.
- Recorder: lines carry exactly the fields above; the existing refusals; nothing under
  `data/held-out/`; the existing `sweep_labels` tests pass unmodified.
- Regression on the frozen sweeps, skipped when the DVC data is not pulled: model-call `queue`
  and `slower` counts by class and reason equal those of the scoring branch's attempt
  `20261112T025531Z-220363` truth files (pinned as numbers in the test).

## Real run and docs

Run both commands on `data/held-out/a06-20261111/capacity-sweeps/` cfg-default and cfg-hardware
from a clean tree. The PR description and `docs/data-card.md` record the `label_version` and the
counts per truth by class and reason. Outputs stay in `.data/`. Docs: a `sweep-truth-v1` section
in `docs/data-card.md`, the recorder in `services/agent/README.md`, the Make target's help text,
and a dated entry in `docs/project-checklist.md` under the held-out comparison item.

Verification: `uv run ruff check .` and the full `uv run pytest` in WSL, with the PostgreSQL test
database started for the run and stopped after.

## After this PR

1. When it merges, `main` is merged into `feature/a06-held-out-scoring`; that branch drops
   `capacity-sweep-truth`, reads `truth.jsonl` from this rule, and refuses any other
   `label_version`.
2. When #70 also merges, the scoring switches to Rohit's M03 model with its M04 calibration on
   `split-v3+932b2df410a7`, as a dated change to its own spec for Benoy's approval, then reruns and
   is pushed for Rohit's review.

Out of scope: any change to `labels-v2`, `sweep-analyse`'s `label_validation`, the split, the
features, or the scoring code.

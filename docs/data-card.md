# AgentWatch step dataset — data card

Status: draft schema `steps-v1.2`, labels `labels-v1.1`. No dataset release exists yet. Column
definitions live in `pipelines/data/src/agentwatch_data/schema.py`; a test fails if this card
misses a column. `validate_rows()` enforces every column's type, nullability, allowed values,
and bounds, and a golden test runs a real recorded run through every stage against it.

`steps-v1.1` added `telemetry_coverage`, `telemetry_status`, `injection_method`,
`agent_model`, `run_passed`, `failed_checks`, and `schema_version` to `steps-v1`; `steps-v1.2`
adds `sampling_interval_seconds` and `measured_sampling_interval_seconds` and judges
`telemetry_coverage` against the declared interval. No existing column changed meaning or type.

## Summary

One row is one measured agent step: a model call, a tool execution, or a retrieval. Each row
carries the step's measured timing, its workload size, the resource readings taken during the
step, and experiment-derived labels naming the resource that caused excess waiting.

| Property | Value |
|---|---|
| Unit | One agent step |
| Format | Parquet, with a `manifest.json` per release |
| Targets | `label_resource` (classification), `label_excess_fraction` (regression) |
| Classes | `none`, `cpu_sandbox`, `inference`, `retrieval` |
| Split | 70/15/15, grouped by `task_id` so replays of a task never cross splits |
| Current coverage | `none` and `cpu_sandbox` from the CPU and coding experiments only |

## Sources

Every row records its origin in `source`.

| `source` | Meaning | Allowed in the final test split |
|---|---|---|
| `measured` | Real runs from AgentWatch experiments | Yes |
| `simulated` | Generated rows calibrated to measured and public agent traces | No |
| `rcaeval` | External RCAEval fault cases, evaluated as a separate track | No |

### Extraction guarantees

Measured runs come from the ingestion API (`GET /v1/runs` and each run's `samples.jsonl`) or
from local run directories named `cpu`, `coding`, or `llm`, or starting with `cpu-`,
`coding-`, or `llm-`, anywhere under a data directory.
`pipelines/data/src/agentwatch_data/extract.py` enforces:

- Every run, sample, and API history row is validated **strictly** against the ingestion
  contracts. Values of the wrong JSON type are rejected rather than coerced: `true` is not a
  number, `"10"` is not an integer, and an integer is not a timestamp. A run that fails is
  recorded as a rejection with its reason; the other runs still extract.
- The evidence must match its declared kind: a `coding` run declares `tests_per_step`, and a
  `cpu` run does not. A run in an `llm-*` directory must carry the agent evidence
  (`result["agent"]`); without it the LLM loop did not finish.
- Experiment check names must be lowercase identifiers (`[a-z][a-z0-9_]*`), so `failed_checks`
  stays unambiguous.
- Network failures are retried with bounded exponential backoff for timeouts, connection
  errors, and HTTP 429, 500, 502, 503, and 504. If a source still cannot be read, or returns a malformed response,
  extraction stops with an error instead of producing a partial dataset. A history page
  without `next_cursor` counts as malformed, never as the last page.
- A run found more than once, locally or through the API, is kept once. Copies whose
  normalized evidence disagrees are all rejected, including copies already set aside as
  duplicates, and that run ID stays rejected when the extraction is later combined with other
  sources.
- Each run records whether it carried resource samples: `present`, `empty`, or `missing`
  (no samples file). The API cannot distinguish an absent file from an empty one.
- Identical samples at the same timestamp are collapsed; disagreeing samples reject the run.

Flattening and labeling also reject untrustworthy runs one at a time, with a reason, while the
other runs continue (`flatten_runs` and `label_rows` return rows and rejections). Only invalid
label rules or an unreadable source stop a build, so one bad run can never block every later
build.

## Releases

Build a release with `make dataset`; set `DATASET_SOURCES` to combine sources, for example
`make dataset DATASET_SOURCES="local:.data api:http://127.0.0.1:8090"`. The command runs
`python -m agentwatch_data.build` (`pipelines/data/src/agentwatch_data/build.py`), prints the
release directory, and exits 1 with a reason when the build fails.

A release is a directory under `.data/datasets/` named by its `dataset_version`:

| File | Contents |
|---|---|
| `steps.parquet` | Every row, every schema column, with Arrow types and nullability taken from `schema.py`; rows ordered by run, then condition, then baseline repeat |
| `manifest.json` | Dataset, schema, label, and builder versions; build time; row, run, and task counts; label, basis, condition, and telemetry-status counts; per-source run and rejection counts; duplicate copies; every rejection with its stage and reason; SHA-256 of the canonical rows and of the Parquet file |

- `dataset_version` is `<schema version>+<first 12 hex digits of the rows' SHA-256>`, so the
  same evidence, rules, and code always produce the same version. A changed run, rule, or label
  fingerprint produces a new one.
- Building a version that already exists verifies its manifest hashes instead of rewriting it;
  a mismatch fails the build.
- A build writes everything into a staging directory and renames it into place only when it is
  complete, so a failed build never leaves a partial release.
- The build fails, writing nothing, when a source cannot be read, the label rules are invalid,
  any row violates the schema, or no usable rows remain. Individual untrustworthy runs are
  rejected and listed in the manifest instead.
- Host paths never appear in a manifest: local sources are labeled by directory name, such as
  `<local:coding-demo>`, and the home directory by `~`.

## Column roles

Only **feature** columns may be model inputs. Labels, rule output, and lineage columns are kept
for training targets, baseline comparison, and traceability.

### Identity

| Column | Type | Description |
|---|---|---|
| `run_id` | string | Experiment run UUID |
| `step_id` | string | Step name within the run. Values such as `contended` reveal the condition, so it is never a feature |
| `task_id` | string | Task identity and the split grouping key |
| `kind` | string | Experiment kind recorded by the ingestion API |
| `source` | string | Data origin, see above |
| `recorded_at` | timestamp (UTC), nullable | When the run was saved |

### Features

| Column | Type | Description |
|---|---|---|
| `queue_seconds` | float | Measured wait before a worker started |
| `execution_seconds` | float | Worker start to finish wall time; includes scheduling delays, so it is not pure work |
| `duration_seconds` | float | Submission to finish wall time |
| `cpu_seconds` | float | Process and child CPU time |
| `queue_fraction` | float | `queue_seconds / duration_seconds` |
| `workload_iterations` | int | Fixed work size of the step |
| `tests_per_step` | int, nullable | Tests executed by the tool step |
| `worker_capacity` | int | Sandbox workers available |
| `cpu_pressure_max_avg10` | float, nullable | Maximum host CPU PSI `some avg10` (%) during the step |
| `cpu_pressure_available` | bool | Whether PSI was readable during the step |
| `cgroup_throttled_usec_delta` | int, nullable | cgroup v2 throttled time gained during the step |
| `cgroup_available` | bool | Whether a throttling delta was measured: at least two samples in the window, every one with a readable counter, and no decrease between consecutive readings |
| `telemetry_sample_count` | int | Resource samples inside the step window |
| `telemetry_coverage` | float, nullable | Samples inside the window ÷ whole declared sampling intervals in the step (`floor(duration / sampling_interval_seconds)`), capped at 1. Judged against the declared interval because a measured one cannot reveal uniform sample loss. Null when telemetry is not `present`, no interval is declared, or the step is shorter than one interval |

Missing readings stay null and are paired with an availability flag. They are never stored as
zero, because a measured zero and an unavailable reading mean different things. On macOS, PSI and
cgroup v2 readings are always unavailable.

### Labels

| Column | Type | Description |
|---|---|---|
| `label_resource` | string | Resource that caused excess waiting, or `none` |
| `label_excess_fraction` | float | `max(0, min(1, (T − B) / T))`, where T is the step duration and B the clean baseline mean |

Labels come from the experiment design and a frozen noise threshold: a step is labeled with a
resource only when contention was applied and its slowdown exceeds the threshold. They do not
come from the rule baseline, so model and rule can be compared fairly.

The rules live in `pipelines/data/src/agentwatch_data/labels.toml` (`labels-v1.1`):

- B is the mean duration of the run's own `baseline` steps; at least two are required.
- The noise threshold is `max(min_seconds, stdev_multiplier × baseline stdev)`, currently
  `max(0.01 s, 3 × stdev)`.
- A `contended` step gets the resource its experiment kind contends (`cpu_sandbox` for the CPU
  and coding experiments) only when `T − B` exceeds the threshold; otherwise `none`. Its
  `injection_method` is the declared method (`worker_queue_blocker`: a competing task submitted
  ahead to a single-worker pool); other steps record `none`.
- `baseline` and `clean` steps are always `none`, because no contention was applied.
- `label_version` is the rules version plus the first 8 hex digits of a fingerprint of the
  validated rule values and the labeling code (`labels.py`, line endings normalized). A retuned
  threshold or a changed labeling rule produces a new version; comments, formatting, equivalent
  spellings such as `3` and `3.0`, and Windows line endings do not.
- A step's condition comes from its position in the run result (`baseline.steps`, `clean`,
  `contended`), and its `step_id` must agree; a renamed step fails instead of changing role.
- Runs with unknown conditions, unknown or mixed kinds, repeated step IDs, fewer than two
  baseline steps, or any duration that is not a finite positive number (NaN, infinity, zero,
  booleans, or non-numeric) fail labeling instead of being guessed. Mixed contention is not
  defined in `labels-v1.1`.
- A run that failed `same_workload_output` did different work under contention, so its
  slowdown is not attributable and the run is rejected from the dataset with that reason. Other failed checks are recorded in
  `failed_checks` but do not block labels: dropping runs because the rule disagreed would select
  the dataset by the rule's own answer.
- Rule files must have exactly the expected keys and finite numbers; NaN, infinity, booleans,
  misspelled or unknown keys, and invalid TOML are rejected before any labeling runs.

`label_basis` records why each row got its label:

| `label_basis` | Meaning |
|---|---|
| `baseline_reference` | Baseline step used to compute B |
| `clean_within_noise` | Clean control within the noise threshold |
| `clean_exceeded_noise` | Clean control slower than the threshold; a data-quality signal, still `none` |
| `contention_exceeded_threshold` | Contention applied and the slowdown exceeded the threshold |
| `contention_below_threshold` | Contention applied but the slowdown stayed within noise, so `none` |

### Rule baseline

| Column | Type | Description |
|---|---|---|
| `rule_prediction` | string | Resource named by the threshold rule (the `resource` field in run results) |
| `rule_version` | string | Version of the rule that produced it |

### Lineage

| Column | Type | Description |
|---|---|---|
| `condition` | string | Experiment condition from the step's position: `baseline`, `clean`, or `contended` |
| `workload` | string | Human-readable workload description |
| `injection_method` | string | How contention was applied to the step (`worker_queue_blocker`), or `none` |
| `agent_model` | string, nullable | Model that invoked the tool in LLM runs, such as `qwen3:1.7b`; null otherwise. Distinguishes LLM-driven coding runs, which share `kind` with direct coding runs |
| `run_passed` | bool | Whether every experiment check passed |
| `failed_checks` | string, nullable | Comma-separated names of failed experiment checks, in alphabetical order |
| `telemetry_status` | string | Run telemetry: `present`, `empty`, `missing` (no samples file), or `no_overlap` (samples exist but none fall inside any step) |
| `sampling_interval_seconds` | float, nullable | Interval the run's sampler declared in its step telemetry (0.25 s for the current worker); null when not recorded |
| `measured_sampling_interval_seconds` | float, nullable | Median gap between the run's samples. Compare with the declared interval to spot sampler drift or bursts; never used for coverage |
| `baseline_mean_seconds` | float | Clean baseline mean duration B used for the regression label |
| `noise_threshold_seconds` | float | Run-specific slowdown threshold derived from the frozen rules |
| `label_basis` | string | Why the step received its label; see the table above |
| `label_version` | string | Labeling rules version plus fingerprint, for example `labels-v1.1+1a2b3c4d` |
| `schema_version` | string | Dataset schema version the row was built with (`steps-v1.2`) |
| `dataset_version` | string | Dataset release this row belongs to |
| `source_sha256` | string | SHA-256 of the run's results and its samples as received, before contract normalization, serialized as canonical JSON (sorted keys, samples ordered by timestamp). Any change to a value produces a new hash; key order, sample order, and whitespace do not |

## Leakage rules

- `step_id`, `condition`, `rule_prediction`, `baseline_mean_seconds`, `label_basis`,
  `injection_method`, `run_passed`, and `failed_checks` are never features.
- Injector configuration and logs are used for labeling only.
- All clean and contended replays of a task stay in the same split.
- Preprocessing is fitted on the training split only.

## Known limitations

- Current measured data covers one fixed task and two classes.
- **Labels are trivially predictable on current data.** With only the CPU queue experiment, the
  contention directly creates queue waiting: on 7 real runs (49 rows), `label_resource` equals
  `rule_prediction` on every row, and `queue_fraction` alone separates the classes (lowest
  positive 0.847, highest negative 0.003). Any model will look near-perfect and model-versus-rule
  comparisons carry no information until inference, retrieval, and mixed contention exist. The
  split stage must run a leakage check before any evaluation.
- **Clean controls sometimes exceed the noise threshold** (`clean_exceeded_noise`): 2 of 7 real
  runs recorded on 2026-10-08 (+0.047 s and +0.035 s against thresholds near 0.02 s) and 0 of 7
  recorded on 2026-10-09. The rate varies between collections, so it is reported per release
  and never tuned away.
- **Telemetry is sparse for short steps.** The worker samples every 250 ms while many steps last
  0.2–0.5 s, so 20 of 42 rows in six real runs had 0–1 samples. `telemetry_coverage` exposes
  this; faster sampling belongs to the worker.
- `task_id` is an interim placeholder (`kind:workload:iterations`): one project at different
  work sizes becomes several tasks, so a grouped split cannot yet guarantee unseen tasks.
- Hardware telemetry is unavailable on macOS hosts.
- Experiment labels are operational proxies and still need natural-overload validation.

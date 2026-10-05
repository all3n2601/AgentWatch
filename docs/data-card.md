# AgentWatch step dataset — data card

Status: draft schema `steps-v1`. No dataset release exists yet. Column definitions live in
`pipelines/data/src/agentwatch_data/schema.py`; a test fails if this card misses a column.

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
| `cgroup_available` | bool | Whether cgroup counters were readable |
| `telemetry_sample_count` | int | Resource samples inside the step window |

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

### Rule baseline

| Column | Type | Description |
|---|---|---|
| `rule_prediction` | string | Resource named by the threshold rule (the `resource` field in run results) |
| `rule_version` | string | Version of the rule that produced it |

### Lineage

| Column | Type | Description |
|---|---|---|
| `condition` | string | Experiment condition: `baseline`, `clean`, or `contended` |
| `workload` | string | Human-readable workload description |
| `baseline_mean_seconds` | float | Clean baseline mean duration B used for the regression label |
| `noise_threshold_seconds` | float | Frozen slowdown threshold used for labeling |
| `label_version` | string | Version of the labeling rules |
| `dataset_version` | string | Dataset release this row belongs to |
| `source_sha256` | string | Hash of the source evidence file |

## Leakage rules

- `step_id`, `condition`, `rule_prediction`, and `baseline_mean_seconds` are never features.
- Injector configuration and logs are used for labeling only.
- All clean and contended replays of a task stay in the same split.
- Preprocessing is fitted on the training split only.

## Known limitations

- Current measured data covers one fixed task and two classes.
- Hardware telemetry is unavailable on macOS hosts.
- Experiment labels are operational proxies and still need natural-overload validation.

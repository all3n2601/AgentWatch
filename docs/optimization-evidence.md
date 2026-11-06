# Optimization experiment evidence

Status: O04, implemented 2026-11-05 in `pipelines/data` and verified on synthetic experiments
only. No optimization experiment has been recorded yet: runner admission limits (O03) and the
hardware evaluation that produces batches (O06) do not exist, and `opt-contract-v1` is still
proposed. Nothing here is an optimization result.

O04 fixes the evidence format for comparing the baseline concurrency policy with candidate
limits under [the optimization contract](optimization-contract.md). Before running anything,
an experiment preregisters a **manifest**. Each attempted batch then writes one **batch
record**. `compare` validates both and writes a deterministic **comparison record** of the
observed paired effects. It never evaluates the contract's gates: O06 applies those once the
contract is accepted, so every record says `gates: not_evaluated`.

## Commands

```bash
make optimization-schedule MANIFEST=<experiment>/manifest.json   # the batch order to run
make compare-optimization EXPERIMENT=<experiment>                # validate and compare
```

`compare-optimization` writes `.data/reports/optimization/<first 12 hex of the manifest
hash>/comparison.json`. It exits 1, writing nothing, when the manifest or the contract is
invalid. Rejected batches and unpaired claims do not make it fail: they are recorded in the
report and make the affected comparisons insufficient. The modules are
`agentwatch_data.optimization_manifest`, `optimization_batches` and `optimization_comparison`.

## Experiment directory

```text
<experiment>/
  manifest.json        written once, before the first batch
  batches/<NNN>.json   one per attempted batch, NNN = schedule index (warm-ups and failures too)
  <run dirs>/          optional agent run evidence that batch tasks link to
```

Experiments are validated in place. A manifest is identified by the SHA-256 of its parsed
contents as canonical JSON, the same method as the contracts; a manifest holding NaN or
infinity has no canonical form and is refused. Every batch record names that hash, so a
manifest edited after batches ran matches none of them. Hidden files in `batches/` (a
writer's staging files) are not read, but they are listed in the report. Copying experiments
into a DVC-versioned store is a separate follow-up.

## Manifest (`opt-experiment-manifest-v1`)

Every key is required unless noted; unknown keys are rejected.

| Key | Content | Checked against `opt-contract-v1` |
|---|---|---|
| `format`, `experiment_id`, `created_at` | `opt-experiment-manifest-v1`, a UUID, an RFC 3339 timestamp with zone | |
| `contract` | `{version, sha256}`: the full hash of the contract the experiment was designed for | Must equal the current contract; a stale or edited contract is refused |
| `purpose` | `development` (choosing settings) or `evaluation` (frozen tasks) | Only evaluation comparisons can be sufficient for gating |
| `workload.tasks` | `{task_id, content_sha256, configuration_digest}` per frozen task (`<name>-r<rev>`; the agent's task hash and run configuration digest), at most 256 | Evaluation: at least `min_evaluation_tasks`, and no task name shared with `workload.development_tasks` (another revision of a development task is the same task) |
| `workload.arrivals` | `{method: "fixed_offsets_v1", tolerance_seconds, offsets: [{task_id, offset_seconds}]}`, one offset per task, starting at 0 | `require_preregistered_arrivals` |
| `runner` | `{version}` | |
| `model` | `{backend: ollama\|vllm, name, digest}` | |
| `hardware` | `{host_id, cpu_model, logical_cpus, memory_bytes, os, kernel, gpus: [{name, driver_version, memory_total_bytes}]}` | |
| `power` | `{source: ac\|battery, profile}`; an unknown power state is refused | |
| `state` | `{tests, retrieval}`: identifiers of the test and retrieval state | |
| `conditions` | `[{name, injectors: [{kind, method, intensity}]}]` | Exactly the contract's conditions; contended ones preregister injectors, clean ones none |
| `policies` | `baseline: {limits: {}}` and `candidates: [{name, limits}]` | Baseline unlimited; each candidate sets exactly one control to a preregistered candidate limit (1, 2, 4); no limit is an absent key, never `null` or 0 |
| `queue_bounds` | `{<control>: {max_length, max_wait_seconds}}` for every control | `require_declared_queue_bounds`; admission waits are judged against these |
| `repetitions` | `{pairs, warmup_batches, seed}`; at most 1,000 pairs and 100 warm-ups | `pairs ≥ min_paired_batches`, and `pairs × tasks ≥ min_completed_tasks_per_configuration` |

Factorial experiments, which change both controls at once, are outside this format. The
contract reports them as not evaluated in v1.

**Policy hash.** A policy is identified by
`canonical_sha256({target: "coding-agent", limits, enforcement_scope})`. Two stored O02 policy
versions with the same limits configure the runner identically, so operator, reason and time
are audit data, not identity.

**Schedule.** `optimization-schedule` derives the batch order from the seed. For each condition
in manifest order, its warm-up batches run at baseline limits, followed by every (candidate,
repetition) pair in a seeded random order. A pair's two batches are adjacent, and a seeded
coin decides which runs first.

## Batch record (`opt-batch-v1`)

| Key | Content |
|---|---|
| `manifest_sha256`, `batch_id` | The manifest's hash; a UUID kept for lineage |
| `index`, `kind`, `condition`, `configuration`, `candidate`, `pair_index` | The schedule slot, which must match `optimization-schedule` exactly (`pair_index` a strict integer) |
| `policy` | `{sha256, limits, version}`: the applied policy; `version` is the O02 version when the runner recorded it, else `null` |
| `status`, `reason` | `completed`, `failed`, `aborted` or `not_started`; every non-completed batch records a reason |
| `started_at`, `finished_at` | RFC 3339 times, required for a started batch, absent for one that never started |
| `clock_id` | The monotonic clock every task tick in the batch comes from |
| `model`, `hardware`, `power`, `state` | Observed identity, in the manifest's shapes; must equal the manifest |
| `injection` | `{active, cleanup_confirmed}`, each `true`, `false` or `null` |
| `admission` | `{starved_admissions, leaked_permits}`, `null` until the runner can observe them (O03) |
| `tasks` | One entry per frozen task: `task_id`, `run_id`, `state` (`completed`, `failed`, `timed_out`, `cancelled`, `not_started`), `acceptance` (`passed`, `failed`, or `null` when unknown), `cold`, `submitted_tick_ns`, `finished_tick_ns`, `admission_waits` (`{<control>: seconds or null}`), and optionally `evidence_dir` |

Checks beyond the shapes:

- **Order.** Batches run one at a time in schedule order. A batch that starts before the
  previously scheduled batch finished is rejected, so a pair's batches did not overlap or run
  in reverse.
- **Injection.** A completed contended batch must show injection active with cleanup confirmed.
  A completed clean batch must not show injection.
- **Arrivals.** Each task's submission, measured from the batch's first submission, must be
  within `tolerance_seconds` of its preregistered offset.
- **Run identity.** No two tasks, in one batch or across batches, share a `run_id`. When a task
  links `evidence_dir`, its `agent-evidence.json` must show the same `run_id`, task ID, task
  content hash and configuration digest as the manifest. The link must stay inside the
  experiment directory, with no symlinks.

**Rejections.** A record that cannot be used is kept in the report, with the SHA-256 of the
bytes that were parsed and one code. The codes:
- `malformed`, `inconsistent_status`;
- `manifest_mismatch`, `schedule_mismatch`, `policy_mismatch`;
- `model_mismatch`, `hardware_mismatch`, `power_mismatch`, `state_mismatch`;
- `injection_mismatch`, `arrival_mismatch`, `order_mismatch`;
- `task_mismatch` (not every frozen task exactly once), `task_identity_mismatch`;
- `duplicate_identity`: a shared batch ID or run ID; neither copy can be trusted, so all are rejected.

## Comparison record (`opt-comparison-v1`)

**Pairing.** A pair is compared only when both scheduled batches have accepted, completed
records with a positive measured batch time. Every other pair is listed as unpaired with its
reason (missing, rejected, failed or aborted, or unmeasured). Every record is listed with its
hash, accepted or rejected. Batch time runs from the first task submission to the last task
finish on the batch's monotonic clock. A missing tick leaves it unmeasured, never zero.

Per condition and candidate:

| Field | Meaning |
|---|---|
| `attempts` | For each configuration, every accepted batch by status and its tasks by state, complete pair or not |
| `outcome_changes` | `new_failures` (the baseline completed a task the candidate failed, cancelled or never started), `new_timeouts`, `newly_failing_acceptance` and `acceptance_unknown`. Counted over every pair whose two records were accepted, including failed or aborted batches, so a candidate that breaks its batch still shows what it caused |
| `per_pair` | For complete pairs: baseline and candidate batch seconds, batch IDs, and `reduction = 1 − candidate / baseline` |
| `effects.median_batch_time_reduction` | The contract's improvement statistic: `1 − median(candidate batch times) / median(baseline batch times)` |
| `effects.median_batch_time_reduction_interval` | Percentile paired bootstrap of that statistic, resampling whole pairs, with the contract's resamples and level and the manifest's seed; `null` with fewer than 2 pairs |
| `effects.median_pair_reduction` | The median of the per-pair reductions. A different statistic, reported beside the contract's, not instead of it |
| `effects.throughput_ratio` | Completed tasks per batch-second, candidate over baseline |
| `effects.p95_warm_task_latency_ratio` | Nearest-rank p95 of warm, completed task latencies; cold tasks are counted separately and excluded |
| `effects.baseline_batch_time_cv` | Variability of baseline batch times. A proxy only: the contract's stability gate is the root step's CV (`max_baseline_root_step_cv`), which needs per-task root step durations |
| `effects.baseline`, `effects.candidate` | Per-side counts, medians and per-control admission waits: median, maximum, how many exceeded the declared `max_wait_seconds`, and how many were unobserved. Starvation and leak totals are `null` when any batch could not observe them |
| `sufficient`, `insufficient_reasons` | `false` whenever the experiment has any rejected or missing batch, any pair is unpaired, the complete pairs fall short of the contract's minimum pairs or completed tasks, or the experiment is for development |

These are observed differences on the declared workload, model and hardware only. They do not
establish a general causal improvement, and `sufficient` is not a pass.

## Attribution features

Policies, conditions, injectors and pair design identify what was compared. They must never
become model inputs. `DESIGN_FIELDS` names them, and a test checks that none is a feature of
the step dataset, the replay labels or `step-features-v1`.

## Handoffs

| Consumer | Uses |
|---|---|
| Benoy (O06) | Writes manifests and batch records in these formats, runs the schedule, then applies the contract's gates to comparison records. The improvement statistic and its interval should be confirmed against the contract before gating |
| Akshayaa (O03) | Records the applied policy version, per-control admission waits, starvation and leaked permits per batch, so those fields stop being `null` |
| Meghana and Allen (O05) | Show comparison records by manifest hash and batch UUID, with unpaired, rejected and missing evidence shown as such |
| Rohit | Reproducible evaluation inputs: the manifest hash, contract hash and seed reproduce the schedule, bootstrap and record |

## Known limitations

- Synthetic experiments only. The real producer (O06) and admission statistics (O03) don't exist yet.
- The agent has two fixed tasks. An evaluation manifest needs at least two frozen tasks that
  share no name with its development tasks, so a valid evaluation needs more tasks first.
- "Frozen evaluation tasks are used once" is not enforced across experiments: no registry of
  used evaluation tasks exists yet.
- Hardware, model and power identity are declared by the producer. O04 compares them but cannot
  measure them.
- The order check uses each batch's recorded wall-clock start and finish. It does not bound the
  gap between a pair's two batches.
- Batch time includes cold requests. Cold tasks are excluded only from warm tail latency, and
  are counted.
- No gate is evaluated, and no candidate is promoted, here.

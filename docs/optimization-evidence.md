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
report. The modules are `agentwatch_data.optimization_manifest`, `optimization_batches` and
`optimization_comparison`.

## Experiment directory

```text
<experiment>/
  manifest.json        written once, before the first batch
  batches/<NNN>.json   one per attempted batch, NNN = schedule index (warm-ups and failures too)
  <run dirs>/          optional agent run evidence that batch tasks link to
```

Experiments are validated in place. A manifest is identified by the SHA-256 of its parsed
contents as canonical JSON, the same method as the contracts. Every batch record names that
hash, so a manifest edited after batches ran matches none of them. Copying experiments into a
DVC-versioned store is a separate follow-up.

## Manifest (`opt-experiment-manifest-v1`)

Every key is required unless noted; unknown keys are rejected.

| Key | Content | Checked against `opt-contract-v1` |
|---|---|---|
| `format`, `experiment_id`, `created_at` | `opt-experiment-manifest-v1`, a UUID, an RFC 3339 timestamp with zone | |
| `contract` | `{version, sha256}`: the full hash of the contract the experiment was designed for | Must equal the current contract; a stale or edited contract is refused |
| `purpose` | `development` (choosing settings) or `evaluation` (frozen tasks, used once) | Only evaluation comparisons can be sufficient for gating |
| `workload.tasks` | `{task_id, content_sha256, configuration_digest}` per frozen task (`<name>-r<rev>`; the agent's task hash and run configuration digest) | Evaluation: at least `min_evaluation_tasks`, disjoint from `workload.development_tasks` |
| `workload.arrivals` | `{method: "fixed_offsets_v1", offsets: [{task_id, offset_seconds}]}`, one per task, starting at 0 | `require_preregistered_arrivals` |
| `runner` | `{version}` | |
| `model` | `{backend: ollama\|vllm, name, digest}` | |
| `hardware` | `{host_id, cpu_model, logical_cpus, memory_bytes, os, kernel, gpus: [{name, driver_version, memory_total_bytes}]}` | |
| `power` | `{source: ac\|battery, profile}`; an unknown power state is refused | |
| `state` | `{tests, retrieval}`: identifiers of the test and retrieval state | |
| `conditions` | `[{name, injectors: [{kind, method, intensity}]}]` | Exactly the contract's conditions; contended ones preregister injectors, clean ones none |
| `policies` | `baseline: {limits: {}}` and `candidates: [{name, limits}]` | Baseline unlimited; each candidate sets exactly one control to a preregistered candidate limit (1, 2, 4); no limit is an absent key, never `null` or 0 |
| `queue_bounds` | `{<control>: {max_length, max_wait_seconds}}` for every control | `require_declared_queue_bounds` |
| `repetitions` | `{pairs, warmup_batches, seed}` | `pairs ≥ min_paired_batches`, and `pairs × tasks ≥ min_completed_tasks_per_configuration` |

**Policy hash.** A policy is identified by
`canonical_sha256({target: "coding-agent", limits, enforcement_scope})`. Two stored O02 policy
versions with the same limits configure the runner identically, so operator, reason and time
are audit data, not identity.

**Schedule.** `optimization-schedule` derives the batch order from the seed. For each condition
in manifest order, its warm-up batches run at baseline limits, followed by every (candidate,
repetition) pair in a seeded random order. A pair's two batches run back to back, and a seeded
coin decides which runs first. Records are checked against this order rather than trusted.

## Batch record (`opt-batch-v1`)

| Key | Content |
|---|---|
| `manifest_sha256`, `batch_id` | The manifest's hash; a UUID kept for lineage |
| `index`, `kind`, `condition`, `configuration`, `candidate`, `pair_index` | The schedule slot, which must match `optimization-schedule` exactly |
| `policy` | `{sha256, limits, version}`: the applied policy; `version` is the O02 version when the runner recorded it, else `null` |
| `status`, `reason` | `completed`, `failed`, `aborted` or `not_started`; every non-completed batch records a reason |
| `started_at`, `finished_at`, `clock_id` | RFC 3339 timestamps; the monotonic clock all task ticks come from |
| `model`, `hardware`, `power`, `state` | Observed identity, in the manifest's shapes; must equal the manifest |
| `injection` | `{active, cleanup_confirmed}`, `null` when not applicable or not observed |
| `admission` | `{starved_admissions, leaked_permits}`, `null` until the runner can observe them (O03) |
| `tasks` | One entry per frozen task: `task_id`, `run_id`, `state` (`completed`, `failed`, `timed_out`, `cancelled`, `not_started`), `acceptance` (`passed`, `failed`, or `null` when unknown), `cold`, `submitted_tick_ns`, `finished_tick_ns`, `admission_wait_seconds`, and optionally `evidence_dir` |

When a task links `evidence_dir`, its `agent-evidence.json` must show the same `run_id`, task
ID, task content hash and configuration digest as the manifest. Links must stay inside the
experiment directory, with no symlinks.

**Rejections.** A record that cannot be used is kept in the report with one code. The codes:
- `malformed`, `inconsistent_status`;
- `manifest_mismatch`, `schedule_mismatch`, `policy_mismatch`;
- `model_mismatch`, `hardware_mismatch`, `power_mismatch`, `state_mismatch`;
- `task_mismatch` (not every frozen task exactly once), `task_identity_mismatch`;
- `duplicate_identity`: two records share a batch ID or run ID; neither can be trusted, so both are rejected.

## Comparison record (`opt-comparison-v1`)

**Pairing.** A pair is compared only when both scheduled batches have accepted, completed
records with a positive measured batch time. Every other pair is listed as unpaired with its
reason (missing, rejected, failed, or unmeasured), and every record is listed with its file
hash, accepted or rejected. Batch time runs from the first task submission to the last task
finish on the batch's monotonic clock. A missing tick leaves it unmeasured, never zero.

Per condition and candidate, over complete pairs:

| Field | Meaning |
|---|---|
| `per_pair` | Baseline and candidate batch seconds, batch IDs, and `reduction = 1 − candidate / baseline` |
| `median_reduction`, `reduction_interval` | Median reduction, with a percentile paired bootstrap over pairs using the contract's resamples and level and the manifest's seed; `null` with fewer than 2 pairs |
| `throughput_ratio` | Completed tasks per batch-second, candidate over baseline |
| `p95_warm_task_latency_ratio` | Nearest-rank p95 of warm, completed task latencies; cold tasks are counted separately and excluded |
| `new_failures`, `new_timeouts`, `newly_failing_acceptance`, `acceptance_unknown` | Per task, within each pair, compared with the baseline batch |
| `baseline_batch_cv` | Variability of baseline batch times, reported for the stability gate |
| `baseline`, `candidate` | Per-side counts, medians, admission wait and starvation/leak totals; a total is `null` when any batch could not observe it |
| `sufficient`, `insufficient_reasons` | Whether the complete pairs reach the contract's minimum pairs and completed tasks, and the experiment is an evaluation |

These are observed differences on the declared workload, model and hardware only. They do not
establish a general causal improvement, and `sufficient` is not a pass.

## Attribution features

Policies, conditions, injectors and pair design identify what was compared. They must never
become model inputs. `DESIGN_FIELDS` names them, and a test checks that none is a feature of
the step dataset, the replay labels or `step-features-v1`.

## Handoffs

| Consumer | Uses |
|---|---|
| Benoy (O06) | Writes manifests and batch records in these formats, runs the schedule, then applies the contract's gates to comparison records |
| Akshayaa (O03) | Records the applied policy version, admission waits, starvation and leaked permits per batch, so those fields stop being `null` |
| Meghana and Allen (O05) | Show comparison records by manifest hash and batch UUID, with unpaired, rejected and missing evidence shown as such |
| Rohit | Reproducible evaluation inputs: the manifest hash, contract hash and seed reproduce the schedule, bootstrap and record |

## Known limitations

- Synthetic experiments only. The real producer (O06) and admission statistics (O03) don't exist yet.
- The agent has two fixed tasks. An evaluation manifest needs at least two frozen tasks that are
  disjoint from its development tasks, so a valid evaluation needs more tasks first.
- Hardware, model and power identity are declared by the producer. O04 compares them but cannot
  measure them.
- Batch time includes cold requests. Cold tasks are excluded only from warm tail latency, and
  are counted.
- No gate is evaluated, and no candidate is promoted, here.

# D03 — Validated, versioned dataset v1 (`steps-v1.3`)

Status: draft for owner review, 2026-11-05. Board task D03 (owner Allen, partner reviewer
Akshay). Handoff: an immutable dataset and feature version to Rohit (M02). Rohit reviews the
evaluation contract change.

## Intent

Rohit's M02 rule-baseline evaluation and later model work need one dataset release that holds
the real replay experiments, labeled with the frozen `labels-v2` rules, featurized with the
shared `step-features-v1` code, and split with the frozen `split-v2` hold-out. That release has
to be rebuildable from versioned evidence with matching lineage, and it has to come with
statistics that show schema, missingness, class balance and leakage, so nobody trains on a
dataset whose defects are hidden.

Board acceptance: rebuild produces matching lineage; schema, missingness, class balance and
leakage checks pass.

## What is decided

- **Evidence exists.** Benoy recorded replay experiments on the GPU laptop, delivered as two
  zips with `SHA256SUMS` (`d03-2026-11-04.zip` SHA-256 `5ee876a3…b5e0`, `d03-2026-11-05.zip`
  SHA-256 `9e82c41b…94b5`). All checksums verify, and `labels-v2` reproduces his counts
  exactly. Together with his A05 attempts they are 54 experiments:

  | Task | Experiments | Labelable | Labeled rows |
  | --- | --- | --- | --- |
  | `fix-window-count-r1` | 33 | 25 (8 failed `stable_baseline`) | 3,575 |
  | `fix-mean-empty-r1` | 21 | 21 | 2,028 |

  The 45-experiment batch was declared before recording (count, interleaved order, stop rule)
  and ran in one pass at `c6b4333`.
- **`eval-contract-v2` is accepted by Rohit**, who told the owner directly on 2026-11-05.
  There is no written record; the contract file and docs cite that date and Rohit confirms it
  by approving this PR. Its four items
  ([proposal](../../evaluation-contract-v2-proposal.md)) apply: `labels-v2` is evaluated,
  `mixed` stays a label value outside the four prediction classes, `split-v2` is selected with
  `group_key = "task_group"` and the frozen `split-assignment.toml`, and every threshold is
  unchanged. The `mixed` scoring rule (item 2) is Rohit's M02 decision; D03 does not need it.
- **One combined release.** Per the accepted contract, a `steps-v1.3` release carries both row
  kinds: the 50 existing CPU and coding runs (`labels-v1.1`) and the replay experiments
  (`labels-v2`). Under `split-v2` every CPU and coding row goes to test (held-out method
  `worker_queue_blocker`), with `fix-window-count`; training is the `fix-mean-empty` replays.

## Goals

1. Replay experiments are curated into the immutable evidence store, validated before they are
   added, and versioned with DVC like runs are today.
2. `make dataset` builds one `steps-v1.3` release from the whole store: CPU and coding rows as
   today, plus one row per labeled replay leaf step with `step-features-v1` features joined
   from that replay's own journal.
3. The release passes the TFDV gate against a newly frozen, reviewed `steps-v1.3` schema.
4. `make split-dataset` produces a `split-v2` split under `eval-contract-v2`, records the D02
   independence, held-out and mixed-in-test checks, and `load_split` rebuilds and verifies it
   on every read.
5. Validation statistics show schema, missingness per column and row kind, class balance per
   split against the contract's minimums, and leakage.
6. Building and splitting twice from the same evidence gives the same dataset version, Parquet
   bytes, split id and per-split `rows_sha256`; a fresh clone that pulls the version does the
   same.
7. The data card, pipeline README, project checklist, evaluation contract docs and an ADR
   describe what is in v1 and what it cannot support.

## Non-goals

- New features. `step-features-v1` is used as merged (B02). Reading GPU, inference-server or
  retrieval metrics is a new feature version and a separate task.
- The `mixed` scoring rule, the rule baseline, training or any evaluation (M02, Rohit).
  `pipelines/model` changes only to read the v2 contract. `make rule-baseline` and
  `make train-candidates` on a v1.3 split stop with a clear error naming the unscored
  `mixed` label until M02 defines its scoring; their v1.2 behavior and tests are unchanged.
  *(designer's call)*
- Changing `labels.toml`, `labels-v2.toml`, `split-assignment.toml` or any contract threshold.
- Recording more experiments, or tuning anything after seeing the data.
- Per-replay read tolerance (a read defect in one replay still rejects its experiment).

## Anti-goals

- Filling a missing reading with zero or a guessed value, or reusing the CPU/coding feature
  window for replay rows.
- Dropping the 8 failed experiments or any failed attempt from the evidence store; they are
  rejected with reasons in the build report.
- Treating a passing gate or split as model readiness. With one training task, validation is
  empty and dependent gates are `inconclusive`.
- Breaking the current `steps-v1.2` release, its split, or anything that reads it.

## Constraints

- `AGENTS.md` measurement rules: null is not zero; queue, execution and total time stay
  distinct; timing units and clock domains are preserved; synthetic fixtures live only in
  tests; differing configurations are exposed, not merged silently.
- Releases are immutable and content-addressed; the evidence and dataset stores are
  append-only, and integrity refuses anything it cannot rebuild.
- Shared interfaces touched: the release schema (`schema.py`, `schema.json`), the evaluation
  contract, the frozen TFDV schema, and the feature list that `pipelines/model` derives from
  `schema.feature_columns()`. All are described in the PR.

## Design

### 1. Evidence store: replay experiment entries

- New entry kind beside the existing `<kind>-<run_id>/` run entries, named from the
  experiment's own content, not its delivery folder:
  `replay-<task name>-<started_at as %Y%m%dT%H%M%SZ>-<first 6 hex of experiment.json's SHA-256>`.
  Delivery names such as `attempt2` are not unique or meaningful, and the store cannot rename
  an entry later. *(designer's call)* The entry holds the experiment directory **byte for
  byte**: `experiment.json`,
  `source/{agent-evidence.json,live-journal.jsonl}`,
  `replays/<NN>/{agent-evidence.json,live-journal.jsonl}` and `injections/<NN>.json`. Any other
  file (for example vLLM metric dumps) is refused.
- `make add-evidence RUN=<dir>` accepts an experiment directory (it has an `experiment.json`).
  Before copying, it reads the experiment with `replay_evidence.read_experiment` and every
  journal line strictly. An unreadable experiment is refused with its reason; a readable
  experiment that failed its protocol checks **is** added, because labeling rejects it with a
  recorded reason and the failure is evidence.
- Same rules as run entries: staged and renamed atomically, never changed, an identical re-add
  does nothing, different content under a stored name is refused, no symlinks.
- Benoy's console logs, `batch-index.csv` and `labels-summary.txt` are delivery records, not
  evidence. The data card cites the two zip hashes and his notes instead.

### 2. Replay rows

For each stored experiment the build runs `label_experiment` (unchanged), then, per labeled
row, joins features:

- Parse that replay's `live-journal.jsonl` records into `StepEvent` and `ResourceSample` with
  the SDK models, and call `agentwatch_features.compute_feature_rows(events, samples,
  declared_interval)`. The declared interval is `configuration.sampling_interval_seconds` from
  the same replay's `agent-evidence.json`; when it is missing, coverage stays null.
- Join on `(run_id, step_id)`. A labeled step with no terminal feature row is rejected
  (`feature_row_missing`).
- Consistency gate: the feature row's duration, queue and execution seconds must equal the
  `experiment.json` timings the label used, within 1e-6 s. A mismatch rejects the step
  (`feature_timing_mismatch`): its label and features would describe different windows. The
  spike over two real experiments found 0 mismatches in 294 rows.

### 3. `steps-v1.3` columns

`steps-v1.2` columns are all kept, with the same names, types and roles. Changes:

| Column | Change | Why |
| --- | --- | --- |
| `workload_iterations`, `worker_capacity` | become nullable | Live registrations do not record them; null on replay rows |
| `cpu_seconds` | becomes nullable | The real journals record no process CPU time for model calls, retrievals and most tool steps |
| `rule_prediction`, `rule_version` | become nullable | Replay evidence has no rule output; M02 computes rules separately |
| `kind` | adds `agent_replay` | Row kind |
| `condition` | keeps `baseline`, `clean`, `contended` | Replay roles map baseline→`baseline`, clean_control→`clean`, contended→`contended` |
| `label_resource` | adds `mixed` | `labels-v2` |
| `label_basis` | union with the `labels-v2` bases | |
| `telemetry_status` | adds `unaligned` | `step-features-v1` status |
| `label_version` | per row: `labels-v1.1+…` or `labels-v2+…` | Two rule sets in one release |

New columns, all ID or LINEAGE, all in `LEAKY_COLUMNS` where they encode design:

| Column | Role | CPU/coding rows | Replay rows |
| --- | --- | --- | --- |
| `feature_version` | LINEAGE | `steps-v1.2` | `step-features-v1` |
| `task_group` | ID | `task_id` | task name without `-r<revision>` |
| `experiment`, `plan_sha256`, `replay_index`, `step_type` | ID | null | from labels |
| `replay_role`, `replay_condition` | LINEAGE, leaky | null | `role`, `condition` from labels |
| `label_resources` | LABEL | resource or empty | from labels; the empty string is allowed for this column only |
| `control_exceeded_noise` | LINEAGE | null | from labels |
| `configuration_sha256` | LINEAGE | null | the replay's registered configuration digest |

Replay rows fill the rest as follows: `source = "measured"`; `recorded_at` from the replay's
`run_created`/first record time; `workload` = `agent replay <task_id>`; `run_passed` = the
experiment's recorded `passed`; `failed_checks` from labels; `sampling_interval_seconds` =
declared, `measured_sampling_interval_seconds` = the feature row's observed interval;
`source_sha256` = canonical hash of that replay's two files plus `experiment.json`.

Feature columns stay the 14 `steps-v1.2` names. `feature_version` says which definition each
row's values follow; they use different windows (inclusive submission-to-finish versus
`[started, finished)`) and inputs, so they are labeled, not presented as one definition. This
is the explicit, versioned mapping `docs/step-features.md` requires.

### 4. Release and manifest

- `SCHEMA_VERSION = "steps-v1.3"`; the release name and digest rules are unchanged, so a new
  version appears and `steps-v1.2+5954f97ef8b2` stays untouched.
- `manifest.json` adds `label_versions` (counts per label version), `feature_version` counts,
  `kind` counts, `label_resources` per-resource counts, and `missing`: null count per column.
  All content-derived, so the manifest stays deterministic.
- `integrity.py` rebuilds replay rows from stored experiments the same way it rebuilds runs.

### 5. TFDV gate

- New frozen schema `pipelines/validation/schema/steps-v1.3.pbtxt`, generated with
  `make freeze-dataset-schema` from the first v1.3 release and reviewed in the PR.
  `steps.pbtxt` stays for v1.2. `VALIDATION_SCHEMA` defaults to the v1.3 file.
- Statistics are written per attempt, as today; they are the release's validation statistics.
  Drift against the last passing v1.2 release is reported and does not fail the gate.

### 6. `split-v2` and `eval-contract-v2`

- `eval-contract-v1` is kept unchanged as `pipelines/model/evaluation-contract-v1.toml`, and
  `steps.pbtxt` stays, so `split-v1+58e921fa846a` on the v1.2 release and the #45/#46 results
  still rebuild when those paths are passed. The contract's hash covers its content, not its
  path. *(designer's call)*
- `pipelines/model/evaluation-contract.toml` becomes `eval-contract-v2`, status `accepted`:
  `schema_version = "steps-v1.3"`, `label_rules = ["labels-v1.1", "labels-v2"]`,
  `[splits] method = "split-v2"`, `group_key = "task_group"`,
  `assignment = "pipelines/data/src/agentwatch_data/split-assignment.toml"`. Thresholds are
  unchanged. `agentwatch_model.contract` reads the same file and is updated with it.
- `splits.load_contract` accepts `method` (default `split-v1` for older contracts),
  `assignment`, `task_group`, and a list of label rules. `make_split` and `load_split` call
  `split_assignment.partition_v2` for `split-v2`; the split id prefix is the method. The split
  file records the assignment's SHA-256 and the independence, held-out and mixed-in-test
  checks. `load_split` rebuilds the split and refuses any difference, as today.
- The import cycle (`split_assignment` imports `splits`) is broken by moving the shared hashing
  and split names into a small module both import.

### 7. Validation statistics and checks

Recorded deterministically in the manifest (release level) and the split file (split level):

| Check | Where | Pass condition |
| --- | --- | --- |
| Schema | build `validate_rows`, TFDV gate | No row or anomaly fails |
| Missingness | manifest `missing`, per `feature_version` in the split summary | Reported; nulls must match documented causes (below) |
| Class balance | split `class_balance` | Test meets the contract minimums: 30 per class, 100 clean, 30 mixed |
| Leakage | split `leakage` | No single feature's training AUC above 0.98 |
| Independence, hold-out, mixed in test | split `split_v2_checks` | `passed`, `met`, `met` |

How `mixed` counts *(designer's call, conservative; Rohit checks it in review)*:

- **Leakage:** one-vs-rest per class on membership in `label_resources`, so a `cpu,inference`
  step is positive for both `cpu_sandbox` and `inference`, and `none` means an empty
  `label_resources`. Counting a mixed step as a negative for its own resources would add label
  noise and lower the AUCs, making a leak harder to detect. For CPU and coding rows
  membership equals `label_resource`, so `split-v1` results do not change.
- **Mixed test minimum (30):** steps labeled `mixed`, meaning confirmed slowed by two or more
  resources, not every step of a mixed-condition replay. The D02 `mixed_in_test` count (rows
  under the `mixed` condition) is recorded beside it.

A **design-encoding report** is added to the split, report-only: among replay rows labeled
`none`, each feature's AUC for separating contended from clean replays. It is not a gate,
because live telemetry legitimately sees injected load on steps that were not slowed.

### 8. Pipeline and versioning

- The DAG and `make` targets need no new stage: acquire accepts experiment directories, and
  build, validate, split and version run as today. Slices gain `feature_version`, `kind` and
  `replay_condition`.
- After review, the evidence is added to `data/evidence/`, the release is built, gated and
  split, `data/*.dvc` are committed, and the snapshot is pushed to the team bucket. The owner
  approved publishing the replay evidence unredacted (2026-11-05), so it is curated as
  received and becomes readable through the `public` remote like the CPU and coding runs.

## Expected results on the real evidence (spike, not yet verified end to end)

- About 5,600 replay rows plus the existing CPU and coding rows.
- **Replay telemetry is mostly null.** Over two experiments, 276 of 294 feature rows had no
  PSI or cgroup reading: model calls and retrievals are `empty` (`step-features-v1` reads only
  CPU metrics, and only on the step's own resource), and millisecond tool steps are
  `no_overlap` at the 250 ms interval. Replay rows therefore carry mostly timing features.
  This is reported, not hidden.
- Test meets the class minimums on `fix-window-count` alone (cpu 127, inference 226,
  retrieval 206, mixed 93). Validation is empty, so calibration and validation-dependent
  gates are `inconclusive`.
- The leakage check may block. If it does, the split is kept as evidence and refused for
  evaluation, as today; nothing is retuned.

## Testing

- Synthetic, labeled-as-synthetic fixtures for: experiment curation (refusals, immutability,
  symlinks, extra files, protocol-failed experiments kept), the feature join (missing row,
  timing mismatch, declared interval missing), v1.3 schema (nullable columns, empty
  `label_resources` only where allowed), manifest statistics, `split-v2` through `make_split`
  and `load_split` (tampering, assignment change, contract change), contract parsing (v1 and
  v2), and integrity rebuild of replay rows.
- Reproduction: build and split twice into empty stores and compare versions, bytes, split id
  and `rows_sha256`; the versioning end-to-end test covers a fresh-clone rebuild.
- Full suites: `pipelines/data`, validation, versioning, orchestration, `pipelines/model`,
  `packages/features`, Ruff.
- The real release is built from Benoy's evidence and its numbers go into the data card.

## Left to the builder

Function and module boundaries, rejection-code names, the manifest field layout, and how the
design-encoding report is laid out, within the rules above.

## Owner decisions (2026-11-05)

1. **Public DVC bucket:** the replay evidence may be pushed unredacted. It holds full model
   conversations for the two synthetic tasks, injection logs and the laptop's resource
   identifiers, and no credentials or user paths (checked by grep before curation).
2. **Rohit's acceptance of `eval-contract-v2`:** given to the owner directly on 2026-11-05,
   not written down. Rohit's PR approval is the written confirmation.

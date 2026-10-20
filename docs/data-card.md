# AgentWatch step dataset — data card

Status: draft schema `steps-v1.2`, labels `labels-v1.1`. Recorded runs are curated into a
versioned evidence store (`make add-evidence`); releases are built from it (`make dataset`),
gated (`make validate-dataset`), split (`make split-dataset`), and versioned with DVC
(`make version-dataset`). Column
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
  number, `"10"` is not an integer, and neither an integer nor a numeric string such as
  `"1791200000"` is a timestamp (`created_at` must be an RFC 3339 date and time with an
  offset). A run that fails is recorded as a rejection with its reason; the other runs still
  extract.
- JSON is parsed strictly: an object that repeats a key is rejected rather than silently
  keeping the last value, and JSON nested more than 64 levels deep (recorded evidence nests 4)
  rejects its run instead of aborting the extraction. The limit is explicit so the result is
  the same on every supported Python version.
- Run directories and evidence files must be real, not symbolic links: a linked run directory
  or a linked `results.json` or `samples.jsonl` is rejected with a reason, never skipped or
  read from outside the data directory.
- The evidence must match its declared kind: a `coding` run declares `tests_per_step`, and a
  `cpu` run does not. A run in an `llm-*` directory must carry the agent evidence
  (`result["agent"]`); without it the LLM loop did not finish.
- Experiment check names must be lowercase identifiers (`[a-z][a-z0-9_]*`), so `failed_checks`
  stays unambiguous.
- The API URL must be `http://` or `https://` with a host. Network failures are retried with
  bounded exponential backoff for timeouts, connection errors, and HTTP 429, 500, 502, 503, and
  504. Each response must arrive within one overall deadline (the request timeout bounds the
  whole body, not each read) and within 64 MiB. If a source still cannot be read, or returns a
  malformed response, extraction stops with an error instead of producing a partial dataset.
- History pages must keep the API's contract: no more runs than the requested limit, and a
  `next_cursor` only after a full page, strictly smaller than the last. A page without
  `next_cursor` counts as malformed, never as the last page.
- A run found more than once, locally or through the API, is kept once. Copies whose
  normalized evidence disagrees are all rejected, including copies already set aside as
  duplicates, and that run ID stays rejected when the extraction is later combined with other
  sources. Which identical copy is kept never depends on source order: an API copy, then the
  earliest saved time, then the most telemetry (`present`, `empty`, `missing`), then the
  smallest raw-evidence hash.
- Each run records whether it carried resource samples: `present`, `empty`, or `missing`
  (no samples file). The API cannot distinguish an absent file from an empty one.
- Identical samples at the same timestamp are collapsed; disagreeing samples reject the run.

Flattening and labeling also reject untrustworthy runs one at a time, with a reason, while the
other runs continue (`flatten_runs` and `label_rows` return rows and rejections), and so does
schema validation, which checks each run's rows on its own. Only invalid label rules, an
unreadable source, no usable rows, or a defect in the builder itself stop a build, so one bad
run can never block every later build.

### Evidence store

Releases are built from `data/evidence/`, a curated store of recorded runs, rather than from
scratch output. `.data/` remains where demos and diagnostics write; a run becomes dataset
evidence only when it is added deliberately:

```sh
make add-evidence RUN=.data/coding-demo       # one or more recorded run directories
```

`python -m agentwatch_data.evidence` (`pipelines/data/src/agentwatch_data/evidence.py`)
validates each run with the same rules as extraction (`read_run`) and copies its
`results.json` and `samples.jsonl` byte for byte into `data/evidence/<kind prefix>-<run_id>/`.
An entry is never changed: adding the same evidence again does nothing (even if the JSON was
reformatted), and different evidence under a stored run ID, or the same run ID under another
kind, is refused. A missing samples file and an empty one are different evidence. Re-running a
demo, which overwrites its output directory, therefore cannot change versioned evidence. Copies
are staged under a hidden `.adding-*` name and renamed into place.

## Releases

Build a release with `make dataset`, which reads the evidence store; set `DATASET_SOURCES` to
combine sources, for example
`make dataset DATASET_SOURCES="local:data/evidence api:http://127.0.0.1:8090"`. Only a release
built from the evidence store alone can be rebuilt from its versioned evidence. The command runs
`python -m agentwatch_data.build` (`pipelines/data/src/agentwatch_data/build.py`), prints the
release directory, and exits 1 with a reason when the build fails.

A release is an immutable directory under `data/datasets/` named by its `dataset_version`.
Every successful build, whether it creates a release or reuses one, also writes its own report
under `data/datasets/builds/`:

| File | Contents |
|---|---|
| `<version>/steps.parquet` | Every row, every schema column, with Arrow types and nullability taken from `schema.py`; rows ordered by run, then condition, then baseline repeat |
| `<version>/schema.json` | The column contract (names, types, roles, nullability, allowed values, bounds), for tools that cannot import `schema.py` |
| `<version>/manifest.json` | Content only: release format (currently 2); dataset, schema, label, and builder versions; row, run, and task counts; label, basis, condition, and telemetry-status counts; SHA-256 of the canonical rows, the Parquet file, and `schema.json`. With the same code and locked dependencies, rebuilding the same evidence reproduces it byte for byte |
| `builds/<build id>.json` | That build's events: build time, the release it created or reused (`reused_release`), per-source run and rejection counts, duplicate copies, and every rejection with its stage and reason |

The command prints the release and report paths. Because rejections live in each build's
report, a rebuild whose accepted rows are unchanged still records evidence rejected that time.

A release is **published** only when a committed build report names it, and consumers should
select releases with `published_releases()` in `build.py`. They are ordered by their latest
report, so the newest is the release the last successful build created or reused, and
`make dataset` followed by `make validate-dataset` checks what was just built. A report
publishes nothing unless it names a real release directory (not a link) whose manifest
carries that version. Builds never delete releases, so
concurrent builds cannot remove each other's datasets. The report is staged before the release
directory appears, so most write failures leave nothing behind; if committing the report still
fails, the build fails and may leave an unpublished release directory, which the next
successful build verifies and publishes.

- `dataset_version` is `<schema version>+<first 12 hex digits of a SHA-256 over the rows' digest,
  the column contract, and the release format>`, so the same evidence, rules, and code always
  produce the same version, whatever the order of the sources and whatever time zone the API
  reports saved times in (they are hashed in UTC, as Parquet stores them). A changed run, rule, label fingerprint, contract, or release format
  produces a new one.
- Releases are never rewritten, so a release in an older format is never upgraded in place. A
  format 1 release (from before `schema.json` existed) stays as it was, and the next build
  publishes a new format 2 release of the same rows beside it, which becomes the newest
  published release.
- Building a version that already exists verifies its format and the hashes of its rows,
  Parquet file, and contract instead of rewriting it; a mismatch fails the build.
- A build writes everything into a staging directory, fsyncs it, and renames it into place only
  when it is complete, so a failed or crashed build never leaves a partial release, and never
  publishes one.
- The build fails, writing nothing, when a source cannot be read, the label rules are invalid,
  no usable rows remain, or the builder itself produces an invalid release. Individual
  untrustworthy runs, including runs whose rows break the schema (`stage: "schema"`), are
  rejected and listed in that build's report instead.
- Host paths never appear in a manifest or report: local sources are labeled by directory name, such as
  `<local:coding-demo>`, and the home directory by `~` (skipped when there is no home, or it
  is `/`, as in some containers).
- Staging directories and staged reports left by a killed build are hidden and harmless, but
  are not yet cleaned up automatically; that sweep is part of the planned hardening pass.

## Quality gate

`make validate-dataset` checks the newest published release with TensorFlow Data Validation
(`pipelines/validation`, a separate project with its own environment because TFDV brings
TensorFlow; it supports CPython 3.13 on Linux x86_64 and macOS arm64). It exits 1 when the gate
fails, so Airflow and scripts can stop on it.

- Every release ships its column contract as `schema.json` (hashed in the manifest). The gate
  verifies the Parquet and contract hashes before reading anything, then refuses the release if
  any Parquet column's physical type differs from the contract. This happens before columns are
  converted for TFDV, so a string column holding `"1"` cannot pass as an integer or boolean.
  It also refuses what TFDV cannot see once columns are converted: NaN in a float column
  (pandas would turn it into a missing value), an empty or whitespace-only string, and a
  release with no rows (TFDV checks nothing on zero examples).
- TFDV statistics are compared with a **frozen schema**,
  `pipelines/validation/schema/steps.pbtxt`. It is generated by `make freeze-dataset-schema`
  and changed only in a reviewed PR. Its domains come from the contract, not from the values
  seen so far: all four resource classes are allowed even though only two occur today, and
  free-text columns such as `workload` and `task_id` have no domain, so new tasks are not
  anomalies. Float columns disallow ±inf. Required columns have a fixed shape (one value in
  every row); nullable columns have none, so a column that is null in only some rows, such as
  `tests_per_step` when coding and CPU runs are mixed, passes.
- **Schema anomalies fail the gate:** a missing or unexpected column, a type change, a value
  outside its range, an unknown category, a missing required value, or a violated dataset
  constraint.
- **Drift is reported but does not fail the gate.** `label_resource`, `condition`,
  `label_basis`, `telemetry_status`, and `kind` are compared with the most recent passing
  release; a category share moving by more than 0.25 (L-infinity, a proposed threshold) is
  listed as drift. New contention or tasks legitimately shift these distributions.
- Each run of the gate is an **attempt**, committed as one immutable directory,
  `data/datasets/validations/<release>/<attempt>/`, holding `result.json` and the TFDV
  statistics it was computed from. The files are written to private staging and the directory
  is renamed into place, so a result is never paired with another attempt's statistics, and
  concurrent attempts never share files. `result.json` records the status, anomalies, drift,
  baseline release, the release's Parquet and contract hashes, the frozen schema's hash, the
  statistics' hash, and the gate and TFDV versions.
- A release's verdict is its **latest committed attempt**, ordered by validation time;
  earlier attempts are kept as evidence. Attempt files are fsynced before and after the
  rename. A committed attempt whose result is missing, unreadable, or names another attempt or
  release, or one validated in the future, makes the verdict **unknown** instead of letting an
  older attempt stand in for it; inspect it, and the verdict returns once it is repaired or
  removed.
- The drift baseline is the most recently validated other release whose verdict is a pass
  against the **same frozen schema**, using the exact statistics of that passing attempt
  (checked against its recorded hash). A release whose verdict is unknown is never a baseline.
- Main-suite tests fail if the frozen schema's columns or the gate's test fixture drift from
  this contract.

## Splits

`make split-dataset` freezes the train, validation, and test splits of the newest published
release (`pipelines/data/src/agentwatch_data/splits.py`).

- **Only gated releases.** The release must verify against its manifest, and its latest
  committed quality-gate attempt must be a pass for exactly its current Parquet and contract
  hashes and for the reviewed frozen schema. Otherwise nothing is written and the command says
  to run `make validate-dataset`.
- **Settings come from the evaluation contract** (`pipelines/model/evaluation-contract.toml`):
  the 70/15/15 fractions, the `task_id` group key, the sources allowed in the test split, and
  the leakage bound. The group key must be `task_id` and the test sources must be known
  sources. The release's schema and label versions must be ones the contract evaluates. The
  split records the contract version and its 8-digit hash.
- **Grouped and stable.** Each task is assigned by hashing `split-v1:<task_id>`, so all of a
  task's runs, baselines, and replays share one split. While the contract's fractions are
  unchanged, a task keeps its split in every later release: new tasks never move old ones, and a
  task once tested is never trained on. Changing the fractions moves tasks near the boundaries,
  so it needs a new contract version and a new test split. The fractions hold over many tasks,
  not for a handful.
- **Measured runs only in test.** A run whose source the contract does not allow in the test
  split is excluded there, not moved to training, and listed under `excluded_runs`.
- **Leakage check on the training split.** For every feature and class, the one-vs-rest AUC
  (Mann-Whitney, ties averaged, a missing reading ranked below every value, folded so that 1.0
  means perfect separation in either direction). Any feature above the contract bound (0.98)
  makes the check `blocked` (compared before rounding); fewer than two classes in training makes
  it `inconclusive`. The split is still written, as evidence, and the command warns that
  evaluation must not use it.
- **Held-out injection methods** are reported: a method counts as held out only if it appears
  in test and not in training or validation while training has another method. The split stage
  reports this requirement but does not force it; assignment by method lands with the second
  injection method.
- The split is written to `data/datasets/splits/<release>/<split id>.json`. Its content is
  deterministic and the split id is a hash of it, so rerunning reuses the file and a changed
  file is refused. It records each task's split, per-split counts (rows, runs, tasks, labels,
  conditions, sources, injection methods), and each split's `rows_sha256`, a hash of the
  release version and its rows' `(run_id, step_id)`: the test-split hash evaluation reports
  record.
- Training and evaluation read rows with `load_split(path, "train", contract, frozen_schema)`.
  On every read it rechecks the quality gate (the release's latest committed verdict must
  still be a pass against `frozen_schema`), then **rebuilds the split** from the release, the
  contract, and that schema and refuses the file unless it matches exactly. An edited
  assignment, exclusion, leakage status, or contract field is refused even if the file's id
  was recomputed, and so is a split kept from before a later failed validation, a schema
  change, or a contract change. There is no option to skip this.
- The verdict is read by the gate's own rules: a committed attempt that is missing its result,
  unreadable, malformed, or validated in the future makes it unknown, and the split stage
  refuses rather than falling back to an older attempt. The gate and the split stage run in
  separate environments, so each has its own reader; one shared table of cases
  (`pipelines/validation/tests/verdict_cases.json`) runs in both test suites, so they cannot
  disagree about a verdict.
- It refuses a split whose leakage check did not pass. Exploration can opt in with
  `allow_unchecked_leakage=True`, which skips only that check; its results must not be reported
  as an evaluation.

## Versioning

The evidence store and the dataset store (releases, build reports, quality-gate attempts, and
split files) are versioned with [DVC](https://dvc.org). Git holds two small pointer files,
`data/evidence.dvc` and `data/datasets.dvc`; DVC keeps the files in its cache and a remote.

```sh
make version-dataset   # verify both stores, then snapshot them; commit the data/*.dvc changes
make push-dataset      # upload the snapshot to the DVC remote
make pull-dataset      # download what this commit versions (TARGET=data/datasets/<release> for one)
make dataset-status    # what is not yet versioned, and (with a remote) what is not yet uploaded
```

- **DVC runs in its own project**, `pipelines/versioning` (DVC 3.67.1 with GCS support, about
  120 packages, its own `uv.lock`, CPython 3.13 and 3.14), so the shared workspace and
  `make bootstrap` are unchanged. The committed `.dvc/config` turns off DVC's usage analytics
  and update checks and holds no remote.
- **Nothing damaged or unreproducible is versioned.** `make version-dataset` first runs
  `python -m agentwatch_data.integrity`, which lists every problem and refuses to continue
  unless:
  - every evidence entry is a valid run under its own `<kind>-<run_id>` name, holding only
    `results.json` and `samples.jsonl`, and no run ID appears twice;
  - every release is exactly what its content produces: it holds only the files its format
    declares (`steps.parquet`, `schema.json`, `manifest.json`; format 1 had no `schema.json`),
    its files match the manifest's hashes, and the row digest, the content-addressed name
    (recomputed from the rows and the contract the release ships), every row's version
    columns, and every content-derived manifest field are recomputed and must match. A release
    edited together with its manifest is therefore refused, not just one whose hashes differ;
  - every release's rows satisfy the frozen column contract of their schema version
    (`pipelines/data/src/agentwatch_data/contracts/<schema version>.json`, byte for byte what
    the builder ships), checked before anything reads them: exactly the contract's columns and
    Arrow types, and every value valid. A release must ship exactly that contract (format 1
    shipped none and is held to its version's), and a schema version with no frozen contract is
    refused. A `steps.parquet` that matches its hash but is not readable Parquet, lacks or
    retypes a column, or holds a value the contract forbids is refused rather than crashing
    the check. A schema change freezes the new contract beside the old ones (a test enforces
    it), so releases of every version stay verifiable;
  - every row of a release names exactly the evidence stored for its run (run ID and
    raw-evidence hash, checked row by row, so one run's rows cannot name different evidence),
    so it can be rebuilt from versioned evidence;
  - every release not yet versioned **is** what its stored evidence builds: its runs are
    rebuilt from the evidence store under the current label rules, through the same code the
    build uses, and must reproduce its rows exactly (and, for the current format, its Parquet
    bytes). Agreeing hashes alone only prove a release names its evidence; this refuses an
    edited value, a dropped row, a re-encoded file, or a release built from other sources (an
    API build records save times the store does not hold). A release built under other label
    rules is refused until it is built again. Releases DVC reports unchanged since the committed
    version (`dvc data status --unchanged`) were rebuilt when they were versioned and the
    append-only check keeps them identical, so they are not rebuilt again, and changing the
    rules or the code later never blocks versioning;
  - every build report is readable and names a release that exists;
  - every quality-gate attempt is readable and holds exactly `result.json` and the
    `statistics.pbtxt` whose hash its result recorded;
  - every split file hashes to its id and belongs to an existing release;
  - nothing in either store is a symbolic link, or a folder where a file belongs (or the
    reverse): a version must hold the content itself, not a pointer to files outside it.
- **The stores are append-only.** Before snapshotting, `make version-dataset` asks DVC
  (`dvc data status --granular`) which already-versioned files were deleted or modified since
  the last snapshot or commit, and refuses if any were. A failed quality-gate verdict, a
  release, or evidence therefore cannot quietly disappear from the next version.
- **Nothing half-written or hidden is versioned.** `.dvcignore` excludes the hidden staging
  names every command writes before renaming its output into place (`.building-*`,
  `.adding-*`, `.attempt-*`, `.*.tmp`) and every other hidden entry inside the stores, such as
  macOS `.DS_Store` files; the dataset build skips hidden directories too.
- **Snapshots are incremental and history stays retrievable.** Every store file is
  immutable and new releases add new directories, so a new version uploads only new files, and
  checking out an older commit and pulling restores the stores exactly as they were then.
- **A versioned release is reproducible.** `pipelines/versioning/tests` versions the recorded
  run, pulls it into a fresh clone, and rebuilds there from the pulled evidence: the dataset
  version is the same and every release file is byte-identical. Quality-gate attempts and split
  files are pulled with the release, so `load_split` can still recheck the gate.
- **Reading a version.** Training and evaluation can `make pull-dataset`, or pull one release
  with what `load_split` needs to recheck it:
  `make pull-dataset TARGET="data/datasets/<release> data/datasets/validations/<release>
  data/datasets/splits/<release>"` (the release folder alone has no gate verdict or split). A
  single file can also be read at any git revision without pulling:
  `dvc.api.read("data/datasets/<release>/steps.parquet", rev="<commit>", mode="rb")`.

**Working with others.** Each pointer file is one checksum for a whole store, so two people
versioning different additions from the same commit change the same line. Because the stores
are append-only, the right merge is always the union, and DVC's merge driver computes it
(`.gitattributes` assigns it to `data/*.dvc`). Once per clone, run `make
dataset-merge-driver`. Then, to bring `main` into a branch that also versioned data:

```sh
git fetch origin
make fetch-dataset           # the driver reads both versions' listings from the DVC cache
git merge origin/main        # data/*.dvc merge into the union, without a conflict
make pull-dataset            # check out the other side's additions
make version-dataset         # confirms the merge lost nothing (append-only check)
```

GitHub's merge button does not run custom merge drivers, so a pull request whose pointer files
conflict with `main` is merged locally this way and pushed. `make pull-dataset` also refuses to
overwrite evidence or releases you added but have not versioned: version (or move) them first,
and never answer with DVC's suggested `--force`, which deletes them.

**Remotes.** No remote is committed until a shared one exists. For local work, add one that
stays out of git: `uv run --project pipelines/versioning dvc remote add --local -d storage
<directory>`. When workstream F provides the bucket, the GCS remote is committed as the default
(`dvc remote add -d gcs gs://<bucket>/agentwatch/dvc`), and the pointer files are committed
with each version. People authenticate with `gcloud auth application-default login`; CI and
Airflow use a service account, never a key in the repository. The bucket uses DVC's
content-addressed layout, so files are found through the pointers rather than by release name.

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
| `cpu_pressure_max_avg10` | float, nullable | Maximum host CPU PSI `some avg10` (%) over the samples inside the step; null unless every one of them carried PSI |
| `cpu_pressure_available` | bool | Whether every sample inside the step carried a PSI reading (and there was at least one) |
| `cgroup_throttled_usec_delta` | int, nullable | cgroup v2 throttled time gained between the first and last samples inside the step, a lower bound for the whole step; `telemetry_coverage` shows how much of the step the samples span |
| `cgroup_available` | bool | Whether a throttling delta was measured: at least two samples in the window, every one with a readable counter, and no decrease between consecutive readings |
| `telemetry_sample_count` | int | Resource samples inside the step window |
| `telemetry_coverage` | float, nullable | Samples inside the window (from submission to finish, both inclusive; a sample exactly where touching steps meet counts in both, so each step's throttling delta starts or ends at the same counter reading) ÷ whole declared sampling intervals in the step (`floor(duration / sampling_interval_seconds)`, in integer nanoseconds), capped at 1. A declared interval must be at least 1 ns. Judged against the declared interval because a measured one cannot reveal uniform sample loss. Null when telemetry is not `present`, no interval is declared, or the step is shorter than one interval |

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
  spellings such as `3` and `3.0`, and Windows line endings do not. The packaged label version
  is pinned in `test_dataset_labels.py`, so any change to the rules or `labels.py` fails CI
  until the PR updates the pin, and a change that can alter labels also bumps `version` (which
  the evaluation contract pins).
- A step's condition comes from its position in the run result (`baseline.steps`, `clean`,
  `contended`), and its `step_id` must agree; a renamed step fails instead of changing role.
- Steps must run one after another in that order, with no overlapping windows, and every step
  must report the same nonblank output `checksum`. Steps may touch (one finishing at the
  nanosecond the next is submitted). A baseline overlapping contention would bias
  the reference, and a step that did different work explains its own slowdown; the producer's
  `same_workload_output` check covers only clean against contended.
- Runs with unknown conditions, unknown or mixed kinds, repeated step IDs, fewer than two
  baseline steps, other than exactly one clean and one contended step, or any duration that
  is not a finite positive number (NaN, infinity, zero, booleans, or non-numeric) fail
  labeling instead of being guessed. Mixed contention is not defined in `labels-v1.1`.
  Flattening always yields exactly one clean and one contended row per run, because the
  ingestion contract has one of each, so that check only guards direct callers of
  `label_rows` and changes no label the pipeline produces.
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
| `sampling_interval_seconds` | float, nullable | Interval the run's sampler declared in its step telemetry (0.25 s for the current worker); null when not recorded. A declared value that is not a finite positive number, or clean and contended values that disagree, rejects the run |
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
- The split stage runs the contract's single-feature leakage check on the training split before
  any evaluation (see [Splits](#splits)).

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
  work sizes becomes several tasks, so a grouped split cannot yet guarantee unseen tasks. The
  recorded coding run is one task, so its whole release lands in the test split, the training
  split is empty, and the leakage check is `inconclusive`.
- Hardware telemetry is unavailable on macOS hosts.
- Experiment labels are operational proxies and still need natural-overload validation.

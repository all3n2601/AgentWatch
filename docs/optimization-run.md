# Running an optimization experiment

Status: O06 batch runner, implemented 2026-11-22 on branch `test/optimization-evaluation`,
verified with test doubles and by a short development pilot on the GPU laptop
([below](#development-pilot-2026-11-22)). `opt-contract-v2` was accepted on 2026-11-25 (#109).
The first evaluation ran on 2026-11-26 ([below](#evaluation-2026-11-26)): no promotion.

The runner executes a preregistered experiment ([experiment evidence](optimization-evidence.md))
on one host and writes the batch records that `make compare-optimization` validates. It applies
each batch's policy through the [concurrency policy](concurrency-policy.md) API. It never
derives the schedule, never evaluates gates, and never tunes anything between batches.

## Steps

1. **Plan.** Measure what this host and runner configuration would preregister:

   ```bash
   make optimization-plan TASKS="fix-window-count fix-slug-case fix-config-precedence fix-fahrenheit-offset" \
     HOST_ID=rtx5070-laptop-wsl2 \
     OPTIMIZATION_ARGS="--model-backend vllm --run-timeout 450 --inference-max-queue 8 --inference-max-wait 300 --test-max-queue 8 --test-max-wait 300"
   ```

   It prints the manifest's `workload.tasks` (each task's content hash and configuration
   digest), `queue_bounds`, `model`, `hardware`, `power` and `state`. Every flag that changes
   a run's configuration digest (model, sampling interval, run timeout, the GPU evidence flags
   `--dcgm-url`, `--gpm-file`, `--vllm-traces-file`) must be passed the same way to the run in
   step 3. The queue-bound flags are the exception: they are plan-only, and the run takes the
   bounds from the manifest's `queue_bounds` (it rejects `--inference-max-queue` and the other
   queue flags). The host label is chosen by the operator and must not contain a user or host
   name.
2. **Preregister.** Write `<experiment>/manifest.json` once, before any batch, from those blocks
   plus the contract's version and hash, purpose, arrivals, conditions and injectors, policies,
   and repetitions. Never edit it afterwards: every record names its hash.
3. **Run.** With the host quiet (on AC power with an unchanged power plan, no other GPU work,
   sleep disabled):

   ```bash
   make optimization-run EXPERIMENT=<experiment> HOST_ID=rtx5070-laptop-wsl2 \
     OPTIMIZATION_ARGS="--model-backend vllm --run-timeout 450"
   ```

   This derives `schedule.json` from the manifest, checks that the checkout's tasks and this
   runner's configuration digests are the preregistered ones and that every injector is one it
   can apply, then runs the slots in order.
4. **Compare.** `make compare-optimization EXPERIMENT=<experiment>` validates every record and
   writes the comparison record.

## Per batch

1. Wait until vLLM has nothing running or waiting (at most 30 s), then pause `--pacing`
   seconds (default 10). With Ollama, only the pause.
2. Read the host's identity. A host that no longer matches the manifest, or cannot be read,
   stops the experiment **before** the batch is attempted: no record is written and the slot
   stays runnable once the host is fixed.
3. Apply the slot's policy and read it back: the baseline has no limits, and a candidate's
   limits are set on the model server this runner calls (`inference_service`, ADR 0015). From
   here the batch is attempted and always gets a record. A policy that cannot be applied gives
   a `not_started` record.
4. Start the condition's injectors (after the declared ramp), write the attempt marker
   `runs/<NNN>/attempt.json` (the record's head, start time and clock), then submit each frozen
   task at its preregistered offset in one runner process: one clock, one admission controller
   and a policy watcher for the batch.
5. Stop the injectors, confirm their cleanup, and write `batches/<NNN>.json`. Run evidence
   stays in `runs/<NNN>/<NN>-<task>/`, and every step is appended to `runner-log.jsonl`.

| Record field | Source |
|---|---|
| `policy.version` | The version read back before the batch |
| `model`, `hardware`, `power`, `state` | Measured before the batch: the served model (vLLM: its revision-pinned name; Ollama: its listed digest), `/proc` CPU model and memory, NVML GPU, Windows power line status and active power plan, the task fixture index |
| `tasks[].submitted_tick_ns` | The batch clock's tick when the runner hands the task to its run |
| `tasks[].finished_tick_ns` | The root step's finish tick among the run's own journal records (its `run_id`); `null` when not journaled |
| `tasks[].state` | The run's state; a run deadline is `timed_out`; `_locally_undelivered` is dropped (delivery is judged per batch, below) |
| `tasks[].acceptance` | The run evidence's `task_passed`; `null` when absent |
| `tasks[].admission_waits` | Each control's longest admission wait in the run; `null` when none was recorded |
| `tasks[].cold` | `true` in warm-up batches only |
| `admission.starved_admissions` | Admissions refused for a full queue or an exceeded wait |
| `admission.leaked_permits` | Permits still open after the batch's drain |
| `injection` | Active only when every injector applied sustained, measured load; cleanup as each injector confirmed it |

A batch is `completed` only when nothing below happened. It is `aborted` when it was cancelled,
an injector did not apply sustained load or its cleanup was not confirmed, the policy changed
while it ran, a candidate's limit was not enforced, or a run's live delivery was incomplete. It
is `failed` when a run raised a runner error or never started, or when the batch itself raised
an error after submission or its record could not be built; such a record reports each task from
the evidence its run left (`cancelled` when the run left none) and records no ticks. Task
failures are outcomes of a completed batch, recorded as they happened.

## Stop, resume and restore

- Any batch that is not `completed` stops the experiment, so a broken host or service does not
  consume the schedule. `Ctrl+C` during a batch cancels its runs (or its injector ramp); that
  batch is recorded as `aborted` (or `not_started` if no task was submitted) and the
  experiment stops. `Ctrl+C` while waiting for the next batch (the idle wait and pacing wake on
  it) stops before that slot's policy is applied: no record is written and the slot stays
  runnable. A cancel after the policy is applied but before submission gives `not_started`.
- Records are never overwritten. Running again resumes at the first unrecorded slot; an
  attempted slot is never rerun to replace its result, and its pair stays unpaired.
- A process that stops mid-batch (sleep, WSL shutdown, a kill) leaves `runs/<NNN>/` without
  `batches/<NNN>.json`. The next run does not rerun that slot into the same directories: with
  the attempt marker it writes a `failed` record from the marker and the evidence the runs
  left (its `finished_at` is when the interruption was found), and without one, a
  `not_started` record. Either stops the experiment.
- On every exit the policy is restored to the baseline (no limits) and read back. The command
  exits non-zero, and the log says so, if that restore failed; check Controls.

## Development pilot (2026-11-22)

A pilot on development tasks only, run to check the runner on hardware and to time batches
before the evaluation is preregistered. It is not evaluation evidence and was never compared.

- **Host:** `rtx5070-laptop-wsl2` (RTX 5070 Laptop, WSL2), on AC power with the Balanced plan,
  sleep disabled, no other GPU process. vLLM `Qwen/Qwen3-1.7B@70d244cc86cc`, `max_num_seqs` 2,
  `gpu_memory_utilization` 0.6. Runner commit `fde030f`, `opt-contract-v1` (proposed).
- **Design:** two `development` manifests with tasks `fix-mean-empty-r1` (offset 0 s) and
  `fix-median-even-r1` (offset 5 s), candidates `inference-1` and `tests-1`, 10 pairs and 1
  warm-up, contended condition `prefill-flood-openai-chat-v1` at concurrency 2 and 2,048
  prompt tokens. One manifest lists the contended condition first, the other the clean one.
  Each was stopped with `SIGINT` after 7 records, so its 8th slot is recorded as `not_started`
  (cancelled during the injector ramp) or `aborted` (cancelled as its tasks finished).
- **Runner behaviour:** all 14 completed batches applied, read back and enforced their policy,
  matched the manifest's host identity, applied and cleaned up their injectors, and had no
  starved admissions or leaked permits. Every run passed its task's tests. The baseline was
  restored and read back after each stop.
- **Batch times** (first submission to last finish):

  | Condition | Configuration | Batches | Batch time |
  |---|---|---|---|
  | contended | baseline | 4 | 14.8–15.0 s |
  | contended | `inference-1` | 2 | 22.3–22.4 s; permit waits up to 4.3 s |
  | contended | `tests-1` | 1 | 14.9 s |
  | clean | baseline | 4 | 7.2–7.5 s |
  | clean | `inference-1`, `tests-1` | 3 | 7.2 s |

- **Planning consequences:** a slot takes about 18 s (clean) to 30 s (contended) with the
  default 10 s pacing, so a full evaluation schedule takes well under an hour. Repeat batches
  differed by about 1%. Under the prefill flood, which reaches vLLM without admission, a limit
  on the agent's own requests only queues them behind the flood; `inference-1` was 50% slower
  in both of its pairs. The design is kept unchanged for the evaluation, and a failing or
  inconclusive candidate is reported as such.
- **Evidence:** `.data/o06-pilot-20261122/{contended-first,clean-first}/` (manifest, schedule,
  batch records, run evidence, runner log), not committed.

## Evaluation (2026-11-26)

The first preregistered evaluation, on the same host and model as the pilot, with the pilot's
design unchanged except for the tasks.

- **Design:** one `evaluation` manifest (`opt-contract-v2`, accepted `0981612d`; runner
  `d84e0b6`) freezing `fix-window-count-r1`, `fix-slug-case-r1`, `fix-config-precedence-r1` and
  `fix-fahrenheit-offset-r1` at offsets 0, 5, 10 and 15 s, with the pilot's tasks named as
  `development_tasks`. Candidates `inference-1` and `tests-1`, clean then contended, 10 pairs
  and 1 warm-up per condition: 82 slots.
- **Runner behaviour:** all 82 batches completed in 58 minutes, every pair was sufficient, and
  the baseline was restored and read back at the end.
- **Result:** no candidate promoted. Median batch time:

  | Condition | Baseline | `inference-1` | `tests-1` |
  |---|---|---|---|
  | clean | 17.0 s | 17.0 s | 17.0 s |
  | contended | 31.9 s | 82.0 s | 31.6 s |

  `inference-1` was about 2.6 times slower under contention, for the reason the pilot showed.
  `tests-1` changed contended batch time by less than 1% and its p95 latency ratio was 1.17.
  The admission overhead verdict is `inconclusive`, so no candidate could pass every gate.
- **Tasks:** `fix-window-count-r1` failed its acceptance tests in every batch and
  `fix-config-precedence-r1` passed only in some contended batches. Its failures in candidate
  batches fail the quality and failure gates for both candidates, and its spread fails or
  leaves inconclusive their contended baseline stability. A later evaluation should freeze
  tasks the model passes reliably.
- **Details and evidence hashes:** O06 in the [project checklist](project-checklist.md).

## Known limitations

- Verified with test doubles (`services/agent/tests/test_optimization_run.py`, and
  `tests/contract/test_optimization_batch_records.py`, which passes every record through the
  O04 checker), the development pilot and one evaluation on one host above.
- Power is read from Windows (natively or through WSL interop). A native Linux or macOS host
  cannot run an experiment yet.
- Supported injectors: `prefill-flood-openai-chat-v1` (kind `inference`) and
  `cpu-spin-processes-v1` (kind `cpu`). Others are refused before the first batch.
- The model's identity is what the server reports, not a hash of the weights it loaded. For
  vLLM the digest is the `@revision` suffix of the served name, set by the operator.
- `hardware.host_id` is the operator's label and `power.profile` is the Windows power plan's
  name; both are free text that ends up in every shared record, so neither should name a
  person or a machine. No host name, user name, MAC address or serial number is collected.
- `tasks[].cold` comes from the schedule (warm-up batches), not from a measurement. That holds
  for vLLM, which keeps its weights loaded. Ollama can unload the model between batches (its
  `keep_alive` default is 5 minutes, and the pre-batch wait is only a pause there), so a pair
  task could hit a cold load and still count as warm. Run Ollama experiments with a
  `keep_alive` longer than the experiment, or treat their warm latencies with that caveat.
- Gates are not evaluated here; the comparison record says `not_evaluated`.

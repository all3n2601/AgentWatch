# Running an optimization experiment

Status: O06 batch runner, implemented 2026-11-22 on branch `test/optimization-evaluation` and
verified with test doubles only. No experiment has been run on hardware yet, and the contract
is still proposed. Nothing here is an optimization result.

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
   a run's configuration digest (model, sampling interval, run timeout, queue bounds, the GPU
   evidence flags `--dcgm-url`, `--gpm-file`, `--vllm-traces-file`) must be passed the same
   way to the run in step 3. The host label is chosen by the operator and must not contain a
   user or host name.
2. **Preregister.** Write `<experiment>/manifest.json` once, before any batch, from those blocks
   plus the contract's version and hash, purpose, arrivals, conditions and injectors, policies,
   and repetitions. Never edit it afterwards: every record names its hash.
3. **Run.** With the host quiet (on AC power with an unchanged power plan, no other GPU work,
   sleep disabled):

   ```bash
   make optimization-run EXPERIMENT=<experiment> HOST_ID=rtx5070-laptop-wsl2 OPTIMIZATION_ARGS="<the same flags>"
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
4. Start the condition's injectors (after the declared ramp), then submit each frozen task at
   its preregistered offset in one runner process: one clock, one admission controller and a
   policy watcher for the batch.
5. Stop the injectors, confirm their cleanup, and write `batches/<NNN>.json`. Run evidence
   stays in `runs/<NNN>/<NN>-<task>/`, and every step is appended to `runner-log.jsonl`.

| Record field | Source |
|---|---|
| `policy.version` | The version read back before the batch |
| `model`, `hardware`, `power`, `state` | Measured before the batch: the served model (vLLM: its revision-pinned name; Ollama: its listed digest), `/proc` CPU model and memory, NVML GPU, Windows power line status and active power plan, the task fixture index |
| `tasks[].submitted_tick_ns` | The batch clock's tick when the runner hands the task to its run |
| `tasks[].finished_tick_ns` | The root step's finish tick in the run's journal; `null` when not journaled |
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
is `failed` when a run raised a runner error or never started. Task failures are outcomes of a
completed batch, recorded as they happened.

## Stop, resume and restore

- Any batch that is not `completed` stops the experiment, so a broken host or service does not
  consume the schedule. `Ctrl+C` cancels the current batch's runs (or its injector ramp); that
  batch is recorded as `aborted` (or `not_started` if no task was submitted) and the
  experiment stops.
- Records are never overwritten. Running again resumes at the first unrecorded slot; an
  attempted slot is never rerun to replace its result, and its pair stays unpaired.
- On every exit the policy is restored to the baseline (no limits) and read back. The command
  exits non-zero, and the log says so, if that restore failed; check Controls.

## Known limitations

- Verified with test doubles only (`services/agent/tests/test_optimization_run.py`, and
  `tests/contract/test_optimization_batch_records.py`, which passes every record through the
  O04 checker). No hardware experiment has run.
- Power is read from Windows (natively or through WSL interop). A native Linux or macOS host
  cannot run an experiment yet.
- Supported injectors: `prefill-flood-openai-chat-v1` (kind `inference`) and
  `cpu-spin-processes-v1` (kind `cpu`). Others are refused before the first batch.
- The model's identity is what the server reports, not a hash of the weights it loaded.
- Gates are not evaluated here; the comparison record says `not_evaluated`.

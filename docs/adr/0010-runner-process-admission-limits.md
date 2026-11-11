# 0010: Enforce concurrency policy as runner-process admission permits

Date: November 9, 2026. Status: proposed for O03 review. Enforces the policy that
[the concurrency policy service](../concurrency-policy.md) (O02) stores, under the semantics of
[the optimization contract](../optimization-contract.md) (`opt-contract-v1`, still proposed).

O02 stores versioned limits for the `coding-agent` target: one for model (inference) requests
and one for test jobs. Nothing limited any work. The runner sent each model request and test
job as soon as its step was submitted, and every runner process ran a single task, so a limit
would have had nothing to act on.

**Admission.** `agentwatch_agent.admission.AdmissionController` is created once per runner
process and shared by that process's runs. A model request and each test job (`run_tests` and
the runner's verification) take a permit after the step is submitted and before it starts, and
give it back in a `finally` on every path. The permit for a model request is handed to the
thread making the request: when a run stops waiting (cancellation, run deadline), the request
can still be streaming from the model server, so its permit is returned only when the request
really ends. The slot is never handed to new work while the server may still be using it.

| Rule | Implementation |
| --- | --- |
| Separate limits | One queue and permit count per control, `inference_requests` and `test_jobs` |
| Fairness | One first-in, first-out queue per control; an arrival is admitted only if nobody waits ahead of it and a permit is free, so no request is overtaken |
| Bounded queues and waits | `max_queue_length` (0 to 1024; 0 refuses instead of queueing) and `max_wait_seconds` (up to 3600 s) per control, declared before the run and part of its configuration digest. A full queue or an exceeded wait fails the step and the run, with the reason |
| Cancellation and deadline | A waiter leaves the queue on cancellation or at the run deadline; cancelled or expired work is never admitted, even with free permits |
| Changes | The watcher reads the active version every second (`--policy-poll`). A new version applies to new admissions only: lowering a limit stops nothing that is running, and new admissions wait until enough permits are returned. Disable or restore to the baseline removes the limit, and waiters are admitted in order |
| Applicability | A control applies only to the service the policy names. The inference control names Ollama; a runner sending its model requests to vLLM records the inference limit as not enforced, with the reason, instead of limiting an unrelated workload. A version with no limit for a control (baseline, disable, restore to the baseline) is also recorded as not enforced, with that reason: `enforced` is true only while a limit is applied |
| Unreadable policy | The policy is read with bounded retries after local setup and before the run is registered (stage `policy`). If it cannot be read or applied, nothing runs (exit 4). During a run, a failed poll keeps the version in force and is counted; an unreachable service never becomes "no limits". A response the runner cannot use stops the watcher and admission: a version older than the one in force, a version whose content changed, a non-retryable HTTP status or a response that fails validation. Queued and new requests are refused at once (`policy_unavailable`), never admitted under limits no longer trusted; admitted work finishes, as when a limit is lowered. Each run then fails with one stable reason (a policy failure, not a runner error) and exits 1 like any other run that started and failed; exit 4 stays "nothing ran" |
| Opt-out | `--no-policy` reads no policy and admits without limits; evidence records `admission.status: not_configured`, never a fabricated version 0 |

**Scope.** Permits count one runner process's work. Two runner processes each enforce the limit
separately, so together they can exceed it. The controller, its process ID and
`enforcement_scope: runner_process` are recorded with every admission; nothing claims a global
limit. A limit across processes would need a shared coordinator with leases that expire when a
runner dies, which is not part of this change.

**Batches.** A process-local limit only matters when one process runs several tasks at once, so
`batch` runs up to 32 tasks concurrently in one process. They share one clock domain (one per
process lifetime, as the shared contract requires) and one controller, and the batch reads the
policy once before any run starts. `batch.json` records each run's outcome, the controller's
state, and whether every permit was returned once the runs closed; a permit still held by an
abandoned request is reported, not assumed returned. A run with no outcome (not started, or a
runner error) records `task_passed: null`: no verification ran, so no task result exists.

**Evidence.** Live StepEvents cannot carry extra fields, so admission evidence is local
(`agent-evidence.json`), keyed by step and span ID. Each admission records the control, the
policy version and limit at arrival and at the decision, whether the limit was enforced, the
queue and active counts at arrival, the queue bounds, the controller, process ID and scope, and
arrival, admission and release ticks on the run's clock, as they stand when the run closes (a
permit still held then by an abandoned request has no release tick yet). Arrival is ticked
before the controller's lock is taken, and waits and holds are differences of these ticks, so
the evidence and coverage measure the same interval. A run's `admission` counts its own
records; the shared controller's state is kept apart under `admission.process` and in
`batch.json`. A step's live queue delay therefore includes its admission wait; `coverage-v4`
splits it into admission wait and the remaining dispatch, apart from the model server's queue
and execution. A record joins a step only when its step and span IDs both match. A step that
needed a permit but has no usable record, or has conflicting records, reports both as unknown,
never as zero.

Consequences: `agent-runner-v4`, and configuration digests change (the configuration records
the admission settings; the policy version, which can change during a run, is evidence). `run`
now needs a reachable policy service unless `--no-policy` is given; the GPU diagnostics call
`run`, so they read the policy from the same worker they already report to. `experiment`,
`overhead` and `sampling-overhead` consult no policy and record no admission. Fixed-limit
evaluation on real hardware is O06; this change makes no claim that any limit improves anything.

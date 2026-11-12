# Instrumented coding agent (A02)

`agentwatch_agent` runs one fixed coding task with a real model and reports it to the
[live lifecycle API](../../docs/live-ingestion.md) as one correlated `live-contracts-v1` run.
Every model call, tool call and retrieval is a step under one root step and one trace.
Configuration, the conversation, model-server metrics and failure/cleanup details are kept in
local evidence keyed by the same step and span IDs.

## Run it

```sh
make infra-up          # PostgreSQL
make dev-worker        # live API on :8090, in its own terminal
make llm-up            # Ollama with qwen3:1.7b (CPU by default)
make agent-tasks       # list tasks and their pinned identities
make agent-run AGENT_TASK=fix-mean-empty
make agent-batch AGENT_TASK=fix-mean-empty REPEAT=2   # concurrent runs under the policy (below)
make agent-experiment AGENT_TASK=fix-mean-empty   # clean and contended replays (below)
```

Settings come from the same variables as the Docker LLM setup: `OLLAMA_URL`, `AGENTWATCH_LLM_MODEL`
and `AGENTWATCH_API_URL`. `--run-timeout` bounds the whole run (default 900 s). On the GPU
laptop, follow the WSL setup in [the GPU access notes](../../docs/gpu-measurement-access.md)
and export `OLLAMA_URL=http://127.0.0.1:11435`. Each run writes
`.data/agent-runs/<task>-<UTC time>-<id>/` containing:

- `agent-evidence.json`: task identity, full configuration and its digest, model endpoint,
  conversation, per-call Ollama metrics, timing breakdown and placement, tool calls
  (including rejected ones), runner verification, resource sampling (achieved intervals,
  missed rounds, sampler cost, GPU identity), delivery statistics, cancellation and cleanup.
- `live-journal.jsonl`: every registration, record and state change sent, in order.
- `vllm-metrics-changes.jsonl` (vLLM backend only): every series of vLLM's `/metrics` that
  changed in each sampling round, the first round in full, bounded at 64 MB. Local evidence for
  analysis beyond the registered metrics; never ingested.

Every model call streams (`model_transport: stream`), and its `stream_timing`
(`model-stream-timing-v1`) records, in the run's clock domain, when the request was sent and
when the reply's headers, chunks and first and last output arrived. Time to first output
includes the server's queue, prefill and the network; it is the runner's observation. With
tools in the request, Ollama 0.35.1 sends a tool-call reply in two or three chunks, so its first
output there is close to the whole request, not a first token.

### vLLM backend

```sh
make llm-down && make vllm-up   # vLLM v0.32.0 + trace collector; they do not share 8 GB
make dcgm-up                    # optional: DCGM device fields on 127.0.0.1:9400
uv run --package agentwatch-agent python -m agentwatch_agent run --task fix-mean-empty   --model-backend vllm --dcgm-url http://127.0.0.1:9400/metrics
```

`--model-backend vllm` (or `AGENTWATCH_MODEL_BACKEND=vllm`) calls `VLLM_MODEL`
(`Qwen/Qwen3-1.7B@70d244cc86cc`, BF16 at a pinned revision) at `VLLM_URL`
(`http://127.0.0.1:8000`). The Compose service fixes that name with its weights, so
changing `VLLM_MODEL` alone makes the calls fail rather than relabel the model. `run`, `experiment` and `sampling-overhead` accept it; the
inference injector then sends competing OpenAI-compatible chat requests
(`competing-openai-chat-v1`). Each model call carries its step's trace context, vLLM's
per-request span (measured queue, prefill, decode, first token, end to end) is read from
`--vllm-traces-file` (default `.data/otel/vllm-traces.jsonl`, written by the Compose
collector) after the run closes, and the call's `timing` becomes `vllm-request-trace-v1`
([ADR 0007](../../docs/adr/0007-vllm-request-traces-and-server-metrics.md)). The model server
is sampled through `/metrics` on a second thread. The collector runs as a non-root user; on a
Linux host, set `AGENTWATCH_OTEL_USER` to the traces folder owner's `$(id -u):$(id -g)`. A vLLM run and an Ollama run never share a
configuration digest: `model_backend` and `model_transport` are part of it.

### GPU profiling activity on WSL2

WSL2 refuses NVML GPM samples, the source of DCGM's profiling fields, so on the GPU laptop a
Windows process samples them ([ADR 0008](../../docs/adr/0008-windows-gpm-exporter-for-wsl2.md)):

```powershell
make gpm-exporter   # on Windows: writes .data/gpm/gpm.prom every 250 ms; Ctrl-C stops
```

Then pass `--gpm-file .data/gpm/gpm.prom` (or `AGENTWATCH_GPM_FILE`) to `run`, `experiment` or
`sampling-overhead` in WSL. Nine `gpu.gpm.*` metrics (SM utilization and occupancy, tensor-core
and DRAM activity, FP32/FP16 pipes, PCIe) are sampled on a `gpu-gpm-*` resource. A round with no
new exporter window records them `unavailable`, never the previous window again, and so does a
run's first round, whose window is only a baseline (the file may be left by a stopped exporter).

The same exit codes apply to `run`, `batch`, `overhead` and `resend`; `batch` exits with the most
severe of its runs' codes, in the order 3, 5, 4, 1, 0:

| Exit code | Meaning |
| --- | --- |
| 0 | Success: the task passed (`run`), or all evidence was delivered and the run is closed |
| 1 | The run failed or was cancelled; its evidence was delivered |
| 2 | Invalid usage: bad arguments, an unknown task or patch, or an unusable journal |
| 3 | Live evidence incomplete: run `make agent-resend JOURNAL=<path>` when the API is back |
| 4 | Nothing ran: local setup failed, the concurrency policy could not be read, or the live API did not register the run |
| 5 | The live API permanently refused some evidence; it stays in the journal and evidence file |
| 6 | `resend` only: the run is open and its journal has no closure (see below) |

Ctrl-C or SIGTERM cancels cleanly: work stops, the workspace and test processes are removed,
unfinished steps end as `cancelled`, and the run is closed with the signal as its reason.
A cancellation requested through the live API is picked up within `cancel_poll_seconds` (1 s).

## Tasks

Tasks live in `src/agentwatch_agent/task_fixtures/`: a small project whose tests fail, Markdown
reference documents, and two or more reviewed candidate patches that all apply cleanly. Only
the documented behaviour (found through retrieval) makes the tests pass.

`task_id` is `<name>-r<revision>`. `task_fixtures/index.toml` pins each task's content
SHA-256; loading refuses a task whose files changed without a revision bump, so a task ID
always covers the same inputs. Worker settings, model, limits and contention belong to the run
configuration, not the task identity. The configuration digest registered with the run is the
SHA-256 of canonical JSON covering task ID and content hash, model, chat options, prompt, tool
definitions, limits, sampling and polling intervals, and runner/contract versions. Endpoint
URLs are excluded: the same configuration can run on different hosts.

## What each live step means

| Step | Type | Resource | Timing | CPU |
| --- | --- | --- | --- | --- |
| `agent-run` | `other` | none | from the run's start (after registration and start-up) to its end | null |
| `model-call-N` | `model_call` | `model-server-*` (`inference`) | runner-side request wall time | null: inference runs in the model server |
| `tool-N-search_docs` | `retrieval` | `docs-index-*` (`retrieval`) | in-process BM25 lookup | null |
| `tool-N-read_file`, `tool-N-apply_patch` | `tool` | `tool-host-*` (`cpu_sandbox`) | in-process | null |
| `tool-N-run_tests`, `verify-tests` | `tool` | `tool-host-*` (`cpu_sandbox`) | test subprocess | test process and children |
| `tool-N-rejected` | `tool` | none | never started | null |

- **Correlation:** all steps share one `trace_id`, are parented to `agent-run`, and use one
  monotonic clock domain (`agent-runner-<uuid>`) for the runner process lifetime.
  `observed_at` is UTC wall time and is never subtracted from ticks.
- **Queue delay** of a model call is runner-side: its admission wait (below), then dispatch;
  local evidence records the two apart, and `coverage` reports them separately. With vLLM, the
  request's own wait in the server's queue is vLLM's measured span attribute, joined after the
  run and `unavailable` when no span arrived. With Ollama, time spent waiting inside the server
  is part of execution wall time, not a measured inference queue. Each Ollama call's
  `timing` (method `ollama-request-timing-v1`) keeps the runner's wall time and Ollama's
  total, load, prompt-evaluation and generation durations, and records the observed queue
  as `unavailable`: Ollama reports none. Its two differences (server time outside load,
  prompt and generation; runner time outside Ollama's total) are estimates across two
  clocks, and an impossible one is reported as `inconsistent`, never clamped.
- **Resource IDs** are opaque and new for every runner process. `model-server-*` identifies
  the model endpoint as used by this run, not an Ollama process lifetime.
- **Resource samples:** one thread reads every probe every 250 ms, each with its own tick in
  the runner's clock domain. The tool host gets five CPU metrics: host PSI `some avg10`, the
  runner cgroup's `throttled_usec`, and three cumulative counters whose increase across a step
  window measures that window: host PSI `some total`, the cgroup's `usage_usec` and the
  cgroup's own PSI `some total` ([ADR 0005](../../docs/adr/0005-cpu-counters-retrieval-timing-and-step-correlation.md)).
  The GPU (`gpu-*`, service `nvidia-nvml`, class `inference`) gets five
  NVML gauges: kernel and memory utilization, memory used, power and SM clock
  ([registry](../../docs/live-ingestion.md), [ADR 0003](../../docs/adr/0003-gpu-and-inference-measurement.md)).
  PSI is host-wide (in WSL2 the whole VM, including Docker), the cgroup counters cover the
  runner's whole cgroup (including its test subprocesses; in WSL2 every process in the distro)
  and NVML covers the whole device (every process on it), so none is step-exclusive. Tests
  still run in-process (`run_project_tests`). When `run_tests` moves into the A03 sandbox
  container (`run_project_tests_isolated`), test CPU and that container's throttling leave
  the runner's cgroup: that integration must update this scope and should register the
  sandbox container as its own resource. On macOS
  CPU metrics are `unsupported` and so are GPU metrics without NVML; a kernel without PSI or
  a cgroup without `cpu.pressure` is `unsupported`, a cpu.stat without a counter is
  `unavailable`, and a negative or unparsable value is `malformed`; a library that fails to initialize,
  lacks an entry point the binding needs, or a missing device is `unavailable`. An Ollama
  `run` always samples these 10 metrics, so it needs a live API that registers them, and the
  longest `--run-timeout` the sample budget allows at 250 ms is 2,000 s. A vLLM run adds the
  model server's 16 `inference.vllm.*` metrics (769 s), and `--dcgm-url` adds 8 `gpu.dcgm.*`
  metrics on a `gpu-dcgm-*` resource (service `nvidia-dcgm`; 588 s with both). HTTP sources
  are read on their own thread with a 200 ms timeout, so a slow scrape never delays the CPU
  and NVML readings; a failed scrape makes that round `unavailable`. None is ever zero. `run` samples GPU index 0 and records
  the device count, name, driver and memory size under `resource_sampling.resources`.
  No retrieval metric exists: retrieval is in-process BM25. Each `search_docs` call instead
  records `retrieval-timing-v1` in its evidence: the search's own monotonic wall and thread
  CPU time (service time), passages scored, and the index queue as `unsupported`, because a
  synchronous in-process index never queues a request. The step's queue delay is runner
  dispatch. `retrieval_index` records the index build during setup, which no step covers.
- **`task_passed`** in the evidence reports only the runner's verification. A run can still be
  `failed` with `task_passed: true` when a runner error (listed in `background_errors` or
  `cleanup_errors`) happened after verification; the run's state and reason say which.
- **Verification:** after the model's final answer, the runner reruns the tests itself
  (`verify-tests`). The run completes only if they pass; the model's claims are never the
  outcome. A wrong patch closes the run as `failed`, and the root step fails with the same
  reason. If the tests cannot finish (a timeout, or the run limit), the run fails with that
  cause instead, never as a test failure.

Model metrics (token counts and Ollama's nanosecond durations) cannot be carried by
`StepEvent`, which rejects extra fields. They are in `agent-evidence.json` under `model_calls`,
keyed by `step_id` and `span_id`.

## Failure, cancellation and delivery

- Invalid model calls (unknown tool, missing or extra arguments, values outside the task's
  allowlists) are never executed. They are recorded as steps that go from `queued` straight
  to `failed`, and the model receives an error so it can correct itself.
- Model transport errors, malformed replies, an empty answer, and the model-call (8),
  tool-call (12) and run (900 s) limits fail the run with a redacted live reason. The test
  limit (60 s) fails the run only during the runner's own `verify-tests`; a model-requested
  `run_tests` that times out is returned to the model as a tool error. The run limit always
  has one wording, `Run exceeded its <N> s time limit`, wherever it is hit. Full error text
  stays in local evidence. Reasons are stripped of paths and control characters.
- A model request, and the optional `/api/ps` placement reading after it, cannot be
  interrupted, so on cancellation or timeout its thread is abandoned. An abandoned model
  request is recorded as `cleanup.abandoned_model_request`, and Ollama stops generating
  within the 512-token `num_predict` bound. An abandoned placement reading is recorded as
  `unavailable`.
- Records are delivered in the background in batches of up to 256, with bounded retries that
  reuse identical content. A batch the service rejects is resent record by record, so only the
  refused records are set aside, such as a step first reported after a remote cancellation
  closed the run. Everything is journaled, and `resend` replays a journal idempotently.
- A run is closed only after its records are delivered. If delivery is still failing at the
  end, the closure is journaled but not sent (sending it would make the service refuse the
  undelivered steps), and the run reports `<state>_locally_undelivered` with exit 3. Once the
  bounded in-memory queue overflows during a long outage, every later record also stays
  journal-only, so step revisions always reach the service in order.
- `resend` replays the records, then the closure, with their original identities. It retries
  a closure once if the run's revision moved. A closure adopted from another actor (such as a
  dashboard cancellation) is journaled before anything else, as `remote_closure`, so `resend`
  can settle the run even if the runner lost contact while adopting it.
- If the service refuses completion although every record was delivered, its refusal is
  permanent (refused or expired step evidence), so the run is closed as `failed` with the
  reason `Completion refused by the live API: ...` and exit 5, both live and on `resend`.
- Records the service refused are permanent too: it rejects identical content again, so both
  `run` and `resend` report them with exit 5, not exit 3.
- If registration times out, the run's outcome on the service is unknown. Nothing runs
  (exit 4), and a `failed` closure is journaled: `resend` then resolves the registration by
  its original request ID and closes the run if the service recorded it. A journal from a
  registration the service refused is not replayable (`resend` exits 2).
- A journal with no closure (the runner was killed, or is still running) makes `resend` exit
  6 without changing the run. Once you know the runner is gone, `resend --fail-unclosed`
  closes the run as `failed`.
- A live API outage when the run moves to `running` fails the run with the recoverable reason
  `Live API unavailable when starting the run`; it exits 3 if the closure could not be
  delivered either, or 1 if the API recovered in time to receive it.
- Steps still open at the end are closed first, before background threads stop and the
  workspace is removed, so their timings never include that shutdown. Every cleanup action
  then runs even if an earlier one fails: failures are listed in `cleanup_errors`, and a
  background thread (flusher, sampler or watcher) that fails is listed in
  `background_errors` and fails the run. The first error is re-raised after the evidence is
  written, except a policy watcher's: that is a policy failure, not a runner bug, and fails
  the run with its own reason (Concurrency limits, below).
- A run failed (rather than cancelled) by another actor ends its open steps as `failed` too.
- Ctrl-C and SIGTERM are handled without printing or blocking, for `run` and `overhead`: a
  repeated signal is ignored rather than waiting on a lock the first one holds. `overhead`
  closes its run as `cancelled` and exits 1, or 3 if the closure could not be delivered.
- `cleanup_confirmed_by_service` stays null. The runner reports what it actually cleaned up,
  but enforced cleanup of the test process belongs to the isolated runner (A03, below).

## Concurrency limits (O03)

`run` and `batch` enforce the [concurrency policy](../../docs/concurrency-policy.md) for this
runner process ([ADR 0010](../../docs/adr/0010-runner-process-admission-limits.md)). Each model
request takes an `inference_requests` permit, and each test job (the `run_tests` tool and
the runner's verification) a `test_jobs` permit, after its step is submitted and before it
starts. Permits are returned on every path; a model request the run stopped waiting for keeps
its permit until the request really ends.

- **Scope:** one process. Several runner processes each enforce the limit separately, so
  together they can exceed it. The inference limit applies only to the service the policy
  names (Ollama); with vLLM it is recorded as not enforced. A version that sets no limit for a
  control (the baseline, disable, or a restore to the baseline) is recorded as not enforced
  too, with that reason.
- **Queues:** first in, first out, bounded per control by `--inference-max-queue` /
  `--test-max-queue` (default 64; 0 refuses instead of queueing) and `--inference-max-wait` /
  `--test-max-wait` (default 600 s). A full queue or an exceeded wait fails the step and the
  run with that reason; cancellation and the run deadline end a wait.
- **Changes:** the active version is read before the run is registered and then every
  `--policy-poll` seconds (1 s). A new version applies to new admissions only; lowering a
  limit stops nothing that is running. A transient poll failure keeps the version in force. A
  response the runner cannot use (an older or changed version, a non-retryable HTTP status, or
  one that fails validation) stops polling and admission at once: queued and new requests are
  refused (`policy_unavailable`), work already admitted finishes, and each run fails with the
  reason `Concurrency policy became unusable during the run; admission stopped` (exit 1, the
  error in `failure_detail`). If the policy cannot be read at the start, nothing runs (exit 4);
  `--no-policy` runs without limits and records `admission.status: not_configured`.
- **Batches:** `batch --task NAME [--task NAME ...] --repeat N` runs up to 32 tasks at once in
  this process, sharing one clock domain and one controller. It writes
  `.data/agent-batches/batch-<UTC time>-<id>/batch.json` (each run's outcome, the controller's
  state, and whether every permit came back) and each run's directory under `runs/`. A run
  with no outcome (not started, or a runner error) records `task_passed: null`, not `false`.
- **Evidence:** `admissions` in `agent-evidence.json` holds one record per admitted or refused
  step: control, outcome, policy version and limit at arrival and at the decision, whether the
  limit was enforced, queue and active counts at arrival, queue bounds, the controller, process
  ID and scope, and arrival, admission and release ticks on the run's clock, as they stand when
  the run closes. Waits and holds are differences of those ticks, the same ones coverage
  uses. `admission.controls` counts this run's own admissions, refusals, longest wait and
  permits held at closure; `admission.process` is the shared controller's state (every run of
  the process, with its policy history), and `admission.policy_watcher` the poller.
  `cleanup.admission_permits_held` counts permits still held by abandoned requests when the
  run closed. Coverage joins a record to a step only when both step and span IDs match; a step
  with more than one record is reported as `inconsistent`.

`experiment`, `overhead` and `sampling-overhead` read no policy and admit without limits.

## Safety boundary

Model output is untrusted. The model can only search the task documents, read files copied
from the fixture, apply one reviewed patch per file, and run the fixture's tests in a disposable
copy. It never supplies code or commands. The tests run in a subprocess with a scrubbed
environment, a timeout and process-group cleanup. That is a trusted-fixture boundary, **not an
isolation sandbox**. The isolated container runner (`agentwatch_sandbox.container`, A03,
[sandbox README](../sandbox/README.md)) is ready for the agent to adopt; until it does, and
until model-written code gets its own review, do not add free-form edits or commands.

## Sampling evidence

`uv run --package agentwatch-agent python -m agentwatch_agent coverage <run dir>` prints a
`coverage-v4` report from the run's journal and evidence. It refuses records of another run
and conflicting records under one identity, and excludes (and counts) records in a clock
domain the run never registered. Each step joins only its own registered resource and
service (`own`); other sampled resources are labeled `context`. Statuses follow the
FeatureRow vocabulary with the service join's meanings: `missing` (not executed, or no
resource or service), `empty` (no sample stream for the resource and service, as for the
model server and docs index), `unaligned`, `no_overlap` and `present`. `sampling` and `context`
are lists of entries carrying their resource and service (and, for `sampling`, clock), so a
resource sampled under two services, or in two clock domains, is never merged. Each executed
step also reports its queue split into admission wait and dispatch (a step that needed a permit
but has no usable admission record reports both as unknown), apart from service time (Ollama's
total, or the search alone), and the remainder as runner handling. It also reports, per
resource, the achieved sampling interval (median, p90, max), late and early rounds, and missed
rounds (rounds the declared cadence expects over the sampled span, minus rounds taken; after a
slow round the sampler takes the overdue rounds back to back, so those count as late and early,
not missed); p90 everywhere is the sorted element at `round(0.9 * (n - 1))`; per executed
step, its `[started, finished)` window, the sampling rounds inside it, coverage against the
declared interval, and min/mean/max of measured gauges (or the counter's increase, null after
a reset). Steps shorter than one interval report `shorter_than_interval`, not 0 or 100%.
Journal lines that fail contract validation are counted, never used. This is producer-side
evidence, not the service's window join (B02).

`... python -m agentwatch_agent sampling-overhead` runs protocol
`resource-sampling-overhead-v1` (declared in `sampling_overhead.py`) against the real model
server, without the live API: after 3 discarded warm-ups, the runner's first model request
for a task alternates 30 times without and 30 times with CPU and GPU sampling (records are
journaled as in a run). It reports wall-time and Ollama-duration medians and p90s, generated
token counts, and the sampler's per-round cost, with no threshold, in
`.data/sampling-overhead/`. A failed or interrupted measurement keeps what it measured, with
its status and error, and exits 1.

## Overhead

`uv run --package agentwatch-agent python -m agentwatch_agent overhead` runs protocol
`agent-overhead-v1` (declared in `overhead.py`): after 3 discarded warm-ups, 30 repetitions
per condition of a task's patched test suite, alternating uninstrumented and instrumented
(60 measured), with the live API running. It reports
wall-time medians and p90s and runner CPU per repetition, with no pass/fail threshold, in
`.data/agent-overhead/`. The fixture suite takes about 55 ms, shorter than one sampling
interval, so the result covers per-step instrumentation, not sustained sampling. Overhead
runs keep the task's own `task_id` (dataset splits group by it) and must be excluded from
training datasets. They are recognizable without runner internals: their single registered
resource uses `service_id` `agentwatch-agent-overhead`, there is no `agent-run` root step, and
their steps are named `overhead-<n>`. Their configuration digest also covers the
`agent-overhead-v1` protocol.

## Replay experiments

`make agent-experiment AGENT_TASK=fix-mean-empty` (or `... python -m agentwatch_agent
experiment --task <name>`) runs protocol `agent-replay-experiment-v1`
([ADR 0006](../../docs/adr/0006-bounded-injectors-and-agent-replay.md)) with the live API and
model server running. It records a source run first, or replays `--source <run dir>`
(`SOURCE=` for make), which must be a completed run of the same task content and model whose
verification passed.

- **Replay (`agent-replay-v1`).** Each model call receives the source's exact input, and the
  source's requested tool calls run in order, whatever the model replies, so every replay has
  the source's step IDs and tool work. The model really generates each time; each replay's
  evidence records under `replay` whether every reply and tool result matched the source
  (test results compare outcome and test count only, since their text includes timings).
  Its configuration digest covers the plan's content (`replay.plan_sha256`), so clean and
  contended replays of one plan register one digest.
- **Schedule.** 1 discarded warm-up, 5 clean baselines, then `cpu`, `inference`, `retrieval`
  and `mixed` (`cpu` + `inference`), each followed by a clean control: 14 replays. Each
  condition's injectors start and ramp for 2 s before its replay and stop after it, bounded to
  the replay's run limit plus 32 s. `--baseline-replays` (at least 3), `--warmup-replays`,
  `--conditions`, `--mixed`, `--ramp-seconds`, `--cpu-workers` (default: every available CPU),
  `--inference-concurrency` (2), `--inference-num-predict` (128) and `--retrieval-threads`
  (2) adjust it; the [injectors](../injectors/README.md) refuse values beyond their bounds.
- **Evidence** in `.data/agent-experiments/<task>-<UTC time>-<id>/`: `experiment.json` (the
  declared protocol, source, every replay's role, condition, run ID, state and step timings,
  baseline variability, per-step comparisons with the baseline, and checks, rewritten after
  every replay), `source/`, `replays/<NN>/` (each replay's evidence and journal, named by
  order only) and `injections/<NN>.json` (each contended replay's injection log, with ticks in
  that replay's clock domain).
- **Variability.** For every step completed in all baselines: count, mean, sample stdev,
  coefficient of variation, min and max of its total duration (submission to finish). The
  declared gate is the `agent-run` CV at most 0.15. Comparisons give each step's difference
  from its baseline mean, ratio and difference in baseline stdevs. They are observations, not
  labels.
- **Checks.** Every baseline, contended replay and clean control is gated. Warm-ups are
  recorded but never gated.
  - `baseline_replays_complete`, `contended_replays_complete` and `clean_controls_complete`:
    each scheduled replay of that role ran and passed verification.
  - `stable_baseline`: every baseline contributed to the CV, which is at most 0.15.
  - `same_steps_every_replay`: each gated replay has readable steps equal to the first
    baseline's.
  - `injectors_cleaned_up`.
  - `injectors_active`: sustained, measured load from every injector until the replay stopped
    it. No injector may stop at its duration bound or backstop. Every CPU worker must report
    and hash rounds must be above zero. The inference child must report no failed request,
    with at least one request completed or still generating at the stop. Retrieval needs
    searches above zero and no thread that errored or ended early. The prefill flood needs at
    least one completed request and no failed one, and the median prompt size the server
    reported must be within 15% of the declared target (a server started with a context
    limit below the prompt fails this); a missing size never counts.
  - `evidence_delivered`.

  A replay whose injector cleanup was not confirmed keeps the state
  `injector_cleanup_unconfirmed` (its run's own state is in `run_state`). The command exits 0
  only when the experiment completed and every check passed, 1 otherwise (evidence is kept),
  3 when a replay's journal needs `resend`, and 2 for invalid settings or a source that cannot
  be replayed.
- **Stopping.** Ctrl-C or SIGTERM cancels the current replay and any ramp, and runs nothing
  further. A replay the live API did not register stops the experiment. An injector whose
  cleanup is not confirmed stops it too (`stopped_injector_cleanup_unconfirmed`); check the
  host for leftover load.

Conditions, roles and injection logs exist only in `experiment.json` and `injections/`. They
are labeling metadata and must never become features. On WSL2, CPU contention also loads the
model server's container, and retrieval contention slows every runner thread (see the
[injector limitations](../injectors/README.md#limitations)).

## Capacity sweep

`capacity-sweep` grows the number of concurrent coding agents on one vLLM configuration and
records what each level does to throughput, pass rate, run duration, measured queueing,
failures and cleanup. It declares that configuration's capacity from a rule fixed in code
before any GPU session, validates the experiment labels against the queue time vLLM measured
for the same request, and then records a prefill-flood injection at half of capacity. The
design is in the [capacity-sweep spec](../../docs/superpowers/specs/2026-11-10-natural-overload-capacity-sweep-design.md);
the declared values are in `sweep_protocol.py` (`natural-overload-sweep-v1`) and are hashed
into `sweep.json`. The version freezes at the first GPU session: no evidence exists under v1
yet, so its values may still change until then, and any change after it is a new version. The
sweep runs without the concurrency policy (`admission.status: not_configured`).

Results describe this laptop (RTX 5070 Laptop, 8 GB, WSL2) and the named configuration only,
never cloud capacity. The configurations are swept separately and never pooled. The measured
results of 2026-11-11 (capacity 3 agents on `cfg-default`, 8 on `cfg-hardware`) are in the
[capacity record](../../docs/capacity-record.md).

**Invocation.** One sweep per configuration, with the live API, vLLM and its trace collector
running. Two sources are required: completed, verified runs of two different tasks
(`fix-mean-empty` and `fix-median-even`), recorded once on vLLM with the declared model and
replayed by every sweep. A source recorded on another backend or model is refused before
anything runs. `fix-window-count` cannot be a source: on vLLM `Qwen/Qwen3-1.7B` it failed
verification 4 times out of 4 (2026-11-11), although it passes on Ollama `qwen3:1.7b`.

```sh
# cfg-default: max-num-seqs 2, gpu-memory-utilization 0.6 (the Compose defaults)
make vllm-up
make agent-capacity-sweep CONFIG=cfg-default SOURCES=".data/sources/mean .data/sources/median"

# cfg-hardware: max-num-seqs 16, gpu-memory-utilization 0.8, then 0.75 or 0.7 if vLLM will not start
make vllm-down
VLLM_MAX_NUM_SEQS=16 VLLM_GPU_MEMORY_UTILIZATION=0.8 make vllm-up
make agent-capacity-sweep CONFIG=cfg-hardware SOURCES=".data/sources/mean .data/sources/median"
```

Give `SOURCES` as paths relative to the repository root: make splits it on spaces, and the
checkout's own path may contain one (`Agent Watch`). Restart vLLM between configurations; the
sweep refuses a server whose reported settings contradict the declared ones. Record each
attempt to start vLLM with `--start-attempt "UTIL:OUTCOME"` (repeatable;
`START_ATTEMPTS="0.8:failed: CUDA out of memory;0.75:started"` for make, which puts the
value in single quotes and splits it on `;`, so an outcome may contain neither `'` nor `;`).
The last attempt must be `started`, and its utilization must be one of the configuration's
declared values (cfg-default: 0.6; cfg-hardware: 0.8, 0.75, 0.7) and the one vLLM reports;
vLLM running at a fallback (anything but the first declared value) without the attempts that
led to it is refused too. The model backend is vLLM only (the default for this command; Ollama is
refused), the served model must be the declared `Qwen/Qwen3-1.7B@70d244cc86cc`, and the flood
only targets a loopback vLLM. `--api-url` (default `AGENTWATCH_API_URL` or
`http://127.0.0.1:8090`) is the live API the runs report to. `--output-dir` (default
`.data/capacity-sweeps`, resolved to an absolute folder when the sweep starts) and
`--run-timeout` apply as for `batch`, but the run timeout defaults to the protocol's 450 s,
and any other value gives a different protocol hash. At 0.25 s with vLLM, DCGM and GPM, the
per-run sample budget allows about 465 s, so the runner's 900 s default would refuse every
run. A run timeout over the sample budget, or one whose flood bound (timeout + 32 s) exceeds
the injector's 1,800 s cap, is refused before anything runs. The
model, `--vllm-traces-file`, `--dcgm-url` and `--gpm-file` arguments are the `run`
command's. vLLM does not report `max_num_seqs`, so the configuration check reads the vLLM
container's launch arguments with `docker inspect` (`--vllm-container`, default
`agentwatch-vllm-1`; `docker` is tried, then `docker.exe` for Docker Desktop through WSL, each
bounded to 10 s): a `--max-num-seqs` or `--gpu-memory-utilization` that contradicts the
declared configuration (or the utilization vLLM reports) refuses the sweep. The launch
arguments count only when the inspected container serves the measured endpoint: it must publish
the `--vllm-url` port on a loopback or all-interfaces address (`serves_endpoint`). Two
containers cannot publish one host port, so another container's flags can never verify the
vLLM being measured. When Docker is unreachable, the inspect fails, the container is not
running or it does not publish the endpoint's port, `max_num_seqs` is recorded
`declared_not_verified` with the reason. There is no resume: a stopped sweep is kept and a new
invocation is a new sweep.

**Output.** `.data/capacity-sweeps/<configuration>-<UTC time>-<id>/`:

- `sweep.json`, rewritten after every batch: the protocol and its hash (including the vLLM
  image digest, the runners' 0.25 s sampling interval, their run limits and 10 s span wait,
  the `/metrics` read timeout, idle-poll interval and read retries, the p90 rule and the
  flood's prompt-size tolerance); the code's git revision (`code_revision`, with `dirty` for
  local changes) and `workload_sha256` over the two sources' `plan_sha256`, recorded but not
  hashed; declared and verified configuration, the container's launch arguments
  (`configuration.launch`) and the start attempts; host (platform, kernel, CPU count, GPU name
  and driver when NVML is available, otherwise recorded as unavailable); sources; every batch
  with its sources, per-run outcomes and classifications, vLLM counter changes and the sweep
  process's CPU and wall time around the batch (`sweep_process`: the whole process, every
  runner thread included); `server.read_failures` (readings that failed every retry);
  hard-stop events; level summaries; excluded batches; the capacity decision; label
  validation; the `max_num_seqs` check; `analysis_errors`; and the final `state`.
- `batches/<NN>-<phase>-n<N>-r<repeat>/` with each batch's `batch.json` and `runs/` (each
  run's evidence and journal).
- `injections/<NN>.json`: each injected batch's injection log, in that batch's clock domain.
- `analysis.json`, only after `sweep-analyse` (below).

Level summaries report, beside throughput, pass rate, durations, spans and device readings,
`agent_failures` (a run ended by the agent's own decision: the model-call or tool-call limit,
or a reply with neither a tool call nor an answer; it counts against the pass rate and is never
a hard stop), `live_api_failures`, `runs_without_evidence` (runs whose `agent-evidence.json`
is missing, unreadable or has no `model_calls` list, so their spans and replies are absent,
not zero), `batches_without_server_changes` (any such batch makes the level's vLLM counter
sums unknown, never a partial sum) and
`client_outside_server_seconds`: each model call's runner time outside vLLM's request span, as
the runner records it (client wall time minus the span's end to end, two durations on two
clocks, so an estimate). Calls without a measured span, or whose span outlasts the client wall
time, are counted `unavailable`.

**Phases.** Configuration check; one discarded warm-up batch; the ramp over N = 1, 2, 3, 4,
6, 8, 12 and 16 concurrent replays (3 batches per level; N = 1 runs 3 per source, so each
source has three baselines); the capacity decision; three live batches of free agents at
capacity, on the tasks of the declared sources (alternating within a batch and rotating
between repeats, as in the ramp); and three prefill-flood batches, each followed by a clean
control, at `max(1, floor(capacity / 2))`. vLLM's `/metrics` is read before and after every
batch, each reading tried 3 times 1 s apart. After each batch the driver waits until vLLM is
idle (polling every second, through failed readings, for at most 30 s), then pauses 10 s.

**Hard stops.** A run that fails for an infrastructure reason (the model server, the run
limit), vLLM still busy after the idle wait, unconfirmed injector cleanup and a vLLM restart.
Live-API problems stop the sweep or exclude the batch but never count against vLLM's capacity
(owner decision 2026-11-11): a run with incomplete delivery or a needed `resend`, refused
records or completion, a run the API never started, or the API unreachable when a run started
is classified `live_api_failure`, its batch is excluded (cause `live_api`), and the sweep stops
(`stopped_resend_needed`, `stopped_run_not_started` or `stopped_live_api_failure`). A vLLM
state that cannot be read within its retries is not a hard stop either: the sweep stops with
`stopped_server_state_unavailable`, and a batch without its reading after it is excluded. A
real infrastructure failure in a batch that also stopped the sweep is still that level's hard
stop.

**Capacity rule.** Capacity is the level before the first level that fails any criterion:
no hard-stop event, throughput at least 10% above the previous level's (N = 1 has none to
beat), pass rate no more than 20 points below N = 1's, p90 run duration at most half the run
limit, and `evidence_complete` (no batch without a throughput and no run without a
duration, so missing evidence never reads as slow or as passing). N = 1 also needs at least
one passed run. The ramp continues past the knee so the whole curve is recorded. The
`result` is `measured`, `below_1` (capacity is `null`), `at_least_ceiling` (every level
passed; capacity 16) or `incomplete`: no knee was found because, before any level failed,
the recorded ramp ended short of the ceiling, or a level without a hard stop has fewer usable
batches than it declares (6 at N = 1, 3 above), or the first failing level has no measured
failure, so what it measured is unknown. A level has no measured failure when it has no hard
stop and fails only the evidence criteria (`readable`, `evidence_complete`,
`summary_present`) and criteria marked `unmeasured`. A criterion is `unmeasured` when its
value is missing: no throughput, pass rate or run duration, or no readable batch at N = 1.
Such a criterion does not pass, but it never counts as an observed slowdown. An incomplete
result's capacity is the last fully passing level and must be read as a lower bound;
`incomplete_reason` says why, and `batch_counts` gives each level's usable, declared and
excluded batches. `first_failing_reasons` names the first failing level's hard-stop codes,
failed criteria and the unmeasured ones among them. A ramp batch with a live-API failure, or cut short by a
stop that is not a hard stop (cancellation, an unreadable server, a runner error), is excluded
from level metrics and listed in `excluded_batches` with its causes, never read as zero
throughput; its level then has fewer usable batches than declared, so the result is
`incomplete`.

**Label validation.** For model-call steps of N >= 2 ramp runs, a step is contended by the
label rule when its duration exceeds its N = 1 baseline mean by more than
`max(0.01 s, 3 x baseline stdev)`. The reference is the same threshold applied to the queue
time vLLM measured for that request. `sweep.json` reports the confusion matrix, precision,
recall, Spearman correlation and the share of excess explained by queueing versus longer
prefill and decode, per level and overall, and counts steps excluded for lacking a measured
span. The verdict is `validated` when precision and recall are both at least 0.90,
`not_validated` otherwise and `inconclusive` when either is undefined. Tool steps have no
measured reference and are reported `unvalidated`.

**States and exit codes.**

| `state` | Exit | Meaning |
| --- | --- | --- |
| `completed` | 0 | Every phase ran (a skipped live or injected phase when capacity is below 1 is recorded). With `analysis_errors`, exit 1. |
| `refused_configuration` | 2 | vLLM contradicts a declared setting or reports no start time. Invalid arguments, an unreplayable source, two sources of one task and a bad start attempt also exit 2 before anything runs. |
| `stopped_resend_needed` | 3 | A run's journal needs `resend`; this outranks the other stops. |
| `stopped_live_api_failure` | 1 | The live API refused a run's records or completion, or was unreachable when a run started. |
| `stopped_vllm_restart` | 1 | vLLM restarted during the sweep. |
| `stopped_injector_cleanup_unconfirmed` | 1 | An injector's stop was not confirmed; check the host for leftover load. |
| `stopped_injector_start_failed` | 1 | The flood failed to start (its cleanup confirmed); the batch never ran, and the injection log is kept. |
| `stopped_run_not_started` | 1 | A run was not registered with the live API. |
| `stopped_warmup_failed` | 1 | A hard stop in the warm-up; nothing about capacity was measured. |
| `stopped_server_busy_after_idle_wait` | 1 | Outside the ramp, vLLM stayed busy after the idle wait. |
| `stopped_server_state_unavailable` | 1 | vLLM's metrics could not be read within the retries or the idle wait, in any phase. |
| `cancelled` | 1 | Ctrl-C or SIGTERM cancelled the current batch and any injector. |
| `failed` | 1 | An unexpected error; it is recorded and then re-raised. |

A hard stop inside the ramp ends the ramp only (`ramp_end: hard_stop`); the later phases
still run when capacity is at least 1. `ramp_end` is `ceiling`, `hard_stop`, `stopped` or
`cancelled`. `stop_reason` and `stop_causes` give what stopped a sweep. A stopped or
cancelled sweep still analyses the ramp batches it recorded: `analysis_scope` is `full_ramp`,
`partial_ramp` (read the capacity decision as covering the recorded levels only) or `none`.
`failed` can hide unconfirmed cleanup: an injector whose stop was not confirmed is recorded
before the error is re-raised, so after any `failed` sweep check `hard_stops` for
`injector_cleanup_unconfirmed` and the host for leftover load.

An analysis that raises never replaces the sweep's `state` or costs its evidence: a batch
whose metrics raise is recorded with status `analysis_error` and fails its level's `readable`
criterion while the sweep continues, and a failed capacity decision or label validation is
recorded in `analysis_errors` (a failed decision skips the live and injected phases, which
need a capacity). A `completed` sweep with analysis errors exits 1.

**Offline analysis.** `sweep-analyse <sweep folder>` rebuilds the analysis of a recorded
sweep from its folder alone, without vLLM, a GPU or the live API: every ramp batch's metrics
read again from disk, level summaries (integer level keys while analysing, ramp batches only,
with the same exclusions as the sweep), the capacity decision, label validation and the
`max_num_seqs` check. It reads run folders under the sweep folder only (`batches/<NN>/runs/`),
never the path a runner recorded, so a copied sweep is analysed from its own files and a run
missing from the copy is counted missing. `analysis_scope` follows the sweep's own mapping
(`none` for a sweep that never ran a batch). It writes `analysis.json` beside `sweep.json`,
which it never modifies, with `protocol_sha256_matches` false if the recorded protocol was
edited. Use it after a `failed` sweep, an analysis error, or to re-check one. It exits 0 when
`analysis.json` is written and 2, with the reason, when the folder has no usable `sweep.json`
or a field the analysis reads is malformed.

```sh
uv run --package agentwatch-agent python -m agentwatch_agent sweep-analyse \
  .data/capacity-sweeps/<configuration>-<UTC time>-<id>
```

**Record tables.** `sweep-record <sweep folder> [<sweep folder> ...]` prints the Markdown tables
of the [capacity record](../../docs/capacity-record.md) for each sweep, in order: the verified
configuration, the ramp, the capacity decision, label validation, the live confirmation and the
injected condition. It reads `sweep.json` and the injection logs only and writes nothing.
`--coverage-jsonl FILE` also prints the sampling coverage tables (`coverage` on the first run of
the first ramp batch at each level, read under the sweep folder) and writes those raw reports to
FILE. A value that was not recorded prints as `n/a`, never 0. It exits 0 when the tables are
printed, and 2, with the reason, when a folder's record, injection log or runs cannot support
them. The tables use `±`, `≥` and `✗`, so redirect its output to a UTF-8 file.

```sh
uv run --package agentwatch-agent python -m agentwatch_agent sweep-record \
  .data/capacity-sweeps/<cfg-default sweep> .data/capacity-sweeps/<cfg-hardware sweep> \
  --coverage-jsonl .data/coverage-per-level.jsonl > tables.md
```

**Evidence status.** Natural-overload and prefill-flood runs are held-out test evidence
(`held_out: true`): never use them for training, calibration, threshold selection or
tuning. `rules-v1` predicts only `cpu_sandbox` or `none`; it cannot predict inference, and
this command does not score it. Scoring rules and models on this evidence follows in a later
change. Concurrent replays share prompt prefixes, so they load vLLM less than free agents;
the live confirmation shows the difference.

## Tests

```sh
uv run pytest services/agent/tests services/sandbox/tests
AGENTWATCH_TEST_DATABASE_URL=postgresql://agentwatch:agentwatch_dev@localhost:5432/agentwatch_test \
  uv run pytest services/agent/tests/test_live_integration.py
```

Unit tests use a scripted model and an in-memory live server. Both are labeled test doubles,
and their output is never run evidence. The integration tests serve the real worker API with
uvicorn, on an isolated PostgreSQL schema, through the SDK `LiveClient`. Real-model runs are
recorded under [the project checklist](../../docs/project-checklist.md).

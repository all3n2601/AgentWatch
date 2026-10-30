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

`--model-backend vllm` (or `AGENTWATCH_MODEL_BACKEND=vllm`) serves `VLLM_MODEL`
(`Qwen/Qwen3-1.7B@70d244cc86cc`, BF16 at a pinned revision) from `VLLM_URL`
(`http://127.0.0.1:8000`). `run`, `experiment` and `sampling-overhead` accept it; the
inference injector then sends competing OpenAI-compatible chat requests
(`competing-openai-chat-v1`). Each model call carries its step's trace context, vLLM's
per-request span (measured queue, prefill, decode, first token, end to end) is read from
`--vllm-traces-file` (default `.data/otel/vllm-traces.jsonl`, written by the Compose
collector) after the run closes, and the call's `timing` becomes `vllm-request-trace-v1`
([ADR 0007](../../docs/adr/0007-vllm-request-traces-and-server-metrics.md)). The model server
is sampled through `/metrics` on a second thread. The collector runs as a non-root user; on a
Linux host, set `AGENTWATCH_OTEL_USER` to the traces folder owner's `$(id -u):$(id -g)`. A vLLM run and an Ollama run never share a
configuration digest: `model_backend` and `model_transport` are part of it.

The same exit codes apply to `run`, `overhead` and `resend`:

| Exit code | Meaning |
| --- | --- |
| 0 | Success: the task passed (`run`), or all evidence was delivered and the run is closed |
| 1 | The run failed or was cancelled; its evidence was delivered |
| 2 | Invalid usage: bad arguments, an unknown task or patch, or an unusable journal |
| 3 | Live evidence incomplete: run `make agent-resend JOURNAL=<path>` when the API is back |
| 4 | Nothing ran: local setup failed, or the live API did not register the run |
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
- **Queue delay** of a model call is runner-side dispatch only. With vLLM, the request's own
  wait in the server's queue is vLLM's measured span attribute, joined after the run and
  `unavailable` when no span arrived. With Ollama, time spent waiting inside the server is
  part of execution wall time, not a measured inference queue. Each Ollama call's
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
  written.
- A run failed (rather than cancelled) by another actor ends its open steps as `failed` too.
- Ctrl-C and SIGTERM are handled without printing or blocking, for `run` and `overhead`: a
  repeated signal is ignored rather than waiting on a lock the first one holds. `overhead`
  closes its run as `cancelled` and exits 1, or 3 if the closure could not be delivered.
- `cleanup_confirmed_by_service` stays null. The runner reports what it actually cleaned up,
  but enforced cleanup of the test process belongs to the isolated runner (A03, below).

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
`coverage-v3` report from the run's journal and evidence. It refuses records of another run
and conflicting records under one identity, and excludes (and counts) records in a clock
domain the run never registered. Each step joins only its own registered resource and
service (`own`); other sampled resources are labeled `context`. Statuses follow the
FeatureRow vocabulary with the service join's meanings: `missing` (not executed, or no
resource or service), `empty` (no sample stream for the resource and service, as for the
model server and docs index), `unaligned`, `no_overlap` and `present`. `sampling` and `context`
are lists of entries carrying their resource and service (and, for `sampling`, clock), so a
resource sampled under two services, or in two clock domains, is never merged. Each executed step also
reports its dispatch queue apart from service time (Ollama's total, or the search alone) and
the remainder as runner handling. It also reports, per resource, the achieved sampling
interval (median, p90, max), late and early rounds, and missed rounds (rounds the declared
cadence expects over the sampled span, minus rounds taken; after a slow round the sampler
takes the overdue rounds back to back, so those count as late and early, not missed); p90
everywhere is the sorted element at `round(0.9 * (n - 1))`; per executed step,
its `[started, finished)` window, the sampling rounds inside it, coverage against the
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
    searches above zero and no thread that errored or ended early.
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

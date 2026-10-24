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
- **Queue delay** of a model call is runner-side dispatch only. Time spent waiting inside
  Ollama is part of execution wall time, not a measured inference queue. Each call's
  `timing` (method `ollama-request-timing-v1`) keeps the runner's wall time and Ollama's
  total, load, prompt-evaluation and generation durations, and records the observed queue
  as `unavailable`: Ollama reports none. Its two differences (server time outside load,
  prompt and generation; runner time outside Ollama's total) are estimates across two
  clocks, and an impossible one is reported as `inconsistent`, never clamped.
- **Resource IDs** are opaque and new for every runner process. `model-server-*` identifies
  the model endpoint as used by this run, not an Ollama process lifetime.
- **Resource samples:** one thread reads every probe every 250 ms, each with its own tick in
  the runner's clock domain. The tool host gets CPU PSI `some avg10` and the runner cgroup's
  `throttled_usec`; the GPU (`gpu-*`, service `nvidia-nvml`, class `inference`) gets five
  NVML gauges: kernel and memory utilization, memory used, power and SM clock
  ([registry](../../docs/live-ingestion.md), [ADR 0003](../../docs/adr/0003-gpu-and-inference-measurement.md)).
  PSI is host-wide, the counter covers the runner's whole cgroup and NVML covers the whole
  device (every process on it), so none is step-exclusive. On macOS CPU metrics are
  `unsupported` and so are GPU metrics without NVML; a library that fails to initialize or
  a missing device is `unavailable`. None is ever zero. `run` samples GPU index 0 and records
  the device count, name, driver and memory size under `resource_sampling.resources`.
  No retrieval metric exists yet: retrieval is in-process BM25.
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
  but isolation and enforced cleanup belong to A03/O03.

## Safety boundary

Model output is untrusted. The model can only search the task documents, read files copied
from the fixture, apply one reviewed patch per file, and run the fixture's tests in a disposable
copy. It never supplies code or commands. The tests run in a subprocess with a scrubbed
environment, a timeout and process-group cleanup. That is a trusted-fixture boundary, **not an
isolation sandbox**: do not add free-form edits or commands without the A03 isolation design.

## Sampling evidence

`uv run --package agentwatch-agent python -m agentwatch_agent coverage <run dir>` prints a
`coverage-v1` report built only from the run's journal: per resource, the achieved sampling
interval (median, p90, max), late rounds and an estimate of missed rounds; per executed step,
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

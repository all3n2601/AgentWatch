# Bounded contention injectors (A05)

`agentwatch_injectors` applies controlled load to one designated resource for a replay
experiment ([ADR 0006](../../docs/adr/0006-bounded-injectors-and-agent-replay.md)). The agent's
`experiment` command uses it; see the [agent README](../agent/README.md#replay-experiments).
It has no dependency on the agent or the live API.

| Injector | Method | Contends | Load | Hard resource bounds |
| --- | --- | --- | --- | --- |
| `CpuSpinInjector` | `cpu-spin-processes-v1` | host CPUs (`cpu_sandbox`) | busy SHA-256 child processes | workers ≤ CPUs available to the process |
| `CompetingInferenceInjector` | `competing-ollama-generate-v1` | the run's model server (`inference`) | non-streaming `/api/generate` requests from one child process | loopback URL only; concurrency ≤ 4; `num_predict` ≤ 256; request timeout ≤ 120 s |
| `IndexQueryFloodInjector` | `in-process-index-query-flood-v1` | the runner process's interpreter, via an identical retrieval index (`retrieval`) | searches from threads in the runner's process | threads ≤ 4; 1–64 queries of ≤ 200 characters |

Every injector also has `max_duration_seconds` (at most 1,800 s).

## Lifecycle

- `start()` launches the workers and returns once every one applies load (at most 15 s,
  or the start fails and everything it started is stopped). The log records `started_tick_ns`
  and `active_tick_ns`.
- A watchdog stops the injector `max_duration_seconds` after `active_tick_ns`
  (`stop_reason: duration_bound`), even if the caller never does.
- Child processes stop themselves at a backstop of the bound plus 20 s, counted from their
  own start, so load ends even if the supervising process is killed.
- `stop()` is idempotent. It sets the stop event and waits 2 s for reports and exits,
  then terminates, then kills (2 s each). A thread cannot be killed, so a retrieval thread
  still running is reported, never assumed stopped.
- Cleanup is confirmed only when every worker is gone. Otherwise `stop()` raises
  `InjectorCleanupError` on every call, and the experiment stops with
  `stopped_injector_cleanup_unconfirmed`.
- `InjectionSet` runs one injector (isolated) or several (mixed) for one replay. It ramps for
  `ramp_seconds`, which cancellation interrupts, and stops every injector even when one fails.

## Injection log (`injection-log-v1`)

Each injector's log has its `injector`, `method`, `contended_resource_class`, `target`
(scope only: no paths, credentials or process IDs), `bounds` and `clock_id`. It also has
`started_tick_ns`, `active_tick_ns`, `stop_requested_tick_ns` and `stopped_tick_ns`, the
wall times `started_at` and `stopped_at`, `stop_reason`, measured `activity`, `cleanup` and
`errors`. Ticks are in the clock passed in. The experiment passes the replayed run's own
clock domain, so injection and step windows compare without crossing clocks. Activity that
was not reported (a worker stopped before reporting) is marked missing, never counted as
zero.

Injection logs are labeling metadata. They must never become model features.

## Limitations

- CPU load is host-wide. In WSL2 it loads the whole VM, including Docker containers such as
  the model server, so it can also slow inference.
- Inference contention shares Ollama's request slots and the device. Ollama reports no queue
  wait, so the waiting itself is not observed directly. Stopping closes the child's
  connections; requests in flight are counted as abandoned.
- Retrieval is an in-process BM25 index, not a networked service. Its contention slows every
  thread of the runner's process (sampling, delivery and dispatch too), so it is not
  retrieval-exclusive.
- Injector overhead has not been measured separately from the contention it applies.

## Tests

```sh
uv run pytest services/injectors/tests
```

Real child processes and threads are started, stopped, bounded and killed. The model server is
a loopback test double, so its responses are synthetic.

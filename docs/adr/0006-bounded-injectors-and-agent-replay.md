# 0006: Bound contention injectors in time and resources, and replay recorded agent runs

Date: October 25, 2026. Status: proposed for A05 review.

Labels need a step measured cleanly and under contention while it does the same work. An
agent run cannot provide that by itself: the model chooses the next tool call, so two runs of
one task can take different paths. Contention also has to be safe to apply on a shared laptop:
load that outlives an experiment, or targets the wrong service, corrupts every later
measurement and can disrupt unrelated work.

**Replay.** `agent-replay-v1` takes a completed run whose verification passed and replays it
call by call. Each model call receives the exact recorded input, and the tool calls the source
model requested run in order, whatever the model replies this time. Step IDs, inputs and
tool work are therefore the same in every replay. The model still generates on every call,
because that is the inference work being measured. Whether its reply matches the source is
recorded per call, never assumed. The source must be of the same task content, model, chat
options and tool definitions, and its conversation must account for every call. Otherwise
the plan is refused. A replay is an ordinary live run that reuses the runner's lifecycle,
cleanup, sampling and delivery. Its configuration digest covers the plan's content, not the
condition, so clean and contended replays of one plan share one workload digest.

**Experiment protocol.** `agent-replay-experiment-v1` declares, before measuring, discarded
warm-ups, at least three clean baselines, and each condition (CPU, inference, retrieval, mixed)
followed by a clean control that exposes drift. The variability statistic is each step's
coefficient of variation across the baselines. The gate is the root step's CV of at most
0.15, the upper end of the implementation plan's 10–15% target. Per-step comparisons with the
baseline are reported as observations, not labels. Labeling rules belong to the dataset
pipeline (D02).

**Injection metadata stays out of features.** Conditions, roles and injection logs live
only in the experiment manifest and its `injections/` files. Replay directories are named by
order, and nothing the live service or the replay evidence records names the condition.
Injection logs record ticks in the replayed run's own clock domain, so labeling can compare
the injection window with step windows without crossing clocks.

**Bounds and cleanup.** Every injector has a validated hard duration (at most 1,800 s; the
experiment uses the replay's run limit plus ramp plus a 30 s margin) and resource caps: CPU
workers at most the CPUs available, inference concurrency at most 4 with `num_predict` at most
256 and a 120 s request timeout, and retrieval threads at most 4. `start()` returns only once
the load is running, and a watchdog stops it at the bound. Child processes also stop
themselves at a backstop, so load ends if the supervisor dies. Stopping escalates from the
stop event to terminate and then kill. Cleanup is confirmed only when every worker is gone;
otherwise `stop()` raises every time, and the experiment stops. The inference injector
accepts only loopback model servers and uses the run's own model and `num_ctx`, so it
contends designated local capacity and never reloads the model.

**Honest scope.** CPU contention is host-wide (in WSL2, the whole VM, including the model
server's container). Retrieval is an in-process BM25 index, so retrieval contention floods an
identical index in the runner's own process: the runner's searches wait for its interpreter,
and so do its other threads. The log names that scope. A retrieval service in its own
container would give retrieval-exclusive contention; none exists yet.

Consequences: a new workspace package, `services/injectors`, holds the injectors with no
dependency on the agent. The agent runner's loop is extracted into `_converse` so a replay can
substitute it; normal runs are unchanged. The repository still has no labels for agent
replays, mixed-contention label rules or ties: D02 consumes the manifest and injection logs.
Injector overhead and coverage at other hosts are not established.

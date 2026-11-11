# A06 Natural overload and concurrency growth — design

Status: approved in chat 2026-11-10 (Benoy), section by section. Branch
`feature/natural-overload-capacity-sweep`. Partner reviewer: Akshayaa.

## Intent

Grow the number of concurrent coding agents on the GPU laptop step by step, under two
declared vLLM configurations, and record what each level does to throughput, variability,
measured contention, failures and cleanup. Declare the measured capacity of each
configuration from a rule fixed before any GPU session, instead of assuming that eight agents
fit. Use the same sweep to validate the experiment labels against a measured reference:
under natural overload (contention from the agents themselves, no injector), does the
labeling rule's "slower than its clean baseline" agree with the queue time vLLM measured for
the same request? Also record a new, held-out GPU injection method at a fixed level below
capacity, so later evaluations have injected evidence no training row has seen.

The results describe this laptop (RTX 5070 Laptop, 8 GB, WSL2) and these configurations
only, not cloud readiness. Fresh evidence goes to Akshay (dataset) and Rohit (evaluation);
Meghana shows the configurations.

## Findings that shaped the design

- `rules-v1` (`pipelines/model/src/agentwatch_model/rules.toml`) predicts only
  `cpu_sandbox` or `none`; it never predicts `inference` or `retrieval`.
- `steps-v1.3` (`pipelines/validation/schema/steps-v1.3.pbtxt`) has no vLLM queue, GPU or
  server feature. vLLM's per-request queue time exists only in each run's
  `agent-evidence.json` (`vllm-request-trace-v1` timing). Scoring the rules or a model on
  natural-overload GPU steps is therefore not meaningful until inference wait features and
  an inference rule exist (Benoy's M2 PR 4).
- The only held-out injection method in `split-v2` is `worker_queue_blocker` (CPU/coding).
  Both existing agent injectors (`competing-openai-chat-v1`, `cpu-spin-processes-v1`) appear
  in training replays, so neither is a held-out method for agent runs. ADR 0007 requires
  `split-v3` to change the held-out method.
- `batch` (O03, #60) runs up to 32 `AgentRunner`s in one process with one clock and one
  admission controller; replay runners are ordinary runners, so `batch` can run replays.
- vLLM v0.32 caches prefixes by default and Compose does not disable it. Identical prompts
  (concurrent replays of one source, or an injector repeating one prompt) share prefill work.

## Scope and pull requests

Three PRs, merged in this order:

1. **`feature/prefill-flood-injector`** — the held-out GPU injection method in
   `services/injectors`, with ADR 0011. Independent of the sweep and reviewable alone.
2. **`feature/natural-overload-capacity-sweep`** (this branch) — the `capacity-sweep`
   command, its analysis, the real GPU sessions and the capacity record. Uses PR 1 only for
   the injected condition.
3. **Later, a new branch from `main`** — score `rules-v1`, the M2 inference rule and the
   learned candidates on the same frozen evidence. Rohit's #61 (calibration and abstention,
   `746ec26`) and #62 (`mixed-scoring-v1`, `64c4e2c`) merged on 2026-11-10
   (00:48 and 00:59 UTC on 2026-11-11); PR 3 still waits
   for the M2 PR 4 inference features and rule, and records the versions it scores with.

Out of scope: policy-limit evaluation (O06; the sweep runs `--no-policy`), changing
`split-assignment.toml` or labels (Rohit and Akshay's decision), CPU/retrieval cases
(Akshayaa), and any claim about cloud capacity.

## Declared protocol `natural-overload-sweep-v1`

Fixed in code (`sweep.py`) and hashed into `sweep.json` before any GPU session. Changing any
value is a new protocol version. The version freezes at the first GPU session: no evidence
exists under v1 yet, so its values may still change under it until then (Ruling 20).

- **Model:** vLLM v0.32.0 (`deploy/compose/compose.yaml` image digest), `Qwen/Qwen3-1.7B` BF16
  at revision `70d244cc86cc`, `max-model-len=4096`, default prefix caching.
- **Configurations,** swept separately and never pooled:
  - `cfg-default`: `max-num-seqs=2`, `gpu-memory-utilization=0.6` (the Compose defaults).
  - `cfg-hardware`: `max-num-seqs=16`, `gpu-memory-utilization=0.8`. If vLLM does not start
    at 0.8, the operator tries 0.75, then 0.7; every attempt is recorded and the value that
    starts is the one used. No other fallback.
- **Workload:** concurrent replays (`agent-replay-v1`) of two frozen sources, one passing
  `run` of `fix-mean-empty` and one of `fix-window-count`, both recorded once under
  `cfg-default` and never re-recorded. Sources alternate within a batch.
- **Ramp:** N = 1, 2, 3, 4, 6, 8, 12, 16 concurrent replays (16 is the ceiling, below
  `batch`'s 32). One discarded warm-up batch per configuration. Three batches per level,
  except N = 1, which runs three batches per source (six), so each source has the three
  baselines the noise rule needs.
- **Pacing:** after each batch, wait until vLLM reports 0 running and 0 waiting requests (at
  most 30 s), then pause a fixed 10 s.
- **Hard stop** — the ramp ends at the first level with any of:
  - a run that fails for an infrastructure reason: out of memory, a model-server error or
    refused request, or the 900 s run limit;
  - unconfirmed cleanup: vLLM still busy after the 30 s idle wait, or injector cleanup not
    confirmed;
  - a vLLM restart (its `process_start_time_seconds` changed).

  Task failures (a wrong patch) do not stop the ramp; they are recorded. Live-API problems
  (incomplete delivery or a needed `resend`, refused records or completion, a run never
  started, the API unreachable at a run's start) stop the sweep or exclude the batch but never
  count against vLLM's capacity (owner decision 2026-11-11): such a run is
  `live_api_failure`, its batch is excluded and counted, and its level's result is
  `incomplete`. A vLLM state that cannot be read within the declared retries is not a hard stop
  either (Ruling 18): the sweep stops with `stopped_server_state_unavailable`.
- **Capacity:** levels are checked in order; capacity is the level before the first level
  that fails any criterion:
  - no hard-stop event in any of its batches;
  - throughput (passed runs per batch-minute, batch mean) at least 10% above the previous
    level's (N = 1 has no previous level and passes this criterion);
  - pass rate no more than 20 percentage points below N = 1's;
  - p90 run duration at most 50% of the run limit (450 s).
  - N = 1 must have at least one passed run, otherwise capacity is "below 1".

  The ramp continues past the knee to a hard stop or the ceiling, so the whole curve is
  recorded. If N = 1 fails, capacity is "below 1"; if every level passes, it is "at least 16
  (ceiling)".
- **Live confirmation:** three `batch` runs of free agents (live runs, alternating the two
  tasks) at the configuration's capacity N. Skipped, and recorded as skipped, when capacity
  is below 1.
- **Injected condition:** at N_inj = max(1, ⌊capacity / 2⌋), three batches with the prefill
  flood running alternate with three clean control batches at the same N, flood first.
  Injector settings: `concurrency=2`, `prompt_tokens=2048`, ramp 2 s, stopped after the
  batch, duration bound = run limit + 32 s. Skipped when capacity is below 1.
- **Admission:** `--no-policy`, recorded as `admission.status: not_configured`.
- **Held-out status:** every natural-overload and injected run is held-out test evidence.
  It must never be used for training, calibration, threshold selection or tuning. Whether
  `split-v3` holds them out formally is Rohit and Akshay's decision. Release prerequisite
  (Akshay, review of PR #64): no sweep evidence enters a dataset release until `split-v3`
  holds it out. That covers `prefill-flood-openai-chat-v1` in `held_out_injection_methods`, and
  also the natural-overload replays, which carry no injection method, so the hold-out must
  also match them by their sweep protocol (`natural-overload-sweep-v1`), not by method alone.

## PR 1: prefill-flood injector

New module `services/injectors/src/agentwatch_injectors/prefill.py`. The competing-chat
child loop in `inference.py` is refactored, without changing its behavior or its report, into
a request loop that takes a picklable request profile, so both injectors share one loop
instead of two copies.

- `PrefillFloodInjector(BoundedInjector)`, method `prefill-flood-openai-chat-v1`,
  `resource_class = "inference"`, vLLM's `/v1/chat/completions` only.
- One child process runs `concurrency` workers. Each loops: a non-streaming chat request
  whose user message is a unique nonce, `request {worker}-{sequence}-{run_token}` (a random
  token per injector run), followed by fixed,
  non-sensitive filler text sized to `prompt_tokens`, with `max_tokens=1`. Almost all the
  work is prefill and KV-cache allocation. The nonce comes first so the prefix cache cannot
  reuse an earlier request's prefill.
- Reuses `BoundedInjector`, `processes` (child group, halt, backstop) and `loopback_url`.
- Hard bounds, refused beyond them: loopback server only; `concurrency` ≤ 4;
  `prompt_tokens` ≤ 3,072; `max_tokens` fixed at 1; request timeout ≤ 120 s;
  `max_duration_seconds` ≤ 1,800 with the standard backstop.
- Filler sizing: a fixed numbered-line text whose token count per line is declared; the
  injector records the target and vLLM's reported `usage.prompt_tokens` per completion
  (min, median, max), so the log shows the prefill the server actually received.
- Activity: requests sent, completed, failed and abandoned at stop. Unreported activity is
  missing, never zero. Prompts are never logged; only counts and token sizes.
- Start succeeds once every worker has sent its first request; an unreachable server fails
  the start and halts what it started. A non-200 or unusable reply is a `PrefillRejected`
  failure followed by the existing 0.5 s backoff.
- `injectors_active` for this method (at least one completed request, no failed request) is
  implemented in PR 2 (capacity-sweep plan Task 1); until then `experiment._sustained` does
  not accept kind "prefill".
- ADR 0011 records why it is the GPU held-out method, how it differs from competing chat
  (prefill-heavy versus decode-heavy), the prefix-cache defence, and that formal hold-out
  needs `split-v3`.

## PR 2: the `capacity-sweep` command

### Components

- `services/agent/src/agentwatch_agent/sweep.py` — the protocol declaration, configuration
  check and driver.
- `services/agent/src/agentwatch_agent/sweep_analysis.py` — per-level metrics, the capacity
  decision and label validation, computed only from recorded evidence; rerunnable on any
  sweep folder without the GPU.
- `cli.py` — `capacity-sweep` subcommand; Makefile target `agent-capacity-sweep`.
- Reused: `AgentBatch`, the replay runner and the experiment's source validation,
  `InjectionSet`, the vLLM metrics reader and `write_json_atomic`.

### Invocation

```
capacity-sweep --configuration cfg-default|cfg-hardware \
  --source <fix-mean-empty run dir> --source <fix-window-count run dir> \
  [--vllm-traces-file ...] [--dcgm-url ...] [--gpm-file ...]
```

One invocation per configuration; the operator restarts vLLM with that configuration first.
Output: `.data/capacity-sweeps/<configuration>-<UTC time>-<id>/`.

### Phases

1. **Configuration check.** Read `/v1/models` (`max_model_len`) and `/metrics`
   (`vllm:cache_config_info`: memory utilization, KV block count and size, prefix caching).
   A declared setting vLLM contradicts refuses the sweep (exit 2). vLLM v0.32 reports
   `gpu_memory_utilization`, `enable_prefix_caching`, `block_size`, `num_gpu_blocks`,
   `kv_cache_size_tokens` and `kv_cache_max_concurrency`, but not `max_num_seqs`, which is
   read from the vLLM container's launch arguments (`docker inspect`, bounded, no shell) when
   Docker is reachable, a contradiction refusing the sweep, and is otherwise recorded as
   `declared_not_verified`; the analysis also checks that no sampled
   `vllm:num_requests_running` exceeded it, and a higher reading is reported as a
   contradiction. `kv_cache_size_tokens` is recorded as the measured KV-cache capacity
   (7,376 tokens at 0.6 in the 2026-10-27 evidence).
2. **Warm-up** batch (N = 1), recorded, never analysed.
3. **Ramp**, as declared, until a hard stop or the ceiling.
4. **Capacity decision** by `sweep_analysis` on the recorded ramp.
5. **Live confirmation** at capacity N.
6. **Injected condition** at N_inj.

### Batches

Each batch is one `AgentBatch` of N runners sharing the process clock, with no admission
controller. Folder: `batches/<NN>-<phase>-n<N>-r<repeat>/` holding `batch.json` and
`runs/`. Before and after each batch the driver reads vLLM's process start time,
prefix-cache queries and hits, preemptions, and running and waiting requests, and records
the changes. Injection logs go to `injections/<NN>.json`, with ticks in the batch's clock.

### `sweep.json`

Rewritten atomically after every batch. Holds the protocol and its hash; declared and
verified configuration with vLLM start attempts; host (GPU name, driver, memory, CPU count,
kernel); sources (run IDs, `plan_sha256`); every batch (phase, N, repeat, batch ID, folder,
per-run outcomes, wall time, vLLM counter changes); injection log paths; hard-stop events;
the capacity decision with each criterion's value and verdict; and the sweep state:
`completed`, `stopped_hard_limit`, `stopped_ceiling`, `stopped_injector_cleanup_unconfirmed`,
`cancelled` or `failed`.

There is no resume. A stopped or crashed sweep keeps what it recorded; a new invocation is a
new sweep in a new folder, and both are kept.

## Analysis

### Per level

Each statistic is reported as the mean across batches with sample stdev and CV. p90 is the
sorted element at `round(0.9 · (n − 1))`, as in `coverage`.

- Throughput (passed runs per batch-minute), pass rate, reply-match rate.
- `agent-run` duration p50, p90, max.
- Measured contention from each model call's vLLM span: queue, prefill, decode, first
  token, end to end. Calls without a span are counted `unavailable` and excluded, never 0.
- Server: preemptions, prefix-cache hit rate.
- Device and host: NVML utilization, memory used, SM clock, power; GPM SM occupancy when
  present; CPU PSI `some avg10` and cgroup throttling. SM clock drift by level is reported
  because a laptop GPU throttles when hot.
- Failures split into task and infrastructure failures; cleanup results.

### Label validation

Over model-call steps of replays at N ≥ 2:

- **Baseline:** per source and step ID, mean and sample stdev of step duration across the
  N = 1 replays of that source (three).
- **Proxy label** (the `labels-v1.1` noise rule): contended when (duration − baseline
  mean) exceeds max(0.01 s, 3 × baseline stdev).
- **Reference:** contended when the step's vLLM-measured queue time exceeds the same
  threshold.
- **Reported** per level and per configuration: the confusion matrix, label precision and
  recall against the reference, the Spearman correlation between excess duration and
  measured queue, and the share of excess explained by queueing versus longer prefill or
  decode (batching slows service without any queue).
- **Declared verdict:** for a configuration, the label is "validated as an inference-wait
  proxy" when precision ≥ 0.90 and recall ≥ 0.90 over all its N ≥ 2 ramp steps; otherwise
  "not validated". Steps without a span are excluded and counted.
- Tool steps have no measured reference; their label rates are reported as `unvalidated`.
- `rules-v1` is not scored in PR 2 (it needs `steps-v1.3` features built by `pipelines/`;
  recomputing them here would duplicate that logic). The docs state that `rules-v1` cannot
  predict `inference`; PR 3 scores rules and models on the frozen evidence.

## Errors and exit codes

- Invalid settings exit 2 before anything runs: unknown configuration, unreplayable source,
  two sources of the same task, or a contradicted vLLM setting.
- A hard-stop event ends the ramp, not the sweep: phases 4–6 still run at or below the last
  safe level when capacity ≥ 1. A vLLM restart or unconfirmed injector cleanup stops the whole
  sweep. A run whose journal needs `resend` stops the sweep with exit 3.
- Exit 0 when the sweep completed and every phase ran; 1 when it stopped or failed (evidence
  kept); 2 invalid settings; 3 `resend` needed.
- An unexpected error is recorded in `sweep.json`, the current batch's runs are cancelled,
  and the error is re-raised. Ctrl-C or SIGTERM cancels the current batch and any injector
  and runs nothing further (`cancelled`).

## Testing

Labeled test doubles only: the scripted model, the in-memory live server, a fake `/metrics`
server, a fake HTTP chat server and a fake injector. No GPU, network or live model in unit
tests.

- **Injector (PR 1):** bounds and non-loopback refusals; unique prompt prefixes; prompt size
  near target; failed start halts and reports cleanup; idempotent stop; missing activity
  distinct from zero; `PrefillRejected` counting.
- **Driver (PR 2):** phase order; N = 1 runs three batches per source; sources alternate;
  each hard-stop trigger ends the ramp; vLLM restart and unconfirmed injector cleanup stop
  the sweep; cancellation; idle-wait timeout; `sweep.json` rewritten after each batch;
  contradicted configuration refused; unreported setting recorded `declared_not_verified`.
- **Analysis (PR 2),** on fixed fixtures: capacity edge cases (knee, a later level passing
  again, below 1, ceiling); throughput and CV; the p90 rule; the label confusion matrix at
  the threshold boundary; missing spans excluded and counted; the 0.90 verdict on both sides.
- **Checks:** `uv run ruff check .` and the full `uv run pytest` in WSL Ubuntu against the
  dedicated `agentwatch_test` PostgreSQL database.

## Documentation and handoff

- Agent README: a "Capacity sweep" section. Injectors README: a row for the new method.
- `docs/capacity-record.md`: measured capacity per configuration, hardware, the curve and
  every failure, written only from real sessions.
- `docs/project-checklist.md`: A06 progress with dates and run IDs. "Evaluate natural
  overload" stays open until PR 3.
- `docs/data-card.md`: natural-overload and prefill-flood runs are held-out test evidence.
- A drafted note for Rohit and Akshay asking that `split-v3` hold out
  `prefill-flood-openai-chat-v1` and the natural-overload runs; Benoy sends it.
- Redacted evidence bundles in `C:\Users\benoy\AgentWatch-evidence\a06-<date>\` for Akshay
  and Rohit; the configuration table for Meghana.
- PR descriptions state that capacity describes this laptop and these configurations only.

## Risks and limitations

- Concurrent replays share prefixes, so they load vLLM less than free agents; prefix-cache
  hit rates are recorded at every level and the live confirmation shows the difference.
- Replays force the source's tool calls, so pass rate mostly shows whether generation still
  finishes in time, not decision quality under load; the live confirmation covers that.
- Laptop thermals vary across a multi-hour session; SM clock and power are kept and drift is
  reported, not corrected.
- WSL2 PSI and cgroup counters cover the whole VM, including the vLLM container.
- Each configuration needs a 2–4 hour GPU session, serialized through Benoy.

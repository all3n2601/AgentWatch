# Instrumentation overhead acceptance protocol (A07)

Status: proposed, declared 2026-11-16 before any A07 measurement. Task A07, owner Akshayaa,
partner reviewer Benoy. Protocol version `instrumentation-overhead-acceptance-v1`.

This protocol fixes, before evaluation, how A07 decides whether AgentWatch's instrumentation
stays within its overhead budget and whether sampling covers the steps it profiles. It reuses the
existing measurement commands and adds a decision rule; it does not change what they measure.
The thresholds below must not be tuned after results are seen. A change needs a new protocol
version, and results under the old version are kept.

## Budget

| Gate | Threshold | Source |
| --- | --- | --- |
| Instrumentation overhead | Median wall-time increase at most 5% versus uninstrumented | `max_median_instrumentation_overhead = 0.05` in [`optimization-contract.toml`](../pipelines/model/optimization-contract.toml), shared with [the evaluation contract](evaluation-contract.md) |
| Baseline stability | Coefficient of variation of uninstrumented wall time per repetition at most 15% | Value borrowed from `max_baseline_root_step_cv = 0.15` in the same file (see below) |
| Sampling coverage | Limits in [Sampling coverage](#sampling-coverage) | Declared here (proposed) |

The 5% budget is still proposed under O01 (Rohit). If O01 adopts a different value, the A07
report states both the verdict under this protocol and the value O01 accepted. It does not
re-label old results.

The stability gate borrows the value of `max_baseline_root_step_cv`, which the contract defines
for a different statistic: the CV of the root step's total duration across baseline batches.
Here it gates the CV of per-repetition uninstrumented wall time. The analysis reads the value
from that key, so a later change to it moves this gate too. The A07 report states the value
used, and a change to either gate's meaning needs a dedicated key and a new protocol version.

**What the budget means here.** The contract compares the median increase with 5%. This
protocol requires the 95% interval's upper bound to be at most 5% before calling an attempt
`within_budget`. That is stricter: a true increase just under 5% will usually be
`inconclusive`, not `within_budget`, especially on the short `agent-overhead-v1` workload,
where A04 measured -0.4% to +4.9%. The intent is that a pass means the overhead is shown to be
within budget, not merely estimated to be.

## Workloads

Each workload is an existing command with a protocol declared in its module. A07 does not
change their repetitions, order or warm-ups.

| Measurement | Command | Protocol | What it covers | Runs on |
| --- | --- | --- | --- | --- |
| Per-step instrumentation | `python -m agentwatch_agent overhead` | `agent-overhead-v1` ([`overhead.py`](../services/agent/src/agentwatch_agent/overhead.py)) | Sampler, live delivery and one step lifecycle around the ~55 ms fixture test suite. Shorter than one sampling interval, so not sustained sampling | Akshayaa (CPU, Linux container) |
| Sustained sampling, CPU | `python -m agentwatch_agent sampling-overhead` against CPU-only Ollama | `resource-sampling-overhead-v1` ([`sampling_overhead.py`](../services/agent/src/agentwatch_agent/sampling_overhead.py)) | CPU probes for a model request lasting several sampling intervals | Akshayaa (CPU, Linux container) |
| Sustained sampling, GPU (Ollama) | `sampling-overhead` with GPU device arguments | `resource-sampling-overhead-v1` | CPU and GPU probes during a GPU model request | Benoy (RTX 5070 Laptop, WSL2) |
| Sustained sampling, GPU (vLLM) | `sampling-overhead` with the vLLM model arguments | `resource-sampling-overhead-v2` | Adds the HTTP service probes (vLLM `/metrics`, DCGM) on their own thread | Benoy (RTX 5070 Laptop, WSL2) |

CPU probes read Linux PSI and cgroup v2 files, so the CPU runs use a Linux container, not macOS
directly. Each measurement describes its host only: a container on a laptop is not a GKE node,
and the laptop GPU is not an L4.

## Attempts

- Each measurement is run in **two attempts** on each host. An attempt is one invocation of the
  command: 3 discarded warm-ups, then 30 uninstrumented and 30 instrumented repetitions,
  alternating. A04 saw a 54 ms shift between attempts, larger than the effect, so one attempt
  cannot show that the difference is stable.
- Every attempt is reported, including failed and interrupted ones with their status and error.
  An attempt is not discarded or replaced because of its result. If an attempt fails for an
  infrastructure reason, for example the model server not starting, a further attempt may be
  made, but the failed one stays in the report with its reason. At most **four** attempts are
  made per measurement and host (two planned, plus at most two after infrastructure failures);
  if two attempts have still not completed, the verdict is `inconclusive`.
- A completed attempt needs the declared 30 pairs. One with fewer (for example a file run with
  non-default repetitions) is `inconclusive`, whatever its interval.
- Nothing else runs on the measured GPU or in the measured container during an attempt.
  Contention injectors stay off.

## Decision rule

For each completed attempt, using runner wall time per repetition:

1. **Relative increase** `r = median(instrumented) / median(uninstrumented) − 1`.
2. **Interval.** A paired percentile bootstrap of `r`, 10,000 resamples, 95% level, with a
   recorded seed. A pair is one uninstrumented repetition and the instrumented repetition that
   follows it in the alternating order. This uses the same method and settings as the
   [optimization contract](optimization-contract.md).
3. **Attempt verdict**:
   - `inconclusive` if the uninstrumented coefficient of variation is above 15%.
   - `within_budget` if the interval's upper bound is at most 0.05.
   - `over_budget` if the interval's lower bound is above 0.05.
   - `inconclusive` otherwise.

For each measurement and host, the verdict is `within_budget` or `over_budget` only when at
least two attempts completed and every completed attempt has that verdict. It is
`not_evaluated` when no attempt could be evaluated (all failed, or none saved per-repetition
walls), and `inconclusive` otherwise. A negative `r` is reported as measured. It does not show
that instrumentation speeds anything up.

Attempts are combined only when they share the protocol, task, model and request, sampling
interval, sampled probe set (the metrics in the attempt's availability counts) and an explicit
host label given to the analysis. The OS and machine type alone cannot tell two Linux x86_64
hosts apart. The same file given twice is refused rather than counted as a second attempt.

Also reported, without a gate: p90 wall time per condition, runner CPU per repetition
(`agent-overhead-v1`), the sampler's per-round cost, Ollama's or vLLM's own durations, and
generated token counts. At temperature 0, identical output is not guaranteed. Token counts do
not change the verdict, but a longer reply takes longer on its own, so when the generated
counts differ between conditions the A07 report states it as a limitation of that attempt's
wall-time comparison.

The measurement commands apply no threshold (their `PROTOCOL` keeps `"threshold": null`, which
is part of their configuration digest). The verdict is computed from their saved output by
`agentwatch_data.overhead_verdict` (#93). That needs per-repetition wall times: `sampling-overhead`
already saves them, and `overhead` saves them from #93 on, so `overhead.json` files recorded
earlier are `not_evaluated`. The analysis writes its report whatever the verdict and exits 0;
it exits 1 only when an attempt file or the contract is invalid.

## Sampling coverage

Proposed limits, judged per attempt from each sampler's interval summary (`interval_summary`
in `resources.py`) and from `python -m agentwatch_agent coverage` for full runs. The measurement
reports do not save these summaries today (`overhead` keeps only the round count, the longest
gap and availability counts); #94 adds them under `sampling_coverage`. The declared interval is
`DEFAULT_SAMPLING_INTERVAL_SECONDS`, 0.25 s.

| Measure | Limit | Field |
| --- | --- | --- |
| Achieved interval, median | Within ±10% of declared (0.225–0.275 s) | `observed_interval_median_seconds` |
| Achieved interval, maximum | At most 2× declared (0.5 s) | `observed_interval_max_seconds` |
| Missed rounds | At most 1% of expected rounds | `missed_rounds`, `expected_rounds` |
| Late rounds (gap above 1.5× declared) | Reported, no limit | `late_rounds` |
| Unavailable readings | Reported per metric with counts, no limit | availability counts |
| Steps shorter than one interval | Count and share reported. Such steps get no coverage fraction | `coverage` output |

The coverage verdict is `met`, `not_met` or `not_evaluated`. It is `not_evaluated` when fewer
than **10** gaps were measured in total (a sampler needs at least two rounds to measure a gap),
so a verdict never rests on one or two gaps. In a full run, a probe that was configured but
recorded no sample at all makes the verdict `not_met`. Missing, unsupported and unavailable
readings stay distinct from measured zero. A metric that was never available is reported as
unavailable, not as covered.

`agent-overhead-v1` coverage is expected to be `not_evaluated`. Each instrumented repetition
starts its own sampler, which reads once immediately, and the ~55 ms workload ends before a
second round, so no gap is measured. That is a property of the workload, not missing evidence.
Coverage for sustained sampling comes from `sampling-overhead` and from full runs.

## Identical settings for paired GPU runs

Benoy records these for every GPU attempt. Both conditions already run within one invocation,
so they share every setting by construction. The record lets the two attempts, and the Ollama
and vLLM measurements, be checked for consistency.

The output files already contain the protocol version, task ID, host OS and machine type,
sampling interval and the probes sampled; `overhead` also records `runner_version`, and
`sampling-overhead` the model name and `request_sha256`. Everything else below is recorded by
hand in the attempt notes that accompany the shared files:

- Repository commit SHA.
- Model digest or pinned revision, backend, and server image digest.
- Chat options (`DEFAULT_CHAT_OPTIONS`: temperature 0, `num_ctx` 4096, `num_predict` 512) and
  the 0.25 s sampling interval.
- GPU model, driver version, WSL kernel, and whether the laptop was on mains power.
- Whether `--gpm-file` was used.
- Start and end time of each attempt (UTC).

Attempts whose settings differ are reported separately and are not combined into one verdict.

## Shared evidence and redaction

Shared evidence is the JSON each command writes under `.data/agent-overhead/` or
`.data/sampling-overhead/`, the coverage output, and the A07 report. Before sharing, each file
must contain none of:

- Credentials, tokens, API keys, or the values of environment variables.
- Absolute host paths, user names, or host names. The host is recorded only as OS and machine
  type, but `overhead` also records its journal's path, which is absolute when `--output-dir`
  is. Run with a relative `--output-dir`, and share a copy made with `redact` (#95), which
  replaces host paths.
- Email addresses or any real user data. The workloads are repository fixtures, so prompts and
  code come from the repository.

A later A07 change adds a check that scans an evidence bundle for these. Until it is merged,
the sharer makes the copy with `redact`, reviews each file by hand, and records in the report
that they did.

## Outputs and handoff

The A07 report gives, per measurement and host: every attempt with its status, `r` and its
interval, the attempt and combined verdicts, the coverage verdict with its measured values,
the recorded settings, and the limitations. Data-quality notes (missing samples, unavailable
metrics, short-step coverage) go to Akshay. The overhead report goes to Rohit for O01.

## Limitations

- One laptop GPU and one CPU container. The results do not establish cloud or L4 behaviour.
- `agent-overhead-v1`'s workload is shorter than one sampling interval, so it measures
  per-step instrumentation cost, not sustained sampling.
- In `sampling-overhead`, each sampled request starts its own sampler, which reads once
  immediately. Sampling is therefore front-loaded compared with a long run.
- Live delivery under sustained sampling is not measured by either command.

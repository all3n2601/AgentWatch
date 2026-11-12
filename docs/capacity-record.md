# Measured agent capacity on the GPU laptop (A06)

This is the record of the A06 natural-overload capacity sweeps run on 2026-11-11. Every number in
the result tables comes from the two sweeps' own `sweep.json` files, their injection logs and
their run folders. The `sweep-record` command generated the Results and Sampling coverage tables
from those files, and they are not edited by hand (see [Reproduce](#reproduce)). They describe this
laptop and these two vLLM configurations only, never cloud capacity. The
[Attribution baseline](#attribution-baseline) scores the rules and the learned classifier on the
same sweeps, and its numbers come from that scoring attempt's outputs.

Design: [natural-overload capacity sweep](superpowers/specs/2026-11-10-natural-overload-capacity-sweep-design.md).
Command: `capacity-sweep` ([agent README](../services/agent/README.md#capacity-sweep)).
Evidence: `data/held-out/a06-20261111/`, versioned with DVC and held out of every dataset release
(see [Reproduce](#reproduce)).

## Summary

| Configuration | vLLM settings | Measured capacity | Why the ramp stopped counting | Label validation |
| --- | --- | --- | --- | --- |
| `cfg-default` | `max_num_seqs` 2, `gpu_memory_utilization` 0.6 | **3 concurrent agents** | Throughput at 4 agents was 6.4% below 3 agents (needs ≥ +10%) | **validated** (precision 0.971, recall 0.999) |
| `cfg-hardware` | `max_num_seqs` 16, `gpu_memory_utilization` 0.8 | **8 concurrent agents** | Throughput at 12 agents was 12.5% below 8 agents | **inconclusive**: vLLM never queued a request |

- **Clean runs.** Both sweeps ran the full ramp from 1 to 16 agents with no hard stop, no
  excluded batch and no failed vLLM state read. Every run passed its task's tests: 175 runs in
  `cfg-default` and 208 in `cfg-hardware`, across the warm-up, ramp (159 each), live, injected
  and control batches.
- **Capacity is the declared rule's result.** It is the last level before the first level that
  fails a criterion. The rule does not look past the first failure. Throughput rose again at
  `cfg-default` N = 6 and at `cfg-hardware` N = 16, and those levels are recorded but do not
  change the result.
- **Two kinds of contention.** `cfg-default` slows agents down by making them wait in vLLM's
  queue. `cfg-hardware` has a slot for every agent, so nothing waits; agents slow each other down
  instead by sharing the GPU inside one batch, which shows up as longer prefill and decode time.
  The label check uses queue time as its reference, so it can judge the first kind and not the
  second (see [Findings](#findings)).
- **What "validated" means here.** The label checked is the noise rule alone: a step is
  contended when its duration exceeds its N = 1 baseline by more than max(0.01 s, 3σ), the
  `[noise]` values of `labels-v2.toml`. A `validated` verdict supports that noise threshold as
  a proxy for inference wait. It does **not** validate the dataset's `labels-v2` under natural
  overload: `labels-v2` labels a resource only under an injected condition, so it would label
  every natural-overload step `none` (`clean_exceeded_noise` at most). PR 3 and later model
  work must not read these verdicts as validation of the dataset labels.
- **Agreement is weakest at the capacity itself.** `cfg-default` is `validated` over all N ≥ 2
  steps, as declared, but not at every level. N = 3, the measured capacity, is `not_validated`
  (precision 0.808), and N = 2 is `inconclusive`. Every level from N = 4 up is `validated`. Where
  the label would be used, at the capacity, it agrees least with queue time.
- **Attribution baseline.** On the 1,740 model calls with a `sweep-truth-v2` truth, `rules-v1`
  missed every call that vLLM queued (miss rate 1.000), as expected, since it has no inference
  class. The frozen M03 classifier with its M04 calibration missed 0.288 of them (95% interval
  0.161–0.482, abstentions counted as misses) and raised an alarm on 0.125 (0.066–0.187) of the
  calls vLLM did not queue above the noise threshold. It abstained on 0.130 of the calls. On
  `cfg-hardware`, where the ramp did not queue, it missed 0.908 (0.770–0.998) of the calls slower
  than their baseline. These are descriptive values with no gate, from one sweep per
  configuration (see [Attribution baseline](#attribution-baseline)).

## Setup

- **Hardware.** One NVIDIA GeForce RTX 5070 Laptop GPU (8,546,942,976 bytes of memory, driver
  617.42), 16 CPUs, WSL2 kernel `6.18.58.1-microsoft-standard-WSL2`. The laptop was on AC power.
- **Model server.** vLLM v0.32.0, image
  `vllm/vllm-openai@sha256:6ee133356d49ed24cee03dd0cda3d3980279a11c107ce3e752c89e76540f41c9`,
  serving `Qwen/Qwen3-1.7B` at revision `70d244cc86cc` with `max-model-len` 4096 and prefix
  caching on. Both configurations were started through the Compose service.
- **Evidence sampled per run.** CPU and NVML, vLLM `/metrics`, DCGM (`--dcgm-url`) and the
  Windows GPM exporter (`--gpm-file`), every 0.25 s. vLLM's per-request spans come from the
  OTLP collector.
- **Code.** Commit `d5eb41a` on `feature/natural-overload-capacity-sweep` (PR #65), with a clean
  working tree, for both sweeps. Both sweeps share one workload hash (`45a15ec9f655`).
- **Protocol.** `natural-overload-sweep-v1`, as declared before any run:
  - **Run limit:** 450 s.
  - **Ramp:** N = 1, 2, 3, 4, 6, 8, 12, 16, with 1 warm-up batch and then 3 batches per level
    (6 at N = 1).
  - **Pacing:** before each batch the server must be idle (bounded 30 s wait), then a 10 s pause.
  - **Policy:** no concurrency policy.
  - **Capacity criteria:**
    - throughput gain of at least 0.10 over the previous level;
    - pass rate no more than 0.20 below N = 1's;
    - p90 run time at most 225 s;
    - no hard stop.
  - **Live confirmation:** 3 batches of free agents at the capacity.
  - **Injected condition:** at max(1, ⌊capacity / 2⌋) agents, 3 prefill-flood batches, each
    followed by a clean control batch.
  - **Label validation:** precision and recall of at least 0.90 against vLLM's measured queue
    time.
  - The hash differs between the configurations because the configuration is part of the
    protocol.
- **Sources.** Two passing runs recorded on vLLM under `cfg-default` and replayed by every sweep:
  - `fix-mean-empty-r1`: run `19fb1536-66a2-4fe8-b9dd-05c8a38c5f68`, 2026-11-11T16:02Z.
  - `fix-median-even-r1`: run `291762be-6b24-4bd4-9a32-7cfafe831f1b`, 2026-11-11T18:40Z.

  Both sources were recorded at commit `d126df0`. `fix-median-even` was then still an
  uncommitted task fixture; it was committed unchanged in `1e8b886`, and its pinned content hash
  (`d14e6e7b…`) is the one the source records. The sweep checks that hash before replaying.

## Findings

1. **`cfg-default` saturates at 3 agents, and the queue is where the time goes.**
   - **Throughput** (passed runs per minute) climbs 19.1 → 35.3 → 47.4 from 1 to 3 agents,
     then stays between 37 and 49 up to 16.
   - **Queue time** rises from 0 ms at N ≤ 2 to a mean of 2,392 ms at N = 16, while prefill and
     decode stay roughly flat.
   - **Label validation** works here. Over all N ≥ 2 steps, the proxy label agrees with measured
     queue time (precision 0.971, recall 0.999, Spearman 0.995), and 1,108.9 s of the 1,182.8 s
     excess is queue time.
   - **Per level:**
     - At N = 2, vLLM has exactly 2 slots, so nothing queues. The 6 steps the label marks are
       false positives, and the level is `inconclusive`.
     - At N = 3 the level is `not_validated` (precision 0.808).
     - Every level from N = 4 up is `validated`.
2. **`cfg-hardware` scales to 8 agents without any queueing.**
   - **Throughput** reaches 125.4 runs per minute at 8 agents, against `cfg-default`'s best
     level of 49.1 (N = 6) and 47.4 at its capacity (N = 3).
   - **No queue.** vLLM's total measured queue time across every N ≥ 2 step was 0.008 s, and the
     most running requests observed was 16, the configured maximum.
   - **The label check cannot judge this.** Steps still got slower than their N = 1 baseline
     (91.6 s in total), and 62.9 s of that is longer prefill and decode: agents slowing each
     other inside one batch. The reference has no positives, so recall is undefined (0 of 0)
     and the declared verdict is `inconclusive`.

   This result is recorded as it came out; the label rule and its reference were not changed
   after seeing it.
3. **The `cfg-hardware` knee coincides with the GPU clocking down.**
   - **At N = 12**, the level that failed, the mean SM clock was 1,537 MHz and mean power 39.4 W.
   - **At N = 8 and N = 16**, they were 2,365 and 2,248 MHz, and 69.9 and 72.7 W.
   - **Throughput** at N = 16 (140.3 runs per minute, CV 0.010) was higher than at N = 8.

   This is an observation, not a cause: the laptop GPU may have dropped to a lower clock state
   during the N = 12 batches. The declared result stays capacity 8, which should be read as
   "the first level where throughput stopped growing on this run". A repeat sweep under the same
   protocol would show whether the N = 12 dip recurs.
4. **The prefill flood slows agents on both configurations.**
   - **Throughput:** at the injected level (N = 1 for `cfg-default`, N = 4 for `cfg-hardware`),
     every flood batch ran far slower than its paired control. On `cfg-default` it was about a
     quarter of the control (4.4–5.4 against 18.2–20.4 runs per minute); on `cfg-hardware`,
     about a sixth to a fifth (12.6–12.8 against 66.3–74.6).
   - **Where the time went:** mainly in decode time, which rose about 6 times on `cfg-default`
     and about 8 times on `cfg-hardware`. Queue time rose too on `cfg-default` (78–141 ms) but stayed small on
     `cfg-hardware` (9–12 ms).
   - **The flood itself:** every flood completed its requests with no failures, a median prompt
     of 2,082–2,084 tokens against the 2,048 target, every reply exactly 1 token, and cleanup
     confirmed.

   These are observed differences between paired batches, not a causal attribution. The
   [Attribution baseline](#prefill-flood-and-its-controls) scores the injected runs
   with the rule and the model.
5. **Live agents run at a similar rate to replays.**
   - **`cfg-default`:** free agents at the capacity (N = 3) ran at 37.2–45.9 runs per minute,
     against 47.4 for replays.
   - **`cfg-hardware`:** at N = 8 they ran at 116.2–135.1, against 125.4 for replays.
   - **Pass rate:** every live run passed.
6. **At 16 agents, the server-metrics sampler falls behind.** `coverage` on one run per level
   (see [Sampling coverage](#sampling-coverage)) shows every sampler on time up to N = 12, apart
   from one late DCGM round at `cfg-hardware` N = 12.
   - **At N = 16, the HTTP sampler misses rounds.** This is the one thread that reads vLLM
     `/metrics`, DCGM and GPM. On `cfg-default` it recorded 38 of about 49–50 expected rounds per
     source: 8–11 late and 11–12 missed, about one in four readings missing. On `cfg-hardware`
     it missed 2–3 of 13–14.
   - **The CPU/NVML sampler stayed on time** at every level.
   - **Results are not affected.** The capacity decision uses run durations, throughput and
     vLLM's per-request spans, which come from traces, not from sampling. The device columns
     come from NVML. The sampled vLLM, DCGM and GPM series at N = 16 are incomplete, and the
     missed rounds are recorded as missing, never as zeros. The 16-agent preflight had predicted
     this: the sampler's median round took 0.23 s of its 0.25 s slot.

## Results

The tables below are `sweep-record`'s output for each sweep's `sweep.json` and its injection logs.
Throughput is passed runs per batch-minute. Durations are the `agent-run` span. vLLM queue,
prefill, decode and end-to-end times come from each model call's vLLM span. The prefix-cache
hit rate, preemptions and device metrics are per level. SM clock drift is the level's mean SM
clock minus N = 1's. "n/a" is a value that was not available, never a zero.

### cfg-default (`cfg-default-20261111T211237Z-2d75ea`)

Sweep `c5a40c4c-e4b3-4dc1-b660-21ca4c88bc93`, 2026-11-11T21:12:40.026401+00:00 to 2026-11-11T21:25:27.287515+00:00; state `completed`, ramp ended `ceiling`, analysis scope `full_ramp`, analysis errors none. Protocol `natural-overload-sweep-v1` (`protocol_sha256` 22e26dc7536f), code d5eb41a, workload `45a15ec9f655`.

#### Configuration as verified

| Setting | Declared | Reported | Status | Source |
| --- | --- | --- | --- | --- |
| `model` | "Qwen/Qwen3-1.7B@70d244cc86cc" | "Qwen/Qwen3-1.7B@70d244cc86cc" | verified | vllm /v1/models |
| `max_model_len` | 4096 | 4096 | verified | vllm /v1/models |
| `gpu_memory_utilization` | [0.6] | 0.6 | verified | vllm /metrics vllm:cache_config_info |
| `enable_prefix_caching` | true | true | verified | vllm /metrics vllm:cache_config_info |
| `max_num_seqs` | 2 | 2 | verified | docker inspect of the vLLM container's launch arguments |
| `launch_gpu_memory_utilization` | [0.6] | 0.6 | verified | docker inspect of the vLLM container's launch arguments |

KV cache: 736 blocks of 16 tokens = 11,776 tokens; vLLM's maximum concurrency at 4096 tokens per request: 2.88. Launch arguments read with `docker.exe` from `agentwatch-vllm-1`, which publishes the measured endpoint's port 8000 (`serves_endpoint: true`). Start attempts: null.

#### Ramp

| N | Batches / runs | Passed | Throughput, runs/min (mean ± sd, CV) | Run p50 / p90 / max, s | vLLM queue mean / p90, ms | Prefill mean, ms | Decode mean, ms | End to end mean, ms | Prefix-cache hit, % | Preemptions | SM clock mean, MHz | SM clock drift, MHz | GPU util mean, % | Power mean, W |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 6 / 6 | 6 | 19.1 ± 1.7 (0.089) | 2.4 / 2.6 / 2.7 | 0.0 / 0.0 | 54.6 | 329.0 | 390.8 | 95.8 | 0 | 2,084 | n/a | 79 | 57.9 |
| 2 | 3 / 6 | 6 | 35.3 ± 2.9 (0.083) | 2.5 / 2.7 / 2.7 | 0.0 / 0.0 | 55.9 | 328.6 | 397.9 | 99.0 | 0 | 2,466 | 382 | 89 | 63.0 |
| 3 | 3 / 9 | 9 | 47.4 ± 3.5 (0.073) | 3.0 / 3.3 / 3.5 | 113.6 / 246.6 | 46.7 | 327.1 | 507.0 | 99.0 | 0 | 2,379 | 295 | 92 | 70.0 |
| 4 | 3 / 12 | 12 | 44.4 ± 5.6 (0.125) | 4.2 / 5.1 / 5.8 | 283.3 / 664.5 | 44.1 | 388.2 | 732.1 | 99.0 | 0 | 1,775 | -309 | 88 | 57.9 |
| 6 | 3 / 18 | 18 | 49.1 ± 6.4 (0.130) | 5.6 / 6.9 / 7.7 | 623.3 / 1,013.4 | 42.9 | 379.8 | 1,073.9 | 99.0 | 0 | 2,099 | 15 | 95 | 60.9 |
| 8 | 3 / 24 | 24 | 47.6 ± 6.7 (0.140) | 8.0 / 10.2 / 10.3 | 1,047.1 / 1,717.7 | 40.1 | 418.2 | 1,539.9 | 99.0 | 0 | 1,735 | -349 | 95 | 47.2 |
| 12 | 3 / 36 | 36 | 45.0 ± 7.3 (0.162) | 12.4 / 15.2 / 15.9 | 1,840.6 / 2,783.2 | 35.7 | 433.2 | 2,330.9 | 99.0 | 0 | 1,797 | -288 | 97 | 49.2 |
| 16 | 3 / 48 | 48 | 37.1 ± 4.0 (0.107) | 15.2 / 18.6 / 19.5 | 2,391.7 / 3,754.2 | 35.5 | 416.0 | 2,861.4 | 99.0 | 0 | 1,816 | -268 | 96 | 50.5 |

Model calls with a vLLM span: 795 measured, 0 without one.

#### Capacity decision

- Result: **measured**, capacity **3**.
- First failing level: 4 (failed criteria: throughput_gain; hard stops: []; unmeasured: []).
- Hard stops: none. Excluded batches: none. vLLM state read failures: 0.
- `max_running_check`: consistent (most running requests observed 2, declared `max_num_seqs` 2, 6,243 samples).

| N | Throughput gain (≥ 0.10) | Pass-rate drop (≤ 0.20) | p90 run, s | Level |
| --- | --- | --- | --- | --- |
| 1 | n/a | 0.000 | 2.6 (limit 225) | pass |
| 2 | 0.846 | 0.000 | 2.7 (limit 225) | pass |
| 3 | 0.343 | 0.000 | 3.3 (limit 225) | pass |
| 4 | -0.064 ✗ | 0.000 | 5.1 (limit 225) | fail |
| 6 | 0.105 | 0.000 | 6.9 (limit 225) | pass |
| 8 | -0.031 ✗ | 0.000 | 10.2 (limit 225) | fail |
| 12 | -0.053 ✗ | 0.000 | 15.2 (limit 225) | fail |
| 16 | -0.177 ✗ | 0.000 | 18.6 (limit 225) | fail |

#### Label validation

| N | TP / FP / FN / TN | Precision | Recall | Spearman | Excess over baseline, s | of which vLLM queue, s | Prefill + decode increase, s | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 0 / 6 / 0 / 24 | 0.000 | n/a | 0.150 | 0.2 | 0.000 | 0.1 | inconclusive |
| 3 | 21 / 5 / 1 / 18 | 0.808 | 0.955 | 0.850 | 5.2 | 4.855 | -0.1 | not_validated |
| 4 | 49 / 5 / 0 / 6 | 0.907 | 1.000 | 0.874 | 20.5 | 17.001 | 2.8 | validated |
| 6 | 84 / 2 / 0 / 4 | 0.977 | 1.000 | 0.956 | 61.5 | 56.101 | 3.3 | validated |
| 8 | 114 / 0 / 0 / 6 | 1.000 | 1.000 | 0.972 | 138.2 | 125.650 | 8.7 | validated |
| 12 | 174 / 2 / 0 / 4 | 0.989 | 1.000 | 0.986 | 352.0 | 331.313 | 15.5 | validated |
| 16 | 234 / 0 / 0 / 6 | 1.000 | 1.000 | 0.994 | 605.4 | 574.014 | 16.8 | validated |
| all N ≥ 2 | 676 / 20 / 1 / 68 | 0.971 | 0.999 | 0.995 | 1,182.8 | 1,108.934 | 47.1 | validated |

Baselines: 10 source steps from the N = 1 replays (0 omitted). Tool steps have no measured reference: 219 of 841 labelled contended (rate 0.260), `unvalidated`.

#### Live confirmation

| N | Batch | Passed / runs | Throughput, runs/min | Batch wall, s |
| --- | --- | --- | --- | --- |
| 3 | 0 | 3 / 3 | 37.2 | 4.8 |
| 3 | 1 | 3 / 3 | 37.9 | 4.7 |
| 3 | 2 | 3 / 3 | 45.9 | 3.9 |

#### Injected condition

| Pair | Batch | N | Passed / runs | Throughput, runs/min | Run median, s | vLLM queue mean, ms | Prefill mean, ms | Decode mean, ms | Flood activity |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | injected | 1 | 1 / 1 | 5.3 | 10.7 | 106.2 | 44.6 | 1,753.9 | 65 done / 0 failed; prompt median 2083; replies not 1 token: 0; cleanup true |
| 0 | control | 1 | 1 / 1 | 20.4 | 2.2 | 0.0 | 66.2 | 285.6 | no flood |
| 1 | injected | 1 | 1 / 1 | 4.4 | 13.0 | 141.3 | 52.9 | 2,188.0 | 78 done / 0 failed; prompt median 2083; replies not 1 token: 0; cleanup true |
| 1 | control | 1 | 1 / 1 | 18.2 | 2.6 | 0.0 | 50.3 | 366.6 | no flood |
| 2 | injected | 1 | 1 / 1 | 5.4 | 10.6 | 77.8 | 113.4 | 1,707.9 | 64 done / 0 failed; prompt median 2084; replies not 1 token: 0; cleanup true |
| 2 | control | 1 | 1 / 1 | 20.1 | 2.2 | 0.0 | 59.0 | 300.0 | no flood |

### cfg-hardware (`cfg-hardware-20261111T213039Z-b4ad4a`)

Sweep `068e27a9-5933-4f99-b465-75502f7e523b`, 2026-11-11T21:30:42.460586+00:00 to 2026-11-11T21:41:09.177749+00:00; state `completed`, ramp ended `ceiling`, analysis scope `full_ramp`, analysis errors none. Protocol `natural-overload-sweep-v1` (`protocol_sha256` 1608deee2a3b), code d5eb41a, workload `45a15ec9f655`.

#### Configuration as verified

| Setting | Declared | Reported | Status | Source |
| --- | --- | --- | --- | --- |
| `model` | "Qwen/Qwen3-1.7B@70d244cc86cc" | "Qwen/Qwen3-1.7B@70d244cc86cc" | verified | vllm /v1/models |
| `max_model_len` | 4096 | 4096 | verified | vllm /v1/models |
| `gpu_memory_utilization` | [0.8, 0.75, 0.7] | 0.8 | verified | vllm /metrics vllm:cache_config_info |
| `enable_prefix_caching` | true | true | verified | vllm /metrics vllm:cache_config_info |
| `max_num_seqs` | 16 | 16 | verified | docker inspect of the vLLM container's launch arguments |
| `launch_gpu_memory_utilization` | [0.8, 0.75, 0.7] | 0.8 | verified | docker inspect of the vLLM container's launch arguments |

KV cache: 1,378 blocks of 16 tokens = 22,048 tokens; vLLM's maximum concurrency at 4096 tokens per request: 5.38. Launch arguments read with `docker.exe` from `agentwatch-vllm-1`, which publishes the measured endpoint's port 8000 (`serves_endpoint: true`). Start attempts: [{"gpu_memory_utilization": 0.8, "outcome": "started"}].

#### Ramp

| N | Batches / runs | Passed | Throughput, runs/min (mean ± sd, CV) | Run p50 / p90 / max, s | vLLM queue mean / p90, ms | Prefill mean, ms | Decode mean, ms | End to end mean, ms | Prefix-cache hit, % | Preemptions | SM clock mean, MHz | SM clock drift, MHz | GPU util mean, % | Power mean, W |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 6 / 6 | 6 | 18.4 ± 1.2 (0.065) | 2.4 / 2.6 / 2.7 | 0.0 / 0.1 | 53.3 | 330.5 | 392.2 | 95.8 | 0 | 2,171 | n/a | 79 | 62.4 |
| 2 | 3 / 6 | 6 | 37.4 ± 1.6 (0.043) | 2.4 / 2.6 / 2.6 | 0.0 / 0.0 | 51.5 | 331.8 | 394.5 | 99.0 | 0 | 2,280 | 109 | 90 | 66.9 |
| 3 | 3 / 9 | 9 | 62.0 ± 3.3 (0.053) | 2.3 / 2.4 / 2.6 | 0.0 / 0.0 | 47.1 | 325.0 | 395.5 | 99.0 | 0 | 2,139 | -32 | 89 | 62.6 |
| 4 | 3 / 12 | 12 | 69.7 ± 4.6 (0.066) | 2.3 / 2.6 / 2.6 | 0.0 / 0.0 | 42.5 | 328.2 | 397.5 | 99.0 | 0 | 2,364 | 193 | 91 | 71.0 |
| 6 | 3 / 18 | 18 | 109.7 ± 4.5 (0.041) | 2.4 / 2.6 / 2.7 | 0.0 / 0.0 | 38.4 | 337.5 | 403.6 | 99.0 | 0 | 2,366 | 195 | 90 | 65.7 |
| 8 | 3 / 24 | 24 | 125.4 ± 9.4 (0.075) | 2.6 / 2.7 / 2.9 | 0.0 / 0.0 | 43.8 | 342.7 | 416.9 | 99.0 | 0 | 2,365 | 194 | 92 | 69.9 |
| 12 | 3 / 36 | 36 | 109.8 ± 6.0 (0.055) | 3.8 / 4.1 / 4.1 | 0.0 / 0.0 | 53.1 | 490.5 | 591.6 | 99.0 | 0 | 1,537 | -634 | 94 | 39.4 |
| 16 | 3 / 48 | 48 | 140.3 ± 1.4 (0.010) | 3.6 / 3.9 / 3.9 | 0.0 / 0.0 | 38.3 | 404.8 | 471.3 | 99.0 | 0 | 2,248 | 77 | 91 | 72.7 |

Model calls with a vLLM span: 795 measured, 0 without one.

#### Capacity decision

- Result: **measured**, capacity **8**.
- First failing level: 12 (failed criteria: throughput_gain; hard stops: []; unmeasured: []).
- Hard stops: none. Excluded batches: none. vLLM state read failures: 0.
- `max_running_check`: consistent (most running requests observed 16, declared `max_num_seqs` 16, 1,853 samples).

| N | Throughput gain (≥ 0.10) | Pass-rate drop (≤ 0.20) | p90 run, s | Level |
| --- | --- | --- | --- | --- |
| 1 | n/a | 0.000 | 2.6 (limit 225) | pass |
| 2 | 1.030 | 0.000 | 2.6 (limit 225) | pass |
| 3 | 0.656 | 0.000 | 2.4 (limit 225) | pass |
| 4 | 0.124 | 0.000 | 2.6 (limit 225) | pass |
| 6 | 0.574 | 0.000 | 2.6 (limit 225) | pass |
| 8 | 0.143 | 0.000 | 2.7 (limit 225) | pass |
| 12 | -0.125 ✗ | 0.000 | 4.1 (limit 225) | fail |
| 16 | 0.278 | 0.000 | 3.9 (limit 225) | pass |

#### Label validation

| N | TP / FP / FN / TN | Precision | Recall | Spearman | Excess over baseline, s | of which vLLM queue, s | Prefill + decode increase, s | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 0 / 9 / 0 / 21 | 0.000 | n/a | -0.396 | 0.1 | 0.000 | 0.1 | inconclusive |
| 3 | 0 / 22 / 0 / 23 | 0.000 | n/a | -0.124 | 0.7 | 0.000 | 0.6 | inconclusive |
| 4 | 0 / 30 / 0 / 30 | 0.000 | n/a | -0.007 | 1.1 | 0.000 | 0.8 | inconclusive |
| 6 | 0 / 63 / 0 / 27 | 0.000 | n/a | -0.037 | 3.6 | 0.001 | 2.7 | inconclusive |
| 8 | 0 / 86 / 0 / 34 | 0.000 | n/a | -0.071 | 6.3 | 0.001 | 4.5 | inconclusive |
| 12 | 0 / 157 / 0 / 23 | 0.000 | n/a | -0.144 | 43.0 | 0.002 | 33.9 | inconclusive |
| 16 | 0 / 201 / 0 / 39 | 0.000 | n/a | -0.048 | 36.7 | 0.003 | 20.4 | inconclusive |
| all N ≥ 2 | 0 / 568 / 0 / 197 | 0.000 | n/a | -0.076 | 91.6 | 0.008 | 62.9 | inconclusive |

Baselines: 10 source steps from the N = 1 replays (0 omitted). Tool steps have no measured reference: 320 of 841 labelled contended (rate 0.380), `unvalidated`.

#### Live confirmation

| N | Batch | Passed / runs | Throughput, runs/min | Batch wall, s |
| --- | --- | --- | --- | --- |
| 8 | 0 | 8 / 8 | 135.1 | 3.6 |
| 8 | 1 | 8 / 8 | 132.6 | 3.6 |
| 8 | 2 | 8 / 8 | 116.2 | 4.1 |

#### Injected condition

| Pair | Batch | N | Passed / runs | Throughput, runs/min | Run median, s | vLLM queue mean, ms | Prefill mean, ms | Decode mean, ms | Flood activity |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | injected | 4 | 4 / 4 | 12.8 | 17.7 | 9.3 | 144.4 | 2,863.5 | 102 done / 0 failed; prompt median 2082; replies not 1 token: 0; cleanup true |
| 0 | control | 4 | 4 / 4 | 66.3 | 2.6 | 0.0 | 53.8 | 348.0 | no flood |
| 1 | injected | 4 | 4 / 4 | 12.7 | 18.4 | 11.9 | 223.0 | 2,884.4 | 103 done / 0 failed; prompt median 2083; replies not 1 token: 0; cleanup true |
| 1 | control | 4 | 4 / 4 | 74.6 | 2.7 | 0.0 | 53.1 | 354.6 | no flood |
| 2 | injected | 4 | 4 / 4 | 12.6 | 18.5 | 9.9 | 192.5 | 2,905.1 | 104 done / 0 failed; prompt median 2083; replies not 1 token: 0; cleanup true |
| 2 | control | 4 | 4 / 4 | 73.3 | 2.7 | 0.0 | 45.9 | 355.3 | no flood |

## Sampling coverage

`coverage` (`coverage-v4`) on the first run of the first ramp batch at each level. Rounds are
sampling rounds recorded, against those expected for the run's length at the declared 0.25 s
interval. Late and missed rounds are counted by the coverage method. The tables are
`sweep-record --coverage-jsonl`'s output, and the raw reports it wrote are in
`data/held-out/a06-20261111/coverage-per-level.jsonl`.

### `cfg-default`

| N | Run | Source | Rounds (expected) | Late | Missed | Interval median / p90 / max, s |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | `88014b98` | nvidia-nvml | 9 (8) | 0 | 0 | 0.249 / 0.252 / 0.276 |
| 1 | `88014b98` | nvidia-dcgm | 9 (8) | 0 | 0 | 0.250 / 0.258 / 0.276 |
| 1 | `88014b98` | nvidia-nvml-gpm | 9 (8) | 0 | 0 | 0.251 / 0.259 / 0.278 |
| 1 | `88014b98` | vllm | 9 (9) | 0 | 0 | 0.250 / 0.250 / 0.250 |
| 1 | `88014b98` | agentwatch-agent-runner | 9 (9) | 0 | 0 | 0.250 / 0.250 / 0.250 |
| 2 | `52d68e58` | nvidia-nvml | 9 (8) | 0 | 0 | 0.249 / 0.256 / 0.257 |
| 2 | `52d68e58` | nvidia-dcgm | 9 (8) | 0 | 0 | 0.246 / 0.257 / 0.262 |
| 2 | `52d68e58` | nvidia-nvml-gpm | 9 (8) | 0 | 0 | 0.246 / 0.259 / 0.269 |
| 2 | `52d68e58` | vllm | 9 (9) | 0 | 0 | 0.250 / 0.250 / 0.250 |
| 2 | `52d68e58` | agentwatch-agent-runner | 9 (9) | 0 | 0 | 0.250 / 0.250 / 0.250 |
| 3 | `c85ba28a` | nvidia-nvml | 12 (11) | 0 | 0 | 0.249 / 0.262 / 0.267 |
| 3 | `c85ba28a` | nvidia-dcgm | 12 (11) | 0 | 0 | 0.248 / 0.267 / 0.284 |
| 3 | `c85ba28a` | nvidia-nvml-gpm | 12 (11) | 0 | 0 | 0.251 / 0.265 / 0.287 |
| 3 | `c85ba28a` | vllm | 12 (12) | 0 | 0 | 0.250 / 0.250 / 0.250 |
| 3 | `c85ba28a` | agentwatch-agent-runner | 12 (12) | 0 | 0 | 0.250 / 0.250 / 0.250 |
| 4 | `53b56dfa` | nvidia-nvml | 13 (12) | 0 | 0 | 0.249 / 0.258 / 0.266 |
| 4 | `53b56dfa` | nvidia-dcgm | 13 (12) | 0 | 0 | 0.248 / 0.261 / 0.283 |
| 4 | `53b56dfa` | nvidia-nvml-gpm | 13 (12) | 0 | 0 | 0.248 / 0.264 / 0.299 |
| 4 | `53b56dfa` | vllm | 13 (13) | 0 | 0 | 0.250 / 0.250 / 0.250 |
| 4 | `53b56dfa` | agentwatch-agent-runner | 13 (13) | 0 | 0 | 0.250 / 0.250 / 0.251 |
| 6 | `3fbe2238` | nvidia-nvml | 23 (23) | 0 | 0 | 0.251 / 0.262 / 0.263 |
| 6 | `3fbe2238` | nvidia-dcgm | 23 (22) | 0 | 0 | 0.254 / 0.268 / 0.283 |
| 6 | `3fbe2238` | nvidia-nvml-gpm | 23 (22) | 0 | 0 | 0.248 / 0.277 / 0.288 |
| 6 | `3fbe2238` | vllm | 23 (23) | 0 | 0 | 0.250 / 0.250 / 0.251 |
| 6 | `3fbe2238` | agentwatch-agent-runner | 23 (23) | 0 | 0 | 0.250 / 0.250 / 0.250 |
| 8 | `06574d1c` | nvidia-nvml | 28 (27) | 0 | 0 | 0.250 / 0.252 / 0.257 |
| 8 | `06574d1c` | nvidia-dcgm | 28 (27) | 0 | 0 | 0.249 / 0.262 / 0.280 |
| 8 | `06574d1c` | nvidia-nvml-gpm | 28 (27) | 0 | 0 | 0.250 / 0.260 / 0.274 |
| 8 | `06574d1c` | vllm | 28 (28) | 0 | 0 | 0.250 / 0.250 / 0.251 |
| 8 | `06574d1c` | agentwatch-agent-runner | 28 (28) | 0 | 0 | 0.250 / 0.250 / 0.255 |
| 12 | `425845b4` | nvidia-nvml | 52 (51) | 0 | 0 | 0.250 / 0.259 / 0.274 |
| 12 | `425845b4` | nvidia-dcgm | 51 (50) | 0 | 0 | 0.251 / 0.285 / 0.317 |
| 12 | `425845b4` | nvidia-nvml-gpm | 51 (50) | 0 | 0 | 0.251 / 0.273 / 0.309 |
| 12 | `425845b4` | vllm | 51 (51) | 0 | 0 | 0.250 / 0.253 / 0.278 |
| 12 | `425845b4` | agentwatch-agent-runner | 52 (52) | 0 | 0 | 0.250 / 0.250 / 0.253 |
| 16 | `fedae905` | nvidia-nvml | 55 (55) | 0 | 0 | 0.252 / 0.303 / 0.338 |
| 16 | `fedae905` | nvidia-dcgm | 38 (49) | 9 | 11 | 0.306 / 0.459 / 0.490 |
| 16 | `fedae905` | nvidia-nvml-gpm | 38 (49) | 11 | 11 | 0.316 / 0.443 / 0.468 |
| 16 | `fedae905` | vllm | 38 (50) | 8 | 12 | 0.313 / 0.456 / 0.531 |
| 16 | `fedae905` | agentwatch-agent-runner | 55 (55) | 0 | 0 | 0.251 / 0.256 / 0.295 |

### `cfg-hardware`

| N | Run | Source | Rounds (expected) | Late | Missed | Interval median / p90 / max, s |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | `83c7b368` | nvidia-nvml | 9 (8) | 0 | 0 | 0.252 / 0.253 / 0.259 |
| 1 | `83c7b368` | nvidia-dcgm | 9 (8) | 0 | 0 | 0.247 / 0.260 / 0.262 |
| 1 | `83c7b368` | nvidia-nvml-gpm | 9 (8) | 0 | 0 | 0.244 / 0.263 / 0.263 |
| 1 | `83c7b368` | vllm | 9 (9) | 0 | 0 | 0.250 / 0.250 / 0.250 |
| 1 | `83c7b368` | agentwatch-agent-runner | 9 (9) | 0 | 0 | 0.250 / 0.250 / 0.250 |
| 2 | `94aeb8b7` | nvidia-nvml | 9 (9) | 0 | 0 | 0.250 / 0.253 / 0.255 |
| 2 | `94aeb8b7` | nvidia-dcgm | 9 (8) | 0 | 0 | 0.248 / 0.262 / 0.271 |
| 2 | `94aeb8b7` | nvidia-nvml-gpm | 9 (8) | 0 | 0 | 0.247 / 0.262 / 0.276 |
| 2 | `94aeb8b7` | vllm | 9 (9) | 0 | 0 | 0.250 / 0.250 / 0.251 |
| 2 | `94aeb8b7` | agentwatch-agent-runner | 9 (9) | 0 | 0 | 0.250 / 0.250 / 0.250 |
| 3 | `a5477550` | nvidia-nvml | 10 (9) | 0 | 0 | 0.250 / 0.254 / 0.258 |
| 3 | `a5477550` | nvidia-dcgm | 10 (9) | 0 | 0 | 0.248 / 0.263 / 0.267 |
| 3 | `a5477550` | nvidia-nvml-gpm | 10 (9) | 0 | 0 | 0.254 / 0.261 / 0.272 |
| 3 | `a5477550` | vllm | 10 (10) | 0 | 0 | 0.250 / 0.250 / 0.250 |
| 3 | `a5477550` | agentwatch-agent-runner | 10 (10) | 0 | 0 | 0.250 / 0.250 / 0.250 |
| 4 | `432fef4e` | nvidia-nvml | 10 (9) | 0 | 0 | 0.251 / 0.278 / 0.282 |
| 4 | `432fef4e` | nvidia-dcgm | 10 (9) | 0 | 0 | 0.251 / 0.264 / 0.266 |
| 4 | `432fef4e` | nvidia-nvml-gpm | 10 (9) | 0 | 0 | 0.253 / 0.262 / 0.268 |
| 4 | `432fef4e` | vllm | 10 (10) | 0 | 0 | 0.250 / 0.251 / 0.251 |
| 4 | `432fef4e` | agentwatch-agent-runner | 10 (10) | 0 | 0 | 0.250 / 0.250 / 0.252 |
| 6 | `f3b80324` | nvidia-nvml | 10 (9) | 0 | 0 | 0.248 / 0.253 / 0.256 |
| 6 | `f3b80324` | nvidia-dcgm | 10 (9) | 0 | 0 | 0.251 / 0.270 / 0.278 |
| 6 | `f3b80324` | nvidia-nvml-gpm | 10 (9) | 0 | 0 | 0.252 / 0.271 / 0.272 |
| 6 | `f3b80324` | vllm | 10 (10) | 0 | 0 | 0.250 / 0.250 / 0.251 |
| 6 | `f3b80324` | agentwatch-agent-runner | 10 (10) | 0 | 0 | 0.250 / 0.250 / 0.250 |
| 8 | `87cc891f` | nvidia-nvml | 10 (9) | 0 | 0 | 0.249 / 0.264 / 0.273 |
| 8 | `87cc891f` | nvidia-dcgm | 10 (10) | 0 | 0 | 0.251 / 0.272 / 0.297 |
| 8 | `87cc891f` | nvidia-nvml-gpm | 10 (10) | 0 | 0 | 0.250 / 0.283 / 0.298 |
| 8 | `87cc891f` | vllm | 10 (10) | 0 | 0 | 0.250 / 0.253 / 0.254 |
| 8 | `87cc891f` | agentwatch-agent-runner | 10 (10) | 0 | 0 | 0.250 / 0.252 / 0.252 |
| 12 | `2cdfddf3` | nvidia-nvml | 16 (15) | 0 | 0 | 0.249 / 0.271 / 0.285 |
| 12 | `2cdfddf3` | nvidia-dcgm | 15 (14) | 1 | 0 | 0.244 / 0.291 / 0.421 |
| 12 | `2cdfddf3` | nvidia-nvml-gpm | 15 (14) | 0 | 0 | 0.235 / 0.363 / 0.374 |
| 12 | `2cdfddf3` | vllm | 15 (15) | 0 | 0 | 0.250 / 0.258 / 0.322 |
| 12 | `2cdfddf3` | agentwatch-agent-runner | 16 (16) | 0 | 0 | 0.250 / 0.251 / 0.254 |
| 16 | `c810108a` | nvidia-dcgm | 11 (14) | 2 | 3 | 0.339 / 0.387 / 0.469 |
| 16 | `c810108a` | nvidia-nvml | 16 (16) | 0 | 0 | 0.254 / 0.303 / 0.327 |
| 16 | `c810108a` | nvidia-nvml-gpm | 11 (14) | 2 | 3 | 0.332 / 0.379 / 0.490 |
| 16 | `c810108a` | vllm | 11 (13) | 2 | 2 | 0.313 / 0.453 / 0.460 |
| 16 | `c810108a` | agentwatch-agent-runner | 16 (16) | 0 | 0 | 0.251 / 0.255 / 0.256 |

## Attribution baseline

This section scores today's attribution on both sweeps: `rules-v1`, and Rohit's frozen M03
classifier with its M04 calibration. It measures how often each one misses inference contention
that vLLM measured, and how often it raises an alarm on a model call that was not contended.
Neither scorer has an inference feature, so a blind spot was expected. The truths, measures and
the absence of any pass/fail gate were declared before any scoring ran
([design](superpowers/specs/2026-11-11-a06-held-out-scoring-design.md), with its section
"Change after #70, #72 and #74"). The results are descriptive: they are observations on one
sweep per configuration, not causal attributions, and the report's gate is
`none (descriptive baseline)`.

Every result below comes from the attempt's `report.json`, with two exceptions: the commit,
times and MLflow run ids in the Run table come from the console output and the `tracking.json`
files, and the hashes of the M03 and M04 output files not listed in the report come from
`sha256sum`. The tables were printed from the report by a script that reads only and rounds to
3 decimals, and none was typed by hand. A rate the report gives as null, because no step is in
its denominator, is printed `n/a`, never 0. Nothing was trained, fitted, tuned or promoted on
this evidence.

### Run

| Item | Value |
| --- | --- |
| Attempt | `20261112T222540Z-d3316f`, status `completed`, in `.data/a06-scoring/scores/20261112T222540Z-d3316f/` |
| When | 2026-11-12 22:25–22:32 UTC (17:25–17:32 local), on the laptop's CPU in WSL Ubuntu |
| Code | Commit `8d414c2` on `feature/a06-held-out-scoring`. The recorder, rule, split and scoring commands each started with no modified, staged or untracked files outside the ignored `.data/` and `data/` stores; see the next rows for M03 and M04 |
| MLflow | Experiment `a06-held-out-scoring`, run `67d710877ec24eb3babea1c8080745cd` |
| Evidence | `data/held-out/a06-20261111/capacity-sweeps/`, both sweeps; protocol `natural-overload-sweep-v1`, `protocol_sha256` `22e26dc7536f…` (`cfg-default`) and `1608deee2a3b…` (`cfg-hardware`); `held_out: true` |
| Truths | `capacity-sweep-steps`, then `sweep-truth` with label version `sweep-truth-v2+c211f3e8`, on each sweep, in `.data/a06-scoring/sweep-truth-v2/{steps,truth}/<sweep folder>/` |
| Truth identity (`truths` in the report) | `cfg-default`: `steps_sha256` `7598a42e00fc…`, `steps_summary_sha256` `8f49eb141ff4…`, `truth_sha256` `92da54bca1be…`; `cfg-hardware`: `0ebe5446d0ef…`, `9c797b592097…`, `069a8303ac37…`. All six equal the hashes pinned by #74 ([data card](data-card.md#sweep-truth-rule-sweep-truth-v2-held-out-a06-sweeps-only)) |
| Model (`model.frozen`) | `training-v1+a34ea0be`, trained on dataset v1 `steps-v1.3+aafae73eb211` (`labels-v2+d803a9c6`) with split `split-v3+932b2df410a7` (SHA-256 `32488a942ad6…`) under `eval-contract-v3` (`1cab0cbe`, accepted). `classifier.txt` `b35152b71b4c…`, `preprocessing.json` `7abf667d18b1…` |
| Calibration (`model.frozen`) | `calibration-v1+08119e78`, `calibration.json` `459ff2a6f0db…`; temperature 0.897, abstention threshold 0.647 on calibrated top-label confidence, both applied unchanged |
| M03 and M04 runs | Rebuilt for this attempt with `train-candidates` and `calibrate` on the same split: MLflow runs `34ef108ec273401b8cfc14105efc9334` (M03) and `5f5782166c4c4cf0b3c9cc4622bdbbab` (M04). Two documentation files (this record and `model-training.md`) were being edited in the tree while they ran; neither command reads them, and their outputs are byte-identical to #72's (next row) |
| Comparison with #72 | M03's `classifier.txt`, `preprocessing.json`, `report.json` (`f98da5478d82…`), `validation_predictions.jsonl` (`c1bf9555c597…`), `predictions.jsonl` and `regressor.txt` equal #72's published hashes ([model training](model-training.md#results-on-dataset-v1)), as does M04's `predictions.jsonl` (`9f6b03fed9fe…`, [calibration](model-calibration.md#results-on-dataset-v1)). M04's `calibration.json` and `report.json` differ from those of #72's rebuild on 2026-11-12 (kept in `.data/review-72/calibration/`) only in `candidates.run_id`, M03's MLflow run id |
| Input hashes (SHA-256) | split, both `truth.jsonl` and `truth.json`, the five M03 and M04 files above, and 696 run evidence and journal files, none missing (all listed in the report's `inputs`) |
| Scored steps | 1,740 model calls (825 `cfg-default`, 915 `cfg-hardware`); refused: none; no scored run or plan appears in the split |
| Scorers | `rules-v1+1bf54c19` (covers `none` and `cpu_sandbox`; `inference` and `retrieval` uncovered) and `lightgbm-classifier-m03-m04` |
| Intervals | 95% percentile bootstrap, 2,000 resamples, seed 0, resampling whole batches; a cell with fewer than 2 batches has none |

The full hashes are in the report. The commands were the recipes of the Make targets in
[Reproduce](#reproduce), as `make -n` prints them, with the recorder and rule outputs written
under `.data/a06-scoring/sweep-truth-v2/` to keep them apart from earlier outputs.
Every scoring attempt is kept:

| When (UTC) | Attempt | Code | Outcome |
| --- | --- | --- | --- |
| 2026-11-12, before 02:27 | First launch | `4658849` | Called `make` inside WSL, which is not installed there; stopped before any command ran and wrote nothing |
| 2026-11-12 02:28 | `20261112T022810Z-7eac8a` | `4658849` | Completed; superseded: its `slower` truth still judged the N = 1 baseline steps, contrary to the spec; fixed in 29df6fc before the next attempt. Its outputs stay in `.data/a06-scoring/scores/20261112T022810Z-7eac8a/`, and its truth files in `.data/a06-scoring/superseded/20261112T022810Z-7eac8a-truth/` |
| 2026-11-12 02:55 | `20261112T025531Z-220363` | `29df6fc` | Completed; superseded: it used a different truth rule (`sweep-truth-v1`, from the agent's former per-step truth command), a different model (a classifier retrained on `split-v2`, uncalibrated and never abstaining) and a different set of scored steps (4,000 steps of every type). Its numbers are not compared with this attempt's. Its outputs stay in `.data/a06-scoring/scores/20261112T025531Z-220363/`. Its truth files are no longer on disk: they are identified by the hashes in that attempt's `report.json` `inputs.truth` (`truth.jsonl` `2c822ddaba90…` and `f113b63d2516…`), and can be regenerated with that attempt's code, the scoring branch's former `capacity-sweep-truth` at `29df6fc`. `.data/a06-scoring/truth/` and `steps/` now hold a later `sweep-truth-v1+3b75882f` labelling from the #74 work |
| 2026-11-12 22:25 | `20261112T222540Z-d3316f` | `8d414c2` | Completed; the attempt this section reports |

### How to read the tables

- **Scored steps.** Only model calls whose truth is a class are scored. Root `agent-run` steps,
  tool, retrieval and other steps, and model calls the rule leaves null are counted by reason
  under [Unscored and refused](#unscored-and-refused).
- **Truths.** `queue` is `inference` when vLLM's measured queue time for the model call is above
  the step's noise threshold, max(0.01 s, 3σ). `slower` is `inference` when the call is slower
  than its N = 1 baseline by more than that threshold; the ramp's N = 1 calls are that baseline,
  so they have no `slower` truth (N = 1 control and injected calls keep theirs).
  `protocol_clean` is `none` for every completed model call of a one-agent batch without a
  flood: ramp N = 1, and the `cfg-default` control batches (`cfg-hardware`'s control batches run
  4 agents).
- **Miss rate** is the share of `inference`-truth steps not predicted `inference`; an abstention
  counts as a miss. **False-alarm rate** is the share of `none`-truth steps predicted another
  class; an abstention is not an alarm but stays in the denominator. Each rate's batch count is
  the number of batches in its denominator.
- **Abstention.** The classifier abstains when its calibrated top-label confidence is below
  M04's threshold. The abstention rate is over all steps with that truth. The rates on answered
  steps leave abstained steps out of numerator and denominator; the report gives them no
  interval. `rules-v1` never abstains (rate 0.000 in every cell).
- **Seen** is `fix-mean-empty`, whose task group is in `split-v3`'s train split (the report's
  `seen_task_groups`). **Unseen** is `fix-median-even`, in no dataset v1 split. Only the unseen
  rows test a task the classifier never trained on.
- **n/a** means the report has no value: a rate with no step in its denominator, or an interval
  over fewer than 2 batches. Neither is a measured 0.

### Miss and false-alarm rates

`queue` truth:

| Scope | Scorer | `inference` steps | Miss rate | 95% interval | Batches | `none` steps | False-alarm rate | 95% interval | Batches |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **overall** | `rules-v1` | 694 | 1.000 | 1.000–1.000 | 24 | 1046 | 0.000 | 0.000–0.000 | 66 |
| `cfg-default` | `rules-v1` | 686 | 1.000 | 1.000–1.000 | 21 | 139 | 0.000 | 0.000–0.000 | 33 |
| `cfg-hardware` | `rules-v1` | 8 | 1.000 | 1.000–1.000 | 3 | 907 | 0.000 | 0.000–0.000 | 33 |
| unseen | `rules-v1` | 340 | 1.000 | 1.000–1.000 | 20 | 520 | 0.000 | 0.000–0.000 | 52 |
| seen | `rules-v1` | 354 | 1.000 | 1.000–1.000 | 23 | 526 | 0.000 | 0.000–0.000 | 54 |
| `cfg-default`, unseen | `rules-v1` | 338 | 1.000 | 1.000–1.000 | 19 | 67 | 0.000 | 0.000–0.000 | 22 |
| `cfg-default`, seen | `rules-v1` | 348 | 1.000 | 1.000–1.000 | 20 | 72 | 0.000 | 0.000–0.000 | 24 |
| `cfg-hardware`, unseen | `rules-v1` | 2 | 1.000 | n/a (1 batch, needs 2) | 1 | 453 | 0.000 | 0.000–0.000 | 30 |
| `cfg-hardware`, seen | `rules-v1` | 6 | 1.000 | 1.000–1.000 | 3 | 454 | 0.000 | 0.000–0.000 | 30 |
| **overall** | classifier | 694 | 0.288 | 0.161–0.482 | 24 | 1046 | 0.125 | 0.066–0.187 | 66 |
| `cfg-default` | classifier | 686 | 0.292 | 0.169–0.479 | 21 | 139 | 0.086 | 0.031–0.159 | 33 |
| `cfg-hardware` | classifier | 8 | 0.000 | 0.000–0.000 | 3 | 907 | 0.131 | 0.071–0.206 | 33 |
| unseen | classifier | 340 | 0.276 | 0.150–0.480 | 20 | 520 | 0.144 | 0.081–0.223 | 52 |
| seen | classifier | 354 | 0.299 | 0.175–0.488 | 23 | 526 | 0.106 | 0.057–0.169 | 54 |
| `cfg-default`, unseen | classifier | 338 | 0.278 | 0.150–0.475 | 19 | 67 | 0.104 | 0.035–0.200 | 22 |
| `cfg-default`, seen | classifier | 348 | 0.305 | 0.178–0.500 | 20 | 72 | 0.069 | 0.000–0.175 | 24 |
| `cfg-hardware`, unseen | classifier | 2 | 0.000 | n/a (1 batch, needs 2) | 1 | 453 | 0.150 | 0.078–0.234 | 30 |
| `cfg-hardware`, seen | classifier | 6 | 0.000 | 0.000–0.000 | 3 | 454 | 0.112 | 0.056–0.183 | 30 |

`slower` truth:

| Scope | Scorer | `inference` steps | Miss rate | 95% interval | Batches | `none` steps | False-alarm rate | 95% interval | Batches |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **overall** | `rules-v1` | 1378 | 1.000 | 1.000–1.000 | 53 | 302 | 0.000 | 0.000–0.000 | 46 |
| `cfg-default` | `rules-v1` | 715 | 1.000 | 1.000–1.000 | 26 | 80 | 0.000 | 0.000–0.000 | 22 |
| `cfg-hardware` | `rules-v1` | 663 | 1.000 | 1.000–1.000 | 27 | 222 | 0.000 | 0.000–0.000 | 24 |
| unseen | `rules-v1` | 649 | 1.000 | 1.000–1.000 | 44 | 181 | 0.000 | 0.000–0.000 | 40 |
| seen | `rules-v1` | 729 | 1.000 | 1.000–1.000 | 52 | 121 | 0.000 | 0.000–0.000 | 40 |
| `cfg-default`, unseen | `rules-v1` | 348 | 1.000 | 1.000–1.000 | 19 | 42 | 0.000 | 0.000–0.000 | 16 |
| `cfg-default`, seen | `rules-v1` | 367 | 1.000 | 1.000–1.000 | 25 | 38 | 0.000 | 0.000–0.000 | 18 |
| `cfg-hardware`, unseen | `rules-v1` | 301 | 1.000 | 1.000–1.000 | 25 | 139 | 0.000 | 0.000–0.000 | 24 |
| `cfg-hardware`, seen | `rules-v1` | 362 | 1.000 | 1.000–1.000 | 27 | 83 | 0.000 | 0.000–0.000 | 22 |
| **overall** | classifier | 1378 | 0.599 | 0.437–0.773 | 53 | 302 | 0.043 | 0.013–0.081 | 46 |
| `cfg-default` | classifier | 715 | 0.312 | 0.184–0.522 | 26 | 80 | 0.087 | 0.027–0.174 | 22 |
| `cfg-hardware` | classifier | 663 | 0.908 | 0.770–0.998 | 27 | 222 | 0.027 | 0.000–0.072 | 24 |
| unseen | classifier | 649 | 0.573 | 0.407–0.749 | 44 | 181 | 0.055 | 0.015–0.114 | 40 |
| seen | classifier | 729 | 0.621 | 0.465–0.784 | 52 | 121 | 0.025 | 0.000–0.054 | 40 |
| `cfg-default`, unseen | classifier | 348 | 0.293 | 0.167–0.494 | 19 | 42 | 0.143 | 0.048–0.281 | 16 |
| `cfg-default`, seen | classifier | 367 | 0.330 | 0.203–0.528 | 25 | 38 | 0.026 | 0.000–0.097 | 18 |
| `cfg-hardware`, unseen | classifier | 301 | 0.897 | 0.745–0.997 | 25 | 139 | 0.029 | 0.000–0.087 | 24 |
| `cfg-hardware`, seen | classifier | 362 | 0.917 | 0.791–1.000 | 27 | 83 | 0.024 | 0.000–0.057 | 22 |

`protocol_clean` truth (false alarms only; it has no `inference` class):

| Scope | Scorer | `none` steps | False-alarm rate | 95% interval | Batches |
| --- | --- | --- | --- | --- | --- |
| **overall** | `rules-v1` | 75 | 0.000 | 0.000–0.000 | 15 |
| `cfg-default` | `rules-v1` | 45 | 0.000 | 0.000–0.000 | 9 |
| `cfg-hardware` | `rules-v1` | 30 | 0.000 | 0.000–0.000 | 6 |
| unseen | `rules-v1` | 35 | 0.000 | 0.000–0.000 | 7 |
| seen | `rules-v1` | 40 | 0.000 | 0.000–0.000 | 8 |
| `cfg-default`, unseen | `rules-v1` | 20 | 0.000 | 0.000–0.000 | 4 |
| `cfg-default`, seen | `rules-v1` | 25 | 0.000 | 0.000–0.000 | 5 |
| `cfg-hardware`, unseen | `rules-v1` | 15 | 0.000 | 0.000–0.000 | 3 |
| `cfg-hardware`, seen | `rules-v1` | 15 | 0.000 | 0.000–0.000 | 3 |
| **overall** | classifier | 75 | 0.000 | 0.000–0.000 | 15 |
| `cfg-default` | classifier | 45 | 0.000 | 0.000–0.000 | 9 |
| `cfg-hardware` | classifier | 30 | 0.000 | 0.000–0.000 | 6 |
| unseen | classifier | 35 | 0.000 | 0.000–0.000 | 7 |
| seen | classifier | 40 | 0.000 | 0.000–0.000 | 8 |
| `cfg-default`, unseen | classifier | 20 | 0.000 | 0.000–0.000 | 4 |
| `cfg-default`, seen | classifier | 25 | 0.000 | 0.000–0.000 | 5 |
| `cfg-hardware`, unseen | classifier | 15 | 0.000 | 0.000–0.000 | 3 |
| `cfg-hardware`, seen | classifier | 15 | 0.000 | 0.000–0.000 | 3 |

### Abstention

Classifier, by truth (`rules-v1` never abstains):

| Truth | Scope | Steps | Abstention rate | Miss rate | Miss rate, answered | False-alarm rate | False-alarm rate, answered |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `queue` | **overall** | 1740 | 0.130 | 0.288 | 0.158 | 0.125 | 0.141 |
| `queue` | `cfg-default` | 825 | 0.145 | 0.292 | 0.161 | 0.086 | 0.095 |
| `queue` | `cfg-hardware` | 915 | 0.117 | 0.000 | 0.000 | 0.131 | 0.149 |
| `queue` | unseen | 860 | 0.152 | 0.276 | 0.140 | 0.144 | 0.169 |
| `queue` | seen | 880 | 0.109 | 0.299 | 0.176 | 0.106 | 0.116 |
| `queue` | `cfg-default`, unseen | 405 | 0.163 | 0.278 | 0.141 | 0.104 | 0.127 |
| `queue` | `cfg-default`, seen | 420 | 0.129 | 0.305 | 0.180 | 0.069 | 0.070 |
| `queue` | `cfg-hardware`, unseen | 455 | 0.143 | 0.000 | 0.000 | 0.150 | 0.175 |
| `queue` | `cfg-hardware`, seen | 460 | 0.091 | 0.000 | 0.000 | 0.112 | 0.124 |
| `slower` | **overall** | 1680 | 0.135 | 0.599 | 0.530 | 0.043 | 0.047 |
| `slower` | `cfg-default` | 795 | 0.151 | 0.312 | 0.183 | 0.087 | 0.096 |
| `slower` | `cfg-hardware` | 885 | 0.120 | 0.908 | 0.894 | 0.027 | 0.029 |
| `slower` | unseen | 830 | 0.157 | 0.573 | 0.485 | 0.055 | 0.062 |
| `slower` | seen | 850 | 0.113 | 0.621 | 0.568 | 0.025 | 0.026 |
| `slower` | `cfg-default`, unseen | 390 | 0.169 | 0.293 | 0.149 | 0.143 | 0.171 |
| `slower` | `cfg-default`, seen | 405 | 0.133 | 0.330 | 0.214 | 0.026 | 0.026 |
| `slower` | `cfg-hardware`, unseen | 440 | 0.145 | 0.897 | 0.876 | 0.029 | 0.031 |
| `slower` | `cfg-hardware`, seen | 445 | 0.094 | 0.917 | 0.908 | 0.024 | 0.026 |
| `protocol_clean` | **overall** | 75 | 0.013 | n/a | n/a | 0.000 | 0.000 |
| `protocol_clean` | `cfg-default` | 45 | 0.000 | n/a | n/a | 0.000 | 0.000 |
| `protocol_clean` | `cfg-hardware` | 30 | 0.033 | n/a | n/a | 0.000 | 0.000 |
| `protocol_clean` | unseen | 35 | 0.029 | n/a | n/a | 0.000 | 0.000 |
| `protocol_clean` | seen | 40 | 0.000 | n/a | n/a | 0.000 | 0.000 |
| `protocol_clean` | `cfg-default`, unseen | 20 | 0.000 | n/a | n/a | 0.000 | 0.000 |
| `protocol_clean` | `cfg-default`, seen | 25 | 0.000 | n/a | n/a | 0.000 | 0.000 |
| `protocol_clean` | `cfg-hardware`, unseen | 15 | 0.067 | n/a | n/a | 0.000 | 0.000 |
| `protocol_clean` | `cfg-hardware`, seen | 15 | 0.000 | n/a | n/a | 0.000 | 0.000 |

### What the scorers predicted

On `inference`-truth steps:

| Truth | Scorer | Scope | Predicted (steps) |
| --- | --- | --- | --- |
| `queue` | `rules-v1` | **overall** | `none` 694 |
| `queue` | `rules-v1` | `cfg-default` | `none` 686 |
| `queue` | `rules-v1` | `cfg-hardware` | `none` 8 |
| `queue` | `rules-v1` | unseen | `none` 340 |
| `queue` | `rules-v1` | seen | `none` 354 |
| `queue` | classifier | **overall** | `abstained` 107, `cpu_sandbox` 31, `inference` 494, `none` 61, `retrieval` 1 |
| `queue` | classifier | `cfg-default` | `abstained` 107, `cpu_sandbox` 31, `inference` 486, `none` 61, `retrieval` 1 |
| `queue` | classifier | `cfg-hardware` | `inference` 8 |
| `queue` | classifier | unseen | `abstained` 54, `cpu_sandbox` 16, `inference` 246, `none` 24 |
| `queue` | classifier | seen | `abstained` 53, `cpu_sandbox` 15, `inference` 248, `none` 37, `retrieval` 1 |
| `slower` | `rules-v1` | **overall** | `none` 1378 |
| `slower` | `rules-v1` | `cfg-default` | `none` 715 |
| `slower` | `rules-v1` | `cfg-hardware` | `none` 663 |
| `slower` | `rules-v1` | unseen | `none` 649 |
| `slower` | `rules-v1` | seen | `none` 729 |
| `slower` | classifier | **overall** | `abstained` 201, `cpu_sandbox` 52, `inference` 553, `none` 533, `retrieval` 39 |
| `slower` | classifier | `cfg-default` | `abstained` 113, `cpu_sandbox` 30, `inference` 492, `none` 79, `retrieval` 1 |
| `slower` | classifier | `cfg-hardware` | `abstained` 88, `cpu_sandbox` 22, `inference` 61, `none` 454, `retrieval` 38 |
| `slower` | classifier | unseen | `abstained` 111, `cpu_sandbox` 29, `inference` 277, `none` 211, `retrieval` 21 |
| `slower` | classifier | seen | `abstained` 90, `cpu_sandbox` 23, `inference` 276, `none` 322, `retrieval` 18 |

On `none`-truth steps:

| Truth | Scorer | Scope | Predicted (steps) |
| --- | --- | --- | --- |
| `queue` | `rules-v1` | **overall** | `none` 1046 |
| `queue` | `rules-v1` | `cfg-default` | `none` 139 |
| `queue` | `rules-v1` | `cfg-hardware` | `none` 907 |
| `queue` | `rules-v1` | unseen | `none` 520 |
| `queue` | `rules-v1` | seen | `none` 526 |
| `queue` | classifier | **overall** | `abstained` 120, `cpu_sandbox` 27, `inference` 59, `none` 795, `retrieval` 45 |
| `queue` | classifier | `cfg-default` | `abstained` 13, `cpu_sandbox` 5, `inference` 6, `none` 114, `retrieval` 1 |
| `queue` | classifier | `cfg-hardware` | `abstained` 107, `cpu_sandbox` 22, `inference` 53, `none` 681, `retrieval` 44 |
| `queue` | classifier | unseen | `abstained` 77, `cpu_sandbox` 19, `inference` 31, `none` 368, `retrieval` 25 |
| `queue` | classifier | seen | `abstained` 43, `cpu_sandbox` 8, `inference` 28, `none` 427, `retrieval` 20 |
| `slower` | `rules-v1` | **overall** | `none` 302 |
| `slower` | `rules-v1` | `cfg-default` | `none` 80 |
| `slower` | `rules-v1` | `cfg-hardware` | `none` 222 |
| `slower` | `rules-v1` | unseen | `none` 181 |
| `slower` | `rules-v1` | seen | `none` 121 |
| `slower` | classifier | **overall** | `abstained` 25, `cpu_sandbox` 6, `none` 264, `retrieval` 7 |
| `slower` | classifier | `cfg-default` | `abstained` 7, `cpu_sandbox` 6, `none` 66, `retrieval` 1 |
| `slower` | classifier | `cfg-hardware` | `abstained` 18, `none` 198, `retrieval` 6 |
| `slower` | classifier | unseen | `abstained` 19, `cpu_sandbox` 6, `none` 152, `retrieval` 4 |
| `slower` | classifier | seen | `abstained` 6, `none` 112, `retrieval` 3 |
| `protocol_clean` | `rules-v1` | **overall** | `none` 75 |
| `protocol_clean` | `rules-v1` | `cfg-default` | `none` 45 |
| `protocol_clean` | `rules-v1` | `cfg-hardware` | `none` 30 |
| `protocol_clean` | `rules-v1` | unseen | `none` 35 |
| `protocol_clean` | `rules-v1` | seen | `none` 40 |
| `protocol_clean` | classifier | **overall** | `abstained` 1, `none` 74 |
| `protocol_clean` | classifier | `cfg-default` | `none` 45 |
| `protocol_clean` | classifier | `cfg-hardware` | `abstained` 1, `none` 29 |
| `protocol_clean` | classifier | unseen | `abstained` 1, `none` 34 |
| `protocol_clean` | classifier | seen | `none` 40 |

### Per concurrency level, ramp model calls

`rules-v1`'s rates are given without intervals; they are 1.000 and 0.000 wherever defined.

`queue` truth:

| Configuration | N | Task | `inference` steps | `rules-v1` miss | Classifier miss | 95% interval | `none` steps | `rules-v1` false alarm | Classifier false alarm | 95% interval |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `cfg-default` | 1 | seen | 0 | n/a | n/a | n/a (no steps) | 15 | 0.000 | 0.000 | 0.000–0.000 |
| `cfg-default` | 1 | unseen | 0 | n/a | n/a | n/a (no steps) | 15 | 0.000 | 0.000 | 0.000–0.000 |
| `cfg-default` | 2 | seen | 0 | n/a | n/a | n/a (no steps) | 15 | 0.000 | 0.000 | 0.000–0.000 |
| `cfg-default` | 2 | unseen | 0 | n/a | n/a | n/a (no steps) | 15 | 0.000 | 0.000 | 0.000–0.000 |
| `cfg-default` | 3 | seen | 13 | 1.000 | 0.923 | 0.667–1.000 | 12 | 0.000 | 0.000 | 0.000–0.000 |
| `cfg-default` | 3 | unseen | 9 | 1.000 | 1.000 | 1.000–1.000 | 11 | 0.000 | 0.091 | 0.000–0.143 |
| `cfg-default` | 4 | seen | 27 | 1.000 | 0.926 | 0.800–1.000 | 3 | 0.000 | 0.333 | 0.000–0.500 |
| `cfg-default` | 4 | unseen | 22 | 1.000 | 0.909 | 0.714–1.000 | 8 | 0.000 | 0.250 | 0.000–0.333 |
| `cfg-default` | 6 | seen | 42 | 1.000 | 0.833 | 0.714–0.929 | 3 | 0.000 | 0.000 | 0.000–0.000 |
| `cfg-default` | 6 | unseen | 42 | 1.000 | 0.714 | 0.571–0.857 | 3 | 0.000 | 0.333 | 0.000–1.000 |
| `cfg-default` | 8 | seen | 57 | 1.000 | 0.333 | 0.111–0.650 | 3 | 0.000 | 0.000 | 0.000–0.000 |
| `cfg-default` | 8 | unseen | 57 | 1.000 | 0.439 | 0.211–0.778 | 3 | 0.000 | 0.000 | 0.000–0.000 |
| `cfg-default` | 12 | seen | 89 | 1.000 | 0.079 | 0.033–0.167 | 1 | 0.000 | 0.000 | n/a (1 batch, needs 2) |
| `cfg-default` | 12 | unseen | 85 | 1.000 | 0.035 | 0.000–0.107 | 5 | 0.000 | 0.200 | 0.000–1.000 |
| `cfg-default` | 16 | seen | 114 | 1.000 | 0.061 | 0.026–0.105 | 6 | 0.000 | 0.000 | 0.000–0.000 |
| `cfg-default` | 16 | unseen | 120 | 1.000 | 0.050 | 0.050–0.050 | 0 | n/a | n/a | n/a (no steps) |
| `cfg-hardware` | 1 | seen | 0 | n/a | n/a | n/a (no steps) | 15 | 0.000 | 0.000 | 0.000–0.000 |
| `cfg-hardware` | 1 | unseen | 0 | n/a | n/a | n/a (no steps) | 15 | 0.000 | 0.000 | 0.000–0.000 |
| `cfg-hardware` | 2 | seen | 0 | n/a | n/a | n/a (no steps) | 15 | 0.000 | 0.000 | 0.000–0.000 |
| `cfg-hardware` | 2 | unseen | 0 | n/a | n/a | n/a (no steps) | 15 | 0.000 | 0.000 | 0.000–0.000 |
| `cfg-hardware` | 3 | seen | 0 | n/a | n/a | n/a (no steps) | 25 | 0.000 | 0.000 | 0.000–0.000 |
| `cfg-hardware` | 3 | unseen | 0 | n/a | n/a | n/a (no steps) | 20 | 0.000 | 0.000 | 0.000–0.000 |
| `cfg-hardware` | 4 | seen | 0 | n/a | n/a | n/a (no steps) | 30 | 0.000 | 0.000 | 0.000–0.000 |
| `cfg-hardware` | 4 | unseen | 0 | n/a | n/a | n/a (no steps) | 30 | 0.000 | 0.000 | 0.000–0.000 |
| `cfg-hardware` | 6 | seen | 0 | n/a | n/a | n/a (no steps) | 45 | 0.000 | 0.022 | 0.000–0.067 |
| `cfg-hardware` | 6 | unseen | 0 | n/a | n/a | n/a (no steps) | 45 | 0.000 | 0.067 | 0.000–0.133 |
| `cfg-hardware` | 8 | seen | 0 | n/a | n/a | n/a (no steps) | 60 | 0.000 | 0.033 | 0.000–0.050 |
| `cfg-hardware` | 8 | unseen | 0 | n/a | n/a | n/a (no steps) | 60 | 0.000 | 0.117 | 0.100–0.150 |
| `cfg-hardware` | 12 | seen | 0 | n/a | n/a | n/a (no steps) | 90 | 0.000 | 0.056 | 0.033–0.067 |
| `cfg-hardware` | 12 | unseen | 0 | n/a | n/a | n/a (no steps) | 90 | 0.000 | 0.033 | 0.000–0.067 |
| `cfg-hardware` | 16 | seen | 0 | n/a | n/a | n/a (no steps) | 120 | 0.000 | 0.150 | 0.125–0.200 |
| `cfg-hardware` | 16 | unseen | 0 | n/a | n/a | n/a (no steps) | 120 | 0.000 | 0.225 | 0.150–0.325 |

`slower` truth (no ramp N = 1 rows: those calls are the baseline):

| Configuration | N | Task | `inference` steps | `rules-v1` miss | Classifier miss | 95% interval | `none` steps | `rules-v1` false alarm | Classifier false alarm | 95% interval |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `cfg-default` | 2 | seen | 6 | 1.000 | 1.000 | 1.000–1.000 | 9 | 0.000 | 0.000 | 0.000–0.000 |
| `cfg-default` | 2 | unseen | 0 | n/a | n/a | n/a (no steps) | 15 | 0.000 | 0.000 | 0.000–0.000 |
| `cfg-default` | 3 | seen | 16 | 1.000 | 0.938 | 0.750–1.000 | 9 | 0.000 | 0.000 | 0.000–0.000 |
| `cfg-default` | 3 | unseen | 10 | 1.000 | 1.000 | 1.000–1.000 | 10 | 0.000 | 0.200 | 0.000–0.333 |
| `cfg-default` | 4 | seen | 28 | 1.000 | 0.929 | 0.800–1.000 | 2 | 0.000 | 0.500 | 0.000–1.000 |
| `cfg-default` | 4 | unseen | 26 | 1.000 | 0.923 | 0.778–1.000 | 4 | 0.000 | 0.500 | 0.000–1.000 |
| `cfg-default` | 6 | seen | 43 | 1.000 | 0.837 | 0.714–0.933 | 2 | 0.000 | 0.000 | 0.000–0.000 |
| `cfg-default` | 6 | unseen | 43 | 1.000 | 0.721 | 0.571–0.857 | 2 | 0.000 | 0.500 | 0.000–1.000 |
| `cfg-default` | 8 | seen | 57 | 1.000 | 0.333 | 0.111–0.650 | 3 | 0.000 | 0.000 | 0.000–0.000 |
| `cfg-default` | 8 | unseen | 57 | 1.000 | 0.439 | 0.211–0.778 | 3 | 0.000 | 0.000 | 0.000–0.000 |
| `cfg-default` | 12 | seen | 89 | 1.000 | 0.079 | 0.033–0.167 | 1 | 0.000 | 0.000 | n/a (1 batch, needs 2) |
| `cfg-default` | 12 | unseen | 87 | 1.000 | 0.057 | 0.000–0.107 | 3 | 0.000 | 0.333 | 0.000–1.000 |
| `cfg-default` | 16 | seen | 114 | 1.000 | 0.061 | 0.026–0.105 | 6 | 0.000 | 0.000 | 0.000–0.000 |
| `cfg-default` | 16 | unseen | 120 | 1.000 | 0.050 | 0.050–0.050 | 0 | n/a | n/a | n/a (no steps) |
| `cfg-hardware` | 2 | seen | 8 | 1.000 | 1.000 | 1.000–1.000 | 7 | 0.000 | 0.000 | 0.000–0.000 |
| `cfg-hardware` | 2 | unseen | 1 | 1.000 | 1.000 | n/a (1 batch, needs 2) | 14 | 0.000 | 0.000 | 0.000–0.000 |
| `cfg-hardware` | 3 | seen | 17 | 1.000 | 1.000 | 1.000–1.000 | 8 | 0.000 | 0.000 | 0.000–0.000 |
| `cfg-hardware` | 3 | unseen | 5 | 1.000 | 1.000 | 1.000–1.000 | 15 | 0.000 | 0.000 | 0.000–0.000 |
| `cfg-hardware` | 4 | seen | 20 | 1.000 | 1.000 | 1.000–1.000 | 10 | 0.000 | 0.000 | 0.000–0.000 |
| `cfg-hardware` | 4 | unseen | 10 | 1.000 | 1.000 | 1.000–1.000 | 20 | 0.000 | 0.000 | 0.000–0.000 |
| `cfg-hardware` | 6 | seen | 36 | 1.000 | 1.000 | 1.000–1.000 | 9 | 0.000 | 0.111 | 0.000–0.333 |
| `cfg-hardware` | 6 | unseen | 27 | 1.000 | 1.000 | 1.000–1.000 | 18 | 0.000 | 0.000 | 0.000–0.000 |
| `cfg-hardware` | 8 | seen | 47 | 1.000 | 1.000 | 1.000–1.000 | 13 | 0.000 | 0.000 | 0.000–0.000 |
| `cfg-hardware` | 8 | unseen | 39 | 1.000 | 1.000 | 1.000–1.000 | 21 | 0.000 | 0.000 | 0.000–0.000 |
| `cfg-hardware` | 12 | seen | 84 | 1.000 | 1.000 | 1.000–1.000 | 6 | 0.000 | 0.000 | n/a (1 batch, needs 2) |
| `cfg-hardware` | 12 | unseen | 73 | 1.000 | 0.986 | 0.958–1.000 | 17 | 0.000 | 0.000 | 0.000–0.000 |
| `cfg-hardware` | 16 | seen | 99 | 1.000 | 1.000 | 1.000–1.000 | 21 | 0.000 | 0.048 | 0.000–0.125 |
| `cfg-hardware` | 16 | unseen | 102 | 1.000 | 1.000 | 1.000–1.000 | 18 | 0.000 | 0.222 | 0.000–0.500 |

### Prefill flood and its controls

`queue` truth:

| Configuration | N | Phase | Scorer | Task | `inference` steps | Miss rate | 95% interval | `none` steps | False-alarm rate | 95% interval |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `cfg-default` | 1 | injected | `rules-v1` | seen | 6 | 1.000 | 1.000–1.000 | 4 | 0.000 | 0.000–0.000 |
| `cfg-default` | 1 | injected | `rules-v1` | unseen | 3 | 1.000 | n/a (1 batch, needs 2) | 2 | 0.000 | n/a (1 batch, needs 2) |
| `cfg-default` | 1 | injected | classifier | seen | 6 | 0.167 | 0.000–0.333 | 4 | 1.000 | 1.000–1.000 |
| `cfg-default` | 1 | injected | classifier | unseen | 3 | 0.333 | n/a (1 batch, needs 2) | 2 | 1.000 | n/a (1 batch, needs 2) |
| `cfg-default` | 1 | control | `rules-v1` | seen | 0 | n/a | n/a (no steps) | 10 | 0.000 | 0.000–0.000 |
| `cfg-default` | 1 | control | `rules-v1` | unseen | 0 | n/a | n/a (no steps) | 5 | 0.000 | n/a (1 batch, needs 2) |
| `cfg-default` | 1 | control | classifier | seen | 0 | n/a | n/a (no steps) | 10 | 0.000 | 0.000–0.000 |
| `cfg-default` | 1 | control | classifier | unseen | 0 | n/a | n/a (no steps) | 5 | 0.000 | n/a (1 batch, needs 2) |
| `cfg-hardware` | 4 | injected | `rules-v1` | seen | 6 | 1.000 | 1.000–1.000 | 24 | 0.000 | 0.000–0.000 |
| `cfg-hardware` | 4 | injected | `rules-v1` | unseen | 2 | 1.000 | n/a (1 batch, needs 2) | 28 | 0.000 | 0.000–0.000 |
| `cfg-hardware` | 4 | injected | classifier | seen | 6 | 0.000 | 0.000–0.000 | 24 | 1.000 | 1.000–1.000 |
| `cfg-hardware` | 4 | injected | classifier | unseen | 2 | 0.000 | n/a (1 batch, needs 2) | 28 | 1.000 | 1.000–1.000 |
| `cfg-hardware` | 4 | control | `rules-v1` | seen | 0 | n/a | n/a (no steps) | 30 | 0.000 | 0.000–0.000 |
| `cfg-hardware` | 4 | control | `rules-v1` | unseen | 0 | n/a | n/a (no steps) | 30 | 0.000 | 0.000–0.000 |
| `cfg-hardware` | 4 | control | classifier | seen | 0 | n/a | n/a (no steps) | 30 | 0.033 | 0.000–0.100 |
| `cfg-hardware` | 4 | control | classifier | unseen | 0 | n/a | n/a (no steps) | 30 | 0.000 | 0.000–0.000 |

`slower` truth:

| Configuration | N | Phase | Scorer | Task | `inference` steps | Miss rate | 95% interval | `none` steps | False-alarm rate | 95% interval |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `cfg-default` | 1 | injected | `rules-v1` | seen | 10 | 1.000 | 1.000–1.000 | 0 | n/a | n/a (no steps) |
| `cfg-default` | 1 | injected | `rules-v1` | unseen | 5 | 1.000 | n/a (1 batch, needs 2) | 0 | n/a | n/a (no steps) |
| `cfg-default` | 1 | injected | classifier | seen | 10 | 0.100 | 0.000–0.200 | 0 | n/a | n/a (no steps) |
| `cfg-default` | 1 | injected | classifier | unseen | 5 | 0.200 | n/a (1 batch, needs 2) | 0 | n/a | n/a (no steps) |
| `cfg-default` | 1 | control | `rules-v1` | seen | 4 | 1.000 | 1.000–1.000 | 6 | 0.000 | 0.000–0.000 |
| `cfg-default` | 1 | control | `rules-v1` | unseen | 0 | n/a | n/a (no steps) | 5 | 0.000 | n/a (1 batch, needs 2) |
| `cfg-default` | 1 | control | classifier | seen | 4 | 1.000 | 1.000–1.000 | 6 | 0.000 | 0.000–0.000 |
| `cfg-default` | 1 | control | classifier | unseen | 0 | n/a | n/a (no steps) | 5 | 0.000 | n/a (1 batch, needs 2) |
| `cfg-hardware` | 4 | injected | `rules-v1` | seen | 30 | 1.000 | 1.000–1.000 | 0 | n/a | n/a (no steps) |
| `cfg-hardware` | 4 | injected | `rules-v1` | unseen | 30 | 1.000 | 1.000–1.000 | 0 | n/a | n/a (no steps) |
| `cfg-hardware` | 4 | injected | classifier | seen | 30 | 0.000 | 0.000–0.000 | 0 | n/a | n/a (no steps) |
| `cfg-hardware` | 4 | injected | classifier | unseen | 30 | 0.000 | 0.000–0.000 | 0 | n/a | n/a (no steps) |
| `cfg-hardware` | 4 | control | `rules-v1` | seen | 21 | 1.000 | 1.000–1.000 | 9 | 0.000 | 0.000–0.000 |
| `cfg-hardware` | 4 | control | `rules-v1` | unseen | 14 | 1.000 | 1.000–1.000 | 16 | 0.000 | 0.000–0.000 |
| `cfg-hardware` | 4 | control | classifier | seen | 21 | 1.000 | 1.000–1.000 | 9 | 0.000 | 0.000–0.000 |
| `cfg-hardware` | 4 | control | classifier | unseen | 14 | 1.000 | 1.000–1.000 | 16 | 0.000 | 0.000–0.000 |

### Unscored and refused

| Item | `cfg-default` | `cfg-hardware` | Reason |
| --- | --- | --- | --- |
| Live-confirmation runs with no truth records | 9 | 24 | Not replays: no plan, so no step baseline |
| Runs without evidence | 0 | 0 | |
| Excluded batches | 0 | 0 | |

Steps with no judged truth, both sweeps together (`unscored.steps`):

| Reason | `queue` | `slower` | `protocol_clean` |
| --- | --- | --- | --- |
| `root_step` (the root `agent-run` step) | 348 | 348 | 348 |
| `no_queue_reference` (tool, retrieval and other steps) | 1,912 | 1,912 | 1,912 |
| `not_applicable` | none (not used by this truth) | 60 (ramp N = 1 model calls) | 1,665 (model calls outside one-agent batches without a flood) |

No model call had `not_completed`, `no_baseline` or `span_not_measured`, and no step was refused
at the feature join (`refused` is empty).

### Observations

These are observed values on one sweep per configuration, not causal attributions.

1. **`rules-v1` misses every inference-truth step and raises no alarm on model calls.** Its miss
   rate is 1.000 under both truths in every slice, as expected: it covers only `none` and
   `cpu_sandbox`. Its false-alarm rate is 0.000 under `queue`, `slower` and `protocol_clean`.
2. **On `cfg-default` the classifier misses fewer queued calls at higher N.** Its `queue` miss
   rate is 0.292 (0.169–0.479), and 0.161 on answered calls. In the ramp it is 0.923 and 1.000
   at N = 3 (seen and unseen), and 0.079 and 0.035 at N = 12. Of the 686 queued model calls it
   predicted `inference` on 486 and abstained on 107; it predicted `none` on 61, `cpu_sandbox`
   on 31 and `retrieval` on 1. No `steps-v1.3` feature describes GPU or vLLM state, and the
   report does not show which features drive these predictions.
3. **On `cfg-hardware` the classifier misses the slowdown that does not queue.** vLLM queued
   only 8 model calls above the threshold, all in the prefill-flood batches, and the classifier
   predicted `inference` on all 8. Of the 663 model calls slower than their baseline it missed
   0.908 (0.770–0.998), 0.894 on answered calls, predicting `none` on 454. In the ramp its
   `slower` miss rate is 1.000 at every level except N = 12 unseen (0.986). The label check
   could not judge this kind of contention either (see [Findings](#findings)).
4. **Classifier `queue` alarms are highest at the top of the `cfg-hardware` ramp.** On ramp
   model calls its `queue` false-alarm rate is between 0.000 and 0.117 up to N = 12, and 0.150
   (seen) and 0.225 (unseen) at N = 16. vLLM did not queue there, but calls ran slower inside
   shared batches. Over all `cfg-hardware` calls it is 0.131 (0.071–0.206), and 0.086
   (0.031–0.159) on `cfg-default`.
5. **Prefill-flood batches.** The classifier predicted a class other than `none` for every
   flood-batch model call that the `queue` truth calls `none` (false-alarm rate 1.000 in every
   injected cell). Those calls were not queued above the threshold, but under the `slower` truth
   every flood-batch call is `inference`, and flood batches ran far slower than their controls
   ([Finding 4](#findings)). On the control batches its `queue` false-alarm rate is 0.000 on
   `cfg-default`, and 0.033 (seen) and 0.000 (unseen) on `cfg-hardware`.
6. **Clean one-agent model calls.** Neither scorer raised an alarm on the 75 `protocol_clean`
   model calls (0.000, interval 0.000–0.000). The classifier abstained on 1 of them, a
   `cfg-hardware` unseen call (abstention rate 0.013 overall).
7. **Abstention.** The classifier abstained on 0.130 of the 1,740 scored model calls: 0.145 on
   `cfg-default` and 0.117 on `cfg-hardware`, 0.152 on the unseen task and 0.109 on the seen
   one. Its abstentions fall mostly on `cfg-default`'s queued calls (107 of 686) and on
   `cfg-hardware`'s unqueued calls (107 of 907). The report gives no interval for abstention.
8. **Seen and unseen tasks.** The rates differ little between the two tasks, and every pair of
   classifier miss and false-alarm intervals shown above for the same configuration overlaps where both exist (the `cfg-hardware` unseen `queue` miss rate rests on 1 batch and has none).
   The largest gap is the classifier's `slower` false alarms on `cfg-default`: 0.143
   (0.048–0.281) unseen against 0.026 (0.000–0.097) seen. One unseen task cannot show how the
   classifier generalises to new tasks.

### Limitations of this baseline

From the report's own `limitations`:

- one laptop, one sweep per configuration;
- dataset v1's replays were recorded mostly on Ollama, A06 on vLLM;
- no `steps-v1.3` feature describes GPU or vLLM state;
- one of the two A06 tasks (`fix-mean-empty`) was seen in training, so only `fix-median-even`
  tests an unseen task.

And from the inputs:

- **One training task group.** `split-v3+932b2df410a7` has one task group in train
  (`fix-mean-empty`, 1,755 steps from 234 runs) and the same group in validation (273 steps;
  [data card](data-card.md#split-class-balance-and-leakage)). So "seen in training" means seen as the only training task
  group, and the seen/unseen contrast is one task against one.
- **Calibration is from dataset v1.** The temperature and threshold were fitted on dataset v1's
  273 validation steps, from replays recorded mostly on Ollama. The report measures no calibration error on
  A06, so the classifier's confidences are not shown to be calibrated on this evidence, and the
  abstention rate here is a consequence of that threshold, not a tuned operating point.
- **Small cells.** Several cells rest on 1–3 batches, and intervals from 2–3 batches are coarse.
  A cell with 1 batch has no interval.
- **The truths are proxies, and they judge model calls only.** `queue` sees only queueing, and
  `slower` only slowdown against the N = 1 replays. Tool, retrieval and root steps are unscored,
  so neither scorer's behaviour on them is measured here; `protocol_clean` covers 75 model calls
  in 15 batches.

The scores stay local in `.data/a06-scoring/` and belong to no dataset release (see the
[data card](data-card.md#held-out-a06-evidence-in-no-release)). M2 PR 4's inference rule will be
scored as a further scorer by the same command.

## Every attempt, including failed and stopped ones

Nothing was deleted. All of these are in `data/held-out/a06-20261111/`.
The scoring attempts, including the superseded one, are listed under
[Attribution baseline, Run](#run).

| When (UTC) | Attempt | Outcome |
| --- | --- | --- |
| 2026-11-11 16:01:22 | Record `fix-mean-empty` and `fix-window-count` with the runner's default 900 s limit | Both refused before starting (exit 2): a 900 s run with vLLM, DCGM and GPM exceeds the 80,000-record sample budget (465 s fits). This led to the 450 s protocol limit. |
| 16:02:14 | Record `fix-mean-empty` (450 s) | Passed; this is the source. |
| 16:02:20–16:02:59 | Record `fix-window-count` (450 s), 4 attempts | All 4 failed verification: the model skipped `test_windows.py`, applied `skip-first-window` and never applied `include-final-window` (runs `83eb2b52…`, `caa21921…`, `6704e7f3…`, `0635b818…`). The same task had passed 12 of 12 on Ollama `qwen3:1.7b`. It was replaced by a new task instead of being reworded. |
| 16:04:02 | Preflight: 16 live agents, no policy (`preflight-batch16/`) | 37.3 s. All 63,427 live events delivered; CPU/NVML sampler 0 of 1,495 rounds late; service sampler 28 late and 4 missed of 1,453 rounds. The 8 `fix-window-count` runs failed verification as above. Not protocol evidence. |
| 18:40:13 | Record `fix-median-even` r1, declared as one design with at most 3 attempts | Passed on the first attempt; this is the source. |
| 18:40:46 | Preflight dry sweep, `cfg-default`, 300 s limit (`preflight-dry-run/`) | Completed, capacity 4 under a non-protocol hash. Meant to stop after N = 1, but the SIGINT went to the `uv` wrapper, which did not forward it, so the whole flow ran. Not protocol evidence. |
| 21:11 | First detached launch of the `cfg-default` protocol sweep | Did not start: no process, log or sweep folder was created. Relaunched. |
| 21:12:12 | `cfg-default` protocol sweep | Completed, exit 0. |
| 21:26:08 | Start vLLM for `cfg-hardware` at `gpu_memory_utilization` 0.8 | Started on the first attempt (ready at 21:30:22). No fallback to 0.75 or 0.7 was needed. |
| 21:30:33 | `cfg-hardware` protocol sweep | Completed, exit 0. |

## Limitations

- **Scope.** One laptop GPU in WSL2, two vLLM configurations, one model and one run per
  configuration. The capacity is not a cloud or multi-GPU number, and a single sweep cannot
  show run-to-run variability of the knee itself.
- **Replays share prompts.** Every replay of a source sends the same prompts, so the prefix
  cache hits 95.8–99.0% of queried tokens. Independent agents with different prompts would
  prefill more and could saturate earlier. The live confirmation batches ran at similar rates,
  but they too ran the same two tasks.
- **What the pass rate means.** Replays force the source's tool calls, so the pass rate mostly
  shows that generation still finished in time, not decision quality under load. The live runs
  cover decisions, at the capacity only.
- **Sampled server metrics at N = 16.** The vLLM, DCGM and GPM samples are incomplete at
  16 agents: about one in four rounds is missed on `cfg-default`. Per-request vLLM spans and
  NVML readings are complete.
- **Thermals.** Laptop GPU clocks varied during the sessions: the SM clock drift column reaches
  −634 MHz at `cfg-hardware` N = 12. Clocks and power are recorded and not corrected.
- **Host counters.** The WSL2 CPU pressure and cgroup counters cover the whole VM, including the
  vLLM container. DCGM profiling fields are unavailable under WSL2; GPM comes from the Windows
  exporter.
- **Label validation covers queueing only.** Its reference is vLLM's queue time, so it cannot
  validate contention that shows up as longer prefill or decode. That is why `cfg-hardware` is
  `inconclusive`. Widening the reference would be a new protocol version, declared before it
  is run. It also validates the noise threshold only, not the dataset's `labels-v2` (see
  [Summary](#summary)).
- **Tool steps are labelled with no reference.** Tool steps have no vLLM queue time to check
  against, yet at N ≥ 2 the noise rule marks 26% of them contended in `cfg-default` (219 of 841)
  and 38% in `cfg-hardware` (320 of 841). This is the step-type spillover the
  [data card](data-card.md) warns about, now measured under natural load. The attribution
  baseline has no reference for these steps either: under `sweep-truth-v2` every truth of a tool,
  retrieval or other step is null `no_queue_reference`, and it counts them as unscored.
- **Held-out evidence.** All natural-overload and `prefill-flood-openai-chat-v1` runs are
  held-out test evidence. They must never be used for training, calibration or tuning, and none
  may enter a dataset release before `split-v3` lists them. `fix-median-even` is kept out of
  training too (Benoy's decision, 2026-11-11): `split-v2` does not name it and would hash it
  into train, so `split-v3` is to put it in `test_task_groups`.

## Reproduce

The evidence is versioned with DVC in `data/held-out/a06-20261111/` (1,917 files, 950 MB,
pointer `data/held-out/a06-20261111.dvc`). Its `README.txt` lists the contents, and
`SHA256SUMS` covers every file. It is held-out test evidence: `make pull-dataset` downloads it
only when it is named, and the dataset build never reads it (see the
[data card](data-card.md#held-out-a06-evidence-in-no-release)).

```sh
# Download it (public HTTPS, no login) and check it:
make pull-dataset TARGET=data/held-out/a06-20261111
(cd data/held-out/a06-20261111 && sha256sum -c --quiet SHA256SUMS)

# Re-run the analysis without the GPU. sweep-analyse writes analysis.json into the folder it
# reads, so analyse a copy and leave the versioned store unchanged:
cp -r data/held-out/a06-20261111/capacity-sweeps /tmp/a06-sweeps
python -m agentwatch_agent sweep-analyse /tmp/a06-sweeps/<sweep folder>

# Regenerate this record's Results and Sampling coverage tables (reads only):
S=data/held-out/a06-20261111/capacity-sweeps
python -m agentwatch_agent sweep-record $S/cfg-default-20261111T211237Z-2d75ea \
  $S/cfg-hardware-20261111T213039Z-b4ad4a --coverage-jsonl /tmp/coverage-per-level.jsonl

# Run the sweeps again. In WSL Ubuntu, with the live API, vLLM and its collector, DCGM and the
# Windows GPM exporter running (see services/agent/README.md, "Capacity sweep"):
H=data/held-out/a06-20261111/sources
python -m agentwatch_agent capacity-sweep --configuration cfg-default \
  --source $H/fix-mean-empty-20261111T160216Z-b526ca \
  --source $H/fix-median-even-20261111T184016Z-64d338 \
  --model-backend vllm --gpm-file .data/gpm/gpm.prom --dcgm-url http://127.0.0.1:9400/metrics
# cfg-hardware: restart vLLM with VLLM_MAX_NUM_SEQS=16 VLLM_GPU_MEMORY_UTILIZATION=0.8 first,
# then the same command with --configuration cfg-hardware --start-attempt "0.8:started".
# Attribution baseline, CPU only, on the frozen evidence in data/held-out/ (dataset v1 pulled).
# 1. Recorder: each scored step with its measurements, unjudged.
H=data/held-out/a06-20261111/capacity-sweeps
D=cfg-default-20261111T211237Z-2d75ea
W=cfg-hardware-20261111T213039Z-b4ad4a
O=.data/a06-scoring/sweep-truth-v2
make agent-capacity-sweep-steps SWEEP=$H/$D OUTPUT=$O/steps/$D
make agent-capacity-sweep-steps SWEEP=$H/$W OUTPUT=$O/steps/$W
# 2. Rule: sweep-truth-v2+c211f3e8 decides every truth and reason.
make sweep-truth STEPS=$O/steps/$D OUTPUT=$O/truth/$D
make sweep-truth STEPS=$O/steps/$W OUTPUT=$O/truth/$W
# 3. Split: freeze split-v3 of dataset v1 (writes split-v3+932b2df410a7.json beside split-v2).
uv run --package agentwatch-data python -m agentwatch_data.splits \
  --release data/datasets/steps-v1.3+aafae73eb211
SPLIT=data/datasets/splits/steps-v1.3+aafae73eb211/split-v3+932b2df410a7.json
# 4. M03 and 5. M04: the frozen classifier and its calibration, into .data/candidates and
# .data/calibration.
make train-candidates SPLIT=$SPLIT
make calibrate SPLIT=$SPLIT
# 6. Score rules-v1 and the M03/M04 classifier on both truth folders.
make score-held-out TRUTH="$O/truth/$D $O/truth/$W" SPLIT=$SPLIT
```

Checked on 2026-11-11 against the DVC copy: `sweep-analyse` reproduces both sweeps' recorded
level summaries, capacity decisions, excluded batches and label validation, with
`protocol_sha256_matches: true`. `sweep-record`'s output is identical to the tables above, and
its coverage reports are byte-identical to `coverage-per-level.jsonl`. A new sweep writes to
`.data/capacity-sweeps/` by default, never into the versioned store.

The attribution baseline's run used these targets' recipes as `make -n` prints them, because the
WSL Ubuntu environment has no `make`. A rerun writes a new attempt folder; the attempt id
includes the start time. The `split-v3` file is not yet versioned: the run of 2026-11-12
regenerated it (SHA-256 `32488a942ad6…`, as the report records) and removed it afterwards, so the
DVC-tracked split folder was left as it was.

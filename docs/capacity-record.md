# Measured agent capacity on the GPU laptop (A06)

This is the record of the A06 natural-overload capacity sweeps run on 2026-11-11. Every number in
the result tables comes from the two sweeps' own `sweep.json` files, their injection logs and
their run folders. The `sweep-record` command generated the Results and Sampling coverage tables
from those files, and they are not edited by hand (see [Reproduce](#reproduce)). They describe this
laptop and these two vLLM configurations only, never cloud capacity.

Design: [natural-overload capacity sweep](superpowers/specs/2026-11-10-natural-overload-capacity-sweep-design.md).
Command: `capacity-sweep` ([agent README](../services/agent/README.md#capacity-sweep)).

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

   These are observed differences between paired batches, not a causal attribution. Scoring the
   injected runs with the rule and the model is A06 PR 3.
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
`.data/a06-20261111/coverage-per-level.jsonl`.

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

## Every attempt, including failed and stopped ones

Nothing was deleted. All of these are under `.data/a06-20261111/` and in the evidence bundle.

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
  [data card](data-card.md) warns about, now measured under natural load. PR 3 should report it.
- **Held-out evidence.** All natural-overload and `prefill-flood-openai-chat-v1` runs are
  held-out test evidence. They must never be used for training, calibration or tuning, and none
  may enter a dataset release before `split-v3` lists them. `fix-median-even` is kept out of
  training too (Benoy's decision, 2026-11-11): `split-v2` does not name it and would hash it
  into train, so `split-v3` is to put it in `test_task_groups`.

## Reproduce

```sh
# In WSL Ubuntu, with the live API, vLLM and its collector, DCGM and the Windows GPM exporter
# running (see services/agent/README.md, "Capacity sweep"):
python -m agentwatch_agent capacity-sweep --configuration cfg-default \
  --source .data/a06-20261111/sources/fix-mean-empty-20261111T160216Z-b526ca \
  --source .data/a06-20261111/sources/fix-median-even-20261111T184016Z-64d338 \
  --model-backend vllm --gpm-file .data/gpm/gpm.prom --dcgm-url http://127.0.0.1:9400/metrics
# cfg-hardware: restart vLLM with VLLM_MAX_NUM_SEQS=16 VLLM_GPU_MEMORY_UTILIZATION=0.8 first,
# then the same command with --configuration cfg-hardware --start-attempt "0.8:started".
# Re-run the analysis of a recorded sweep offline, without the GPU:
python -m agentwatch_agent sweep-analyse .data/a06-20261111/capacity-sweeps/<sweep folder>
# Regenerate this record's Results and Sampling coverage tables (reads only):
S=.data/a06-20261111/capacity-sweeps
python -m agentwatch_agent sweep-record $S/cfg-default-20261111T211237Z-2d75ea \
  $S/cfg-hardware-20261111T213039Z-b4ad4a --coverage-jsonl /tmp/coverage-per-level.jsonl
```

Checked on 2026-11-11: `sweep-record`'s output is identical to the tables above, and its
coverage reports are byte-identical to `coverage-per-level.jsonl`.

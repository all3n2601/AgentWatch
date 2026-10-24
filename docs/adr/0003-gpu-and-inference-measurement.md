# 0003: Sample the GPU through NVML and keep the inference queue unobserved

Date: October 24, 2026. Status: proposed for A04 review.

A04 needs real GPU and inference timing correlated with agent steps on the team's only GPU
machine, an RTX 5070 laptop running the agent in WSL2 and Ollama in Docker (A01,
[the access record](../gpu-measurement-access.md)). DCGM and vLLM are not installed there,
per-process GPU accounting is unavailable under WSL2, and Ollama exposes no request queue.
The A01 probe ran `nvidia-smi` beside the runner; its samples were not ingested and could
only be joined to steps through wall time.

**GPU readings.** The runner reads NVML in-process through a small `ctypes` binding
(`agentwatch_worker.nvml`), on the same sampling thread as CPU PSI and the cgroup counter.
Spawning `nvidia-smi` for each sample cost about 93 ms of CPU on this laptop, enough to
distort the CPU signals sampled beside it, against about 0.14 ms per NVML reading. The
binding uses a few long-stable entry points, so no dependency is added. Five versioned,
whole-device gauges are registered in `live-contracts-v1`: kernel and memory-controller
utilization, memory used, power and SM clock. They are never per process: other processes on
the device (on this laptop, the Windows desktop) are included.

**Identity.** The GPU is registered as its own `inference` resource (`gpu-*`, service
`nvidia-nvml`), not as the model server. The model server is a separate resource whose
placement (`/api/ps`) says how much of the model is in VRAM; model-call steps reference the
server. A GPU reading is therefore evidence about the device the model uses, joined to a step
only by its window in the shared clock domain, not a measurement of that step's own work.

**Inference timing.** Each model call keeps its runner-observed wall time and Ollama's
reported total, load, prompt-evaluation and generation durations. The observed queue is
recorded as unavailable with a reason: Ollama reports none, and inferring one from duration
differences would present an estimate as a measurement. Two differences are kept, labeled as
estimates across two clocks: server time outside load, prompt and generation, and runner time
outside Ollama's total.

**Evidence of sampling quality.** The sampler records achieved intervals, late and missed
rounds and its own per-round thread CPU and wall time. A `coverage` report applies the
contract's `[started, finished)` window rule to the journal, and protocol
`resource-sampling-overhead-v1` alternates sampled and unsampled requests of the runner's
first model call to measure whether sampling changes inference timing. None of these apply a
threshold; O01 owns budgets.

Consequences: hosts without NVML record every GPU metric as `unsupported` (or `unavailable`
when the library fails to initialize), never zero, and the run's configuration lists the
sampled metrics, so GPU and non-GPU runs have different digests. NVML utilization averages
over the driver's own period and cannot resolve sub-period activity or establish saturation.
DCGM, vLLM queue metrics, Qdrant retrieval latency and L4/GCP hardware remain unverified, and
adding any of them needs new versioned metric names rather than redefining these.

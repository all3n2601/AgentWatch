# 0007: Measure inference waiting with vLLM request traces, server metrics and stream timing

Date: October 27, 2026. Status: proposed for M2 review. Extends
[ADR 0003](0003-gpu-and-inference-measurement.md), which stays in force for NVML and Ollama.

ADR 0003 left the inference queue unobserved: Ollama reports no wait, so every model call's
queue is `unavailable`, and the model server had no sample stream, so model calls joined no
telemetry of their own. Rule attribution for inference (M2) needs a measured wait. On the GPU
laptop, vLLM v0.32.0 and dcgm-exporter 4.5.2 both run under WSL2 (measured October 27, 2026;
[the M2 record](../gpu-measurement-access.md#m2-vllm-request-traces-server-metrics-and-dcgm-october-27-2026)).

**Per-request wait: vLLM's trace span, joined by identity.** vLLM records an `llm_request`
span for each request with its engine-measured queue, prefill, decode, inference, first-token
and end-to-end time and its token counts. It parents that span on the request's W3C
`traceparent`. The runner sends each model call with its own step's trace and span ID, so a span
joins exactly one step by identity, never by time. The OpenTelemetry Collector (Compose
`otel-collector`, not published on the host) appends spans as OTLP JSON lines to
`.data/otel/vllm-traces.jsonl`; vLLM's exporter speaks only protobuf, and the collector keeps a
decoder out of the runner. After the run closes, the runner waits at most 10 s for every call's
span and records `vllm-request-trace-v1` per call. A missing span is `unavailable`, and two
spans for one step or a span naming another completion ID is `malformed`. The durations are
vLLM's clocks, kept apart from runner ticks. Live StepEvents cannot carry them, so, like
Ollama's breakdown, they are local evidence (`agent-evidence.json`) keyed by step and span ID.
Moving them into features is B02/M2 work.

**Server state around a step: vLLM's /metrics as samples.** The model server becomes a sampled
`inference` resource (service `vllm`). Sixteen metrics are registered as new versioned names:
running and waiting requests and KV-cache use (gauges), and cumulative sums and counts of
finished requests' queue, prefill, decode, first-token and end-to-end time, preemptions and
tokens (counters). They are server-wide: vLLM observes a request when it finishes, so a
window's increase covers every request finished in it, including competing ones, and a step's
own request usually finishes just after its window ends. They describe contention around a
step; they are not its wait. Every series of every round is also kept, as changes, in
`vllm-metrics-changes.jsonl` (bounded at 64 MB) for analysis beyond the registered metrics.

**Runner-observed timing: streamed calls.** Model calls stream (`model_transport: stream`), and
the runner ticks the request, the reply's headers, every chunk and every chunk with output, in
the run's own clock domain (`model-stream-timing-v1`). Time to first output therefore joins
steps and samples without crossing clocks, for Ollama as well as vLLM. It includes the server's
queue, prefill and the local network; it is an observation, not a server phase.

**DCGM where available.** An optional dcgm-exporter feeds eight `gpu.dcgm.*.v1` metrics on a
separate resource (service `nvidia-dcgm`), so its readings never merge with NVML's. On the
laptop the device fields work (utilization, memory-copy utilization, an energy counter,
temperature), while the profiling fields (SM active and occupancy, tensor and DRAM activity)
read N/A: DCGM lists them, but `NvmlGpmSampleGet` fails (NVML status 19, "in use by another
client", and 17, "operating system", with DCGM as the only client). They are registered
so a supporting host (an L4) measures them, and are `unavailable` here, never zero.

**Isolation of the measurement.** HTTP sources are read on a second sampling thread with a
200 ms fetch timeout, so a slow scrape cannot delay the CPU and NVML readings; a failed fetch
makes that round's readings `unavailable`. A scrape also costs the model server CPU, so the
sampling-overhead protocol gains a v2 that runs these probes (`resource-sampling-overhead-v2`).

Consequences: `agent-runner-v3`. The run configuration records `model_backend` and
`model_transport`, so a vLLM run (BF16 weights, a different server) never shares a workload
digest with an Ollama run (Q4_K_M), and new Ollama runs get new digests; recorded history is
unchanged. A vLLM run sends 26 metrics per round, or 34 with DCGM, so the longest run at
250 ms falls to about 769 s (588 s with DCGM). vLLM and Ollama cannot share the laptop's 8 GB:
the Compose profiles are alternatives. Prefix caching stays at vLLM's default (on), so a repeated
prompt can skip prefill; replays record it rather than disabling it. The inference injector
gains `competing-openai-chat-v1` for vLLM, with the same bounds. Not established: per-process
GPU use, DCGM profiling on any host, vLLM on a non-WSL host, and the effect of the collector on
vLLM's request timing beyond the measured sampling overhead.

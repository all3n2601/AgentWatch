"""Probes that read a service's Prometheus metrics: vLLM's queue and timing, DCGM's device fields.

Each round fetches the exposition once, within a timeout shorter than the sampling interval,
and maps registered metrics to its series. Readings keep their availability: a failed fetch
makes every metric `unavailable`, a series the service did not export is `unavailable` with
its name, and a non-finite or out-of-range value is `malformed`. Nothing is filled with zero.

vLLM's counters are server-wide sums over every request that finished (its histograms observe
a request when it ends), so their increase across a window covers all requests completed in
it, not one step's own request. Per-request queue time comes from vLLM's request traces
(`vllm_traces`); these probes describe the server around a step.
"""

import http.client
import json
import math
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import IO

from agentwatch_sdk.live import METRICS, ResourceRegistration

from agentwatch_agent.clock import MonotonicClock
from agentwatch_agent.prometheus_text import Exposition, ExpositionError, fetch_exposition
from agentwatch_agent.resources import MetricReading, Reading

# Below the 250 ms default interval, so one slow fetch delays at most the round it is in.
DEFAULT_FETCH_TIMEOUT_SECONDS = 0.2
FETCH_ERRORS = (OSError, http.client.HTTPException, ExpositionError)
# The raw change log keeps every series a run saw, for analysis beyond the registered metrics.
# It stops (and says so) at this size, so a long run cannot fill the disk.
MAX_CHANGE_LOG_BYTES = 64 * 1024 * 1024


@dataclass(frozen=True)
class SeriesMetric:
    """One registered metric read as the sum of the matching series of one exposition name."""

    metric: str
    unit: str
    series: str
    # Cumulative: a window's increase is the measurement, a decrease a server restart (reset).
    counter: bool = False


VLLM_METRICS = (
    # Gauges: requests the engine is running and holding in its queue, and KV-cache use (0-1).
    SeriesMetric("inference.vllm.requests_running.v1", "requests", "vllm:num_requests_running"),
    SeriesMetric("inference.vllm.requests_waiting.v1", "requests", "vllm:num_requests_waiting"),
    SeriesMetric("inference.vllm.kv_cache_usage.v1", "fraction", "vllm:kv_cache_usage_perc"),
    # Cumulative sums and counts of finished requests' server-measured phases.
    SeriesMetric(
        "inference.vllm.request_queue.seconds_total.v1",
        "seconds",
        "vllm:request_queue_time_seconds_sum",
        counter=True,
    ),
    SeriesMetric(
        "inference.vllm.request_queue.requests_total.v1",
        "requests",
        "vllm:request_queue_time_seconds_count",
        counter=True,
    ),
    SeriesMetric(
        "inference.vllm.request_prefill.seconds_total.v1",
        "seconds",
        "vllm:request_prefill_time_seconds_sum",
        counter=True,
    ),
    SeriesMetric(
        "inference.vllm.request_prefill.requests_total.v1",
        "requests",
        "vllm:request_prefill_time_seconds_count",
        counter=True,
    ),
    SeriesMetric(
        "inference.vllm.request_decode.seconds_total.v1",
        "seconds",
        "vllm:request_decode_time_seconds_sum",
        counter=True,
    ),
    SeriesMetric(
        "inference.vllm.request_decode.requests_total.v1",
        "requests",
        "vllm:request_decode_time_seconds_count",
        counter=True,
    ),
    SeriesMetric(
        "inference.vllm.first_token.seconds_total.v1",
        "seconds",
        "vllm:time_to_first_token_seconds_sum",
        counter=True,
    ),
    SeriesMetric(
        "inference.vllm.first_token.requests_total.v1",
        "requests",
        "vllm:time_to_first_token_seconds_count",
        counter=True,
    ),
    SeriesMetric(
        "inference.vllm.request_e2e.seconds_total.v1",
        "seconds",
        "vllm:e2e_request_latency_seconds_sum",
        counter=True,
    ),
    SeriesMetric(
        "inference.vllm.request_e2e.requests_total.v1",
        "requests",
        "vllm:e2e_request_latency_seconds_count",
        counter=True,
    ),
    SeriesMetric(
        "inference.vllm.preemptions_total.v1",
        "preemptions",
        "vllm:num_preemptions_total",
        counter=True,
    ),
    SeriesMetric(
        "inference.vllm.prompt_tokens_total.v1", "tokens", "vllm:prompt_tokens_total", counter=True
    ),
    SeriesMetric(
        "inference.vllm.generation_tokens_total.v1",
        "tokens",
        "vllm:generation_tokens_total",
        counter=True,
    ),
)
DCGM_METRICS = (
    SeriesMetric("gpu.dcgm.utilization.gpu.v1", "percent", "DCGM_FI_DEV_GPU_UTIL"),
    SeriesMetric("gpu.dcgm.utilization.mem_copy.v1", "percent", "DCGM_FI_DEV_MEM_COPY_UTIL"),
    SeriesMetric(
        "gpu.dcgm.energy.total_mj.v1",
        "millijoules",
        "DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION",
        counter=True,
    ),
    SeriesMetric("gpu.dcgm.temperature.gpu_c.v1", "celsius", "DCGM_FI_DEV_GPU_TEMP"),
    # Profiling (DCP) ratios, 0-1. Absent where the device cannot sample them (the laptop).
    SeriesMetric("gpu.dcgm.prof.sm_active.v1", "fraction", "DCGM_FI_PROF_SM_ACTIVE"),
    SeriesMetric("gpu.dcgm.prof.sm_occupancy.v1", "fraction", "DCGM_FI_PROF_SM_OCCUPANCY"),
    SeriesMetric("gpu.dcgm.prof.tensor_active.v1", "fraction", "DCGM_FI_PROF_PIPE_TENSOR_ACTIVE"),
    SeriesMetric("gpu.dcgm.prof.dram_active.v1", "fraction", "DCGM_FI_PROF_DRAM_ACTIVE"),
)
SERVICE_COUNTER_METRICS = frozenset(
    entry.metric for entry in VLLM_METRICS + DCGM_METRICS if entry.counter
)


def series_reading(
    exposition: Exposition, entry: SeriesMetric, labels: dict[str, str], source: str
) -> Reading:
    """Sum the series named by `entry` whose labels include `labels`; never invent a value."""
    matching = [
        series
        for series in exposition.named(entry.series)
        if all(series.label(name) == value for name, value in labels.items())
    ]
    if not matching:
        return None, "unavailable", f"{source} did not export {entry.series}"[:512]
    values = [series.value for series in matching]
    if not all(math.isfinite(value) for value in values):
        return None, "malformed", f"{source} reported a non-finite {entry.series}"[:512]
    total = sum(values)
    _resource_class, _unit, minimum, maximum = METRICS[entry.metric]
    if total < minimum or (maximum is not None and total > maximum):
        return None, "malformed", f"{source} {entry.series} is outside its registered range"[:512]
    return total, "measured", None


class ExpositionChangeLog:
    """Append each round's changed series to a JSONL file: the full first round, then deltas.

    Local evidence only, never ingested: it keeps histogram buckets and unregistered series so
    later analysis is not limited to the registered metrics. Ticks are the run's clock domain.
    """

    def __init__(self, path: Path, clock: MonotonicClock):
        self.path = path
        self.clock = clock
        self._stream: IO[str] | None = path.open("a", encoding="utf-8")
        self._last: dict[str, float] = {}
        self.written_bytes = 0
        self.rounds = 0
        self.truncated = False

    def record(self, exposition: Exposition | None, fetch_seconds: float, error: str | None):
        if self._stream is None:
            return
        current = {
            json.dumps([s.name, list(s.labels)], separators=(",", ":")): s.value
            for s in (exposition.series if exposition else ())
            # JSON has no NaN or infinity; the reading itself already reports them malformed.
            if math.isfinite(s.value)
        }
        line = json.dumps(
            {
                "tick_ns": self.clock.tick(),
                "fetch_seconds": fetch_seconds,
                "error": error,
                "changed": {k: v for k, v in current.items() if self._last.get(k) != v},
                "removed": sorted(set(self._last) - set(current)) if exposition else [],
                "unparsed_lines": exposition.unparsed_lines if exposition else None,
            },
            allow_nan=False,
        )
        if self.written_bytes + len(line) + 1 > MAX_CHANGE_LOG_BYTES:
            self.truncated = True
            self.close()
            return
        self._stream.write(line + "\n")
        self.written_bytes += len(line) + 1
        self.rounds += 1
        if exposition is not None:
            self._last = current

    def describe(self) -> dict:
        return {
            "file": self.path.name,
            "rounds": self.rounds,
            "bytes": self.written_bytes,
            "truncated": self.truncated,
        }

    def close(self) -> None:
        if self._stream is not None:
            self._stream.close()
            self._stream = None


class ExpositionProbe:
    """One registered resource read from one service's /metrics endpoint each round."""

    def __init__(
        self,
        resource: ResourceRegistration,
        *,
        url: str,
        source: str,
        entries: tuple[SeriesMetric, ...],
        labels: dict[str, str],
        scope: str,
        fetch_timeout_seconds: float = DEFAULT_FETCH_TIMEOUT_SECONDS,
        change_log: ExpositionChangeLog | None = None,
    ):
        self.resource = resource
        self.url = url
        self.source = source
        self.entries = entries
        self.labels = labels
        self.scope = scope
        self.metrics = tuple((entry.metric, entry.unit) for entry in entries)
        self.fetch_timeout_seconds = fetch_timeout_seconds
        self.change_log = change_log
        self._fetch_seconds: list[float] = []
        self._failures: dict[str, int] = {}

    def read(self) -> list[MetricReading]:
        started = time.monotonic()
        exposition, error = None, None
        try:
            exposition = fetch_exposition(self.url, self.fetch_timeout_seconds)
        except FETCH_ERRORS as failure:
            error = f"{type(failure).__name__}"
            self._failures[error] = self._failures.get(error, 0) + 1
        fetch_seconds = time.monotonic() - started
        self._fetch_seconds.append(fetch_seconds)
        if self.change_log is not None:
            self.change_log.record(exposition, fetch_seconds, error)
        if exposition is None:
            reason = f"{self.source} metrics request failed ({error})"
            return [((e.metric, e.unit), (None, "unavailable", reason)) for e in self.entries]
        return [
            ((e.metric, e.unit), series_reading(exposition, e, self.labels, self.source))
            for e in self.entries
        ]

    def describe(self) -> dict:
        fetches = self._fetch_seconds
        return {
            "source": self.source,
            "series_labels": dict(self.labels),
            "fetch_timeout_seconds": self.fetch_timeout_seconds,
            "fetches": len(fetches),
            "fetch_failures": dict(self._failures),
            "fetch_median_seconds": statistics.median(fetches) if fetches else None,
            "fetch_max_seconds": max(fetches) if fetches else None,
            "change_log": self.change_log.describe() if self.change_log else None,
        }

    def close(self) -> None:
        if self.change_log is not None:
            self.change_log.close()


def vllm_probe(
    resource: ResourceRegistration,
    base_url: str,
    served_model: str,
    change_log: ExpositionChangeLog | None = None,
) -> ExpositionProbe:
    return ExpositionProbe(
        resource,
        url=base_url.rstrip("/") + "/metrics",
        source="vLLM",
        entries=VLLM_METRICS,
        labels={"model_name": served_model},
        scope=(
            "the whole vLLM server for the run's served model: every request it handled, "
            "including competing ones; not one step's own request"
        ),
        change_log=change_log,
    )


def dcgm_probe(
    resource: ResourceRegistration, metrics_url: str, gpu_index: int = 0
) -> ExpositionProbe:
    return ExpositionProbe(
        resource,
        url=metrics_url,
        source="dcgm-exporter",
        entries=DCGM_METRICS,
        labels={"gpu": str(gpu_index)},
        scope="one whole NVIDIA device through DCGM, every process on it; not per process",
    )

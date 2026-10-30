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
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import IO

from agentwatch_sdk.live import METRICS, ResourceRegistration

from agentwatch_agent.clock import MonotonicClock
from agentwatch_agent.prometheus_text import (
    Exposition,
    ExpositionError,
    fetch_exposition,
    read_exposition_file,
)
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
    """Sum the series named by `entry` whose labels include `labels`; never invent a value.

    If any line of that name failed to parse, the reading is malformed: its labels are
    unknown, so a sum of the remaining series could silently undercount.
    """
    if entry.series in exposition.unparsed_names:
        return None, "malformed", f"{source} sent an unparsable {entry.series} line"[:512]
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
    """One registered resource read from one exposition (an endpoint or a file) each round."""

    def __init__(
        self,
        resource: ResourceRegistration,
        *,
        fetch: Callable[[], Exposition],
        source: str,
        entries: tuple[SeriesMetric, ...],
        labels: dict[str, str],
        scope: str,
        fetch_details: dict | None = None,
        change_log: ExpositionChangeLog | None = None,
    ):
        self.resource = resource
        self.fetch = fetch
        self.fetch_details = fetch_details or {}
        self.source = source
        self.entries = entries
        self.labels = labels
        self.scope = scope
        self.metrics = tuple((entry.metric, entry.unit) for entry in entries)
        self.change_log = change_log
        self._fetch_seconds: list[float] = []
        self._failures: dict[str, int] = {}

    def read(self) -> list[MetricReading]:
        exposition, error = self._fetch_logged()
        if exposition is None:
            reason = f"{self.source} metrics request failed ({error})"
            return self._all(None, "unavailable", reason)
        return self._readings(exposition)

    def _all(self, value: float | None, availability: str, reason: str | None):
        return [((e.metric, e.unit), (value, availability, reason)) for e in self.entries]

    def _readings(self, exposition: Exposition) -> list[MetricReading]:
        return [
            ((e.metric, e.unit), series_reading(exposition, e, self.labels, self.source))
            for e in self.entries
        ]

    def _fetch_logged(self) -> tuple[Exposition | None, str | None]:
        started = time.monotonic()
        exposition, error = None, None
        try:
            exposition = self.fetch()
        except FETCH_ERRORS as failure:
            error = f"{type(failure).__name__}"
            self._failures[error] = self._failures.get(error, 0) + 1
        fetch_seconds = time.monotonic() - started
        self._fetch_seconds.append(fetch_seconds)
        if self.change_log is not None:
            self.change_log.record(exposition, fetch_seconds, error)
        return exposition, error

    def describe(self) -> dict:
        fetches = self._fetch_seconds
        return {
            "source": self.source,
            "series_labels": dict(self.labels),
            **self.fetch_details,
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
    url = base_url.rstrip("/") + "/metrics"
    return ExpositionProbe(
        resource,
        fetch=lambda: fetch_exposition(url, DEFAULT_FETCH_TIMEOUT_SECONDS),
        fetch_details={"fetch_timeout_seconds": DEFAULT_FETCH_TIMEOUT_SECONDS},
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
        fetch=lambda: fetch_exposition(metrics_url, DEFAULT_FETCH_TIMEOUT_SECONDS),
        fetch_details={"fetch_timeout_seconds": DEFAULT_FETCH_TIMEOUT_SECONDS},
        source="dcgm-exporter",
        entries=DCGM_METRICS,
        labels={"gpu": str(gpu_index)},
        scope="one whole NVIDIA device through DCGM, every process on it; not per process",
    )


GPM_PREFIX = "agentwatch_gpm"
# (registered metric, unit, the exporter's metric name; agentwatch_worker.gpm.METRICS)
_GPM = (
    ("gpu.gpm.graphics_active.v1", "percent", "graphics_active"),
    ("gpu.gpm.sm_util.v1", "percent", "sm_util"),
    ("gpu.gpm.sm_occupancy.v1", "percent", "sm_occupancy"),
    ("gpu.gpm.tensor_active.v1", "percent", "tensor_active"),
    ("gpu.gpm.dram_bw_util.v1", "percent", "dram_bw_util"),
    ("gpu.gpm.fp32_util.v1", "percent", "fp32_util"),
    ("gpu.gpm.fp16_util.v1", "percent", "fp16_util"),
    ("gpu.gpm.pcie_tx.v1", "mebibytes_per_second", "pcie_tx"),
    ("gpu.gpm.pcie_rx.v1", "mebibytes_per_second", "pcie_rx"),
)
GPM_METRICS = tuple(
    SeriesMetric(metric, unit, f"{GPM_PREFIX}_{name}_{unit}") for metric, unit, name in _GPM
)
_GPM_NAMES = {metric: name for metric, _unit, name in _GPM}


class GpmFileProbe(ExpositionProbe):
    """GPM windows from the exporter's text file (`agentwatch_worker.gpm_exporter`, ADR 0008).

    A round uses a window only once: when the exporter has written no new window since the
    previous round (same `instance` and `sequence`), every reading is `unavailable`, never the
    old window at a new tick. The first window a run reads is only a baseline: a file left by a
    stopped exporter looks the same as a live one, so a window counts once the run has seen the
    file change, which needs no clock shared with Windows. A window covers the exporter's last
    interval on its own clock, ending before the runner's tick by up to that interval plus the
    file's latency; no clock mapping is applied.
    """

    def __init__(self, resource: ResourceRegistration, path: Path, gpu_index: int = 0):
        super().__init__(
            resource,
            fetch=lambda: read_exposition_file(path),
            fetch_details={"file": path.name},
            source="GPM exporter",
            entries=GPM_METRICS,
            labels={"gpu": str(gpu_index)},
            scope=(
                "one whole NVIDIA device through NVML GPM on the Windows host, every process on "
                "it, over the exporter's latest window; not per process"
            ),
        )
        self._label = {"gpu": str(gpu_index)}
        self._last_window: tuple[str, float] | None = None
        self.repeated_windows = 0
        self._windows: list[float] = []
        # Each exporter instance's own CPU readings, so a restart never yields a negative cost.
        self._exporter_cpu: dict[str, list[float]] = {}

    def _one(self, exposition: Exposition, name: str, labels: dict[str, str]) -> float | None:
        values = [
            s.value
            for s in exposition.named(f"{GPM_PREFIX}_{name}")
            if all(s.label(k) == v for k, v in labels.items())
        ]
        return values[0] if len(values) == 1 and math.isfinite(values[0]) else None

    def _instance(self, exposition: Exposition) -> str | None:
        instances = [
            s.label("instance")
            for s in exposition.named(f"{GPM_PREFIX}_exporter_info")
            if s.label("gpu") == self._label["gpu"]
        ]
        return instances[0] if len(instances) == 1 and instances[0] else None

    def read(self) -> list[MetricReading]:
        exposition, error = self._fetch_logged()
        if exposition is None:
            return self._all(None, "unavailable", f"GPM exporter file unreadable ({error})")
        instance = self._instance(exposition)
        sequence = self._one(exposition, "sequence", self._label)
        if instance is None or sequence is None:
            return self._all(
                None, "unavailable", "GPM exporter file has no exporter instance or window sequence"
            )
        window, previous = (instance, sequence), self._last_window
        self._last_window = window
        if previous is None:
            return self._all(
                None,
                "unavailable",
                "First GPM window of this run, kept as a baseline: the file may be left by a "
                "stopped exporter",
            )
        if window == previous:
            self.repeated_windows += 1
            return self._all(None, "unavailable", "No new GPM window since the previous round")
        window_seconds = self._one(exposition, "window_seconds", self._label)
        if window_seconds is not None:
            self._windows.append(window_seconds)
        cpu = self._one(exposition, "exporter_cpu_seconds_total", self._label)
        if cpu is not None:
            self._exporter_cpu.setdefault(instance, []).append(cpu)
        readings = self._readings(exposition)
        for index, ((metric, unit), (value, availability, reason)) in enumerate(readings):
            if availability == "unavailable":
                code = self._one(
                    exposition, "metric_status", self._label | {"metric": _GPM_NAMES[metric]}
                )
                if code is not None and code != 0:
                    reason = f"NVML refused GPM metric {_GPM_NAMES[metric]} (code {code:g})"
                    readings[index] = ((metric, unit), (None, "unavailable", reason))
        return readings

    def describe(self) -> dict:
        spans = [cpu[-1] - cpu[0] for cpu in self._exporter_cpu.values() if len(cpu) > 1]
        return super().describe() | {
            "windows_used": len(self._windows),
            "repeated_windows": self.repeated_windows,
            "exporter_instances": len(self._exporter_cpu),
            "window_median_seconds": statistics.median(self._windows) if self._windows else None,
            "window_max_seconds": max(self._windows) if self._windows else None,
            # The exporter's own process CPU between the first and last window this run used,
            # summed over exporter instances.
            "exporter_cpu_seconds": sum(spans) if spans else None,
        }

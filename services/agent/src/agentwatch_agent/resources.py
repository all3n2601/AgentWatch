"""Sample the run's resources as live ResourceSamples, on one background thread.

Each probe reads one registered resource. CPU readers are shared with the worker's existing
sampler and GPU readings come from NVML. Every reading is emitted with an explicit
availability: macOS has no procfs or NVML, so it records `unsupported` rather than zero, and a
missing cgroup, device or counter stays distinct from a measured value. PSI is host-wide, the
cgroup counter covers the runner's whole cgroup and NVML describes the whole device, so none
of them is step-exclusive.
"""

import statistics
import threading
import time
from collections.abc import Callable
from itertools import pairwise
from pathlib import Path
from typing import Protocol
from uuid import UUID, uuid4

from agentwatch_sdk.live import ResourceRegistration, ResourceSample
from agentwatch_worker.nvml import NvmlDevice, NvmlError, NvmlUnavailable
from agentwatch_worker.sampler import discover_cgroup, read_cpu_stat, read_pressure

from agentwatch_agent.clock import MonotonicClock
from agentwatch_agent.config import CONTRACT_VERSION

PSI_METRIC = ("cpu.psi.some.avg10.v1", "percent")
THROTTLED_METRIC = ("cpu.cgroup.throttled_usec.v1", "microseconds")
GPU_UTILIZATION_METRIC = ("gpu.nvml.utilization.gpu.v1", "percent")
GPU_MEMORY_UTILIZATION_METRIC = ("gpu.nvml.utilization.memory.v1", "percent")
GPU_MEMORY_USED_METRIC = ("gpu.nvml.memory.used_bytes.v1", "bytes")
GPU_POWER_METRIC = ("gpu.nvml.power.usage_watts.v1", "watts")
GPU_SM_CLOCK_METRIC = ("gpu.nvml.clock.sm_mhz.v1", "megahertz")
CPU_METRICS = (PSI_METRIC, THROTTLED_METRIC)
GPU_METRICS = (
    GPU_UTILIZATION_METRIC,
    GPU_MEMORY_UTILIZATION_METRIC,
    GPU_MEMORY_USED_METRIC,
    GPU_POWER_METRIC,
    GPU_SM_CLOCK_METRIC,
)
# A gap this many declared intervals long means at least one sampling round was missed.
LATE_ROUND_FACTOR = 1.5
Reading = tuple[float | None, str, str | None]
MetricReading = tuple[tuple[str, str], Reading]


class Probe(Protocol):
    resource: ResourceRegistration
    metrics: tuple[tuple[str, str], ...]
    scope: str

    def read(self) -> list[MetricReading]: ...
    def describe(self) -> dict: ...
    def close(self) -> None: ...


def _psi_reading(proc: Path) -> Reading:
    if not proc.is_dir():
        return None, "unsupported", "procfs is not available on this platform"
    try:
        value = read_pressure(proc / "pressure/cpu")["some"]["avg10"]
    except FileNotFoundError:
        return None, "unsupported", "CPU PSI is not exposed by this kernel"
    except OSError as error:
        return None, "unavailable", f"CPU PSI could not be read ({type(error).__name__})"
    except (ValueError, KeyError):
        return None, "malformed", "CPU PSI did not contain a numeric some avg10 field"
    if not 0 <= value <= 100:
        return None, "malformed", "CPU PSI some avg10 was outside 0-100 percent"
    return value, "measured", None


def _throttled_reading(proc: Path, stat_path: Path | None) -> Reading:
    if not proc.is_dir():
        return None, "unsupported", "procfs is not available on this platform"
    if stat_path is None:
        return None, "unavailable", "cgroup v2 membership was not found"
    try:
        stat = read_cpu_stat(stat_path)
    except FileNotFoundError:
        return None, "unavailable", "cgroup cpu.stat does not exist (CPU controller disabled)"
    except OSError as error:
        return None, "unavailable", f"cgroup cpu.stat could not be read ({type(error).__name__})"
    except ValueError:
        return None, "malformed", "cgroup cpu.stat was not key/integer lines"
    if "throttled_usec" not in stat:
        return None, "unavailable", "cgroup cpu.stat has no throttled_usec counter"
    if stat["throttled_usec"] < 0:
        return None, "malformed", "cgroup throttled_usec was negative"
    return float(stat["throttled_usec"]), "measured", None


class CpuProbe:
    metrics = CPU_METRICS
    scope = "host CPU PSI; cgroup v2 counter of the runner's own cgroup"

    def __init__(
        self,
        resource: ResourceRegistration,
        proc: Path = Path("/proc"),
        cgroup_root: Path = Path("/sys/fs/cgroup"),
    ):
        self.resource = resource
        self.proc = proc
        self.stat_path = discover_cgroup(proc, cgroup_root)

    def read(self) -> list[MetricReading]:
        return [
            (PSI_METRIC, _psi_reading(self.proc)),
            (THROTTLED_METRIC, _throttled_reading(self.proc, self.stat_path)),
        ]

    def describe(self) -> dict:
        return {"cgroup_found": self.stat_path is not None}

    def close(self) -> None:
        pass


def _nvml_failure(error: NvmlError) -> Reading:
    return None, "unsupported" if error.not_supported else "unavailable", str(error)


def _percent(value: int, name: str) -> Reading:
    if not 0 <= value <= 100:
        return None, "malformed", f"NVML {name} was outside 0-100 percent"
    return float(value), "measured", None


class GpuProbe:
    """Whole-device NVML gauges for one GPU index; the device is opened once, on creation.

    If NVML cannot be opened, every round still emits each metric with the opening failure
    as its reason, so an absent GPU never looks like an idle one.
    """

    metrics = GPU_METRICS
    scope = "one whole NVIDIA device through NVML, every process on it; not per process"

    def __init__(
        self,
        resource: ResourceRegistration,
        open_device: Callable[[], NvmlDevice] = NvmlDevice.open,
    ):
        self.resource = resource
        self.device: NvmlDevice | None = None
        self.unavailable: Reading | None = None
        try:
            self.device = open_device()
        except NvmlUnavailable as error:
            availability = "unsupported" if error.unsupported else "unavailable"
            self.unavailable = (None, availability, str(error))

    def read(self) -> list[MetricReading]:
        if self.device is None:
            return [(metric, self.unavailable) for metric in self.metrics]
        readings: list[MetricReading] = []
        try:
            gpu, memory = self.device.utilization_percent()
            utilization = (_percent(gpu, "GPU utilization"), _percent(memory, "memory utilization"))
        except NvmlError as error:
            utilization = (_nvml_failure(error),) * 2
        readings += zip((GPU_UTILIZATION_METRIC, GPU_MEMORY_UTILIZATION_METRIC), utilization)
        try:
            used, total = self.device.memory_bytes()
            memory_reading: Reading = (
                (float(used), "measured", None)
                if used <= total
                else (None, "malformed", "NVML memory used exceeded total memory")
            )
        except NvmlError as error:
            memory_reading = _nvml_failure(error)
        readings.append((GPU_MEMORY_USED_METRIC, memory_reading))
        for metric, read in (
            (GPU_POWER_METRIC, self.device.power_watts),
            (GPU_SM_CLOCK_METRIC, self.device.sm_clock_mhz),
        ):
            try:
                readings.append((metric, (float(read()), "measured", None)))
            except NvmlError as error:
                readings.append((metric, _nvml_failure(error)))
        return readings

    def describe(self) -> dict:
        if self.device is None:
            _, availability, reason = self.unavailable
            return {"device": None, "availability": availability, "reason": reason}
        identity = self.device.identity
        return {
            "device": {
                "index": identity.index,
                "device_count": identity.device_count,
                "name": identity.name,
                "driver_version": identity.driver_version,
                "memory_total_bytes": identity.memory_total_bytes,
            },
            "availability": "measured",
            "reason": None,
        }

    def close(self) -> None:
        if self.device is not None:
            self.device.close()


def interval_summary(ticks_ns: list[int], declared_seconds: float) -> dict:
    """Achieved gaps between consecutive sampling rounds, against the declared interval."""
    gaps = [(b - a) / 1e9 for a, b in pairwise(ticks_ns)]
    late = [gap for gap in gaps if gap > LATE_ROUND_FACTOR * declared_seconds]
    ordered = sorted(gaps)
    return {
        "sample_rounds": len(ticks_ns),
        "observed_interval_median_seconds": statistics.median(gaps) if gaps else None,
        "observed_interval_p90_seconds": (
            ordered[min(len(ordered) - 1, int(0.9 * len(ordered)))] if ordered else None
        ),
        "observed_interval_max_seconds": max(gaps) if gaps else None,
        "late_rounds": len(late),
        # Rounds that would have fallen inside late gaps at the declared cadence.
        "missed_rounds_estimate": sum(max(round(gap / declared_seconds) - 1, 0) for gap in late),
    }


class ResourceSampler:
    """One sampling thread for every probe, at a fixed cadence, measuring its own cost."""

    def __init__(
        self,
        emit: Callable[[ResourceSample], None],
        *,
        run_id: UUID,
        task_id: str,
        clock: MonotonicClock,
        probes: list[Probe],
        interval_seconds: float,
    ):
        self.emit = emit
        self.run_id = run_id
        self.task_id = task_id
        self.clock = clock
        self.probes = probes
        self.interval_seconds = interval_seconds
        self._round_ticks: list[int] = []
        self._round_wall_ns: list[int] = []
        self._round_thread_cpu_ns: list[int] = []
        self._availability: dict[str, dict[str, int]] = {}
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="resource-sampler", daemon=True)
        self.error: BaseException | None = None

    def sample_once(self) -> None:
        wall_started, cpu_started = time.monotonic_ns(), time.thread_time_ns()
        round_tick = None
        for probe in self.probes:
            # Each probe gets its own tick, taken just before it reads.
            tick = self.clock.tick()
            observed_at = self.clock.now()
            round_tick = tick if round_tick is None else round_tick
            for (metric, unit), (value, availability, reason) in probe.read():
                counts = self._availability.setdefault(metric, {})
                counts[availability] = counts.get(availability, 0) + 1
                self.emit(
                    ResourceSample(
                        contract_version=CONTRACT_VERSION,
                        record_type="resource_sample",
                        run_id=self.run_id,
                        task_id=self.task_id,
                        sample_id=uuid4(),
                        resource_id=probe.resource.resource_id,
                        service_id=probe.resource.service_id,
                        resource_class=probe.resource.resource_class,
                        observed_at=observed_at,
                        clock_id=self.clock.clock_id,
                        tick_ns=tick,
                        metric=metric,
                        unit=unit,
                        value=value,
                        availability=availability,
                        reason=reason,
                    )
                )
        self._round_ticks.append(round_tick)
        # The round's cost includes reading, record validation and the journal write in emit.
        self._round_wall_ns.append(time.monotonic_ns() - wall_started)
        self._round_thread_cpu_ns.append(time.thread_time_ns() - cpu_started)

    def _run(self) -> None:
        next_due = time.monotonic()
        while not self._stop.is_set():
            try:
                self.sample_once()
            # Re-raised by the runner, so sampling never stops silently.
            except Exception as error:  # noqa: BLE001
                self.error = error
                return
            # Schedule against a fixed cadence so slow readings do not accumulate drift.
            next_due += self.interval_seconds
            self._stop.wait(max(next_due - time.monotonic(), 0))

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *_):
        self._stop.set()
        self._thread.join(timeout=10)
        # Only after the thread stopped: a probe must not close under an in-flight read.
        if not self._thread.is_alive():
            for probe in self.probes:
                probe.close()

    def summary(self) -> dict:
        """Declared versus achieved sampling and the sampler's own cost, for evidence."""
        ticks = self._round_ticks
        walls = [ns / 1e9 for ns in self._round_wall_ns]
        cpu = [ns / 1e9 for ns in self._round_thread_cpu_ns]
        span = (ticks[-1] - ticks[0]) / 1e9 if len(ticks) > 1 else None
        return {
            "declared_interval_seconds": self.interval_seconds,
            **interval_summary(ticks, self.interval_seconds),
            "availability_counts": self._availability,
            "resources": [
                {
                    "resource_id": probe.resource.resource_id,
                    "resource_class": probe.resource.resource_class,
                    "metrics": [metric for metric, _ in probe.metrics],
                    "scope": probe.scope,
                }
                | probe.describe()
                for probe in self.probes
            ],
            "round_cost": {
                "method": "sampler-thread monotonic wall and thread CPU per round",
                "wall_median_seconds": statistics.median(walls) if walls else None,
                "wall_max_seconds": max(walls) if walls else None,
                "thread_cpu_median_seconds": statistics.median(cpu) if cpu else None,
                "thread_cpu_total_seconds": sum(cpu),
                # Share of one core the sampler used over the sampled span.
                "thread_cpu_fraction_of_span": sum(cpu[:-1]) / span if span else None,
            },
        }

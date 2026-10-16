"""Sample the tool host's CPU signals as live ResourceSamples.

Readers are shared with the worker's existing sampler. Every reading is emitted with an
explicit availability: macOS has no procfs, so it records `unsupported` rather than zero, and
a missing cgroup or unreadable counter stays distinct from a measured value. PSI is host-wide
and the cgroup counter covers the runner's whole cgroup, so neither is step-exclusive.
"""

import statistics
import threading
import time
from collections.abc import Callable
from pathlib import Path
from uuid import UUID, uuid4

from agentwatch_sdk.live import ResourceRegistration, ResourceSample
from agentwatch_worker.sampler import discover_cgroup, read_cpu_stat, read_pressure

from agentwatch_agent.clock import MonotonicClock
from agentwatch_agent.config import CONTRACT_VERSION

PSI_METRIC = ("cpu.psi.some.avg10.v1", "percent")
THROTTLED_METRIC = ("cpu.cgroup.throttled_usec.v1", "microseconds")
Reading = tuple[float | None, str, str | None]


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


class CpuResourceSampler:
    def __init__(
        self,
        emit: Callable[[ResourceSample], None],
        *,
        run_id: UUID,
        task_id: str,
        clock: MonotonicClock,
        resource: ResourceRegistration,
        interval_seconds: float,
        proc: Path = Path("/proc"),
        cgroup_root: Path = Path("/sys/fs/cgroup"),
    ):
        self.emit = emit
        self.run_id = run_id
        self.task_id = task_id
        self.clock = clock
        self.resource = resource
        self.interval_seconds = interval_seconds
        self.proc = proc
        self.stat_path = discover_cgroup(proc, cgroup_root)
        self._tick_times: list[int] = []
        self._availability: dict[str, dict[str, int]] = {}
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="cpu-sampler", daemon=True)
        self.error: BaseException | None = None

    def sample_once(self) -> None:
        tick = self.clock.tick()
        observed_at = self.clock.now()
        self._tick_times.append(tick)
        readings = (
            (PSI_METRIC, _psi_reading(self.proc)),
            (THROTTLED_METRIC, _throttled_reading(self.proc, self.stat_path)),
        )
        for (metric, unit), (value, availability, reason) in readings:
            counts = self._availability.setdefault(metric, {})
            counts[availability] = counts.get(availability, 0) + 1
            self.emit(
                ResourceSample(
                    contract_version=CONTRACT_VERSION,
                    record_type="resource_sample",
                    run_id=self.run_id,
                    task_id=self.task_id,
                    sample_id=uuid4(),
                    resource_id=self.resource.resource_id,
                    service_id=self.resource.service_id,
                    resource_class=self.resource.resource_class,
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

    def summary(self) -> dict:
        """Declared versus achieved sampling, for overhead and coverage reporting."""
        gaps = [(b - a) / 1e9 for a, b in zip(self._tick_times, self._tick_times[1:])]
        return {
            "resource_id": self.resource.resource_id,
            "declared_interval_seconds": self.interval_seconds,
            "sample_rounds": len(self._tick_times),
            "observed_interval_median_seconds": statistics.median(gaps) if gaps else None,
            "observed_interval_max_seconds": max(gaps) if gaps else None,
            "availability_counts": self._availability,
            "scope": "host CPU PSI; cgroup v2 counter of the runner's own cgroup",
        }

"""Read-only Linux CPU telemetry, sampled alongside local experiments."""

import threading
import time
from pathlib import Path


def read_pressure(path: Path) -> dict:
    result = {}
    for line in path.read_text().splitlines():
        kind, *fields = line.split()
        result[kind] = {
            key: float(value) for key, value in (field.split("=", 1) for field in fields)
        }
    return result


def read_cpu_stat(path: Path) -> dict:
    return {
        key: int(value) for key, value in (line.split() for line in path.read_text().splitlines())
    }


def discover_cgroup(proc: Path, root: Path) -> Path | None:
    try:
        for line in (proc / "self/cgroup").read_text().splitlines():
            if line.startswith("0::"):
                group = (root / line[3:].lstrip("/")).resolve()
                if group.is_relative_to(root.resolve()):
                    return group / "cpu.stat"
    except OSError:
        pass
    return None


class CpuSampler:
    def __init__(
        self,
        proc: Path = Path("/proc"),
        root: Path = Path("/sys/fs/cgroup"),
        interval: float = 0.25,
    ):
        self.pressure_path = proc / "pressure/cpu"
        self.stat_path = discover_cgroup(proc, root)
        self.interval = interval
        self.samples: list[dict] = []
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def sample(self) -> dict:
        row = {
            "timestamp_ns": time.time_ns(),
            "cpu_pressure": None,
            "cgroup_cpu_stat": None,
            "unavailable": [],
        }
        for name, path, reader in [
            ("cpu_pressure", self.pressure_path, read_pressure),
            ("cgroup_cpu_stat", self.stat_path, read_cpu_stat),
        ]:
            try:
                if path is None:
                    raise FileNotFoundError("cgroup v2 membership not found")
                row[name] = reader(path)
            except (OSError, ValueError) as exc:
                row["unavailable"].append(f"{name}: {exc}")
        return row

    def _run(self):
        while not self.stop.is_set():
            self.samples.append(self.sample())
            self.stop.wait(self.interval)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join(timeout=2)
        self.samples.append(self.sample())

    def summarize(self, submitted_ns: int, finished_ns: int) -> dict:
        rows = [row for row in self.samples if submitted_ns <= row["timestamp_ns"] <= finished_ns]
        pressures = [
            row["cpu_pressure"]["some"]["avg10"]
            for row in rows
            if row["cpu_pressure"] and "avg10" in row["cpu_pressure"].get("some", {})
        ]
        stats = [row["cgroup_cpu_stat"] for row in rows if row["cgroup_cpu_stat"]]
        throttled = None
        if len(stats) >= 2 and all("throttled_usec" in stat for stat in (stats[0], stats[-1])):
            delta = stats[-1]["throttled_usec"] - stats[0]["throttled_usec"]
            throttled = delta if delta >= 0 else None
        return {
            "sample_count": len(rows),
            "valid_pressure_samples": len(pressures),
            "max_cpu_pressure_avg10_percent": max(pressures) if pressures else None,
            "valid_cgroup_samples": len(stats),
            "observed_cgroup_throttled_usec_delta": throttled,
            "interval_seconds": self.interval,
            "scope": "host CPU pressure; profiler cgroup v2 counters",
            "status": "available" if pressures else "unavailable_or_no_samples",
        }

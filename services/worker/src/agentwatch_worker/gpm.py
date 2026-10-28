"""NVML GPU Performance Monitoring (GPM): SM, tensor, DRAM and PCIe activity between two samples.

GPM is the source DCGM's profiling fields use. On the GPU laptop the device supports it and the
Windows driver serves samples, but WSL2 refuses them (`nvmlGpmSampleGet` returns "operating
system" and then "in use by another client"), so this binding runs as a Windows process
(`gpm_exporter`) whose output the WSL runner reads (ADR 0008).

A metric is the activity of the whole device between two samples, every process on it, never
one process. Metric IDs are NVML's `nvmlGpmMetricId_t` values; the driver names each one, and
those names were checked on the laptop's driver (617.42) on October 28, 2026.
"""

import ctypes
from dataclasses import dataclass

from agentwatch_worker.nvml import NvmlDevice, NvmlError, NvmlUnavailable

GPM_FUNCTIONS = (
    "nvmlGpmQueryDeviceSupport",
    "nvmlGpmSampleAlloc",
    "nvmlGpmSampleFree",
    "nvmlGpmSampleGet",
    "nvmlGpmMetricsGet",
)
GPM_SUPPORT_VERSION = 1  # NVML_GPM_SUPPORT_VERSION
GPM_METRICS_GET_VERSION = 1  # NVML_GPM_METRICS_GET_VERSION
# The request struct holds NVML_GPM_METRIC_MAX metric slots, which grows across NVML
# releases; a larger array is safe because the driver reads only `numMetrics` of them.
METRIC_SLOTS = 210


@dataclass(frozen=True)
class GpmMetric:
    name: str
    metric_id: int
    unit: str
    # The driver's short name for the ID, checked when a metrics request returns.
    driver_name: str


METRICS = (
    GpmMetric("graphics_active", 1, "percent", "gract"),
    GpmMetric("sm_util", 2, "percent", "smutil"),
    GpmMetric("sm_occupancy", 3, "percent", "smocc"),
    GpmMetric("tensor_active", 5, "percent", "mmaact"),
    GpmMetric("dram_bw_util", 10, "percent", "dram"),
    GpmMetric("fp32_util", 12, "percent", "fp32"),
    GpmMetric("fp16_util", 13, "percent", "fp16"),
    GpmMetric("pcie_tx", 20, "mebibytes_per_second", "pcitx"),
    GpmMetric("pcie_rx", 21, "mebibytes_per_second", "pcirx"),
)


class _Support(ctypes.Structure):
    _fields_ = [("version", ctypes.c_uint), ("isSupportedDevice", ctypes.c_uint)]


class _MetricInfo(ctypes.Structure):
    _fields_ = [
        ("shortName", ctypes.c_char_p),
        ("longName", ctypes.c_char_p),
        ("unit", ctypes.c_char_p),
    ]


class _Metric(ctypes.Structure):
    _fields_ = [
        ("metricId", ctypes.c_uint),
        ("nvmlReturn", ctypes.c_int),
        ("value", ctypes.c_double),
        ("metricInfo", _MetricInfo),
    ]


class _MetricsGet(ctypes.Structure):
    _fields_ = [
        ("version", ctypes.c_uint),
        ("numMetrics", ctypes.c_uint),
        ("sample1", ctypes.c_void_p),
        ("sample2", ctypes.c_void_p),
        ("metrics", _Metric * METRIC_SLOTS),
    ]


@dataclass(frozen=True)
class MetricResult:
    """One metric over one window: a value, or the NVML code that refused it."""

    value: float | None
    code: int


class GpmSampler:
    """Two alternating GPM samples on one open device; each `window()` is the latest interval.

    The first call only takes a sample: a window needs two. Close it exactly once.
    """

    def __init__(self, device: NvmlDevice, metrics: tuple[GpmMetric, ...] = METRICS):
        library = device.library
        missing = [name for name in GPM_FUNCTIONS if not hasattr(library, name)]
        if missing:
            raise NvmlUnavailable(f"NVML library lacks {', '.join(missing)}")
        support = _Support(version=GPM_SUPPORT_VERSION, isSupportedDevice=0)
        code = library.nvmlGpmQueryDeviceSupport(device.handle, ctypes.byref(support))
        if code != 0:
            raise NvmlUnavailable(f"GPM support query failed: {NvmlError('GPM support', code)}")
        if not support.isSupportedDevice:
            raise NvmlUnavailable("This device does not support GPM", unsupported=True)
        self._device = device
        self._library = library
        self.metrics = metrics
        self._samples = [ctypes.c_void_p(), ctypes.c_void_p()]
        self._allocated: list[ctypes.c_void_p] = []
        try:
            for sample in self._samples:
                self._call("nvmlGpmSampleAlloc", library.nvmlGpmSampleAlloc(ctypes.byref(sample)))
                self._allocated.append(sample)
        except BaseException:
            self.close()
            raise
        self._current = 0
        self._have_previous = False

    @staticmethod
    def _call(name: str, code: int) -> None:
        if code != 0:
            raise NvmlError(name, code)

    def window(self) -> dict[str, MetricResult] | None:
        """Take a sample; return each metric since the previous one, or None on the first."""
        sample = self._samples[self._current]
        self._call("nvmlGpmSampleGet", self._library.nvmlGpmSampleGet(self._device.handle, sample))
        previous = self._samples[1 - self._current]
        self._current = 1 - self._current
        if not self._have_previous:
            self._have_previous = True
            return None
        request = _MetricsGet(
            version=GPM_METRICS_GET_VERSION,
            numMetrics=len(self.metrics),
            sample1=previous,
            sample2=sample,
        )
        for slot, metric in zip(request.metrics, self.metrics, strict=False):
            slot.metricId = metric.metric_id
        self._call("nvmlGpmMetricsGet", self._library.nvmlGpmMetricsGet(ctypes.byref(request)))
        results = {}
        for slot, metric in zip(request.metrics, self.metrics, strict=False):
            reported = slot.metricInfo.shortName
            if slot.metricId != metric.metric_id or (
                reported is not None and reported.decode() != metric.driver_name
            ):
                # The ID means something else on this driver: never report it under our name.
                results[metric.name] = MetricResult(None, -1)
            elif slot.nvmlReturn != 0:
                results[metric.name] = MetricResult(None, slot.nvmlReturn)
            else:
                results[metric.name] = MetricResult(slot.value, 0)
        return results

    def close(self) -> None:
        while self._allocated:
            self._library.nvmlGpmSampleFree(self._allocated.pop())

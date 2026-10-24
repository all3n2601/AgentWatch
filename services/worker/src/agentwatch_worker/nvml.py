"""Read-only NVIDIA GPU gauges through NVML, using only the standard library's ctypes.

In-process NVML reads cost about 0.14 ms of CPU each on the GPU laptop (WSL2), while spawning
`nvidia-smi` for every sample cost about 93 ms of CPU, enough to distort the CPU signals
sampled beside it. Only a few long-stable NVML entry points are bound, without argtypes, so
each call passes explicit ctypes values.

Readings describe one whole device, never one process: per-process GPU accounting is
unavailable under WSL2 (docs/gpu-measurement-access.md). NVML utilization is the fraction of
the driver's own sample period (1/6 s to 1 s, product dependent) in which a kernel ran, so
it cannot resolve shorter activity and does not establish saturation.
"""

import ctypes
from collections.abc import Callable
from dataclasses import dataclass

# WSL2 ships the library outside the default search path; Windows names it nvml.dll.
LIBRARY_CANDIDATES = ("libnvidia-ml.so.1", "/usr/lib/wsl/lib/libnvidia-ml.so.1", "nvml.dll")
NVML_SUCCESS = 0
NVML_ERROR_NOT_SUPPORTED = 3
NVML_CLOCK_SM = 1
# Large enough for NVML's device-name (96) and driver-version (80) buffers.
STRING_BUFFER_BYTES = 96
ERROR_NAMES = {
    1: "UNINITIALIZED",
    2: "INVALID_ARGUMENT",
    3: "NOT_SUPPORTED",
    4: "NO_PERMISSION",
    6: "NOT_FOUND",
    7: "INSUFFICIENT_SIZE",
    9: "DRIVER_NOT_LOADED",
    10: "TIMEOUT",
    12: "LIBRARY_NOT_FOUND",
    13: "FUNCTION_NOT_FOUND",
    15: "GPU_IS_LOST",
    16: "RESET_REQUIRED",
    17: "OPERATING_SYSTEM",
    18: "LIB_RM_VERSION_MISMATCH",
    999: "UNKNOWN",
}


class _Utilization(ctypes.Structure):
    _fields_ = [("gpu", ctypes.c_uint), ("memory", ctypes.c_uint)]


class _Memory(ctypes.Structure):
    _fields_ = [
        ("total", ctypes.c_ulonglong),
        ("free", ctypes.c_ulonglong),
        ("used", ctypes.c_ulonglong),
    ]


class NvmlError(Exception):
    """An NVML call returned an error code; the message names the call, never a host path."""

    def __init__(self, function: str, code: int):
        super().__init__(f"{function} returned NVML error {code} ({ERROR_NAMES.get(code, '?')})")
        self.function = function
        self.code = code

    @property
    def not_supported(self) -> bool:
        return self.code == NVML_ERROR_NOT_SUPPORTED


class NvmlUnavailable(Exception):
    """NVML or the requested device cannot be used on this host.

    `unsupported` is true when the platform has no NVIDIA management library at all, as on
    macOS, rather than a library that failed to initialize or a missing device.
    """

    def __init__(self, reason: str, *, unsupported: bool = False):
        super().__init__(reason)
        self.unsupported = unsupported


@dataclass(frozen=True)
class GpuIdentity:
    index: int
    device_count: int
    name: str
    driver_version: str
    memory_total_bytes: int


def _check(function: str, code: int) -> None:
    if code != NVML_SUCCESS:
        raise NvmlError(function, code)


def _text(buffer: ctypes.Array) -> str:
    return buffer.value.decode("utf-8", errors="replace")


class NvmlDevice:
    """One initialized NVML session bound to one device index. Close it exactly once."""

    def __init__(self, library: object, handle: ctypes.c_void_p, identity: GpuIdentity):
        self._library = library
        self._handle = handle
        self.identity = identity
        self._closed = False

    @classmethod
    def open(
        cls,
        index: int = 0,
        *,
        load: Callable[[str], object] = ctypes.CDLL,
        candidates: tuple[str, ...] = LIBRARY_CANDIDATES,
    ) -> "NvmlDevice":
        library = None
        for name in candidates:
            try:
                library = load(name)
                break
            except OSError:
                continue
        if library is None:
            raise NvmlUnavailable("NVML library was not found on this host", unsupported=True)
        code = library.nvmlInit_v2()
        if code != NVML_SUCCESS:
            raise NvmlUnavailable(f"NVML could not initialize: {NvmlError('nvmlInit_v2', code)}")
        try:
            return cls._bind(library, index)
        except BaseException:
            library.nvmlShutdown()
            raise

    @classmethod
    def _bind(cls, library: object, index: int) -> "NvmlDevice":
        try:
            count = ctypes.c_uint()
            _check("nvmlDeviceGetCount_v2", library.nvmlDeviceGetCount_v2(ctypes.byref(count)))
            if not 0 <= index < count.value:
                raise NvmlUnavailable(f"No NVIDIA GPU at index {index} ({count.value} found)")
            handle = ctypes.c_void_p()
            _check(
                "nvmlDeviceGetHandleByIndex_v2",
                library.nvmlDeviceGetHandleByIndex_v2(ctypes.c_uint(index), ctypes.byref(handle)),
            )
            name = ctypes.create_string_buffer(STRING_BUFFER_BYTES)
            _check(
                "nvmlDeviceGetName",
                library.nvmlDeviceGetName(handle, name, ctypes.c_uint(STRING_BUFFER_BYTES)),
            )
            driver = ctypes.create_string_buffer(STRING_BUFFER_BYTES)
            _check(
                "nvmlSystemGetDriverVersion",
                library.nvmlSystemGetDriverVersion(driver, ctypes.c_uint(STRING_BUFFER_BYTES)),
            )
            memory = _Memory()
            _check(
                "nvmlDeviceGetMemoryInfo",
                library.nvmlDeviceGetMemoryInfo(handle, ctypes.byref(memory)),
            )
        except NvmlError as error:
            raise NvmlUnavailable(f"NVML device {index} could not be opened: {error}") from error
        identity = GpuIdentity(
            index=index,
            device_count=count.value,
            name=_text(name),
            driver_version=_text(driver),
            memory_total_bytes=memory.total,
        )
        return cls(library, handle, identity)

    def utilization_percent(self) -> tuple[int, int]:
        """(kernel-busy, memory-controller-busy) percent over NVML's own sample period."""
        rates = _Utilization()
        _check(
            "nvmlDeviceGetUtilizationRates",
            self._library.nvmlDeviceGetUtilizationRates(self._handle, ctypes.byref(rates)),
        )
        return rates.gpu, rates.memory

    def memory_bytes(self) -> tuple[int, int]:
        """(used, total) device memory in bytes, across every process on the device."""
        memory = _Memory()
        _check(
            "nvmlDeviceGetMemoryInfo",
            self._library.nvmlDeviceGetMemoryInfo(self._handle, ctypes.byref(memory)),
        )
        return memory.used, memory.total

    def power_watts(self) -> float:
        milliwatts = ctypes.c_uint()
        _check(
            "nvmlDeviceGetPowerUsage",
            self._library.nvmlDeviceGetPowerUsage(self._handle, ctypes.byref(milliwatts)),
        )
        return milliwatts.value / 1000

    def sm_clock_mhz(self) -> int:
        megahertz = ctypes.c_uint()
        _check(
            "nvmlDeviceGetClockInfo",
            self._library.nvmlDeviceGetClockInfo(
                self._handle, ctypes.c_int(NVML_CLOCK_SM), ctypes.byref(megahertz)
            ),
        )
        return megahertz.value

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._library.nvmlShutdown()

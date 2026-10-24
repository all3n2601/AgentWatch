"""NVML binding against a fake library (a test double; its readings are never evidence)."""

import ctypes

import pytest
from agentwatch_worker.nvml import (
    NVML_CLOCK_SM,
    NvmlDevice,
    NvmlError,
    NvmlUnavailable,
    _Memory,
    _Utilization,
)


class FakeNvml:
    """Implements the NVML calls the binding makes, writing through ctypes byref arguments."""

    def __init__(self, *, init=0, count=1, failures=None):
        self.init_code, self.count = init, count
        self.failures = failures or {}
        self.calls: list[str] = []
        self.clock_types: list[int] = []

    def _code(self, name):
        self.calls.append(name)
        return self.failures.get(name, 0)

    def nvmlInit_v2(self):
        self.calls.append("nvmlInit_v2")
        return self.init_code

    def nvmlShutdown(self):
        return self._code("nvmlShutdown")

    def nvmlDeviceGetCount_v2(self, count):
        count._obj.value = self.count
        return self._code("nvmlDeviceGetCount_v2")

    def nvmlDeviceGetHandleByIndex_v2(self, index, handle):
        handle._obj.value = 0x1000 + index.value
        return self._code("nvmlDeviceGetHandleByIndex_v2")

    def nvmlDeviceGetName(self, _handle, buffer, _length):
        buffer.value = b"Test GPU"
        return self._code("nvmlDeviceGetName")

    def nvmlSystemGetDriverVersion(self, buffer, _length):
        buffer.value = b"617.42"
        return self._code("nvmlSystemGetDriverVersion")

    def nvmlDeviceGetMemoryInfo(self, _handle, memory):
        memory._obj.total, memory._obj.used = 8_547_991_552, 1_702_593_821
        memory._obj.free = memory._obj.total - memory._obj.used
        return self._code("nvmlDeviceGetMemoryInfo")

    def nvmlDeviceGetUtilizationRates(self, _handle, rates):
        rates._obj.gpu, rates._obj.memory = 81, 40
        return self._code("nvmlDeviceGetUtilizationRates")

    def nvmlDeviceGetPowerUsage(self, _handle, milliwatts):
        milliwatts._obj.value = 87_250
        return self._code("nvmlDeviceGetPowerUsage")

    def nvmlDeviceGetClockInfo(self, _handle, clock_type, megahertz):
        self.clock_types.append(clock_type.value)
        megahertz._obj.value = 2662
        return self._code("nvmlDeviceGetClockInfo")


def open_fake(library, index=0):
    return NvmlDevice.open(index, load=lambda _name: library, candidates=("fake",))


def test_open_reads_identity_and_converts_units():
    library = FakeNvml()
    device = open_fake(library)
    assert device.identity.name == "Test GPU"
    assert device.identity.driver_version == "617.42"
    assert (device.identity.index, device.identity.device_count) == (0, 1)
    assert device.utilization_percent() == (81, 40)
    assert device.memory_bytes() == (1_702_593_821, 8_547_991_552)
    assert device.power_watts() == 87.25
    assert device.sm_clock_mhz() == 2662
    assert library.clock_types == [NVML_CLOCK_SM]


def test_missing_library_is_unsupported():
    def load(_name):
        raise OSError("not found")

    with pytest.raises(NvmlUnavailable) as raised:
        NvmlDevice.open(load=load, candidates=("a", "b"))
    assert raised.value.unsupported
    assert "not found on this host" in str(raised.value)


def test_later_candidates_are_tried():
    library, tried = FakeNvml(), []

    def load(name):
        tried.append(name)
        if name == "first":
            raise OSError("missing")
        return library

    NvmlDevice.open(load=load, candidates=("first", "second"))
    assert tried == ["first", "second"]


def test_initialization_failure_is_unavailable_not_unsupported():
    with pytest.raises(NvmlUnavailable) as raised:
        open_fake(FakeNvml(init=9))
    assert not raised.value.unsupported
    assert "DRIVER_NOT_LOADED" in str(raised.value)


@pytest.mark.parametrize(
    ("library", "message"),
    [
        (FakeNvml(count=0), r"No NVIDIA GPU at index 0 \(0 found\)"),
        (FakeNvml(failures={"nvmlDeviceGetName": 15}), "GPU_IS_LOST"),
    ],
)
def test_unusable_device_shuts_nvml_down(library, message):
    with pytest.raises(NvmlUnavailable, match=message):
        open_fake(library)
    assert library.calls[-1] == "nvmlShutdown"


def test_read_errors_carry_the_call_and_code():
    device = open_fake(FakeNvml(failures={"nvmlDeviceGetPowerUsage": 3}))
    with pytest.raises(NvmlError) as raised:
        device.power_watts()
    assert raised.value.not_supported
    assert raised.value.function == "nvmlDeviceGetPowerUsage"
    assert "NOT_SUPPORTED" in str(raised.value)


def test_close_shuts_down_once():
    library = FakeNvml()
    device = open_fake(library)
    device.close()
    device.close()
    assert library.calls.count("nvmlShutdown") == 1


def test_structures_match_the_nvml_abi():
    # nvmlUtilization_t is two unsigned ints; nvmlMemory_t is three unsigned long longs.
    assert ctypes.sizeof(_Utilization) == 2 * ctypes.sizeof(ctypes.c_uint)
    assert ctypes.sizeof(_Memory) == 3 * 8

"""NVML GPM binding and the GPM exporter, against a fake library (synthetic, never evidence)."""

import ctypes
import os

import pytest
from agentwatch_worker import gpm_exporter
from agentwatch_worker.gpm import METRICS, GpmSampler, _MetricsGet, _Support
from agentwatch_worker.gpm_exporter import export, render
from agentwatch_worker.nvml import NvmlError, NvmlUnavailable


class FakeDevice:
    def __init__(self, library):
        self.library = library
        self.handle = ctypes.c_void_p(1)


class FakeGpm:
    """Answers the GPM entry points; `values` maps metric ID to (return code, value)."""

    def __init__(self, supported=1, support_code=0, values=None, names=None, sample_code=0):
        self.supported, self.support_code, self.sample_code = supported, support_code, sample_code
        self.values = values or {m.metric_id: (0, 10.0 * m.metric_id) for m in METRICS}
        self.names = names or {m.metric_id: m.driver_name for m in METRICS}
        self.allocated = self.freed = self.samples = 0
        self._name_buffers = []

    def nvmlGpmQueryDeviceSupport(self, _handle, support):
        ctypes.cast(support, ctypes.POINTER(_Support)).contents.isSupportedDevice = self.supported
        return self.support_code

    def nvmlGpmSampleAlloc(self, _sample):
        self.allocated += 1
        return 0

    def nvmlGpmSampleFree(self, _sample):
        self.freed += 1
        return 0

    def nvmlGpmSampleGet(self, _handle, _sample):
        self.samples += 1
        return self.sample_code

    def nvmlGpmMetricsGet(self, request):
        request = ctypes.cast(request, ctypes.POINTER(_MetricsGet)).contents
        assert request.version == 1
        for slot in request.metrics[: request.numMetrics]:
            code, value = self.values[slot.metricId]
            slot.nvmlReturn, slot.value = code, value
            name = ctypes.create_string_buffer(self.names[slot.metricId].encode())
            self._name_buffers.append(name)
            slot.metricInfo.shortName = ctypes.cast(name, ctypes.c_char_p)
        return 0


def test_the_first_sample_has_no_window_and_later_ones_report_every_metric():
    library = FakeGpm()
    sampler = GpmSampler(FakeDevice(library))
    assert sampler.window() is None
    window = sampler.window()
    assert window["sm_util"].value == 20.0 and window["sm_util"].code == 0
    assert window["dram_bw_util"].value == 100.0
    assert set(window) == {m.name for m in METRICS}
    sampler.close()
    sampler.close()
    assert (library.allocated, library.freed) == (2, 2)


def test_a_metric_nvml_refuses_keeps_its_code_and_no_value():
    library = FakeGpm(values={m.metric_id: (0, 1.0) for m in METRICS} | {5: (3, 0.0)})
    sampler = GpmSampler(FakeDevice(library))
    sampler.window()
    window = sampler.window()
    assert (window["tensor_active"].value, window["tensor_active"].code) == (None, 3)


def test_an_id_the_driver_names_differently_is_never_reported_under_our_name():
    library = FakeGpm(names={m.metric_id: m.driver_name for m in METRICS} | {2: "other"})
    sampler = GpmSampler(FakeDevice(library))
    sampler.window()
    assert sampler.window()["sm_util"] == gpm_exporter.MetricResult(None, -1)


def test_an_unsupported_device_is_unsupported_not_an_error():
    with pytest.raises(NvmlUnavailable) as caught:
        GpmSampler(FakeDevice(FakeGpm(supported=0)))
    assert caught.value.unsupported is True


def test_a_library_without_gpm_is_unavailable():
    class Old:
        pass

    with pytest.raises(NvmlUnavailable, match="lacks nvmlGpmQueryDeviceSupport"):
        GpmSampler(FakeDevice(Old()))


def test_a_refused_sample_raises_with_its_code():
    # WSL2 answers 17 (operating system) and then 19 (in use by another client).
    sampler = GpmSampler(FakeDevice(FakeGpm(sample_code=17)))
    with pytest.raises(NvmlError, match="nvmlGpmSampleGet returned NVML error 17"):
        sampler.window()


def window_of(values):
    return {name: gpm_exporter.MetricResult(value, code) for name, (value, code) in values.items()}


def test_render_omits_refused_metrics_and_keeps_their_status():
    text = render(
        0,
        "a1",
        7,
        0.25,
        window_of({"sm_util": (42.5, 0), "tensor_active": (None, 3)}),
        {"sm_util": "percent", "tensor_active": "percent"},
        1.5,
        0,
    )
    assert 'agentwatch_gpm_exporter_info{gpu="0",instance="a1"} 1' in text
    assert 'agentwatch_gpm_sequence{gpu="0"} 7' in text
    assert 'agentwatch_gpm_sm_util_percent{gpu="0"} 42.500000' in text
    assert "agentwatch_gpm_tensor_active_percent" not in text
    assert 'agentwatch_gpm_metric_status{gpu="0",metric="tensor_active"} 3' in text


class ScriptedSampler:
    metrics = METRICS

    def __init__(self):
        self.calls = 0

    def window(self):
        self.calls += 1
        if self.calls == 1:
            return None
        return {m.name: gpm_exporter.MetricResult(float(self.calls), 0) for m in METRICS}


class FakeTime:
    def __init__(self):
        self.now = 0.0

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += max(seconds, 0.001)


def test_export_writes_one_numbered_window_per_interval(tmp_path):
    clock = FakeTime()
    output = tmp_path / "gpm.prom"
    counts = export(
        ScriptedSampler(),
        output,
        gpu_index=0,
        interval_seconds=0.25,
        duration_seconds=1.0,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
        cpu_seconds=lambda: 0.5,
        instance="a1",
    )
    assert counts == {"windows": 3, "windows_written": 3, "write_failures": 0}
    text = output.read_text()
    assert 'agentwatch_gpm_sequence{gpu="0"} 3' in text
    assert 'agentwatch_gpm_window_seconds{gpu="0"} 0.250000' in text
    assert not list(tmp_path.glob(".*.tmp"))


def test_a_file_held_by_a_reader_is_counted_not_fatal(tmp_path, monkeypatch):
    def refuse(_source, _target):
        raise PermissionError("held by a reader")

    monkeypatch.setattr(os, "replace", refuse)
    clock = FakeTime()
    counts = export(
        ScriptedSampler(),
        tmp_path / "gpm.prom",
        gpu_index=0,
        interval_seconds=0.25,
        duration_seconds=0.6,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
        cpu_seconds=lambda: 0.0,
    )
    assert counts["write_failures"] == counts["windows"] >= 1
    assert counts["windows_written"] == 0
    assert not list(tmp_path.iterdir())


class SlowSampler(ScriptedSampler):
    """The second window takes 0.6 s, overrunning a 0.25 s interval."""

    def __init__(self, clock):
        super().__init__()
        self.clock = clock
        self.sampled_at = []

    def window(self):
        self.sampled_at.append(self.clock.now)
        if len(self.sampled_at) == 2:
            self.clock.now += 0.6
        return super().window()


def test_an_overrun_skips_missed_slots_instead_of_sampling_back_to_back(tmp_path):
    clock = FakeTime()
    sampler = SlowSampler(clock)
    export(
        sampler,
        tmp_path / "gpm.prom",
        gpu_index=0,
        interval_seconds=0.25,
        duration_seconds=1.6,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
        cpu_seconds=lambda: 0.0,
    )
    # Slots at 0.5 and 0.75 were missed; sampling resumes on the grid at 1.0.
    assert sampler.sampled_at[:4] == pytest.approx([0.0, 0.25, 1.0, 1.25])


def test_each_exporter_process_names_a_new_instance(tmp_path):
    instances = set()
    for _ in range(2):
        clock = FakeTime()
        output = tmp_path / "gpm.prom"
        export(
            ScriptedSampler(),
            output,
            gpu_index=0,
            interval_seconds=0.25,
            duration_seconds=0.6,
            monotonic=clock.monotonic,
            sleep=clock.sleep,
            cpu_seconds=lambda: 0.0,
        )
        instances.add(output.read_text().split('instance="')[1].split('"')[0])
    assert len(instances) == 2


@pytest.mark.parametrize("interval", ["0.05", "nan", "inf"])
def test_intervals_below_the_minimum_are_refused(interval):
    with pytest.raises(SystemExit):
        gpm_exporter.main(["--output", "x.prom", "--interval", interval])

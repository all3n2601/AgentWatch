from uuid import uuid4

import pytest
from agent_test_doubles import FakeGpu
from agentwatch_agent.clock import MonotonicClock
from agentwatch_agent.resources import CpuProbe, GpuProbe, ResourceSampler
from agentwatch_sdk.live import ResourceRegistration
from agentwatch_worker.nvml import NvmlUnavailable

RESOURCE = ResourceRegistration(
    resource_id="tool-host-1", service_id="agentwatch-agent-runner", resource_class="cpu_sandbox"
)
GPU_RESOURCE = ResourceRegistration(
    resource_id="gpu-1", service_id="nvidia-nvml", resource_class="inference"
)


def make_sampler(probes, interval_seconds=0.01):
    samples = []
    instance = ResourceSampler(
        samples.append,
        run_id=uuid4(),
        task_id="fix-mean-empty-r1",
        clock=MonotonicClock("sampler-clock"),
        probes=probes,
        interval_seconds=interval_seconds,
    )
    return instance, samples


def sampler(tmp_path, proc, cgroup_root=None):
    return make_sampler([CpuProbe(RESOURCE, proc, cgroup_root or tmp_path / "cgroup")])


def linux_fixture(tmp_path, pressure, cpu_stat, cgroup_pressure=None):
    proc = tmp_path / "proc"
    (proc / "pressure").mkdir(parents=True)
    (proc / "self").mkdir()
    (proc / "self/cgroup").write_text("0::/agent.slice\n")
    if pressure is not None:
        (proc / "pressure/cpu").write_text(pressure)
    group = tmp_path / "cgroup/agent.slice"
    group.mkdir(parents=True)
    if cpu_stat is not None:
        (group / "cpu.stat").write_text(cpu_stat)
    if cgroup_pressure is not None:
        (group / "cpu.pressure").write_text(cgroup_pressure)
    return proc


def by_metric(samples):
    return {sample.metric: sample for sample in samples}


def test_measured_linux_readings(tmp_path):
    proc = linux_fixture(
        tmp_path,
        "some avg10=12.50 avg60=1.00 avg300=0.00 total=5\n",
        "usage_usec 10\nthrottled_usec 0\n",
    )
    instance, samples = sampler(tmp_path, proc)
    instance.sample_once()
    readings = by_metric(samples)
    psi = readings["cpu.psi.some.avg10.v1"]
    throttled = readings["cpu.cgroup.throttled_usec.v1"]
    assert (psi.availability, psi.value, psi.unit) == ("measured", 12.5, "percent")
    # A measured zero stays a measured zero, distinct from unavailable.
    assert (throttled.availability, throttled.value) == ("measured", 0.0)
    assert psi.tick_ns == throttled.tick_ns


def test_cumulative_cpu_counters_are_read_with_the_gauges(tmp_path):
    proc = linux_fixture(
        tmp_path,
        "some avg10=0.00 avg60=0.00 avg300=0.00 total=499359020\n"
        "full avg10=0.00 avg60=0.00 avg300=0.00 total=0\n",
        "usage_usec 26795\nuser_usec 13397\nthrottled_usec 1200\n",
        "some avg10=0.00 avg60=0.00 avg300=0.00 total=138\n",
    )
    instance, samples = sampler(tmp_path, proc)
    instance.sample_once()
    readings = by_metric(samples)
    assert len(samples) == 5
    assert readings["cpu.psi.some.total_usec.v1"].value == 499_359_020.0
    assert readings["cpu.cgroup.usage_usec.v1"].value == 26_795.0
    assert readings["cpu.cgroup.psi.some.total_usec.v1"].value == 138.0
    counters = {s.unit for m, s in readings.items() if m != "cpu.psi.some.avg10.v1"}
    assert counters == {"microseconds"}
    assert len({s.tick_ns for s in samples}) == 1


@pytest.mark.parametrize(
    ("pressure", "cpu_stat", "cgroup_pressure", "expected"),
    [
        # Cgroup PSI disabled: unsupported, while every other reading is measured.
        (
            "some avg10=1.0 total=5\n",
            "usage_usec 1\nthrottled_usec 0\n",
            None,
            {
                "cpu.cgroup.psi.some.total_usec.v1": "unsupported",
                "cpu.cgroup.usage_usec.v1": "measured",
            },
        ),
        # A PSI line without total; a cpu.stat without usage (CPU controller counters absent).
        (
            "some avg10=1.0\n",
            "throttled_usec 0\n",
            "some total=x\n",
            {
                "cpu.psi.some.total_usec.v1": "malformed",
                "cpu.psi.some.avg10.v1": "measured",
                "cpu.cgroup.usage_usec.v1": "unavailable",
                "cpu.cgroup.psi.some.total_usec.v1": "malformed",
            },
        ),
        # A negative counter is malformed, never a measured value.
        (
            "some avg10=1.0 total=-5\n",
            "usage_usec -1\nthrottled_usec 0\n",
            "some total=3\n",
            {"cpu.psi.some.total_usec.v1": "malformed", "cpu.cgroup.usage_usec.v1": "malformed"},
        ),
    ],
)
def test_each_cpu_counter_keeps_its_own_cause(
    tmp_path, pressure, cpu_stat, cgroup_pressure, expected
):
    proc = linux_fixture(tmp_path, pressure, cpu_stat, cgroup_pressure)
    instance, samples = sampler(tmp_path, proc)
    instance.sample_once()
    readings = by_metric(samples)
    for metric, availability in expected.items():
        assert readings[metric].availability == availability, metric
        assert (readings[metric].value is None) == (availability != "measured")


def test_missing_cgroup_leaves_host_psi_measured(tmp_path):
    proc = linux_fixture(tmp_path, "some avg10=1.0 total=7\n", "usage_usec 1\n")
    (proc / "self/cgroup").write_text("1:cpu:/legacy\n")
    instance, samples = sampler(tmp_path, proc)
    instance.sample_once()
    readings = by_metric(samples)
    assert readings["cpu.psi.some.total_usec.v1"].value == 7.0
    for metric in ("cpu.cgroup.usage_usec.v1", "cpu.cgroup.psi.some.total_usec.v1"):
        assert readings[metric].availability == "unavailable"


def test_platform_without_procfs_is_unsupported_not_zero(tmp_path):
    instance, samples = sampler(tmp_path, tmp_path / "no-proc")
    instance.sample_once()
    for sample in samples:
        assert (sample.availability, sample.value) == ("unsupported", None)
        assert "procfs" in sample.reason


@pytest.mark.parametrize(
    ("pressure", "cpu_stat", "psi_state", "cgroup_state"),
    [
        (None, "usage_usec 1\n", "unsupported", "unavailable"),
        ("some avg10=oops\n", "throttled_usec x\n", "malformed", "malformed"),
        ("full avg10=1.0\n", None, "malformed", "unavailable"),
        ("some avg10=150.0\n", "throttled_usec -4\n", "malformed", "malformed"),
    ],
)
def test_unusable_readings_keep_their_cause(tmp_path, pressure, cpu_stat, psi_state, cgroup_state):
    instance, samples = sampler(tmp_path, linux_fixture(tmp_path, pressure, cpu_stat))
    instance.sample_once()
    readings = by_metric(samples)
    assert readings["cpu.psi.some.avg10.v1"].availability == psi_state
    assert readings["cpu.cgroup.throttled_usec.v1"].availability == cgroup_state
    original = ("cpu.psi.some.avg10.v1", "cpu.cgroup.throttled_usec.v1")
    assert all(readings[m].value is None and readings[m].reason for m in original)
    assert all(str(tmp_path) not in (sample.reason or "") for sample in samples), "paths leaked"


def test_missing_cgroup_membership_is_unavailable(tmp_path):
    proc = linux_fixture(tmp_path, "some avg10=1.0\n", "throttled_usec 1\n")
    (proc / "self/cgroup").write_text("1:cpu:/legacy\n")
    instance, samples = sampler(tmp_path, proc)
    instance.sample_once()
    assert by_metric(samples)["cpu.cgroup.throttled_usec.v1"].availability == "unavailable"


def test_background_sampling_reports_achieved_intervals(tmp_path):
    instance, samples = sampler(tmp_path, tmp_path / "no-proc")
    with instance:
        while len(samples) < 25:
            pass
    summary = instance.summary()
    assert summary["sample_rounds"] >= 5
    assert summary["declared_interval_seconds"] == 0.01
    assert summary["observed_interval_median_seconds"] > 0
    assert summary["availability_counts"]["cpu.psi.some.avg10.v1"] == {
        "unsupported": summary["sample_rounds"]
    }
    assert instance.error is None
    assert summary["resources"][0]["resource_id"] == "tool-host-1"
    assert summary["round_cost"]["thread_cpu_total_seconds"] >= 0
    assert summary["round_cost"]["wall_median_seconds"] >= 0


def gpu_readings(device=None, opener=None):
    probe = GpuProbe(GPU_RESOURCE, opener or (lambda: device))
    instance, samples = make_sampler([probe])
    instance.sample_once()
    return probe, by_metric(samples)


def test_measured_gpu_readings_keep_units_and_measured_zero():
    _, readings = gpu_readings(FakeGpu(utilization=(0, 3), used=1_702_593_821, power=87.5))
    utilization = readings["gpu.nvml.utilization.gpu.v1"]
    # An idle GPU is a measured zero, distinct from an unavailable reading.
    assert (utilization.availability, utilization.value, utilization.unit) == (
        "measured",
        0.0,
        "percent",
    )
    assert readings["gpu.nvml.utilization.memory.v1"].value == 3.0
    assert readings["gpu.nvml.memory.used_bytes.v1"].value == 1_702_593_821.0
    power = readings["gpu.nvml.power.usage_watts.v1"]
    assert (power.value, power.unit) == (87.5, "watts")
    assert readings["gpu.nvml.clock.sm_mhz.v1"].unit == "megahertz"
    assert {s.resource_id for s in readings.values()} == {"gpu-1"}
    assert {s.resource_class for s in readings.values()} == {"inference"}


@pytest.mark.parametrize(
    ("error", "availability"),
    [
        (
            NvmlUnavailable("NVML library was not found on this host", unsupported=True),
            "unsupported",
        ),
        (NvmlUnavailable("NVML could not initialize: driver not loaded"), "unavailable"),
    ],
)
def test_unopened_gpu_is_reported_every_round_never_as_zero(error, availability):
    def opener():
        raise error

    probe, readings = gpu_readings(opener=opener)
    assert len(readings) == 5
    for sample in readings.values():
        assert (sample.availability, sample.value, sample.reason) == (
            availability,
            None,
            str(error),
        )
    assert probe.describe() == {"device": None, "availability": availability, "reason": str(error)}


def test_each_gpu_reading_fails_independently():
    # NOT_SUPPORTED (3) is a platform limitation; GPU_IS_LOST (15) is an unavailable device.
    device = FakeGpu(failures={"power": 3, "utilization": 15})
    _, readings = gpu_readings(device)
    assert readings["gpu.nvml.power.usage_watts.v1"].availability == "unsupported"
    assert "NOT_SUPPORTED" in readings["gpu.nvml.power.usage_watts.v1"].reason
    for metric in ("gpu.nvml.utilization.gpu.v1", "gpu.nvml.utilization.memory.v1"):
        assert readings[metric].availability == "unavailable"
        assert "GPU_IS_LOST" in readings[metric].reason
    assert readings["gpu.nvml.memory.used_bytes.v1"].availability == "measured"
    assert readings["gpu.nvml.clock.sm_mhz.v1"].availability == "measured"


def test_impossible_gpu_readings_are_malformed():
    device = FakeGpu(utilization=(101, 50), used=FakeGpu.identity.memory_total_bytes + 1)
    _, readings = gpu_readings(device)
    assert readings["gpu.nvml.utilization.gpu.v1"].availability == "malformed"
    assert readings["gpu.nvml.utilization.memory.v1"].availability == "measured"
    assert readings["gpu.nvml.memory.used_bytes.v1"].availability == "malformed"


def test_one_round_reads_every_probe_with_its_own_tick(tmp_path):
    device = FakeGpu()
    cpu = CpuProbe(RESOURCE, tmp_path / "no-proc", tmp_path / "cgroup")
    instance, samples = make_sampler([cpu, GpuProbe(GPU_RESOURCE, lambda: device)])
    instance.sample_once()
    assert len(samples) == 10
    cpu_tick = {s.tick_ns for s in samples if s.resource_id == "tool-host-1"}
    gpu_tick = {s.tick_ns for s in samples if s.resource_id == "gpu-1"}
    assert len(cpu_tick) == len(gpu_tick) == 1
    assert cpu_tick.pop() <= gpu_tick.pop()
    summary = instance.summary()
    assert summary["sample_rounds"] == 1
    assert [r["resource_id"] for r in summary["resources"]] == ["tool-host-1", "gpu-1"]
    assert summary["resources"][1]["device"]["name"] == "Test GPU"


def test_sampler_closes_probes_once_after_stopping():
    device = FakeGpu()
    instance, _ = make_sampler([GpuProbe(GPU_RESOURCE, lambda: device)])
    with instance:
        pass
    assert device.closed == 1


class ScriptedClock(MonotonicClock):
    def __init__(self, ticks):
        super().__init__("scripted-clock")
        self.ticks = iter(ticks)

    def tick(self):
        return next(self.ticks)


def test_summary_counts_late_and_missed_rounds():
    interval_ns = 10_000_000
    # Gaps of 1, 1, 3 and 1 intervals: one late gap that skipped two rounds.
    ticks = [0, interval_ns, 2 * interval_ns, 5 * interval_ns, 6 * interval_ns]
    instance = ResourceSampler(
        lambda _sample: None,
        run_id=uuid4(),
        task_id="fix-mean-empty-r1",
        clock=ScriptedClock(ticks),
        probes=[GpuProbe(GPU_RESOURCE, FakeGpu)],
        interval_seconds=0.01,
    )
    for _ in ticks:
        instance.sample_once()
    summary = instance.summary()
    assert summary["sample_rounds"] == 5
    assert summary["observed_interval_median_seconds"] == pytest.approx(0.01)
    assert summary["observed_interval_max_seconds"] == pytest.approx(0.03)
    assert (summary["late_rounds"], summary["early_rounds"]) == (1, 0)
    assert (summary["expected_rounds"], summary["missed_rounds"]) == (7, 2)


def test_late_rounds_taken_back_to_back_are_not_missed():
    # The sampler catches up after a slow round: one 3-interval gap, then two overdue
    # rounds 1 ms apart. Every round the cadence expects was taken, only late.
    ms = 1_000_000
    ticks = [0, 10 * ms, 40 * ms, 41 * ms, 42 * ms]
    instance = ResourceSampler(
        lambda _sample: None,
        run_id=uuid4(),
        task_id="fix-mean-empty-r1",
        clock=ScriptedClock(ticks),
        probes=[GpuProbe(GPU_RESOURCE, FakeGpu)],
        interval_seconds=0.01,
    )
    for _ in ticks:
        instance.sample_once()
    summary = instance.summary()
    assert (summary["late_rounds"], summary["early_rounds"]) == (1, 2)
    assert (summary["expected_rounds"], summary["missed_rounds"]) == (5, 0)


def test_summary_without_rounds_reports_no_intervals():
    instance, _ = make_sampler([GpuProbe(GPU_RESOURCE, FakeGpu)])
    summary = instance.summary()
    assert summary["sample_rounds"] == 0
    assert summary["observed_interval_median_seconds"] is None
    assert summary["round_cost"]["thread_cpu_fraction_of_span"] is None

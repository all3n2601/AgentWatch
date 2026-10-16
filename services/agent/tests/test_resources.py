from uuid import uuid4

import pytest
from agentwatch_agent.clock import MonotonicClock
from agentwatch_agent.resources import CpuResourceSampler
from agentwatch_sdk.live import ResourceRegistration

RESOURCE = ResourceRegistration(
    resource_id="tool-host-1", service_id="agentwatch-agent-runner", resource_class="cpu_sandbox"
)


def sampler(tmp_path, proc, cgroup_root=None):
    samples = []
    instance = CpuResourceSampler(
        samples.append,
        run_id=uuid4(),
        task_id="fix-mean-empty-r1",
        clock=MonotonicClock("sampler-clock"),
        resource=RESOURCE,
        interval_seconds=0.01,
        proc=proc,
        cgroup_root=cgroup_root or tmp_path / "cgroup",
    )
    return instance, samples


def linux_fixture(tmp_path, pressure, cpu_stat):
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
    assert all(sample.value is None and sample.reason for sample in samples)
    assert all(str(tmp_path) not in sample.reason for sample in samples), "host paths leaked"


def test_missing_cgroup_membership_is_unavailable(tmp_path):
    proc = linux_fixture(tmp_path, "some avg10=1.0\n", "throttled_usec 1\n")
    (proc / "self/cgroup").write_text("1:cpu:/legacy\n")
    instance, samples = sampler(tmp_path, proc)
    instance.sample_once()
    assert by_metric(samples)["cpu.cgroup.throttled_usec.v1"].availability == "unavailable"


def test_background_sampling_reports_achieved_intervals(tmp_path):
    instance, samples = sampler(tmp_path, tmp_path / "no-proc")
    with instance:
        while len(samples) < 10:
            pass
    summary = instance.summary()
    assert summary["sample_rounds"] >= 5
    assert summary["declared_interval_seconds"] == 0.01
    assert summary["observed_interval_median_seconds"] > 0
    assert summary["availability_counts"]["cpu.psi.some.avg10.v1"] == {
        "unsupported": summary["sample_rounds"]
    }
    assert instance.error is None

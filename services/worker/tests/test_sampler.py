import json

from agentwatch_worker.sampler import CpuSampler, discover_cgroup


def linux_fixture(tmp_path):
    proc = tmp_path / "proc"
    root = tmp_path / "cgroup"
    (proc / "pressure").mkdir(parents=True)
    (proc / "self").mkdir()
    (root / "agentwatch").mkdir(parents=True)
    (proc / "self/cgroup").write_text("0::/agentwatch\n")
    (proc / "pressure/cpu").write_text(
        "some avg10=12.50 avg60=4.00 avg300=1.00 total=200000\n"
        "full avg10=0.00 avg60=0.00 avg300=0.00 total=0\n"
    )
    (root / "agentwatch/cpu.stat").write_text(
        "usage_usec 10000\nnr_throttled 2\nthrottled_usec 400\n"
    )
    return proc, root


def test_linux_measurements_and_window_alignment(tmp_path):
    proc, root = linux_fixture(tmp_path)
    sampler = CpuSampler(proc, root)
    assert discover_cgroup(proc, root) == root / "agentwatch/cpu.stat"
    first = sampler.sample()
    assert first["cpu_pressure"]["some"]["avg10"] == 12.5
    assert first["cgroup_cpu_stat"]["throttled_usec"] == 400
    assert first["unavailable"] == []
    first["timestamp_ns"] = 100
    second = json.loads(json.dumps(first))
    second["timestamp_ns"] = 200
    second["cgroup_cpu_stat"]["throttled_usec"] = 900
    sampler.samples = [first, second]
    summary = sampler.summarize(100, 200)
    assert summary["status"] == "available"
    assert summary["observed_cgroup_throttled_usec_delta"] == 500
    assert sampler.summarize(201, 300)["sample_count"] == 0


def test_missing_metrics_are_null_not_zero(tmp_path):
    sampler = CpuSampler(tmp_path / "absent", tmp_path / "absent-cgroup")
    sample = sampler.sample()
    assert sample["cpu_pressure"] is None
    assert sample["cgroup_cpu_stat"] is None
    assert len(sample["unavailable"]) == 2
    sampler.samples = [sample]
    summary = sampler.summarize(0, sample["timestamp_ns"])
    assert summary["status"] == "unavailable_or_no_samples"
    assert summary["max_cpu_pressure_avg10_percent"] is None


def test_cgroup_membership_cannot_escape_root(tmp_path):
    proc, root = linux_fixture(tmp_path)
    (proc / "self/cgroup").write_text("0::/../../outside\n")
    assert discover_cgroup(proc, root) is None


def test_malformed_metric_does_not_stop_sampler(tmp_path):
    proc, root = linux_fixture(tmp_path)
    (proc / "pressure/cpu").write_text("some avg10=bad\n")
    sample = CpuSampler(proc, root).sample()
    assert sample["cpu_pressure"] is None
    assert sample["cgroup_cpu_stat"] is not None


def test_cpu_counter_reset_is_not_negative_throttling(tmp_path):
    proc, root = linux_fixture(tmp_path)
    sampler = CpuSampler(proc, root)
    first, second = sampler.sample(), sampler.sample()
    second["cgroup_cpu_stat"]["throttled_usec"] = 0
    sampler.samples = [first, second]
    assert (
        sampler.summarize(0, second["timestamp_ns"])["observed_cgroup_throttled_usec_delta"] is None
    )

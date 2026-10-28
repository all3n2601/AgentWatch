"""The runner's GPM file probe, against files the real exporter's renderer writes (synthetic)."""

import pytest
from agentwatch_agent.service_probes import GPM_METRICS, GpmFileProbe
from agentwatch_sdk.live import METRICS, ResourceRegistration
from agentwatch_worker.gpm import METRICS as EXPORTED
from agentwatch_worker.gpm import MetricResult
from agentwatch_worker.gpm_exporter import render

GPU = ResourceRegistration(
    resource_id="gpu-gpm-1", service_id="nvidia-nvml-gpm", resource_class="inference"
)
UNITS = {metric.name: metric.unit for metric in EXPORTED}


def write(path, sequence, values=None, cpu=1.0):
    results = {m.name: MetricResult(12.5, 0) for m in EXPORTED} | (values or {})
    path.write_text(render(0, sequence, 0.25, results, UNITS, cpu, 0))


def readings(probe):
    return {metric: reading for (metric, _unit), reading in probe.read()}


@pytest.mark.parametrize("entry", GPM_METRICS, ids=lambda e: e.metric)
def test_every_gpm_metric_is_registered_with_its_unit(entry):
    resource_class, unit, _minimum, _maximum = METRICS[entry.metric]
    assert (resource_class, unit) == ("inference", entry.unit)


def test_the_probe_reads_every_metric_the_exporter_writes(tmp_path):
    # A name mismatch between the two packages would make every reading unavailable.
    path = tmp_path / "gpm.prom"
    write(path, 1)
    values = readings(GpmFileProbe(GPU, path))
    assert len(values) == len(EXPORTED) == len(GPM_METRICS)
    assert set(values.values()) == {(12.5, "measured", None)}


def test_a_measured_zero_stays_zero(tmp_path):
    path = tmp_path / "gpm.prom"
    write(path, 1, {"sm_util": MetricResult(0.0, 0)})
    assert readings(GpmFileProbe(GPU, path))["gpu.gpm.sm_util.v1"] == (0.0, "measured", None)


def test_a_window_is_used_once_and_a_repeat_is_unavailable(tmp_path):
    path = tmp_path / "gpm.prom"
    probe = GpmFileProbe(GPU, path)
    write(path, 1, cpu=1.0)
    assert readings(probe)["gpu.gpm.sm_util.v1"][1] == "measured"
    repeated = readings(probe)["gpu.gpm.sm_util.v1"]
    assert repeated == (None, "unavailable", "No new GPM window since the previous round")
    write(path, 2, cpu=1.25)
    assert readings(probe)["gpu.gpm.sm_util.v1"][1] == "measured"
    described = probe.describe()
    assert (described["windows_used"], described["repeated_windows"]) == (2, 1)
    assert described["exporter_cpu_seconds"] == pytest.approx(0.25)
    assert described["window_median_seconds"] == pytest.approx(0.25)


def test_a_metric_nvml_refused_names_its_code(tmp_path):
    path = tmp_path / "gpm.prom"
    write(path, 1, {"tensor_active": MetricResult(None, 3)})
    value, availability, reason = readings(GpmFileProbe(GPU, path))["gpu.gpm.tensor_active.v1"]
    assert (value, availability) == (None, "unavailable")
    assert reason == "NVML refused GPM metric tensor_active (code 3)"


def test_a_missing_file_is_unavailable_never_zero(tmp_path):
    values = readings(GpmFileProbe(GPU, tmp_path / "absent.prom"))
    assert {availability for _v, availability, _r in values.values()} == {"unavailable"}
    assert next(iter(values.values()))[2].startswith("GPM exporter file unreadable")


def test_a_file_without_a_sequence_is_unavailable(tmp_path):
    path = tmp_path / "gpm.prom"
    path.write_text('agentwatch_gpm_sm_util_percent{gpu="0"} 50\n')
    reason = readings(GpmFileProbe(GPU, path))["gpu.gpm.sm_util.v1"][2]
    assert reason == "GPM exporter file has no window sequence"


def test_an_out_of_range_percentage_is_malformed(tmp_path):
    path = tmp_path / "gpm.prom"
    write(path, 1, {"dram_bw_util": MetricResult(140.0, 0)})
    assert readings(GpmFileProbe(GPU, path))["gpu.gpm.dram_bw_util.v1"][1] == "malformed"

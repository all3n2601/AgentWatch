import json

import pytest
from agentwatch_agent import service_probes
from agentwatch_agent.clock import MonotonicClock
from agentwatch_agent.prometheus_text import parse_exposition
from agentwatch_agent.resources import COUNTER_METRICS
from agentwatch_agent.service_probes import (
    DCGM_METRICS,
    SERVICE_COUNTER_METRICS,
    VLLM_METRICS,
    ExpositionChangeLog,
    ExpositionProbe,
    SeriesMetric,
    dcgm_probe,
    series_reading,
    vllm_probe,
)
from agentwatch_sdk.live import METRICS, ResourceRegistration
from test_prometheus_text import MetricsServer

SERVER = ResourceRegistration(
    resource_id="model-server-1", service_id="vllm", resource_class="inference"
)
DCGM = ResourceRegistration(
    resource_id="gpu-dcgm-1", service_id="nvidia-dcgm", resource_class="inference"
)
WAITING = SeriesMetric(
    "inference.vllm.requests_waiting.v1", "requests", "vllm:num_requests_waiting"
)
KV = SeriesMetric("inference.vllm.kv_cache_usage.v1", "fraction", "vllm:kv_cache_usage_perc")
# Synthetic vLLM exposition: two engines serve the run's model, one serves another model.
VLLM_TEXT = """\
vllm:num_requests_waiting{engine="0",model_name="m@1"} 2.0
vllm:num_requests_waiting{engine="1",model_name="m@1"} 1.0
vllm:num_requests_waiting{engine="0",model_name="other"} 9.0
vllm:kv_cache_usage_perc{engine="0",model_name="m@1"} 0.25
"""


@pytest.mark.parametrize("entry", VLLM_METRICS + DCGM_METRICS, ids=lambda e: e.metric)
def test_every_probe_metric_is_registered_with_its_unit(entry):
    resource_class, unit, _minimum, _maximum = METRICS[entry.metric]
    assert resource_class == "inference"
    assert unit == entry.unit


def test_counters_are_exactly_the_cumulative_service_metrics():
    assert SERVICE_COUNTER_METRICS == {e.metric for e in VLLM_METRICS + DCGM_METRICS if e.counter}
    assert "inference.vllm.requests_waiting.v1" not in SERVICE_COUNTER_METRICS
    assert "inference.vllm.request_queue.seconds_total.v1" in SERVICE_COUNTER_METRICS
    assert "gpu.dcgm.energy.total_mj.v1" in SERVICE_COUNTER_METRICS
    assert not SERVICE_COUNTER_METRICS & COUNTER_METRICS


def test_matching_series_are_summed_and_other_models_ignored():
    exposition = parse_exposition(VLLM_TEXT)
    assert series_reading(exposition, WAITING, {"model_name": "m@1"}, "vLLM") == (
        3.0,
        "measured",
        None,
    )


def test_measured_zero_stays_zero():
    exposition = parse_exposition('vllm:num_requests_waiting{model_name="m@1"} 0.0\n')
    assert series_reading(exposition, WAITING, {"model_name": "m@1"}, "vLLM") == (
        0.0,
        "measured",
        None,
    )


def test_an_unexported_series_is_unavailable_never_zero():
    exposition = parse_exposition(VLLM_TEXT)
    value, availability, reason = series_reading(exposition, WAITING, {"model_name": "x"}, "vLLM")
    assert (value, availability) == (None, "unavailable")
    assert reason == "vLLM did not export vllm:num_requests_waiting"


@pytest.mark.parametrize("text", ["NaN", "+Inf", "-Inf"])
def test_non_finite_values_are_malformed(text):
    exposition = parse_exposition(f'vllm:num_requests_waiting{{model_name="m@1"}} {text}\n')
    value, availability, _reason = series_reading(exposition, WAITING, {"model_name": "m@1"}, "v")
    assert (value, availability) == (None, "malformed")


@pytest.mark.parametrize(("entry", "text"), [(KV, "1.5"), (WAITING, "-1")])
def test_values_outside_the_registered_range_are_malformed(entry, text):
    exposition = parse_exposition(f'{entry.series}{{model_name="m@1"}} {text}\n')
    value, availability, reason = series_reading(exposition, entry, {"model_name": "m@1"}, "v")
    assert (value, availability) == (None, "malformed")
    assert "outside its registered range" in reason


def test_a_failed_fetch_makes_every_metric_unavailable():
    probe = vllm_probe(SERVER, "http://127.0.0.1:1", "m@1")
    readings = probe.read()
    assert len(readings) == len(VLLM_METRICS)
    assert {availability for _, (_, availability, _) in readings} == {"unavailable"}
    assert readings[0][1][2].startswith("vLLM metrics request failed (")
    described = probe.describe()
    assert described["fetches"] == 1 and sum(described["fetch_failures"].values()) == 1


def test_a_probe_reads_its_series_from_the_endpoint():
    with MetricsServer(VLLM_TEXT.encode()) as server:
        probe = ExpositionProbe(
            SERVER,
            url=server.url,
            source="vLLM",
            entries=(WAITING, KV),
            labels={"model_name": "m@1"},
            scope="test",
        )
        readings = dict(probe.read())
    assert readings[(WAITING.metric, "requests")] == (3.0, "measured", None)
    assert readings[(KV.metric, "fraction")] == (0.25, "measured", None)


def test_dcgm_profiling_fields_absent_from_the_export_are_unavailable():
    # Synthetic: the laptop's exporter omits DCP fields that DCGM reads as N/A.
    text = 'DCGM_FI_DEV_GPU_UTIL{gpu="0"} 12\nDCGM_FI_DEV_GPU_UTIL{gpu="1"} 99\n'
    with MetricsServer(text.encode()) as server:
        readings = dict(dcgm_probe(DCGM, server.url, gpu_index=0).read())
    assert readings[("gpu.dcgm.utilization.gpu.v1", "percent")] == (12.0, "measured", None)
    value, availability, reason = readings[("gpu.dcgm.prof.sm_active.v1", "fraction")]
    assert (value, availability) == (None, "unavailable")
    assert "DCGM_FI_PROF_SM_ACTIVE" in reason


def test_the_change_log_keeps_the_first_round_then_only_changes(tmp_path):
    log = ExpositionChangeLog(tmp_path / "changes.jsonl", MonotonicClock("c"))
    log.record(parse_exposition("a 1\nb 2\n"), 0.01, None)
    log.record(parse_exposition("a 1\nb 3\n"), 0.01, None)
    log.record(None, 0.2, "TimeoutError")
    log.record(parse_exposition("b 3\n"), 0.01, None)
    log.close()
    lines = [json.loads(line) for line in (tmp_path / "changes.jsonl").read_text().splitlines()]
    assert [len(line["changed"]) for line in lines] == [2, 1, 0, 0]
    assert lines[1]["changed"] == {'["b",[]]': 3.0}
    assert lines[2]["error"] == "TimeoutError" and lines[2]["removed"] == []
    # A failed fetch does not reset what the next round is compared with.
    assert lines[3]["removed"] == ['["a",[]]']
    assert log.describe() | {"bytes": 0} == {
        "file": "changes.jsonl",
        "rounds": 4,
        "bytes": 0,
        "truncated": False,
    }


def test_the_change_log_stops_at_its_size_bound(tmp_path, monkeypatch):
    monkeypatch.setattr(service_probes, "MAX_CHANGE_LOG_BYTES", 150)
    log = ExpositionChangeLog(tmp_path / "changes.jsonl", MonotonicClock("c"))
    for value in range(10):
        log.record(parse_exposition(f"a {value}\n"), 0.01, None)
    assert log.truncated is True
    assert (tmp_path / "changes.jsonl").stat().st_size <= 150
    assert log.describe()["rounds"] < 10


def test_an_unparsable_line_of_a_registered_series_is_malformed_not_absent():
    # One engine's line is broken: summing the other would undercount the waiting requests.
    text = (
        'vllm:num_requests_waiting{engine="0",model_name="m@1"} 2.0\n'
        'vllm:num_requests_waiting{engine="1",model_name="m@1" 1.0\n'
    )
    exposition = parse_exposition(text)
    value, availability, reason = series_reading(exposition, WAITING, {"model_name": "m@1"}, "v")
    assert (value, availability) == (None, "malformed")
    assert reason == "v sent an unparsable vllm:num_requests_waiting line"
    # Other series in the same exposition are unaffected.
    kv = parse_exposition(text + 'vllm:kv_cache_usage_perc{model_name="m@1"} 0.5\n')
    assert series_reading(kv, KV, {"model_name": "m@1"}, "v")[1] == "measured"

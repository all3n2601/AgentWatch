"""Reading and verifying vLLM's server state against the declared sweep configuration.

The exposition below is synthetic. Its `vllm:cache_config_info` labels copy the label set the
laptop's vLLM v0.32 reported on 2026-10-27; it is not recorded-run evidence.
"""

import socket
from dataclasses import asdict

import pytest
from agentwatch_agent.sweep_analysis import server_changes
from agentwatch_agent.sweep_protocol import CONFIGURATIONS, MODEL_ID, MODEL_REVISION
from agentwatch_agent.vllm_server import (
    ServerState,
    ServerStateUnavailable,
    read_server_state,
    verify_configuration,
)
from vllm_test_server import FakeVllm

SERVED = f"{MODEL_ID}@{MODEL_REVISION}"


def exposition(gpu_memory_utilization="0.6", running=0.0, waiting=0.0, extra=""):
    labels = (
        'block_size="16",cache_dtype="auto",calculate_kv_scales="False",'
        'cpu_offload_gb="0",enable_prefix_caching="True",engine="0",'
        f'gpu_memory_utilization="{gpu_memory_utilization}",is_attention_free="False",'
        'kv_cache_max_concurrency="1.80078125",kv_cache_size_tokens="7376",'
        'num_cpu_blocks="None",num_gpu_blocks="461",prefix_caching_hash_algo="sha256",'
        'sliding_window="None",swap_space="4"'
    )
    model = f'engine="0",model_name="{SERVED}"'
    return f"""\
# HELP process_start_time_seconds Start time of the process since unix epoch in seconds.
# TYPE process_start_time_seconds gauge
process_start_time_seconds 1.79314e+09
vllm:cache_config_info{{{labels}}} 1.0
vllm:num_requests_running{{{model}}} {running}
vllm:num_requests_waiting{{{model}}} {waiting}
vllm:prefix_cache_queries_total{{{model}}} 1200.0
vllm:prefix_cache_hits_total{{{model}}} 800.0
vllm:num_preemptions_total{{{model}}} 0.0
vllm:request_success_total{{{model},finished_reason="stop"}} 40.0
vllm:request_success_total{{{model},finished_reason="length"}} 2.0
{extra}"""


def models(max_model_len=4096, name=SERVED):
    return {
        "object": "list",
        "data": [{"id": name, "root": MODEL_ID, "max_model_len": max_model_len}],
    }


def read(text):
    with FakeVllm([]) as server:
        server.metrics_text = text
        return read_server_state(server.url + "/metrics")


def test_server_state_reads_start_time_cache_config_counters_and_requests():
    state = read(exposition(running=1.0, waiting=2.0))
    assert state.start_time_seconds == 1.79314e09
    assert state.cache_config["kv_cache_size_tokens"] == "7376"
    assert state.running == 1.0 and state.waiting == 2.0
    # Summed over label sets: request successes are split by finish reason.
    assert state.counters == {
        "vllm:prefix_cache_queries_total": 1200.0,
        "vllm:prefix_cache_hits_total": 800.0,
        "vllm:num_preemptions_total": 0.0,
        "vllm:request_success_total": 42.0,
    }
    # The analysis consumes the state in exactly this shape.
    assert set(asdict(state)) == {
        "start_time_seconds",
        "cache_config",
        "counters",
        "running",
        "waiting",
    }
    assert server_changes(asdict(state), asdict(state))["request_successes"] == 0


def test_a_missing_series_is_none_not_zero():
    text = "\n".join(
        line
        for line in exposition().splitlines()
        if not line.startswith(("vllm:num_preemptions_total", "vllm:num_requests_waiting"))
    )
    state = read(text)
    assert state.counters["vllm:num_preemptions_total"] is None
    assert state.waiting is None
    assert state.running == 0.0


def test_a_malformed_or_non_finite_series_is_none_not_a_reading():
    text = exposition().replace(
        f'vllm:num_requests_running{{engine="0",model_name="{SERVED}"}} 0.0',
        f'vllm:num_requests_running{{engine="0",model_name="{SERVED}"}} NaN',
    )
    text = text.replace("process_start_time_seconds 1.79314e+09", "process_start_time_seconds x")
    state = read(text)
    assert state.running is None
    assert state.start_time_seconds is None


def test_reported_settings_are_verified_and_max_num_seqs_is_not():
    state = read(exposition())
    result = verify_configuration(CONFIGURATIONS["cfg-default"], models(), state)
    settings = result["settings"]
    assert settings["max_model_len"] == {
        "declared": 4096,
        "reported": 4096,
        "status": "verified",
        "source": "vllm /v1/models",
    }
    assert settings["gpu_memory_utilization"]["status"] == "verified"
    assert settings["gpu_memory_utilization"]["reported"] == 0.6
    assert settings["enable_prefix_caching"]["status"] == "verified"
    assert settings["model"]["status"] == "verified"
    assert settings["max_num_seqs"]["status"] == "declared_not_verified"
    assert settings["max_num_seqs"]["reported"] is None
    assert result["contradicted"] == []
    assert result["kv_cache_size_tokens"] == 7376
    assert result["num_gpu_blocks"] == 461
    assert result["kv_cache_max_concurrency"] == 1.80078125
    assert result["block_size"] == 16


def test_a_contradicted_setting_is_reported():
    state = read(exposition(gpu_memory_utilization="0.9"))
    result = verify_configuration(CONFIGURATIONS["cfg-default"], models(max_model_len=8192), state)
    assert result["settings"]["gpu_memory_utilization"]["status"] == "contradicted"
    assert result["settings"]["max_model_len"]["status"] == "contradicted"
    assert result["contradicted"] == ["gpu_memory_utilization", "max_model_len"]


def test_a_fallback_memory_value_is_accepted_for_cfg_hardware():
    state = read(exposition(gpu_memory_utilization="0.75"))
    result = verify_configuration(CONFIGURATIONS["cfg-hardware"], models(), state)
    setting = result["settings"]["gpu_memory_utilization"]
    assert setting["status"] == "verified"
    assert setting["declared"] == [0.8, 0.75, 0.7]
    assert setting["reported"] == 0.75
    # 0.75 is a cfg-hardware fallback, never a cfg-default value.
    default = verify_configuration(CONFIGURATIONS["cfg-default"], models(), state)
    assert default["settings"]["gpu_memory_utilization"]["status"] == "contradicted"


def test_a_model_that_is_not_served_contradicts_the_declaration():
    result = verify_configuration(
        CONFIGURATIONS["cfg-default"], models(name="other/model"), read(exposition())
    )
    assert result["settings"]["model"]["status"] == "contradicted"
    assert result["settings"]["max_model_len"]["status"] == "declared_not_verified"
    assert "model" in result["contradicted"]


def test_unreported_cache_config_is_declared_not_verified_not_contradicted():
    state = ServerState(
        start_time_seconds=1.0, cache_config={}, counters={}, running=0.0, waiting=0.0
    )
    result = verify_configuration(CONFIGURATIONS["cfg-default"], models(), state)
    for name in ("gpu_memory_utilization", "enable_prefix_caching", "max_num_seqs"):
        assert result["settings"][name]["status"] == "declared_not_verified"
    assert result["kv_cache_size_tokens"] is None
    assert result["contradicted"] == []


def test_an_unreachable_metrics_endpoint_raises_server_state_unavailable():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    # Nothing listens on the port once the probe socket is closed.
    with pytest.raises(ServerStateUnavailable, match="metrics"):
        read_server_state(f"http://127.0.0.1:{port}/metrics", timeout_seconds=2.0)


def test_an_error_status_raises_server_state_unavailable():
    with FakeVllm([]) as server, pytest.raises(ServerStateUnavailable):
        read_server_state(server.url + "/not-metrics")

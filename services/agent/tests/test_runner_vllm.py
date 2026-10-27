"""The runner with a vLLM backend, against loopback and in-memory test doubles only."""

import json
from pathlib import Path

from agent_test_doubles import SOLVE_MEAN, ScriptedModel
from agentwatch_agent.config import RunLimits
from agentwatch_agent.model import VllmChatModel
from agentwatch_agent.runner import VLLM_CHANGE_LOG_FILE
from agentwatch_agent.service_probes import VLLM_METRICS
from test_runner import evidence, make_runner
from vllm_test_server import MODEL, FakeVllm, text_reply, tool_reply

SOLVE_MEAN_VLLM = [
    tool_reply("search_docs", '{"query": "mean of an empty input"}'),
    tool_reply("read_file", '{"path": "stats.py"}'),
    tool_reply("apply_patch", '{"patch_id": "return-none-for-empty"}'),
    tool_reply("run_tests", "{}"),
    text_reply("Applied return-none-for-empty; all 5 tests pass."),
]


def vllm_runner(task, server, fake, tmp_path, **overrides):
    model = VllmChatModel(fake.url, MODEL, num_ctx=4096)
    settings = {
        "vllm_traces_file": tmp_path / "otel" / "vllm-traces.jsonl",
        "trace_wait_seconds": 2.0,
        # 900 s at 0.06 s x 21 metrics would exceed the sample budget; 120 s fits.
        "limits": RunLimits(run_timeout_seconds=120),
    } | overrides
    return make_runner(task, server, model, tmp_path, **settings)


def test_a_vllm_run_joins_each_calls_measured_queue_and_samples_the_server(
    mean_task, server, tmp_path
):
    (tmp_path / "otel").mkdir()
    with FakeVllm(SOLVE_MEAN_VLLM, traces_file=tmp_path / "otel" / "vllm-traces.jsonl") as fake:
        outcome = vllm_runner(mean_task, server, fake, tmp_path).run()
    assert (outcome.state, outcome.task_passed, outcome.delivery_complete) == (
        "completed",
        True,
        True,
    )
    run = server.only_run()
    resources = {r.resource_id: r for r in run.registration.resources}
    model_call = run.steps["model-call-1"]
    assert resources[model_call.resource_id].service_id == "vllm"
    # Every request carried its own step's span as the parent of vLLM's request span.
    spans = [run.steps[f"model-call-{i}"].span_id for i in range(1, 6)]
    assert fake.traceparents == [f"00-{model_call.trace_id}-{span}-01" for span in spans]
    assert all(request["stream"] for request in fake.requests)

    saved = evidence(outcome)
    assert saved["model_backend"] == "vllm"
    assert saved["configuration"]["model_backend"] == "vllm"
    assert saved["configuration"]["model_transport"] == "stream"
    assert saved["server_traces"]["complete"] is True
    for call in saved["model_calls"]:
        timing = call["timing"]
        assert timing["status"] == "reported"
        assert timing["observed_queue_seconds"] == 0.002
        assert timing["observed_queue_availability"] == "measured"
        assert timing["server_trace"]["request_id"] == timing["completion_id"]
        assert call["stream_timing"]["time_to_first_output_seconds"] is not None
        assert call["model_placement"]["context_matches_num_ctx"] is True

    sampled = {
        record["metric"]
        for record in run.records.values()
        if record.get("record_type") == "resource_sample"
    }
    assert {entry.metric for entry in VLLM_METRICS} <= sampled
    service = saved["service_sampling"]
    assert service["availability_counts"]["inference.vllm.requests_running.v1"]["measured"] >= 1
    assert service["resources"][0]["resource_id"] == model_call.resource_id
    assert (Path(outcome.evidence_path).parent / VLLM_CHANGE_LOG_FILE).is_file()


def test_spans_that_never_arrive_leave_the_queue_unavailable(mean_task, server, tmp_path):
    (tmp_path / "otel").mkdir()
    with FakeVllm(SOLVE_MEAN_VLLM) as fake:  # no collector: nothing is exported
        outcome = vllm_runner(mean_task, server, fake, tmp_path, trace_wait_seconds=0.3).run()
    saved = evidence(outcome)
    assert outcome.state == "completed"
    assert saved["server_traces"]["complete"] is False
    for call in saved["model_calls"]:
        assert call["timing"]["observed_queue_seconds"] is None
        assert call["timing"]["observed_queue_availability"] == "unavailable"


def test_an_unusable_metrics_exposition_records_unavailable_readings(mean_task, server, tmp_path):
    with FakeVllm(SOLVE_MEAN_VLLM) as fake:
        fake.metrics_text = "not an exposition {"
        outcome = vllm_runner(mean_task, server, fake, tmp_path, vllm_traces_file=None).run()
    saved = evidence(outcome)
    assert outcome.state == "completed"
    counts = saved["service_sampling"]["availability_counts"]
    assert set(counts["inference.vllm.requests_running.v1"]) == {"unavailable"}
    assert saved["server_traces"] == {"status": "not_configured"}


def test_backend_and_transport_change_the_configuration_digest(mean_task, server, tmp_path):
    with FakeVllm([]) as fake:
        vllm = vllm_runner(mean_task, server, fake, tmp_path)._configuration()
    ollama = make_runner(mean_task, server, ScriptedModel(SOLVE_MEAN), tmp_path)._configuration()
    assert vllm.model_backend == "vllm" and ollama.model_backend == "ollama"
    assert vllm.digest() != ollama.digest()
    assert json.dumps(vllm.as_dict())  # serializable evidence

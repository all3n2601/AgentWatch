"""vLLM client, request translation and stream timing, against a loopback test double."""

import json

import pytest
from agentwatch_agent import vllm as vllm_module
from agentwatch_agent.clock import MonotonicClock
from agentwatch_agent.model import StreamTimer, VllmChatModel, joined_vllm_timing, parse_reply
from agentwatch_agent.vllm import (
    StreamAssembler,
    VllmResponseError,
    chat_request,
    openai_messages,
    parse_completion,
    served_model,
    traceparent,
    vllm_base_url,
    vllm_chat,
)
from vllm_test_server import MODEL, FakeVllm, text_reply, tool_reply

CANONICAL = [
    {"role": "system", "content": "sys"},
    {"role": "user", "content": "task"},
    {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {"function": {"name": "search_docs", "arguments": {"query": "mean"}}},
            {"function": {"name": "read_file", "arguments": {"path": "stats.py"}}},
        ],
    },
    {"role": "tool", "tool_name": "search_docs", "content": '{"results": []}'},
    {"role": "tool", "tool_name": "read_file", "content": '{"text": "x"}'},
    {"role": "assistant", "content": "done"},
]


def payload(messages=CANONICAL, **options):
    return {
        "model": MODEL,
        "messages": messages,
        "tools": [{"type": "function", "function": {"name": "search_docs"}}],
        "stream": False,
        "think": False,
        "options": {"temperature": 0, "num_ctx": 4096, "num_predict": 512} | options,
    }


def test_tool_results_answer_their_calls_by_deterministic_ids():
    translated = openai_messages(CANONICAL)
    calls = translated[2]["tool_calls"]
    assert [call["id"] for call in calls] == ["call-2-0", "call-2-1"]
    assert calls[0]["function"] == {"name": "search_docs", "arguments": '{"query": "mean"}'}
    assert translated[2]["content"] is None
    assert [m.get("tool_call_id") for m in translated[3:5]] == ["call-2-0", "call-2-1"]
    assert "tool_name" not in translated[3]
    # A replay of the same messages sends the same request.
    assert openai_messages(CANONICAL) == translated


@pytest.mark.parametrize(
    "messages",
    [
        [{"role": "tool", "tool_name": "x", "content": "{}"}],
        [CANONICAL[2], {"role": "user", "content": "interrupts"}],
        [CANONICAL[2], CANONICAL[3], {"role": "assistant", "content": "too early"}],
    ],
)
def test_histories_whose_results_do_not_match_their_calls_are_refused(messages):
    with pytest.raises(ValueError):
        openai_messages(messages)


def test_options_map_to_openai_fields_and_num_ctx_is_not_sent():
    body = chat_request(payload(), stream=True)
    assert body["temperature"] == 0 and body["max_tokens"] == 512
    assert "num_ctx" not in json.dumps(body)
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert body["stream"] is True and body["stream_options"] == {"include_usage": True}
    assert "stream_options" not in chat_request(payload(), stream=False)


def test_a_completion_comes_back_in_the_canonical_shape():
    document = FakeVllm.completion(tool_reply("search_docs", '{"query": "q"}'), "chatcmpl-1")
    response = parse_completion(document)
    assert response["message"]["tool_calls"] == [
        {"function": {"name": "search_docs", "arguments": {"query": "q"}}}
    ]
    reply = parse_reply(response)
    assert reply.calls[0].arguments == {"query": "q"}
    assert reply.metrics["vllm"]["id"] == "chatcmpl-1"


def test_arguments_that_are_not_json_are_passed_through_for_the_dispatcher():
    response = parse_completion(FakeVllm.completion(tool_reply("read_file", "{not json"), "c"))
    assert response["message"]["tool_calls"][0]["function"]["arguments"] == "{not json"


@pytest.mark.parametrize(
    "document", [[], {"choices": []}, {"choices": [{"message": "x"}]}, {"choices": ["x"]}]
)
def test_unusable_completions_are_refused(document):
    with pytest.raises(VllmResponseError):
        parse_completion(document)


def test_streamed_tool_call_fragments_are_joined_by_index():
    assembler = StreamAssembler()
    outputs = [
        assembler.add(chunk)
        for chunk in FakeVllm.chunks(tool_reply("read_file", '{"path": "stats.py"}'), "c-1")
    ]
    response = assembler.response()
    assert response["message"]["tool_calls"] == [
        {"function": {"name": "read_file", "arguments": {"path": "stats.py"}}}
    ]
    assert response["vllm"]["id"] == "c-1" and response["vllm"]["finish_reason"] == "stop"
    assert response["vllm"]["usage"] == {"prompt_tokens": 40, "total": 45}
    # The role-only first chunk and the usage chunk carry no output.
    assert outputs[0] is False and outputs[-1] is False and sum(outputs) == 3


def test_a_streamed_tool_call_without_an_index_is_refused():
    with pytest.raises(VllmResponseError, match="index"):
        StreamAssembler().add({"choices": [{"delta": {"tool_calls": [{"function": {}}]}}]})


def test_streaming_reports_events_and_sends_the_trace_context():
    events = []
    with FakeVllm([text_reply("all tests pass")]) as server:
        response = vllm_chat(
            server.url,
            payload(),
            timeout_seconds=5,
            trace_context=traceparent("a" * 32, "b" * 16),
            on_event=events.append,
        )
    assert response["message"]["content"] == "all tests pass "
    assert server.traceparents == [f"00-{'a' * 32}-{'b' * 16}-01"]
    assert events[0] == "response"
    assert events.count("output") == 3 and events.count("chunk") == 6


def test_without_an_observer_one_completion_is_requested():
    with FakeVllm([text_reply("answer")]) as server:
        response = vllm_chat(server.url, payload(), timeout_seconds=5)
    assert server.requests[0]["stream"] is False
    assert server.traceparents == [None]
    assert response["message"]["content"] == "answer"


def test_a_stream_without_its_done_marker_is_refused():
    with FakeVllm([text_reply("x")]) as server:
        server.omit_done = True
        with pytest.raises(VllmResponseError, match="before \\[DONE\\]"):
            vllm_chat(server.url, payload(), timeout_seconds=5, on_event=lambda _event: None)


def test_an_oversized_stream_is_refused(monkeypatch):
    monkeypatch.setattr(vllm_module, "MAX_STREAM_BYTES", 50)
    with (
        FakeVllm([text_reply("one two three four five six seven")]) as server,
        pytest.raises(VllmResponseError, match="exceeded 50 bytes"),
    ):
        vllm_chat(server.url, payload(), timeout_seconds=5, on_event=lambda _event: None)


def test_placement_reports_the_served_context_and_checks_num_ctx():
    with FakeVllm([]) as server:
        matching = VllmChatModel(server.url, MODEL, num_ctx=4096).observe_placement()
        different = VllmChatModel(server.url, MODEL, num_ctx=8192).observe_placement()
        missing = VllmChatModel(server.url, "not/served", num_ctx=4096).observe_placement()
    assert matching["status"] == "measured" and matching["max_model_len"] == 4096
    assert matching["context_matches_num_ctx"] is True
    assert different["context_matches_num_ctx"] is False
    assert missing["status"] == "unavailable" and "context_matches_num_ctx" not in missing


def test_an_unreachable_server_has_unavailable_placement():
    placement = VllmChatModel("http://127.0.0.1:1", MODEL).observe_placement()
    assert placement["status"] == "unavailable"


@pytest.mark.parametrize("models", [[], {"data": "x"}])
def test_malformed_model_lists_are_malformed(models):
    assert served_model(models, MODEL)["status"] == "malformed"


@pytest.mark.parametrize("url", ["127.0.0.1:8000", "ftp://host", "http://"])
def test_vllm_urls_must_be_http(url):
    with pytest.raises(ValueError):
        vllm_base_url(url)


def test_call_time_timing_waits_for_the_span_and_never_claims_a_queue():
    timing = VllmChatModel("http://x", MODEL).request_timing({"vllm": {"id": "c-1"}}, 0.5)
    assert timing["status"] == "pending_trace"
    assert timing["observed_queue_seconds"] is None
    assert timing["observed_queue_availability"] == "unavailable"
    assert timing["completion_id"] == "c-1"


def test_a_joined_span_supplies_the_measured_queue_and_service_time():
    call = VllmChatModel("http://x", MODEL).request_timing({"vllm": {"id": "c-1"}}, 0.04)
    joined = joined_vllm_timing(
        call, {"status": "measured", "queue_seconds": 0.002, "e2e_seconds": 0.033}
    )
    assert joined["observed_queue_seconds"] == 0.002
    assert joined["observed_queue_availability"] == "measured"
    assert joined["server_total_seconds"] == 0.033
    assert joined["client_outside_server_seconds"] == pytest.approx(0.007)
    assert joined["status"] == "reported"


def test_a_missing_span_leaves_the_queue_unavailable_with_its_reason():
    call = VllmChatModel("http://x", MODEL).request_timing({}, 0.04)
    joined = joined_vllm_timing(call, {"status": "unavailable", "problems": ["no span"]})
    assert joined["observed_queue_seconds"] is None
    assert joined["observed_queue_availability"] == "unavailable"
    assert joined["observed_queue_reason"] == "no span"
    assert joined["server_total_seconds"] is None


def test_server_time_beyond_the_runner_wall_gives_no_difference():
    call = VllmChatModel("http://x", MODEL).request_timing({}, 0.01)
    joined = joined_vllm_timing(call, {"status": "measured", "queue_seconds": 0, "e2e_seconds": 1})
    assert joined["client_outside_server_seconds"] is None


class StepClock(MonotonicClock):
    """Ticks advance 10 ms per reading; synthetic test timing."""

    def __init__(self):
        super().__init__("timer-clock")
        self.ticks = 0

    def tick(self):
        self.ticks += 10_000_000
        return self.ticks


def test_stream_timer_measures_first_output_on_the_runners_clock():
    timer = StreamTimer(StepClock())
    for event in ("request", "response", "chunk", "chunk", "output", "chunk", "output"):
        timer(event)
    summary = timer.summary()
    assert summary["clock_id"] == "timer-clock"
    assert summary["request_tick_ns"] == 10_000_000
    assert summary["time_to_response_seconds"] == pytest.approx(0.01)
    assert summary["time_to_first_chunk_seconds"] == pytest.approx(0.02)
    assert summary["time_to_first_output_seconds"] == pytest.approx(0.04)
    assert summary["output_offsets_seconds"] == [pytest.approx(0.04), pytest.approx(0.06)]
    assert summary["output_gap_median_seconds"] == pytest.approx(0.02)
    assert (summary["chunks"], summary["output_chunks"]) == (3, 2)
    assert summary["output_offsets_truncated"] is False


def test_a_request_that_never_answered_has_no_output_timing():
    timer = StreamTimer(StepClock())
    timer("request")
    summary = timer.summary()
    assert summary["time_to_first_output_seconds"] is None
    assert summary["output_span_seconds"] is None and summary["output_offsets_seconds"] == []


def test_recorded_output_ticks_are_bounded(monkeypatch):
    monkeypatch.setattr("agentwatch_agent.model.MAX_RECORDED_OUTPUT_TICKS", 2)
    timer = StreamTimer(StepClock())
    timer("request")
    for _ in range(5):
        timer("output")
    summary = timer.summary()
    assert summary["output_chunks"] == 5 and len(summary["output_offsets_seconds"]) == 2
    assert summary["output_offsets_truncated"] is True

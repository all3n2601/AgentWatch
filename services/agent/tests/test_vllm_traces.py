"""Joining vLLM request spans to model-call steps; spans are synthetic test fixtures."""

import json

import pytest
from agentwatch_agent import vllm_traces
from agentwatch_agent.vllm_traces import (
    collect_request_timings,
    join_spans,
    request_spans,
    span_timing,
    trace_files,
)
from vllm_test_server import span_line

TRACE = "1" * 32
OTHER_TRACE = "2" * 32
STEP_A, STEP_B = "a" * 16, "b" * 16


def spans_in(text: str) -> list[dict]:
    return [
        span
        for line in text.splitlines()
        for resource in json.loads(line)["resourceSpans"]
        for scope in resource["scopeSpans"]
        for span in scope["spans"]
    ]


def test_only_this_traces_request_spans_are_read(tmp_path):
    path = tmp_path / "vllm-traces.jsonl"
    startup = json.loads(span_line(TRACE, "", "x"))
    startup["resourceSpans"][0]["scopeSpans"][0]["spans"][0]["name"] = "Overall Loading"
    path.write_text(
        span_line(TRACE, STEP_A, "c-1")
        + span_line(OTHER_TRACE, STEP_B, "c-2")
        + json.dumps(startup)
        + "\n"
        + '{"resourceSpans": [ truncated line still being written '
        + TRACE
        + "\n"
    )
    spans = request_spans([path], TRACE)
    assert [span["parentSpanId"] for span in spans] == [STEP_A]


def test_rotated_backups_are_read_with_the_current_file(tmp_path):
    (tmp_path / "vllm-traces-2026-10-27T23-00-00.000.jsonl").write_text(
        span_line(TRACE, STEP_A, "c-1")
    )
    (tmp_path / "vllm-traces.jsonl").write_text(span_line(TRACE, STEP_B, "c-2"))
    (tmp_path / "unrelated.jsonl").write_text(span_line(TRACE, "c" * 16, "c-3"))
    files = trace_files(tmp_path, "vllm-traces.jsonl")
    assert len(files) == 2
    assert {s["parentSpanId"] for s in request_spans(files, TRACE)} == {STEP_A, STEP_B}


def test_trace_files_beyond_the_read_bound_are_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(vllm_traces, "MAX_TRACE_BYTES", 10)
    path = tmp_path / "vllm-traces.jsonl"
    path.write_text(span_line(TRACE, STEP_A, "c-1"))
    with pytest.raises(ValueError, match="exceed"):
        request_spans([path], TRACE)


def test_a_span_supplies_the_requests_measured_phases():
    timing = span_timing(spans_in(span_line(TRACE, STEP_A, "c-1"))[0], "c-1")
    assert timing["status"] == "measured" and timing["problems"] == []
    assert timing["queue_seconds"] == 0.002
    assert (timing["prefill_seconds"], timing["decode_seconds"]) == (0.01, 0.02)
    assert timing["e2e_seconds"] == 0.033 and timing["first_token_seconds"] == 0.012
    assert (timing["prompt_tokens"], timing["completion_tokens"]) == (40, 5)
    assert timing["request_id"] == "c-1"


def test_a_measured_zero_queue_stays_zero():
    timing = span_timing(spans_in(span_line(TRACE, STEP_A, "c-1", queue=0.0))[0], "c-1")
    assert timing["status"] == "measured" and timing["queue_seconds"] == 0.0


def test_a_span_for_a_different_request_is_malformed():
    timing = span_timing(spans_in(span_line(TRACE, STEP_A, "c-1"))[0], "c-9")
    assert timing["status"] == "malformed"
    assert timing["problems"] == ["span request ID differs from the reply's completion ID"]


@pytest.mark.parametrize("value", [{"doubleValue": -0.1}, {"stringValue": "1"}, {}])
def test_unusable_durations_are_malformed_not_converted(value):
    span = spans_in(span_line(TRACE, STEP_A, "c-1"))[0]
    span["attributes"][0]["value"] = value
    timing = span_timing(span, "c-1")
    assert timing["status"] == "malformed" and timing["queue_seconds"] is None


def test_missing_and_duplicate_spans_are_distinguished():
    spans = spans_in(span_line(TRACE, STEP_A, "c-1") + span_line(TRACE, STEP_A, "c-1"))
    joined = join_spans(spans, {STEP_A: "c-1", STEP_B: "c-2"})
    assert joined[STEP_A]["status"] == "malformed"
    assert joined[STEP_B]["status"] == "unavailable"


def test_collection_waits_for_late_spans_within_its_bound(tmp_path):
    path = tmp_path / "vllm-traces.jsonl"
    clock = {"now": 0.0}

    def sleep(seconds):
        clock["now"] += seconds
        path.write_text(span_line(TRACE, STEP_A, "c-1"))  # exported during the wait

    collected = collect_request_timings(
        path, TRACE, {STEP_A: "c-1"}, wait_seconds=5, sleep=sleep, monotonic=lambda: clock["now"]
    )
    assert collected["complete"] is True and collected["spans_found"] == 1
    assert clock["now"] == pytest.approx(vllm_traces.POLL_SECONDS)


def test_collection_gives_up_at_its_bound_and_reports_what_is_missing(tmp_path):
    clock = {"now": 0.0}

    def sleep(seconds):
        clock["now"] += seconds

    collected = collect_request_timings(
        tmp_path / "vllm-traces.jsonl",
        TRACE,
        {STEP_A: "c-1"},
        wait_seconds=1,
        sleep=sleep,
        monotonic=lambda: clock["now"],
    )
    assert collected["complete"] is False
    assert collected["calls"][STEP_A]["status"] == "unavailable"
    assert clock["now"] >= 1


def test_lines_that_are_not_otlp_objects_are_skipped(tmp_path):
    path = tmp_path / "vllm-traces.jsonl"
    path.write_text(
        f'["{TRACE}"]\n'
        f'{{"resourceSpans": ["{TRACE}", {{"scopeSpans": "x"}}]}}\n'
        + span_line(TRACE, STEP_A, "c-1")
    )
    assert [s["parentSpanId"] for s in request_spans([path], TRACE)] == [STEP_A]


def test_a_file_rotated_away_mid_read_is_skipped(tmp_path):
    missing = tmp_path / "vllm-traces-2026-10-27T23-00-00.000.jsonl"
    current = tmp_path / "vllm-traces.jsonl"
    current.write_text(span_line(TRACE, STEP_A, "c-1"))
    assert len(request_spans([missing, current], TRACE)) == 1

import argparse
import http.client
import io
import json
import threading
import urllib.error
from datetime import datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from agentwatch_worker import ollama
from agentwatch_worker.ollama import (
    model_placement,
    normalized_model_name,
    observe_model_placement,
    ollama_base_url,
)


def loaded(size=1000, size_vram=1000, name="qwen3:1.7b"):
    return {
        "models": [
            {
                "name": name,
                "model": name,
                "size": size,
                "size_vram": size_vram,
                "digest": "8f68893c685c",
                "details": {"quantization_level": "Q4_K_M"},
                "context_length": 4096,
            }
        ]
    }


@pytest.mark.parametrize(
    ("size_vram", "fraction"),
    [(1000, 1.0), (0, 0.0), (250, 0.25)],
)
def test_model_placement_reports_ollama_memory_split(size_vram, fraction):
    placement = model_placement(loaded(size_vram=size_vram), "qwen3:1.7b")
    assert placement["status"] == "measured"
    assert placement["gpu_memory_fraction"] == fraction
    assert placement["size_vram_bytes"] == size_vram
    assert placement["digest"] == "8f68893c685c"


def test_unloaded_model_is_unavailable_not_cpu():
    placement = model_placement({"models": []}, "qwen3:1.7b")
    assert placement["status"] == "unavailable"
    assert "gpu_memory_fraction" not in placement


@pytest.mark.parametrize(
    ("name", "normalized"),
    [
        ("qwen3", "qwen3:latest"),
        ("qwen3:1.7b", "qwen3:1.7b"),
        (" qwen3:1.7b ", "qwen3:1.7b"),
        ("library/qwen3:1.7b", "qwen3:1.7b"),
        ("registry.ollama.ai/library/qwen3", "qwen3:latest"),
        ("user/custom", "user/custom:latest"),
        ("localhost:5000/team/model", "localhost:5000/team/model:latest"),
        ("localhost:5000/team/model:v2", "localhost:5000/team/model:v2"),
    ],
)
def test_model_names_are_compared_in_ollamas_canonical_form(name, normalized):
    assert normalized_model_name(name) == normalized


@pytest.mark.parametrize("requested", ["qwen3", "qwen3:latest", "library/qwen3"])
def test_untagged_or_namespaced_model_matches_its_listed_name(requested):
    placement = model_placement(loaded(name="qwen3:latest"), requested)
    assert placement["status"] == "measured"


def test_different_tag_of_the_same_model_is_not_matched():
    placement = model_placement(loaded(name="qwen3:latest"), "qwen3:1.7b")
    assert placement["status"] == "unavailable"


def test_non_string_listed_names_are_ignored():
    response = loaded()
    response["models"].insert(0, {"name": None, "model": 7, "size": 1, "size_vram": 1})
    assert model_placement(response, "qwen3:1.7b")["status"] == "measured"


@pytest.mark.parametrize(
    "response",
    [
        {},
        [],
        loaded(size=0, size_vram=0),
        loaded(size_vram=2000),
        loaded(size_vram=-1),
        loaded(size_vram=True),
        loaded(size="1000"),
    ],
)
def test_inconsistent_placement_is_malformed(response):
    placement = model_placement(response, "qwen3:1.7b")
    assert placement["status"] == "malformed"
    assert "gpu_memory_fraction" not in placement


@pytest.mark.parametrize(
    ("failure", "reported"),
    [
        (urllib.error.URLError("connection refused"), "connection refused"),
        (ConnectionResetError("connection reset"), "connection reset"),
        (http.client.IncompleteRead(b"partial"), "IncompleteRead"),
    ],
)
def test_placement_request_failure_is_recorded_as_unavailable(failure, reported):
    def unreachable():
        raise failure

    placement = observe_model_placement(unreachable, "qwen3:1.7b")
    assert placement["status"] == "unavailable"
    assert placement["reason"].startswith("placement request failed:")
    assert reported in placement["reason"]
    assert "gpu_memory_fraction" not in placement
    assert datetime.fromisoformat(placement["observed_at"]).utcoffset() == timedelta(0)
    assert placement["read_seconds"] >= 0


def respond_with(body: bytes):
    def urlopen(*_args, **_kwargs):
        return io.BytesIO(body)

    return urlopen


@pytest.mark.parametrize("body", [b"\xff", b"not json", b'{"models": '])
def test_undecodable_placement_response_is_malformed(monkeypatch, body):
    monkeypatch.setattr(ollama.urllib.request, "urlopen", respond_with(body))
    placement = observe_model_placement(
        lambda: ollama.ollama_loaded_models("http://ollama.test"), "qwen3:1.7b"
    )
    assert placement["status"] == "malformed"
    assert placement["reason"].startswith("placement response is not valid UTF-8 JSON:")
    assert "gpu_memory_fraction" not in placement


def test_placement_programming_errors_are_not_contained():
    def broken():
        raise KeyError("models")

    with pytest.raises(KeyError):
        observe_model_placement(broken, "qwen3:1.7b")


@pytest.mark.parametrize(
    "url", ["127.0.0.1:11434", "ftp://127.0.0.1:11434", "http://", "localhost", ""]
)
def test_ollama_url_must_be_http(url):
    with pytest.raises(argparse.ArgumentTypeError):
        ollama_base_url(url)


@pytest.mark.parametrize("url", ["http://127.0.0.1:11435", "https://ollama.example:443/"])
def test_valid_ollama_urls_are_accepted(url):
    assert ollama_base_url(url) == url


# Values below are illustrative nanosecond counts, not recorded evidence.
REPORTED = {
    "total_duration": 540_000_000,
    "load_duration": 40_000_000,
    "prompt_eval_duration": 150_000_000,
    "eval_duration": 330_000_000,
    "eval_count": 60,
}


def test_request_timing_separates_reported_service_time_from_an_unobserved_queue():
    timing = ollama.request_timing(REPORTED, client_wall_seconds=0.6)
    assert timing["status"] == "reported"
    assert timing["observed_queue_seconds"] is None
    assert timing["observed_queue_availability"] == "unavailable"
    assert timing["server_total_seconds"] == pytest.approx(0.54)
    assert timing["server_prompt_eval_seconds"] == pytest.approx(0.15)
    assert timing["server_unattributed_seconds"] == pytest.approx(0.02)
    assert timing["client_outside_server_seconds"] == pytest.approx(0.06)
    assert timing["problems"] == []


def test_missing_durations_leave_their_differences_null():
    metrics = {key: value for key, value in REPORTED.items() if key != "load_duration"}
    timing = ollama.request_timing(metrics, client_wall_seconds=0.6)
    assert timing["status"] == "partial"
    assert timing["server_load_seconds"] is None
    assert timing["server_unattributed_seconds"] is None
    # The total alone still bounds the runner's time outside the server.
    assert timing["client_outside_server_seconds"] == pytest.approx(0.06)
    assert timing["problems"] == ["load_duration missing"]


@pytest.mark.parametrize("value", [-1, 1.5, True, "10", None])
def test_invalid_durations_are_not_converted(value):
    timing = ollama.request_timing(REPORTED | {"eval_duration": value}, client_wall_seconds=0.6)
    assert timing["status"] == "partial"
    assert timing["server_eval_seconds"] is None
    assert timing["problems"] == ["eval_duration is not a nonnegative integer"]


def test_impossible_duration_sums_are_inconsistent_not_clamped():
    oversized = REPORTED | {"eval_duration": 900_000_000}
    timing = ollama.request_timing(oversized, client_wall_seconds=0.5)
    assert timing["status"] == "inconsistent"
    assert timing["server_unattributed_seconds"] is None
    # Ollama's 0.54 s total cannot fit inside a 0.5 s request observed by the runner.
    assert timing["client_outside_server_seconds"] is None
    assert len(timing["problems"]) == 2


def test_differences_within_the_timing_tolerance_are_kept():
    timing = ollama.request_timing(REPORTED, client_wall_seconds=0.54 - 5e-7)
    assert timing["status"] == "reported"
    assert timing["client_outside_server_seconds"] == pytest.approx(-5e-7)


class StreamingOllama:
    """Loopback /api/chat test double that streams NDJSON chunks; synthetic, never evidence."""

    def __init__(self, chunks: list[dict]):
        owner = self
        self.payloads: list[dict] = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                owner.payloads.append(
                    json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                )
                self.send_response(200)
                self.send_header("Content-Type", "application/x-ndjson")
                self.end_headers()
                for chunk in chunks:
                    self.wfile.write((json.dumps(chunk) + "\n").encode())

            def log_message(self, *_):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.server.shutdown()
        self.server.server_close()


FINAL_CHUNK = {
    "message": {"role": "assistant", "content": ""},
    "done": True,
    "done_reason": "stop",
    "total_duration": 3_000_000,
    "load_duration": 1_000,
    "prompt_eval_duration": 1_000_000,
    "eval_duration": 1_500_000,
    "eval_count": 3,
}


def test_a_streamed_chat_is_returned_in_the_single_reply_shape():
    call = {"function": {"name": "read_file", "arguments": {"path": "stats.py"}}}
    chunks = [
        {"message": {"role": "assistant", "content": "Read"}, "done": False},
        {"message": {"role": "assistant", "content": " it"}, "done": False},
        {"message": {"role": "assistant", "content": "", "tool_calls": [call]}, "done": False},
        FINAL_CHUNK,
    ]
    events = []
    with StreamingOllama(chunks) as server:
        response = ollama.ollama_chat_stream(
            server.url, {"model": "m", "stream": False}, events.append
        )
    assert server.payloads[0]["stream"] is True
    assert response["message"] == {"role": "assistant", "content": "Read it", "tool_calls": [call]}
    assert response["total_duration"] == 3_000_000 and response["eval_count"] == 3
    assert events == ["response", "chunk", "output", "chunk", "output", "chunk", "output", "chunk"]
    timing = ollama.request_timing(response, 0.004)
    assert timing["status"] == "reported" and timing["server_total_seconds"] == 0.003


def test_a_stream_that_ends_without_its_final_chunk_is_refused():
    chunks = [{"message": {"role": "assistant", "content": "partial"}, "done": False}]
    with (
        StreamingOllama(chunks) as server,
        pytest.raises(RuntimeError, match="without a final chunk"),
    ):
        ollama.ollama_chat_stream(server.url, {"model": "m"}, lambda _event: None)


def test_a_stream_error_chunk_is_refused():
    with (
        StreamingOllama([{"error": "model not found"}]) as server,
        pytest.raises(RuntimeError, match="Invalid Ollama stream chunk"),
    ):
        ollama.ollama_chat_stream(server.url, {"model": "m"}, lambda _event: None)

import math
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from agentwatch_agent import prometheus_text
from agentwatch_agent.prometheus_text import ExpositionError, fetch_exposition, parse_exposition

# Synthetic exposition text for parser tests; not recorded evidence.
EXPOSITION = """\
# HELP vllm:num_requests_waiting Number of requests waiting to be processed.
# TYPE vllm:num_requests_waiting gauge
vllm:num_requests_waiting{engine="0",model_name="m@1"} 2.0
vllm:request_queue_time_seconds_sum{engine="0",model_name="m@1"} 7.4e-05 1793142726000
plain_metric 3
"""


def test_series_keep_names_labels_and_values():
    exposition = parse_exposition(EXPOSITION)
    assert exposition.unparsed_lines == 0
    waiting = exposition.named("vllm:num_requests_waiting")
    assert [(s.label("model_name"), s.value) for s in waiting] == [("m@1", 2.0)]
    # A trailing timestamp is allowed and ignored.
    assert exposition.named("vllm:request_queue_time_seconds_sum")[0].value == 7.4e-05
    assert exposition.named("plain_metric")[0].labels == ()


def test_label_values_are_unescaped():
    exposition = parse_exposition('m{a="x\\"y",b="line\\nnext",c="back\\\\slash",d="{,}"} 1\n')
    series = exposition.named("m")[0]
    assert series.label("a") == 'x"y'
    assert series.label("b") == "line\nnext"
    assert series.label("c") == "back\\slash"
    assert series.label("d") == "{,}"


def test_non_finite_values_are_kept_for_the_caller_to_judge():
    exposition = parse_exposition("a NaN\nb +Inf\nc -Inf\n")
    assert math.isnan(exposition.named("a")[0].value)
    assert exposition.named("b")[0].value == math.inf
    assert exposition.named("c")[0].value == -math.inf


@pytest.mark.parametrize(
    "line",
    [
        "1bad 3",
        'm{a="unterminated} 1',
        "m{a=unquoted} 1",
        'm{a="x\\q"} 1',
        "m not-a-number",
        "m 1 2 3",
        "m",
    ],
)
def test_unparsable_lines_are_counted_not_guessed(line):
    exposition = parse_exposition(f"good 1\n{line}\n")
    assert exposition.unparsed_lines == 1
    assert [s.name for s in exposition.series] == ["good"]


def test_too_many_lines_are_refused(monkeypatch):
    monkeypatch.setattr(prometheus_text, "MAX_EXPOSITION_LINES", 3)
    with pytest.raises(ExpositionError, match="more than 3 lines"):
        parse_exposition("a 1\nb 2\nc 3\nd 4\n")


class MetricsServer:
    """Loopback /metrics test double; its bodies are synthetic."""

    def __init__(self, body: bytes, status: int = 200):
        owner = self
        self.body, self.status = body, status

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(owner.status)
                self.send_header("Content-Length", str(len(owner.body)))
                self.end_headers()
                self.wfile.write(owner.body)

            def log_message(self, *_):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/metrics"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.server.shutdown()
        self.server.server_close()


def test_fetch_reads_a_bounded_exposition():
    with MetricsServer(EXPOSITION.encode()) as server:
        exposition = fetch_exposition(server.url, timeout_seconds=2)
    assert len(exposition.series) == 3


def test_an_oversized_exposition_is_refused(monkeypatch):
    monkeypatch.setattr(prometheus_text, "MAX_EXPOSITION_BYTES", 10)
    with (
        MetricsServer(b"metric_name 1\n") as server,
        pytest.raises(ExpositionError, match="larger than 10 bytes"),
    ):
        fetch_exposition(server.url, timeout_seconds=2)


def test_a_non_utf8_exposition_is_refused():
    with MetricsServer(b"m 1\n\xff\n") as server, pytest.raises(ExpositionError, match="not UTF-8"):
        fetch_exposition(server.url, timeout_seconds=2)


def test_an_http_error_propagates_as_a_transport_failure():
    with MetricsServer(b"down", status=503) as server, pytest.raises(OSError):
        fetch_exposition(server.url, timeout_seconds=2)

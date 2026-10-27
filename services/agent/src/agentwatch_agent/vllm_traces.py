"""Join vLLM's per-request spans to the run's model-call steps (`vllm-request-trace-v1`).

The runner sends each model call with its step's trace context; vLLM records one
`llm_request` span per request whose parent is that step's span, with its own measured queue,
prefill, decode and end-to-end time. The OpenTelemetry Collector appends those spans as OTLP
JSON lines to a file (deploy/compose/otel-collector.yaml). Spans join a step only by the run's
trace ID and the step's span ID, never by time.

The durations are vLLM's engine timestamps for that one request: measured by the server,
not the runner, so they are not compared with runner ticks. A span's wall times are vLLM's
host clock, kept as recorded. A missing span is `unavailable`; two spans for one step, a span
whose request ID differs from the reply's, or an unusable duration is `malformed`.
"""

import json
import math
import time
from collections.abc import Callable, Iterable
from pathlib import Path

from agentwatch_agent.vllm import REQUEST_TIMING_METHOD

SPAN_NAME = "llm_request"
# vLLM attribute -> evidence field, all seconds (vLLM v0.32.0 output_processor.do_tracing).
DURATIONS = {
    "gen_ai.latency.time_in_queue": "queue_seconds",
    "gen_ai.latency.time_in_model_prefill": "prefill_seconds",
    "gen_ai.latency.time_in_model_decode": "decode_seconds",
    "gen_ai.latency.time_in_model_inference": "inference_seconds",
    "gen_ai.latency.time_to_first_token": "first_token_seconds",
    "gen_ai.latency.e2e": "e2e_seconds",
}
COUNTS = {
    "gen_ai.usage.prompt_tokens": "prompt_tokens",
    "gen_ai.usage.completion_tokens": "completion_tokens",
}
# Bound the read: the collector rotates at 64 MB with 3 backups (otel-collector.yaml).
MAX_TRACE_BYTES = 320 * 1024 * 1024
POLL_SECONDS = 0.25


def _value(attribute: dict) -> object:
    value = attribute.get("value") if isinstance(attribute, dict) else None
    if not isinstance(value, dict):
        return None
    for kind in ("doubleValue", "stringValue", "boolValue"):
        if kind in value:
            return value[kind]
    if "intValue" in value:
        # OTLP JSON encodes 64-bit integers as strings.
        try:
            return int(value["intValue"])
        except (TypeError, ValueError):
            return None
    return None


def trace_files(directory: Path, name: str) -> list[Path]:
    """The current trace file and its rotated backups (`<stem>-<timestamp><suffix>`)."""
    stem, suffix = Path(name).stem, Path(name).suffix
    return sorted(p for p in directory.glob(f"{stem}*{suffix}") if p.is_file())


def request_spans(paths: Iterable[Path], trace_id: str) -> list[dict]:
    """Every `llm_request` span of one trace, read from OTLP JSON lines."""
    spans, budget = [], MAX_TRACE_BYTES
    for path in paths:
        try:
            size = path.stat().st_size
        except FileNotFoundError:
            continue  # rotated away between listing and reading
        budget -= size
        if budget < 0:
            raise ValueError(f"trace files exceed {MAX_TRACE_BYTES} bytes")
        with path.open(encoding="utf-8", errors="replace") as stream:
            for line in stream:
                # Cheap filter first: most lines belong to other runs.
                if trace_id not in line:
                    continue
                try:
                    document = json.loads(line)
                except json.JSONDecodeError:
                    continue  # a line still being written
                for resource in document.get("resourceSpans") or []:
                    for scope in resource.get("scopeSpans") or []:
                        for span in scope.get("spans") or []:
                            if span.get("traceId") == trace_id and span.get("name") == SPAN_NAME:
                                spans.append(span)
    return spans


def span_timing(span: dict, response_id: object) -> dict:
    attributes = {a.get("key"): _value(a) for a in span.get("attributes") or []}
    timing: dict = {
        "method": REQUEST_TIMING_METHOD,
        "span_id": span.get("spanId"),
        "request_id": attributes.get("gen_ai.request.id"),
        "server_start_unix_nano": span.get("startTimeUnixNano"),
        "server_end_unix_nano": span.get("endTimeUnixNano"),
        "clock": "vLLM engine timestamps for this request; not comparable with runner ticks",
    }
    problems = []
    for key, field in DURATIONS.items():
        value = attributes.get(key)
        valid = isinstance(value, float | int) and not isinstance(value, bool)
        if not valid or not math.isfinite(value) or value < 0:
            problems.append(f"{key} missing or not a nonnegative number")
            value = None
        timing[field] = value
    for key, field in COUNTS.items():
        value = attributes.get(key)
        timing[field] = value if isinstance(value, int) and value >= 0 else None
    if response_id is not None and timing["request_id"] != response_id:
        problems.append("span request ID differs from the reply's completion ID")
    timing["status"] = "malformed" if problems else "measured"
    timing["problems"] = problems
    return timing


def join_spans(spans: list[dict], calls: dict[str, object]) -> dict[str, dict]:
    """Per model-call span ID (the step), its vLLM timing or why there is none."""
    by_parent: dict[str, list[dict]] = {}
    for span in spans:
        by_parent.setdefault(span.get("parentSpanId"), []).append(span)
    joined = {}
    for span_id, response_id in calls.items():
        found = by_parent.get(span_id, [])
        if not found:
            joined[span_id] = {
                "method": REQUEST_TIMING_METHOD,
                "status": "unavailable",
                "problems": ["no vLLM request span names this step as its parent"],
            }
        elif len(found) > 1:
            joined[span_id] = {
                "method": REQUEST_TIMING_METHOD,
                "status": "malformed",
                "problems": [f"{len(found)} vLLM request spans name this step as their parent"],
            }
        else:
            joined[span_id] = span_timing(found[0], response_id)
    return joined


def collect_request_timings(
    traces_file: Path,
    trace_id: str,
    calls: dict[str, object],
    *,
    wait_seconds: float,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
) -> dict:
    """Wait (bounded) for every call's span to be exported, then join what arrived.

    `calls` maps each model-call step's span ID to its reply's completion ID (or None when
    the request never returned one). Spans arrive after the request ends: vLLM exports within
    0.5 s and the collector flushes within 0.2 s (Compose configuration).
    """
    deadline = monotonic() + wait_seconds
    while True:
        spans = request_spans(trace_files(traces_file.parent, traces_file.name), trace_id)
        joined = join_spans(spans, calls)
        complete = all(entry["status"] != "unavailable" for entry in joined.values())
        if complete or monotonic() >= deadline:
            return {
                "method": REQUEST_TIMING_METHOD,
                "traces_file": traces_file.name,
                "spans_found": len(spans),
                "complete": complete,
                "calls": joined,
            }
        sleep(POLL_SECONDS)

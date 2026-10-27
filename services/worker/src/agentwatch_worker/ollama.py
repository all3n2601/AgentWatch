"""Ollama HTTP client helpers shared by the agent runner and the GPU diagnostics.

Placement readings describe where Ollama reports the model's memory, not GPU utilization or
saturation. Server-reported durations are Ollama's nanoseconds, kept unconverted.
"""

import argparse
import http.client
import json
import time
import urllib.parse
import urllib.request
from collections.abc import Callable
from datetime import UTC, datetime

# Bounds each chat request's socket waits; generation is further bounded by num_predict.
CHAT_TIMEOUT_SECONDS = 300
# Server-reported chat response fields kept as evidence.
OLLAMA_METRIC_KEYS = (
    "total_duration",
    "load_duration",
    "prompt_eval_count",
    "prompt_eval_duration",
    "eval_count",
    "eval_duration",
    "done_reason",
)
REQUEST_TIMING_METHOD = "ollama-request-timing-v1"
# Ollama's documented per-request durations, in nanoseconds.
SERVER_DURATION_KEYS = {
    "server_total_seconds": "total_duration",
    "server_load_seconds": "load_duration",
    "server_prompt_eval_seconds": "prompt_eval_duration",
    "server_eval_seconds": "eval_duration",
}
# The live contracts' timing-breakdown tolerance.
TIMING_TOLERANCE_SECONDS = 1e-6
QUEUE_UNAVAILABLE_REASON = (
    "Ollama reports no request queue wait; any wait is inside its total or load duration"
)
PLACEMENT_TIMEOUT_SECONDS = 10
# Order matters: the registry prefix can precede the namespace.
DEFAULT_MODEL_PREFIXES = ("registry.ollama.ai/", "library/")


def ollama_base_url(value: str) -> str:
    """Validate the --url/OLLAMA_URL boundary so later requests cannot fail on its form."""
    parsed = urllib.parse.urlsplit(value)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise argparse.ArgumentTypeError(
            f"expected an http(s) URL such as http://127.0.0.1:11434, got {value!r}"
        )
    return value


def ollama_chat(url: str, payload: dict) -> dict:
    request = urllib.request.Request(
        url.rstrip("/") + "/api/chat",
        data=json.dumps(payload, allow_nan=False).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=CHAT_TIMEOUT_SECONDS) as response:
        result = json.load(response)
    if result.get("error") or not isinstance(result.get("message"), dict):
        raise RuntimeError(f"Invalid Ollama response: {result}")
    return result


def ollama_chat_stream(url: str, payload: dict, on_event: Callable[[str], None]) -> dict:
    """Stream one chat request and return it in the non-streamed response shape.

    `on_event` receives "response" when the reply's headers arrive, "chunk" for each NDJSON
    chunk and "output" when a chunk carries text or tool calls, so the caller can time the
    first output on its own clock. The final chunk carries Ollama's durations.
    """
    request = urllib.request.Request(
        url.rstrip("/") + "/api/chat",
        data=json.dumps(payload | {"stream": True}, allow_nan=False).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    content: list[str] = []
    calls: list = []
    final: dict | None = None
    with urllib.request.urlopen(request, timeout=CHAT_TIMEOUT_SECONDS) as response:
        on_event("response")
        for raw in response:
            if not raw.strip():
                continue
            chunk = json.loads(raw)
            if not isinstance(chunk, dict) or chunk.get("error"):
                raise RuntimeError(f"Invalid Ollama stream chunk: {chunk}")
            message = chunk.get("message") if isinstance(chunk.get("message"), dict) else {}
            text, chunk_calls = message.get("content"), message.get("tool_calls")
            if isinstance(text, str) and text:
                content.append(text)
            if isinstance(chunk_calls, list):
                calls.extend(chunk_calls)
            on_event("chunk")
            if (isinstance(text, str) and text) or chunk_calls:
                on_event("output")
            if chunk.get("done") is True:
                final = chunk
                break
    if final is None:
        raise RuntimeError("Ollama stream ended without a final chunk")
    message = {"role": "assistant", "content": "".join(content)}
    if calls:
        message["tool_calls"] = calls
    return {key: value for key, value in final.items() if key != "message"} | {"message": message}


def _nonnegative_or_none(value: float) -> float | None:
    return value if value >= -TIMING_TOLERANCE_SECONDS else None


def request_timing(metrics: dict, client_wall_seconds: float) -> dict:
    """Split one chat request's latency into what was observed and what Ollama reported.

    The runner observes only its own wall time around the request. Ollama reports its total,
    model-load, prompt-evaluation and generation durations but no queue wait, so the observed
    queue stays unavailable instead of being inferred. Two derived differences are estimates
    across two clocks (the runner's and Ollama's): server time outside load, prompt and
    generation, and runner time outside Ollama's total (HTTP, JSON and thread scheduling).
    A difference below zero beyond the timing tolerance is reported as inconsistent.
    """
    timing: dict = {
        "method": REQUEST_TIMING_METHOD,
        "client_wall_seconds": client_wall_seconds,
        "observed_queue_seconds": None,
        "observed_queue_availability": "unavailable",
        "observed_queue_reason": QUEUE_UNAVAILABLE_REASON,
    }
    problems, inconsistent = [], False
    for field, key in SERVER_DURATION_KEYS.items():
        value = metrics.get(key)
        if key not in metrics:
            problems.append(f"{key} missing")
            value = None
        elif isinstance(value, bool) or not isinstance(value, int) or value < 0:
            problems.append(f"{key} is not a nonnegative integer")
            value = None
        timing[field] = None if value is None else value / 1e9
    parts = [timing[field] for field in SERVER_DURATION_KEYS]
    total, *components = parts
    unattributed = outside = None
    if None not in parts:
        unattributed = _nonnegative_or_none(total - sum(components))
        if unattributed is None:
            inconsistent = True
            problems.append("load, prompt and generation exceed Ollama's total duration")
    if total is not None:
        outside = _nonnegative_or_none(client_wall_seconds - total)
        if outside is None:
            inconsistent = True
            problems.append("Ollama's total duration exceeds the runner's request wall time")
    timing |= {
        "server_unattributed_seconds": unattributed,
        "client_outside_server_seconds": outside,
        "derived_differences": "estimates across runner and Ollama clocks",
        "status": "inconsistent" if inconsistent else ("partial" if problems else "reported"),
        "problems": problems,
    }
    return timing


def ollama_loaded_models(url: str) -> dict:
    with urllib.request.urlopen(
        url.rstrip("/") + "/api/ps", timeout=PLACEMENT_TIMEOUT_SECONDS
    ) as response:
        return json.load(response)


def normalized_model_name(name: str) -> str:
    """Return Ollama's canonical form of a model reference for comparison.

    Ollama resolves an untagged name to `:latest` and omits its default registry and
    `library/` namespace, so `qwen3` is listed by /api/ps as `qwen3:latest`.
    """
    name = name.strip()
    for prefix in DEFAULT_MODEL_PREFIXES:
        name = name.removeprefix(prefix)
    # A colon in the final path segment is a tag; one before a slash is a registry port.
    if ":" not in name.rsplit("/", 1)[-1]:
        name += ":latest"
    return name


def model_placement(loaded: dict, model: str) -> dict:
    """Summarize where Ollama reports the model's memory, from one /api/ps response.

    Ollama's own "100% GPU" display is size_vram / size and assumes size_vram <= size, so a
    larger value is treated as malformed. This is model placement, not GPU utilization or
    saturation.
    """
    source = {"source": "ollama /api/ps"}
    models = loaded.get("models") if isinstance(loaded, dict) else None
    if not isinstance(models, list):
        return source | {"status": "malformed", "reason": "response has no models list"}
    wanted = normalized_model_name(model)
    entry = next(
        (
            item
            for item in models
            if isinstance(item, dict)
            and any(
                isinstance(listed, str) and normalized_model_name(listed) == wanted
                for listed in (item.get("name"), item.get("model"))
            )
        ),
        None,
    )
    if entry is None:
        return source | {"status": "unavailable", "reason": "model is not currently loaded"}
    size, size_vram = entry.get("size"), entry.get("size_vram")
    valid = all(
        isinstance(value, int) and not isinstance(value, bool) for value in (size, size_vram)
    )
    if not valid or size <= 0 or not 0 <= size_vram <= size:
        return source | {
            "status": "malformed",
            "reason": "size/size_vram are not consistent byte counts",
        }
    details = entry.get("details") if isinstance(entry.get("details"), dict) else {}
    return source | {
        "status": "measured",
        "size_bytes": size,
        "size_vram_bytes": size_vram,
        "gpu_memory_fraction": size_vram / size,
        "digest": entry.get("digest"),
        "quantization_level": details.get("quantization_level"),
        "context_length": entry.get("context_length"),
    }


def observe_model_placement(fetch_loaded: Callable[[], dict], model: str) -> dict:
    observed_at = datetime.now(UTC).isoformat()
    started = time.monotonic_ns()
    # Only transport and response-decoding failures are contained: placement is ancillary
    # evidence, so a failed reading must not change a run's outcome or hide its own error.
    # Programming errors still propagate.
    try:
        placement = model_placement(fetch_loaded(), model)
    except (OSError, http.client.HTTPException) as error:
        placement = {
            "source": "ollama /api/ps",
            "status": "unavailable",
            "reason": f"placement request failed: {error}",
        }
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        placement = {
            "source": "ollama /api/ps",
            "status": "malformed",
            "reason": f"placement response is not valid UTF-8 JSON: {error}",
        }
    return placement | {
        "observed_at": observed_at,
        "read_seconds": (time.monotonic_ns() - started) / 1e9,
    }

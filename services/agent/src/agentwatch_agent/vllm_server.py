"""Read vLLM's server state from `/metrics` and verify it against a declared configuration.

The capacity sweep reads this state before and after every batch: the process start time
(a change means vLLM restarted), the cache configuration it reports, the cumulative counters
whose increases are recorded per batch, and the running and waiting request counts the idle
wait polls. A series that is missing, malformed or non-finite is None, never 0.

vLLM v0.32 reports its cache configuration as the labels of `vllm:cache_config_info`; it does
not report `max_num_seqs`. That is verified from the running container's launch arguments
(`vllm_launch`) when Docker is reachable, and is otherwise `declared_not_verified`.
"""

import http.client
import math
from dataclasses import dataclass

from agentwatch_agent.prometheus_text import Exposition, ExpositionError, fetch_exposition
from agentwatch_agent.sweep_protocol import (
    METRICS_READ_TIMEOUT_SECONDS,
    MODEL_ID,
    MODEL_REVISION,
    VllmConfiguration,
)
from agentwatch_agent.vllm import served_model
from agentwatch_agent.vllm_launch import READ

START_TIME_METRIC = "process_start_time_seconds"
CACHE_CONFIG_METRIC = "vllm:cache_config_info"
RUNNING_METRIC = "vllm:num_requests_running"
WAITING_METRIC = "vllm:num_requests_waiting"
COUNTER_METRICS = (
    "vllm:prefix_cache_queries_total",
    "vllm:prefix_cache_hits_total",
    "vllm:num_preemptions_total",
    "vllm:request_success_total",
)
DEFAULT_TIMEOUT_SECONDS = METRICS_READ_TIMEOUT_SECONDS  # declared, and hashed, by the protocol
SERVED_MODEL_NAME = f"{MODEL_ID}@{MODEL_REVISION}"
METRICS_SOURCE = "vllm /metrics vllm:cache_config_info"
LAUNCH_SOURCE = "docker inspect of the vLLM container's launch arguments"
NOT_INSPECTED = {"status": "not_configured", "reason": "no vLLM container was inspected"}

VERIFIED = "verified"
CONTRADICTED = "contradicted"
DECLARED_NOT_VERIFIED = "declared_not_verified"


class ServerStateUnavailable(RuntimeError):
    """vLLM's `/metrics` could not be fetched or read as an exposition."""


@dataclass(frozen=True)
class ServerState:
    start_time_seconds: float | None
    cache_config: dict[str, str]
    counters: dict[str, float | None]
    running: float | None
    waiting: float | None


def _total(exposition: Exposition, name: str) -> float | None:
    """The sum over a metric's label sets; None if it is absent, malformed or non-finite."""
    if name in exposition.unparsed_names:
        return None
    values = [series.value for series in exposition.named(name)]
    if not values or not all(math.isfinite(value) for value in values):
        return None
    return sum(values)


def _start_time(exposition: Exposition) -> float | None:
    if START_TIME_METRIC in exposition.unparsed_names:
        return None
    series = exposition.named(START_TIME_METRIC)
    # One process exports one start time; several would leave a restart undecidable.
    if len(series) != 1 or not math.isfinite(series[0].value):
        return None
    return series[0].value


def read_server_state(
    metrics_url: str, timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
) -> ServerState:
    try:
        exposition = fetch_exposition(metrics_url, timeout_seconds)
    # URLError, HTTPError and socket timeouts are OSErrors.
    except (OSError, http.client.HTTPException, ExpositionError) as error:
        raise ServerStateUnavailable(
            f"vLLM metrics at {metrics_url} unavailable: {type(error).__name__}: {error}"[:500]
        ) from error
    cache = exposition.named(CACHE_CONFIG_METRIC)
    return ServerState(
        start_time_seconds=_start_time(exposition),
        cache_config=dict(cache[0].labels) if cache else {},
        counters={name: _total(exposition, name) for name in COUNTER_METRICS},
        running=_total(exposition, RUNNING_METRIC),
        waiting=_total(exposition, WAITING_METRIC),
    )


def _float(text: str | None) -> float | None:
    try:
        value = float(text) if text is not None else None
    except ValueError:
        return None
    return value if value is not None and math.isfinite(value) else None


def _int(text: str | None) -> int | None:
    try:
        return int(text) if text is not None else None
    except ValueError:
        return None


def _boolean(text: str | None) -> bool | None:
    return {"true": True, "false": False}.get(text.lower()) if text is not None else None


def _setting(declared: object, reported: object, matches: bool, source: str, **extra) -> dict:
    if reported is None:
        status = DECLARED_NOT_VERIFIED
    else:
        status = VERIFIED if matches else CONTRADICTED
    return {"declared": declared, "reported": reported, "status": status, "source": source} | extra


def _launch_problem(launch: dict) -> str | None:
    """Why `launch` (a `vllm_launch.read_launch_arguments` result) cannot describe this vLLM."""
    if launch.get("status") != READ:
        return f"launch arguments {launch.get('status')}: {launch.get('reason')}"
    if launch.get("running") is not True:
        return f"container {launch.get('container')} is not running"
    return None


def _max_num_seqs(declared: int, reported: int | None, launch: dict) -> dict:
    """From `/metrics` when vLLM reports it, else from the running container's launch flag."""
    if reported is not None:
        return _setting(declared, reported, reported == declared, METRICS_SOURCE)
    problem = _launch_problem(launch)
    launched = launch.get("max_num_seqs") if problem is None else None
    if launched is None:
        reason = problem or "the launch arguments do not set --max-num-seqs"
        return _setting(
            declared,
            None,
            False,
            LAUNCH_SOURCE,
            reason=f"vLLM does not report max_num_seqs and {reason}; running requests are checked",
        )
    return _setting(declared, launched, launched == declared, LAUNCH_SOURCE)


def _launch_utilization(declared: tuple[float, ...], reported: float | None, launch: dict) -> dict:
    """The launch flag must be a declared value and, when vLLM reports one, the same value."""
    problem = _launch_problem(launch)
    launched = launch.get("gpu_memory_utilization") if problem is None else None
    if launched is None:
        reason = problem or "the launch arguments do not set --gpu-memory-utilization"
        return _setting(list(declared), None, False, LAUNCH_SOURCE, reason=reason)
    matches = any(math.isclose(launched, value) for value in declared) and (
        reported is None or math.isclose(launched, reported)
    )
    return _setting(list(declared), launched, matches, LAUNCH_SOURCE)


def verify_configuration(
    declared: VllmConfiguration, models: dict, state: ServerState, launch: dict | None = None
) -> dict:
    """Each declared setting `verified`, `contradicted` or `declared_not_verified` (unreported).

    `gpu_memory_utilization` is verified when vLLM reports one of the declared values, because
    the operator starts the first of the declared fallbacks that runs. `launch` is the vLLM
    container's launch arguments (`vllm_launch`); they verify `max_num_seqs`, and their
    utilization must match. The measured KV-cache capacity is recorded alongside, from the same
    exposition.
    """
    launch = NOT_INSPECTED if launch is None else launch
    served = served_model(models, SERVED_MODEL_NAME)
    config = state.cache_config
    memory = _float(config.get("gpu_memory_utilization"))
    prefix_caching = _boolean(config.get("enable_prefix_caching"))
    max_num_seqs = _int(config.get("max_num_seqs"))
    if served["status"] == "measured":
        model = _setting(SERVED_MODEL_NAME, served["served_model"], True, served["source"])
    else:
        # A list without the declared model contradicts it; a malformed list says nothing.
        model = {
            "declared": SERVED_MODEL_NAME,
            "reported": None,
            "status": CONTRADICTED if served["status"] == "unavailable" else DECLARED_NOT_VERIFIED,
            "source": served["source"],
            "reason": served["reason"],
        }
    reported_length = served.get("max_model_len")
    settings = {
        "model": model,
        "max_model_len": _setting(
            declared.max_model_len,
            reported_length if isinstance(reported_length, int) else None,
            reported_length == declared.max_model_len,
            served["source"],
        ),
        "gpu_memory_utilization": _setting(
            list(declared.gpu_memory_utilization),
            memory,
            memory is not None
            and any(math.isclose(memory, value) for value in declared.gpu_memory_utilization),
            METRICS_SOURCE,
        ),
        "enable_prefix_caching": _setting(
            declared.enable_prefix_caching,
            prefix_caching,
            prefix_caching == declared.enable_prefix_caching,
            METRICS_SOURCE,
        ),
        "max_num_seqs": _max_num_seqs(declared.max_num_seqs, max_num_seqs, launch),
        "launch_gpu_memory_utilization": _launch_utilization(
            declared.gpu_memory_utilization, memory, launch
        ),
    }
    return {
        "settings": settings,
        "contradicted": sorted(
            name for name, setting in settings.items() if setting["status"] == CONTRADICTED
        ),
        "kv_cache_size_tokens": _int(config.get("kv_cache_size_tokens")),
        "num_gpu_blocks": _int(config.get("num_gpu_blocks")),
        "kv_cache_max_concurrency": _float(config.get("kv_cache_max_concurrency")),
        "block_size": _int(config.get("block_size")),
    }

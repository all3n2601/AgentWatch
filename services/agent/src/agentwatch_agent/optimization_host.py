"""Live wiring for the optimization batch runner on this host (O06).

The runner takes its host as functions (`optimization_run.Host`); these build them against the
real model server, injectors and task fixtures. Each reading is measured, never defaulted.
"""

import json
import threading
import time
import urllib.parse
import urllib.request
from collections.abc import Callable, Sequence
from pathlib import Path

from agentwatch_injectors.base import BoundedInjector
from agentwatch_injectors.cpu import CPU_METHOD, CpuSpinInjector
from agentwatch_injectors.prefill import PREFILL_METHOD, PrefillFloodInjector
from agentwatch_worker.nvml import NvmlDevice

from agentwatch_agent.clock import MonotonicClock
from agentwatch_agent.host_identity import hardware_identity, power_identity
from agentwatch_agent.optimization_experiment import ExperimentError
from agentwatch_agent.replay import canonical_sha256
from agentwatch_agent.runner import AgentRunner
from agentwatch_agent.tasks import Task
from agentwatch_agent.vllm import served_model, vllm_models
from agentwatch_agent.vllm_server import ServerStateUnavailable, read_server_state

DEFAULT_IDLE_TIMEOUT_SECONDS = 30.0
DEFAULT_PACING_SECONDS = 10.0
IDLE_POLL_SECONDS = 0.5
HTTP_TIMEOUT_SECONDS = 10.0


def injector_factory(*, endpoint: str, served_model_name: str) -> Callable:
    """Build the manifest's injectors against this host's model server."""

    def make(planned: dict, clock: MonotonicClock, bound_seconds: float) -> BoundedInjector:
        intensity = planned.get("intensity") or {}
        if planned.get("method") == PREFILL_METHOD:
            return PrefillFloodInjector(
                clock,
                bound_seconds,
                url=endpoint,
                model=served_model_name,
                concurrency=intensity["concurrency"],
                prompt_tokens=intensity["prompt_tokens"],
            )
        if planned.get("method") == CPU_METHOD:
            return CpuSpinInjector(clock, bound_seconds, workers=intensity["workers"])
        raise ExperimentError(f"No injector for method {planned.get('method')!r}")

    return make


def vllm_model_identity(url: str, name: str) -> dict:
    """The served model; its digest is the pinned revision in the served name (`<id>@<rev>`)."""
    served = served_model(vllm_models(url), name)
    if served.get("status") != "measured":
        raise RuntimeError(f"vLLM does not serve {name}: {served.get('reason')}")
    _model_id, separator, revision = name.rpartition("@")
    if not separator or not revision:
        raise RuntimeError(f"Serve a revision-pinned model name (<id>@<revision>), not {name}")
    return {"backend": "vllm", "name": name, "digest": revision}


def ollama_model_identity(url: str, name: str) -> dict:
    """The installed model's digest from Ollama's model list."""
    with urllib.request.urlopen(
        urllib.parse.urljoin(url.rstrip("/") + "/", "api/tags"), timeout=HTTP_TIMEOUT_SECONDS
    ) as response:
        listing = json.load(response)
    for model in listing.get("models") or []:
        if isinstance(model, dict) and name in (model.get("name"), model.get("model")):
            digest = model.get("digest")
            if isinstance(digest, str) and digest:
                return {"backend": "ollama", "name": name, "digest": digest}
    raise RuntimeError(f"Ollama does not list {name} with a digest")


def task_fixture_state() -> dict:
    """The manifest's `state` block: the pinned task fixtures, and no retrieval index."""
    index = Path(__file__).resolve().parent / "task_fixtures" / "index.toml"
    return {
        "tests": f"task-fixtures-{canonical_sha256(index.read_text(encoding='utf-8'))[:16]}",
        # The coding agent has no retrieval tool, so no index state can differ between batches.
        "retrieval": "not-used",
    }


def _pause(cancel: threading.Event, seconds: float) -> bool:
    """Wait `seconds`, or less if `cancel` is set; True when cancelled."""
    return cancel.wait(seconds)


def vllm_idle_waiter(
    metrics_url: str,
    *,
    timeout_seconds: float = DEFAULT_IDLE_TIMEOUT_SECONDS,
    pacing_seconds: float = DEFAULT_PACING_SECONDS,
    pause: Callable[[threading.Event, float], bool] = _pause,
    monotonic: Callable[[], float] = time.monotonic,
) -> Callable[[threading.Event], None]:
    """Wait until vLLM has nothing running or waiting, then a fixed pause, before a batch.

    Returns early once `cancel` is set, so an operator's Ctrl+C between batches is honoured
    before the next slot is attempted (the runner checks it on return).
    """

    def wait(cancel: threading.Event) -> None:
        deadline = monotonic() + timeout_seconds
        while True:
            try:
                state = read_server_state(metrics_url)
            except ServerStateUnavailable as error:
                raise RuntimeError(f"vLLM metrics unreadable: {error}") from error
            if state.running == 0 and state.waiting == 0:
                break
            if monotonic() >= deadline:
                raise RuntimeError(
                    f"vLLM still had {state.running} running and {state.waiting} waiting "
                    f"after {timeout_seconds:g} s"
                )
            if pause(cancel, IDLE_POLL_SECONDS):
                return
        pause(cancel, pacing_seconds)

    return wait


def plan_document(
    tasks: Sequence[Task], runners: Sequence[AgentRunner], identity: dict, bounds: dict
) -> dict:
    """The manifest blocks this host and runner configuration produce, to preregister."""
    return {
        "workload": {
            "tasks": [
                {
                    "task_id": task.task_id,
                    "content_sha256": task.content_sha256,
                    "configuration_digest": runner.configuration_digest(),
                }
                for task, runner in zip(tasks, runners, strict=True)
            ]
        },
        "queue_bounds": {
            control: {
                "max_length": bound.max_queue_length,
                "max_wait_seconds": bound.max_wait_seconds,
            }
            for control, bound in bounds.items()
        },
        **identity,
    }


def pacing_waiter(
    pacing_seconds: float, pause: Callable[[threading.Event, float], bool] = _pause
) -> Callable[[threading.Event], None]:
    """A fixed pause before each batch, for model servers without a queue to read (Ollama);
    it ends early once `cancel` is set."""

    def wait(cancel: threading.Event) -> None:
        pause(cancel, pacing_seconds)

    return wait


def identity_observer(
    *,
    model: Callable[[], dict],
    host_label: str,
    open_gpu: Callable[[], NvmlDevice] | None,
) -> Callable[[], dict]:
    """Read the manifest's model, hardware, power and state blocks, measured at each call."""

    def observe() -> dict:
        return {
            "model": model(),
            "hardware": hardware_identity(host_label, open_gpu=open_gpu),
            "power": power_identity(),
            "state": task_fixture_state(),
        }

    return observe

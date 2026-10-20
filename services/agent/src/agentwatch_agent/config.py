"""Declared run configuration, its limits, and the digest registered with the live run."""

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field

from agentwatch_worker.ollama import CHAT_TIMEOUT_SECONDS

RUNNER_VERSION = "agent-runner-v1"
CONTRACT_VERSION = "live-contracts-v1"
DEFAULT_MODEL = "qwen3:1.7b"
# Greedy decoding with bounded output. Temperature 0 does not by itself make runs repeatable.
# num_ctx matches the LLM demo's 4096; recorded runs peak near 900 prompt tokens. Lowering it
# from 8192 reduced model memory (2.3 to 1.8 GiB). CPU inference speed in Docker on the
# development Mac still varied 3-10x between runs with this setting; the cause is unknown.
DEFAULT_CHAT_OPTIONS = {"temperature": 0, "num_ctx": 4096, "num_predict": 512}
MODEL_SOCKET_TIMEOUT_SECONDS = CHAT_TIMEOUT_SECONDS
DEFAULT_SAMPLING_INTERVAL_SECONDS = 0.25
DEFAULT_CANCEL_POLL_SECONDS = 1.0
# The live API retains at most 100,000 records per run; keep sampling well below that so step
# events always fit. Each sampling round emits one record per CPU metric.
MAX_SAMPLE_RECORDS_PER_RUN = 80_000
SAMPLE_RECORDS_PER_ROUND = 2


def _positive_int(name: str, value: object) -> None:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _positive_seconds(name: str, value: object) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{name} must be finite and positive seconds")


def validate_sample_budget(run_timeout_seconds: float, sampling_interval_seconds: float) -> None:
    """Refuse settings whose resource samples could exceed the live per-run record budget."""
    _positive_seconds("run_timeout_seconds", run_timeout_seconds)
    _positive_seconds("sampling_interval_seconds", sampling_interval_seconds)
    rounds = run_timeout_seconds / sampling_interval_seconds
    if rounds * SAMPLE_RECORDS_PER_ROUND > MAX_SAMPLE_RECORDS_PER_RUN:
        longest = MAX_SAMPLE_RECORDS_PER_RUN / SAMPLE_RECORDS_PER_ROUND * sampling_interval_seconds
        raise ValueError(
            f"A {run_timeout_seconds:g} s run sampled every {sampling_interval_seconds:g} s "
            f"exceeds the {MAX_SAMPLE_RECORDS_PER_RUN}-record sample budget per live run; "
            f"use at most {longest:g} s"
        )


@dataclass(frozen=True)
class RunLimits:
    """Bounds that end a run as failed instead of letting a model loop indefinitely."""

    max_model_calls: int = 8
    max_tool_calls: int = 12
    test_timeout_seconds: float = 60
    run_timeout_seconds: float = 900
    max_tool_output_chars: int = 4000

    def __post_init__(self):
        _positive_int("max_model_calls", self.max_model_calls)
        _positive_int("max_tool_calls", self.max_tool_calls)
        _positive_int("max_tool_output_chars", self.max_tool_output_chars)
        _positive_seconds("test_timeout_seconds", self.test_timeout_seconds)
        _positive_seconds("run_timeout_seconds", self.run_timeout_seconds)


@dataclass(frozen=True)
class RunConfiguration:
    """Everything that determines the workload, excluding where its services are hosted.

    Endpoint URLs are recorded in run evidence but excluded from the digest: the same model and
    task served from another host is the same configuration on different resources.
    """

    task_id: str
    task_content_sha256: str
    model: str
    system_prompt: str
    tools: tuple[dict, ...]
    chat_options: dict = field(default_factory=lambda: dict(DEFAULT_CHAT_OPTIONS))
    limits: RunLimits = field(default_factory=RunLimits)
    sampling_interval_seconds: float = DEFAULT_SAMPLING_INTERVAL_SECONDS
    cancel_poll_seconds: float = DEFAULT_CANCEL_POLL_SECONDS
    model_socket_timeout_seconds: float = MODEL_SOCKET_TIMEOUT_SECONDS
    runner_version: str = RUNNER_VERSION
    contract_version: str = CONTRACT_VERSION

    def __post_init__(self):
        if not self.model.strip():
            raise ValueError("model must be nonblank")
        _positive_seconds("sampling_interval_seconds", self.sampling_interval_seconds)
        _positive_seconds("cancel_poll_seconds", self.cancel_poll_seconds)
        validate_sample_budget(self.limits.run_timeout_seconds, self.sampling_interval_seconds)

    def as_dict(self) -> dict:
        return asdict(self)

    def digest(self) -> str:
        """Lowercase SHA-256 of canonical JSON; key order and whitespace do not matter."""
        canonical = json.dumps(
            self.as_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        return hashlib.sha256(canonical.encode()).hexdigest()

"""Prefill contention: long-prompt, one-token requests to the run's own vLLM server.

A child process keeps `concurrency` non-streaming chat requests in flight, each with a prompt of
about `prompt_tokens` tokens and `max_tokens=1`. The server spends its time prefilling long
prompts rather than decoding, which is the opposite of `CompetingInferenceInjector`'s
generation-heavy load, so the two are different injection methods. Each prompt starts with a
nonce line unique to the worker, request and injector run, so vLLM's prefix cache cannot reuse
the prefill of an earlier request, including one from an earlier run against the same server.
The per-request digits come before the run token, within the first KV-cache block: vLLM caches
whole 16-token blocks, and the chat template, the word "request" and a leading run token would
fill block 0 with text shared by every request of a run (measured on 2026-11-10 against vLLM
v0.32, see ADR 0011). Only loopback endpoints are accepted, as for the competing injector.

The prompt size is an estimate from `FILLER_TOKENS_PER_LINE`; the server's reported
`usage.prompt_tokens` is what the log records. Prompts are fixed, non-sensitive text and are
never logged.
"""

import math
import statistics
import urllib.parse
import uuid
from dataclasses import dataclass

from agentwatch_injectors.base import (
    BACKSTOP_GRACE_SECONDS,
    BoundedInjector,
    InjectionClock,
    require_count,
    require_seconds,
)
from agentwatch_injectors.inference import (
    MAX_REQUEST_TIMEOUT_SECONDS,
    SERVERS,
    GenerateRejected,
    loopback_url,
    request_flood,
)
from agentwatch_injectors.processes import (
    ChildGroup,
    halt_children,
    new_group,
    start_children,
)

PREFILL_METHOD = "prefill-flood-openai-chat-v1"
CHAT_PATH = SERVERS["vllm"][0]
MAX_PREFILL_CONCURRENCY = 4
MAX_PROMPT_TOKENS = 3072
# "length" is the normal end of a `max_tokens=1` reply; anything else (abort, error, ...) is not
# a finished completion.
FINISHED_REASONS = ("stop", "length")
# Changing FILLER_LINE, FILLER_TOKENS_PER_LINE or the nonce format (`filler_prompt`) changes the
# load, so it needs a new method version (`prefill-flood-openai-chat-v2`).
# Measured 2026-11-10 with vLLM's /tokenize for Qwen/Qwen3-1.7B (ADR 0011): every filler line
# below is 19 tokens. It is an estimate for other tokenizers and chat templates.
FILLER_TOKENS_PER_LINE = 19
FILLER_LINE = "{number:04d} the quick brown fox jumps over the lazy dog near the quiet river bank"


class PrefillRejected(GenerateRejected):
    """The model server answered, but not with a finished chat completion."""


def filler_prompt(run_token: str, worker: int, sequence: int, prompt_tokens: int) -> str:
    """A nonce line, then enough fixed numbered lines for about `prompt_tokens` tokens."""
    line_count = math.ceil(prompt_tokens / FILLER_TOKENS_PER_LINE)
    lines = [f"request {worker}-{sequence}-{run_token}"]
    lines.extend(FILLER_LINE.format(number=number) for number in range(1, line_count + 1))
    return "\n".join(lines) + "\n"


def _count_stats(counts: list[int], median_key: str) -> dict | None:
    if not counts:
        return None
    return {
        "count": len(counts),
        "min": min(counts),
        # median_low keeps the median an integer token count that was really reported.
        median_key: statistics.median_low(counts),
        "max": max(counts),
    }


@dataclass(frozen=True)
class PrefillProfile:
    """A unique long prompt every time; the measurement is the server's prompt token count."""

    model: str
    prompt_tokens: int
    # Random per injector run: worker and sequence counters restart at 0 in every run.
    run_token: str
    rejection = PrefillRejected

    @property
    def path(self) -> str:
        return CHAT_PATH

    def payload(self, worker: int, sequence: int) -> dict:
        return {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": filler_prompt(self.run_token, worker, sequence, self.prompt_tokens),
                }
            ],
            "max_tokens": 1,
            "temperature": 0,
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": False},
        }

    def measure(self, result: object) -> int | None:
        if not isinstance(result, dict) or result.get("error"):
            raise PrefillRejected("model server returned an unusable chat response")
        choices = result.get("choices")
        if (
            not isinstance(choices, list)
            or not choices
            or not isinstance(choices[0], dict)
            or choices[0].get("finish_reason") not in FINISHED_REASONS
        ):
            raise PrefillRejected("model server returned no finished choice")
        usage = result.get("usage") if isinstance(result.get("usage"), dict) else {}
        count = usage.get("prompt_tokens")
        is_count = isinstance(count, int) and not isinstance(count, bool) and count > 0
        return count if is_count else None

    def measure_secondary(self, result: object) -> int | None:
        """The reply's `usage.completion_tokens`, `None` when it is missing or not a count."""
        usage = result.get("usage") if isinstance(result, dict) else None
        count = usage.get("completion_tokens") if isinstance(usage, dict) else None
        is_count = isinstance(count, int) and not isinstance(count, bool) and count >= 0
        return count if is_count else None

    def summarize(
        self, measurements: list[int], unmeasured: int, secondary: list[int | None]
    ) -> dict:
        completion_tokens = [count for count in secondary if count is not None]
        return {
            "prompt_tokens_target": self.prompt_tokens,
            "prompt_tokens": _count_stats(measurements, "median"),
            "completions_without_prompt_tokens": unmeasured,
            "completion_tokens": _count_stats(completion_tokens, "median_low"),
            # A reply that reported no count is counted apart, never as one token.
            "completions_not_one_token": sum(1 for count in completion_tokens if count != 1),
            "completions_without_completion_tokens": len(secondary) - len(completion_tokens),
        }


class PrefillFloodInjector(BoundedInjector):
    kind = "prefill"
    method = PREFILL_METHOD
    resource_class = "inference"

    def __init__(
        self,
        clock: InjectionClock,
        max_duration_seconds: float,
        *,
        url: str,
        model: str,
        concurrency: int = 2,
        prompt_tokens: int = 2048,
        request_timeout_seconds: float = 60.0,
    ):
        super().__init__(clock, max_duration_seconds)
        self.url = loopback_url(url)
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be a nonblank model name")
        self.model = model
        self.concurrency = require_count("concurrency", concurrency, MAX_PREFILL_CONCURRENCY)
        self.prompt_tokens = require_count("prompt_tokens", prompt_tokens, MAX_PROMPT_TOKENS)
        self.request_timeout_seconds = require_seconds(
            "request_timeout_seconds", request_timeout_seconds, MAX_REQUEST_TIMEOUT_SECONDS
        )
        self._group: ChildGroup | None = None

    def bounds(self) -> dict:
        return {
            "concurrency": self.concurrency,
            "prompt_tokens": self.prompt_tokens,
            "filler_lines": math.ceil(self.prompt_tokens / FILLER_TOKENS_PER_LINE),
            "max_tokens": 1,
            "request_timeout_seconds": self.request_timeout_seconds,
            "backstop_seconds": self._backstop_seconds(),
        }

    def target(self) -> dict:
        # Only the host and port are recorded, never credentials or paths from the URL.
        return {
            "scope": "loopback model server",
            "server": "vllm",
            "model": self.model,
            "endpoint_host": urllib.parse.urlsplit(self.url).hostname,
            "endpoint_port": urllib.parse.urlsplit(self.url).port,
        }

    def _backstop_seconds(self) -> float:
        return self.max_duration_seconds + BACKSTOP_GRACE_SECONDS

    def _launch(self) -> None:
        # Drawn per launch, never logged, so no launch can reuse an earlier token.
        run_token = uuid.uuid4().hex[:12]
        # Owned before the child starts, so a failed start halts and reports it.
        self._group = new_group()
        start_children(
            self._group,
            request_flood,
            [
                (
                    self.url,
                    PrefillProfile(self.model, self.prompt_tokens, run_token),
                    self.concurrency,
                    self.request_timeout_seconds,
                )
            ],
            self._backstop_seconds(),
            "agentwatch-prefill-injector",
        )

    def _halt(self) -> tuple[dict, dict]:
        if self._group is None:
            return {"requests_started": 0}, {"confirmed": True, "children": []}
        reports, cleanup = halt_children(self._group)
        report = reports[0] if reports else None
        if report is None:
            # The child was stopped before it reported: its counts are unknown, not zero.
            return {"report": "missing"}, cleanup
        return {"report": "received"} | report, cleanup

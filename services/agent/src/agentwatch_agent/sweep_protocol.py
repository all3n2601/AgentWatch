"""Declared protocol `natural-overload-sweep-v1` for the natural-overload capacity sweep.

Pure declarations and schedules, no I/O. Any changed value is a new protocol
version: the hash of `protocol()` is recorded with every sweep so results are
only compared under identical declared settings.
"""

from dataclasses import dataclass

from agentwatch_agent.replay import canonical_sha256

PROTOCOL_VERSION = "natural-overload-sweep-v1"

VLLM_VERSION = "v0.32.0"
MODEL_ID = "Qwen/Qwen3-1.7B"
MODEL_REVISION = "70d244cc86cc"
MAX_MODEL_LEN_TOKENS = 4096

RAMP_LEVELS: tuple[int, ...] = (1, 2, 3, 4, 6, 8, 12, 16)
WARMUP_BATCHES = 1
BATCHES_PER_LEVEL = 3
# N = 1 has one agent, so it runs 3 batches per source (6) to cover both tasks.
BASELINE_BATCHES_PER_SOURCE = 3
SOURCE_COUNT = 2

IDLE_WAIT_SECONDS = 30.0  # wait for 0 running and 0 waiting requests, at most
PACING_SECONDS = 10.0  # fixed pause after the server is idle

MIN_THROUGHPUT_GAIN = 0.10  # over the previous level
MAX_PASS_RATE_DROP = 0.20  # below the N = 1 pass rate
MAX_P90_FRACTION_OF_RUN_LIMIT = 0.5
MIN_PASSED_RUNS_AT_ONE_AGENT = 1  # N = 1 with fewer passed runs gives capacity "below 1"

LIVE_CONFIRMATION_BATCHES = 3
INJECTED_PAIRS = 3  # flood batches, each followed by a clean control

INJECTOR_NAME = "PrefillFloodInjector"
INJECTOR_CONCURRENCY = 2
INJECTOR_PROMPT_TOKENS = 2048
INJECTOR_RAMP_SECONDS = 2.0
INJECTOR_BOUND_MARGIN_SECONDS = 32.0  # bound = run limit + this

# Contended when (duration - baseline mean) > max(floor, multiplier * baseline stdev).
LABEL_THRESHOLD_FLOOR_SECONDS = 0.01
LABEL_THRESHOLD_STDEV_MULTIPLIER = 3.0
VALIDATED_MIN_PRECISION = 0.90
VALIDATED_MIN_RECALL = 0.90

PHASES = ("warmup", "ramp", "live", "injected", "control")


@dataclass(frozen=True)
class VllmConfiguration:
    name: str
    max_num_seqs: int
    # Tried in order; the first value that starts is used and every attempt is recorded.
    gpu_memory_utilization: tuple[float, ...]
    max_model_len: int = MAX_MODEL_LEN_TOKENS
    enable_prefix_caching: bool = True


CONFIGURATIONS: dict[str, VllmConfiguration] = {
    "cfg-default": VllmConfiguration("cfg-default", 2, (0.6,)),
    "cfg-hardware": VllmConfiguration("cfg-hardware", 16, (0.8, 0.75, 0.7)),
}


@dataclass(frozen=True)
class Batch:
    phase: str
    level: int
    repeat: int
    sources: tuple[int, ...]  # sources[i] is the source index for run i
    injected: bool

    def __post_init__(self) -> None:
        if self.phase not in PHASES:
            raise ValueError(f"unknown batch phase {self.phase!r}")


WARMUP_BATCH = Batch(phase="warmup", level=1, repeat=0, sources=(0,), injected=False)


def level_sources(level: int, repeat: int) -> tuple[int, ...]:
    """Sources alternate across a level's agents and rotate between repeats."""
    return tuple((run + repeat) % SOURCE_COUNT for run in range(level))


def ramp_batches(level: int) -> list[Batch]:
    if level < 1:
        raise ValueError("level must be at least 1")
    if level == 1:
        return [
            Batch("ramp", 1, repeat, (repeat % SOURCE_COUNT,), False)
            for repeat in range(BASELINE_BATCHES_PER_SOURCE * SOURCE_COUNT)
        ]
    return [
        Batch("ramp", level, repeat, level_sources(level, repeat), False)
        for repeat in range(BATCHES_PER_LEVEL)
    ]


def live_batches(capacity: int) -> list[Batch]:
    return [
        Batch("live", capacity, repeat, level_sources(capacity, repeat), False)
        for repeat in range(LIVE_CONFIRMATION_BATCHES)
    ]


def injected_level(capacity: int) -> int:
    return max(1, capacity // 2)


def injected_batches(level: int) -> list[Batch]:
    """Flood batches alternating with clean controls, flood first."""
    batches: list[Batch] = []
    for pair in range(INJECTED_PAIRS):
        sources = level_sources(level, pair)
        batches.append(Batch("injected", level, pair, sources, True))
        batches.append(Batch("control", level, pair, sources, False))
    return batches


def protocol(configuration: VllmConfiguration, run_timeout_seconds: float) -> dict:
    return {
        "version": PROTOCOL_VERSION,
        "vllm_version": VLLM_VERSION,
        "model": {"id": MODEL_ID, "revision": MODEL_REVISION},
        "configuration": {
            "name": configuration.name,
            "max_num_seqs": configuration.max_num_seqs,
            "gpu_memory_utilization": list(configuration.gpu_memory_utilization),
            "max_model_len": configuration.max_model_len,
            "enable_prefix_caching": configuration.enable_prefix_caching,
        },
        "run_timeout_seconds": run_timeout_seconds,
        "ramp_levels": list(RAMP_LEVELS),
        "warmup_batches": WARMUP_BATCHES,
        "batches_per_level": BATCHES_PER_LEVEL,
        "baseline_batches_per_source": BASELINE_BATCHES_PER_SOURCE,
        "idle_wait_seconds": IDLE_WAIT_SECONDS,
        "pacing_seconds": PACING_SECONDS,
        "policy": "none",
        "capacity_criteria": {
            "min_throughput_gain": MIN_THROUGHPUT_GAIN,
            "max_pass_rate_drop": MAX_PASS_RATE_DROP,
            "max_p90_fraction_of_run_limit": MAX_P90_FRACTION_OF_RUN_LIMIT,
            "min_passed_runs_at_one_agent": MIN_PASSED_RUNS_AT_ONE_AGENT,
        },
        "max_p90_run_seconds": MAX_P90_FRACTION_OF_RUN_LIMIT * run_timeout_seconds,
        "live_confirmation_batches": LIVE_CONFIRMATION_BATCHES,
        "injected_pairs": INJECTED_PAIRS,
        "injected_level_rule": "max(1, floor(capacity / 2))",
        "injector": {
            "name": INJECTOR_NAME,
            "concurrency": INJECTOR_CONCURRENCY,
            "prompt_tokens": INJECTOR_PROMPT_TOKENS,
            "ramp_seconds": INJECTOR_RAMP_SECONDS,
            "bound_seconds": run_timeout_seconds + INJECTOR_BOUND_MARGIN_SECONDS,
        },
        "label_rule": {
            "contended_when": "duration - baseline_mean > max(floor_seconds, "
            "stdev_multiplier * baseline_stdev)",
            "floor_seconds": LABEL_THRESHOLD_FLOOR_SECONDS,
            "stdev_multiplier": LABEL_THRESHOLD_STDEV_MULTIPLIER,
            "reference": "measured vLLM queue > the same threshold",
        },
        "validated_when": {
            "min_precision": VALIDATED_MIN_PRECISION,
            "min_recall": VALIDATED_MIN_RECALL,
        },
        "held_out": True,
    }


def protocol_sha256(document: dict) -> str:
    return canonical_sha256(document)

"""The declared natural-overload sweep protocol: schedules and the hashed declaration."""

import functools
import tomllib
from dataclasses import asdict
from pathlib import Path

import pytest
from agentwatch_agent import sweep_protocol, vllm_server
from agentwatch_agent.config import DEFAULT_SAMPLING_INTERVAL_SECONDS, RunLimits
from agentwatch_agent.experiment import PREFILL_PROMPT_TOKENS_TOLERANCE
from agentwatch_agent.runner import DEFAULT_TRACE_WAIT_SECONDS
from agentwatch_agent.sweep_protocol import (
    CONFIGURATIONS,
    P90_RULE,
    VLLM_IMAGE,
    Batch,
    injected_batches,
    injected_level,
    protocol,
    protocol_sha256,
    ramp_batches,
)


def test_one_agent_runs_three_batches_per_source():
    assert [b.sources for b in ramp_batches(1)] == [(0,), (1,), (0,), (1,), (0,), (1,)]


def test_sources_alternate_and_rotate_between_repeats():
    assert [b.sources for b in ramp_batches(3)] == [(0, 1, 0), (1, 0, 1), (0, 1, 0)]


def test_ramp_batches_carry_phase_level_and_repeat():
    batches = ramp_batches(2)
    assert [(b.phase, b.level, b.repeat, b.injected) for b in batches] == [
        ("ramp", 2, 0, False),
        ("ramp", 2, 1, False),
        ("ramp", 2, 2, False),
    ]
    assert isinstance(batches[0], Batch)


@pytest.mark.parametrize("capacity,level", [(1, 1), (2, 1), (3, 1), (8, 4), (16, 8)])
def test_the_injected_level_is_half_capacity_at_least_one(capacity, level):
    assert injected_level(capacity) == level


def test_injected_batches_alternate_flood_first():
    batches = injected_batches(4)
    assert [b.phase for b in batches] == ["injected", "control"] * 3
    assert [b.injected for b in batches] == [True, False] * 3
    assert all(b.level == 4 for b in batches)


def test_the_protocol_hash_changes_with_any_declared_value():
    a = protocol(CONFIGURATIONS["cfg-default"], 900.0)
    b = protocol(CONFIGURATIONS["cfg-hardware"], 900.0)
    assert protocol_sha256(a) != protocol_sha256(b)
    assert protocol_sha256(a) != protocol_sha256(protocol(CONFIGURATIONS["cfg-default"], 600.0))
    assert protocol_sha256(a) == protocol_sha256(protocol(CONFIGURATIONS["cfg-default"], 900.0))
    assert a["capacity_criteria"] == {
        "min_throughput_gain": 0.10,
        "max_pass_rate_drop": 0.20,
        "max_p90_fraction_of_run_limit": 0.5,
        "min_passed_runs_at_one_agent": 1,
    }
    assert a["ramp_levels"] == [1, 2, 3, 4, 6, 8, 12, 16]
    assert a["held_out"] is True


def test_configurations_and_injector_are_declared():
    assert CONFIGURATIONS["cfg-default"].max_num_seqs == 2
    assert CONFIGURATIONS["cfg-default"].gpu_memory_utilization == (0.6,)
    assert CONFIGURATIONS["cfg-hardware"].max_num_seqs == 16
    assert CONFIGURATIONS["cfg-hardware"].gpu_memory_utilization == (0.8, 0.75, 0.7)
    document = protocol(CONFIGURATIONS["cfg-default"], 900.0)
    assert document["injector"] == {
        "name": "PrefillFloodInjector",
        "concurrency": 2,
        "prompt_tokens": 2048,
        "ramp_seconds": 2.0,
        "bound_seconds": 932.0,
        "prompt_tokens_tolerance": PREFILL_PROMPT_TOKENS_TOLERANCE,
    }
    assert document["max_p90_run_seconds"] == 450.0


COMPOSE_FILE = Path(__file__).resolve().parents[3] / "deploy" / "compose" / "compose.yaml"
LABELS_V2_FILE = (
    Path(__file__).resolve().parents[3]
    / "pipelines"
    / "data"
    / "src"
    / "agentwatch_data"
    / "labels-v2.toml"
)


def test_the_measurement_settings_are_hashed_into_the_protocol(monkeypatch):
    document = protocol(CONFIGURATIONS["cfg-default"], 900.0)
    assert document["sampling_interval_seconds"] == DEFAULT_SAMPLING_INTERVAL_SECONDS
    assert document["vllm_image"] == VLLM_IMAGE
    assert document["p90_rule"] == P90_RULE
    before = protocol_sha256(document)
    monkeypatch.setattr(sweep_protocol, "SAMPLING_INTERVAL_SECONDS", 0.5)
    assert protocol_sha256(protocol(CONFIGURATIONS["cfg-default"], 900.0)) != before


def test_the_declared_vllm_image_is_the_compose_services_digest():
    lines = COMPOSE_FILE.read_text(encoding="utf-8").splitlines()
    service = lines.index("  vllm:")
    image = next(line for line in lines[service:] if line.strip().startswith("image:"))
    assert image.split("image:", 1)[1].strip() == VLLM_IMAGE


def test_the_label_thresholds_are_the_dataset_noise_rule():
    """Review of #65 (Akshay): the sweep copies `labels-v2.toml [noise]`, so pin it to the file.

    A later label version that changes the noise rule then fails here instead of drifting from
    what the sweep validated.
    """
    noise = tomllib.loads(LABELS_V2_FILE.read_text(encoding="utf-8"))["noise"]
    assert sweep_protocol.LABEL_THRESHOLD_FLOOR_SECONDS == noise["min_seconds"]
    assert sweep_protocol.LABEL_THRESHOLD_STDEV_MULTIPLIER == noise["stdev_multiplier"]


def test_the_sweeps_bounds_and_run_limits_are_hashed_into_the_protocol(monkeypatch):
    """Bug hunt: waits, retries and run limits the sweep depends on were not declared."""
    document = protocol(CONFIGURATIONS["cfg-default"], 900.0)
    assert document["server_state"] == {
        "metrics_read_timeout_seconds": vllm_server.DEFAULT_TIMEOUT_SECONDS,
        "idle_poll_seconds": sweep_protocol.IDLE_POLL_SECONDS,
        "read_attempts": sweep_protocol.STATE_READ_ATTEMPTS,
        "read_retry_pause_seconds": sweep_protocol.STATE_READ_RETRY_PAUSE_SECONDS,
    }
    assert document["trace_wait_seconds"] == DEFAULT_TRACE_WAIT_SECONDS
    # The limits every sweep runner gets: the declared ones with this sweep's run timeout.
    assert document["run_limits"] == asdict(RunLimits(run_timeout_seconds=900.0))
    assert sweep_protocol.sweep_run_limits(900.0) == RunLimits(run_timeout_seconds=900.0)
    before = protocol_sha256(document)
    for name, value in (
        ("METRICS_READ_TIMEOUT_SECONDS", 5.0),
        ("IDLE_POLL_SECONDS", 0.5),
        ("STATE_READ_ATTEMPTS", 5),
        ("STATE_READ_RETRY_PAUSE_SECONDS", 2.0),
        ("TRACE_WAIT_SECONDS", 3.0),
    ):
        with monkeypatch.context() as patched:
            patched.setattr(sweep_protocol, name, value)
            changed = protocol_sha256(protocol(CONFIGURATIONS["cfg-default"], 900.0))
        assert changed != before, name
    with monkeypatch.context() as patched:
        patched.setattr(sweep_protocol, "RunLimits", functools.partial(RunLimits, max_tool_calls=4))
        assert protocol_sha256(protocol(CONFIGURATIONS["cfg-default"], 900.0)) != before


def test_the_protocol_version_states_when_it_freezes():
    source = Path(sweep_protocol.__file__).read_text(encoding="utf-8")
    line = next(n for n, text in enumerate(source.splitlines()) if "PROTOCOL_VERSION =" in text)
    nearby = "\n".join(source.splitlines()[max(0, line - 4) : line + 1])
    assert "first GPU session" in nearby

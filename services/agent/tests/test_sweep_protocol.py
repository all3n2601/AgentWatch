"""The declared natural-overload sweep protocol: schedules and the hashed declaration."""

import pytest
from agentwatch_agent.sweep_protocol import (
    CONFIGURATIONS,
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
    }
    assert document["max_p90_run_seconds"] == 450.0

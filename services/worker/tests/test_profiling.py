import pytest
from agentwatch_worker.profiling import StepMeasurement, attribute_cpu_queue


def test_queue_attribution_and_fraction_use_direct_timestamps():
    step = StepMeasurement("test", 1_000_000_000, 4_000_000_000, 5_000_000_000, 0.8, "abc")
    result = attribute_cpu_queue(step, 0.01)
    assert result["resource"] == "cpu_sandbox"
    assert result["queue_seconds"] == 3
    assert result["execution_seconds"] == 1
    assert result["queue_fraction"] == 0.75


def test_small_dispatch_delay_is_not_contention():
    step = StepMeasurement("test", 0, 1_000_000, 1_000_000_000, 0.8, "abc")
    assert attribute_cpu_queue(step, 0.01)["resource"] == "none"


@pytest.mark.parametrize("start,end", [(2, 1), (-1, 2), (1, 1)])
def test_rejects_invalid_timestamps(start, end):
    with pytest.raises(ValueError):
        attribute_cpu_queue(StepMeasurement("test", 0, start, end, 0, "abc"), 0.01)

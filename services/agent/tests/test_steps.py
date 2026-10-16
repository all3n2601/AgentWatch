from uuid import uuid4

import pytest
from agentwatch_agent.clock import MonotonicClock
from agentwatch_agent.redaction import redact_reason
from agentwatch_agent.steps import StepRecorder, new_span_id, new_trace_id
from agentwatch_sdk.live import ResourceRegistration, validate_transition

RESOURCE = ResourceRegistration(
    resource_id="tool-host-x", service_id="agentwatch-agent-runner", resource_class="cpu_sandbox"
)


@pytest.fixture
def recorder():
    events = []
    recorder = StepRecorder(
        run_id=uuid4(),
        task_id="fix-mean-empty-r1",
        trace_id=new_trace_id(),
        clock=MonotonicClock("test-clock"),
        emit=events.append,
    )
    recorder.events = events
    return recorder


def assert_valid_lifecycle(events):
    previous = {}
    for event in events:
        validate_transition(previous.get(event.step_id), event)
        previous[event.step_id] = event


def test_completed_step_has_three_contiguous_revisions(recorder):
    root = recorder.submit("agent-run", "other", parent=None, resource=None)
    step = recorder.submit("tool-1-run_tests", "tool", parent=root, resource=RESOURCE)
    recorder.start(step)
    recorder.complete(step, cpu_seconds=0.25, cpu_scope="process_and_children")
    events = [e for e in recorder.events if e.step_id == step.step_id]
    assert [(e.sequence, e.state) for e in events] == [
        (0, "queued"),
        (1, "running"),
        (2, "completed"),
    ]
    assert events[2].cpu_seconds == 0.25
    assert events[2].parent_step_id == "agent-run"
    assert {e.trace_id for e in recorder.events} == {recorder.trace_id}
    assert len({e.event_id for e in recorder.events}) == len(recorder.events)
    assert_valid_lifecycle(recorder.events)


def test_failure_before_start_is_revision_one_without_start_or_cpu(recorder):
    step = recorder.submit("tool-1-rejected", "tool", parent=None, resource=None)
    recorder.fail(step, "Rejected tool call", cpu_seconds=1.0, cpu_scope="process")
    failed = recorder.events[-1]
    assert (failed.sequence, failed.state) == (1, "failed")
    assert failed.started_tick_ns is None
    assert failed.cpu_seconds is None and failed.cpu_scope is None
    assert_valid_lifecycle(recorder.events)


def test_cancellation_after_start_is_revision_two(recorder):
    step = recorder.submit("model-call-1", "model_call", parent=None, resource=None)
    recorder.start(step)
    recorder.cancel(step, "Cancelled from the dashboard")
    assert (recorder.events[-1].sequence, recorder.events[-1].state) == (2, "cancelled")
    assert_valid_lifecycle(recorder.events)


def test_unfinished_lists_children_before_parents(recorder):
    root = recorder.submit("agent-run", "other", parent=None, resource=None)
    recorder.start(root)
    child = recorder.submit("model-call-1", "model_call", parent=root, resource=None)
    done = recorder.submit("model-call-0", "model_call", parent=root, resource=None)
    recorder.fail(done, "x")
    assert [s.step_id for s in recorder.unfinished()] == [child.step_id, root.step_id]


def test_invalid_transitions_and_duplicates_are_refused(recorder):
    root = recorder.submit("agent-run", "other", parent=None, resource=None)
    with pytest.raises(ValueError):
        recorder.complete(root)
    with pytest.raises(ValueError, match="already exists"):
        recorder.submit("agent-run", "other", parent=None, resource=None)
    recorder.fail(root, "x")
    with pytest.raises(ValueError):
        recorder.cancel(root, "again")
    with pytest.raises(ValueError, match="finished step"):
        recorder.submit("late", "tool", parent=root, resource=None)


def test_reasons_are_redacted_before_they_are_shared(recorder):
    step = recorder.submit("tool-1", "tool", parent=None, resource=None)
    recorder.fail(step, "boom\x00 in /Users/alice/secret/project/stats.py\nline 2")
    assert recorder.events[-1].reason == "boom in <path> line 2"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("", "No reason was recorded"),
        ("\x00\x01", "No reason was recorded"),
        ("C:\\Users\\bob\\file.txt failed", "<path> failed"),
        ("ratio 3/4 kept", "ratio 3/4 kept"),
        ("\ud800 lone", "? lone"),
    ],
)
def test_redaction_cases(raw, expected):
    assert redact_reason(raw) == expected


def test_redaction_bounds_length():
    reason = redact_reason("x" * 2000)
    assert len(reason) == 512
    assert reason.endswith("…")


def test_trace_and_span_ids_are_nonzero_lowercase_hex():
    for value, length in ((new_trace_id(), 32), (new_span_id(), 16)):
        assert len(value) == length
        assert int(value, 16) != 0
        assert value == value.lower()


def test_clock_is_monotonic_and_timezone_aware():
    clock = MonotonicClock()
    first, second = clock.tick(), clock.tick()
    assert 0 <= first <= second
    assert clock.now().tzinfo is not None
    assert clock.clock_id.startswith("agent-runner-")

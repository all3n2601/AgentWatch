"""Runner-process admission (O03): bounded, first-in-first-out permits under a policy version.

Policies here are synthetic test doubles, never recorded-run evidence.
"""

import threading
import time
from urllib.error import HTTPError, URLError

import pytest
from agent_test_doubles import FakePolicySource, policy_version
from agentwatch_agent.admission import (
    AdmissionController,
    AdmissionRefused,
    AdmissionSettings,
    PolicyError,
    PolicyWatcher,
    QueueBounds,
)
from agentwatch_agent.clock import MonotonicClock
from agentwatch_agent.delivery import RetryPolicy

SERVICES = {"inference_requests": "ollama", "test_jobs": "agentwatch-agent-runner"}
NO_SLEEP = RetryPolicy(attempts=3, initial_backoff_seconds=0, max_backoff_seconds=0)
CLOCK = MonotonicClock()


def controller(policy=None, inference_service="ollama", **bounds):
    settings = AdmissionSettings(
        inference_requests=QueueBounds(**bounds), test_jobs=QueueBounds(**bounds)
    )
    made = AdmissionController(settings, inference_service_id=inference_service)
    made.apply(policy or policy_version(0))
    return made


def acquire(made, control="test_jobs", step="step", *, cancel=None, deadline=None, **extra):
    return made.acquire(
        control,
        step_id=step,
        span_id=f"span-{step}",
        resource_id=f"resource-{control}",
        service_id=extra["service_id"]
        if "service_id" in extra
        else made.governed_services[control],
        clock=CLOCK,
        cancel=cancel or threading.Event(),
        # Short, so a regression that stops honouring a bound fails instead of hanging.
        deadline_monotonic=deadline if deadline is not None else time.monotonic() + 10,
    )


def wait_for(condition, timeout=5.0):
    """Wait for state another thread changes; bounded, so a regression fails instead of hanging."""
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, "condition not reached"
        threading.Event().wait(0.005)


def queued(made, control="test_jobs"):
    return made.summary()["controls"][control]["queued"]


def active(made, control="test_jobs"):
    return made.summary()["controls"][control]["active"]


class Waiter(threading.Thread):
    """Acquire in a thread; keeps the permit or the refusal for the test to inspect."""

    def __init__(self, made, step, control="test_jobs", **kwargs):
        super().__init__(daemon=True)
        self.made, self.step, self.control, self.kwargs = made, step, control, kwargs
        self.permit = None
        self.refused = None
        self.done = threading.Event()

    def run(self):
        try:
            self.permit = acquire(self.made, self.control, self.step, **self.kwargs)
        except AdmissionRefused as refused:
            self.refused = refused
        finally:
            self.done.set()


def test_a_limit_bounds_concurrent_admissions_and_every_permit_returns():
    made = controller(policy_version(1, tests=2))
    lock, running, peak = threading.Lock(), [0], [0]

    def work(index):
        permit = acquire(made, step=f"s{index}")
        try:
            with lock:
                running[0] += 1
                peak[0] = max(peak[0], running[0])
            threading.Event().wait(0.02)
            with lock:
                running[0] -= 1
        finally:
            permit.release()

    threads = [threading.Thread(target=work, args=(i,)) for i in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(10)
    summary = made.summary()["controls"]["test_jobs"]
    assert peak[0] == 2
    assert (summary["admitted"], summary["active"], summary["queued"]) == (8, 0, 0)
    assert made.open_permits() == 0


def test_admission_is_first_in_first_out_so_no_waiter_is_overtaken():
    made = controller(policy_version(1, tests=1))
    holder = acquire(made, step="holder")
    waiters = []
    for index in range(5):
        waiter = Waiter(made, f"w{index}")
        waiter.start()
        wait_for(lambda count=index + 1: queued(made) == count)
        waiters.append(waiter)
    order = []
    holder.release()
    for waiter in waiters:
        assert waiter.done.wait(5)
        order.append(waiter.step)
        assert waiter.permit.snapshot()["queued_ahead_at_arrival"] == int(waiter.step[1:])
        waiter.permit.release()
    assert order == ["w0", "w1", "w2", "w3", "w4"]
    assert made.open_permits() == 0


def test_a_full_queue_refuses_at_once_without_disturbing_the_waiters():
    made = controller(policy_version(1, tests=1), max_queue_length=1)
    holder = acquire(made, step="holder")
    first = Waiter(made, "first")
    first.start()
    wait_for(lambda: queued(made) == 1)
    with pytest.raises(AdmissionRefused) as refused:
        acquire(made, step="second")
    assert refused.value.outcome == "queue_full"
    assert refused.value.record["max_queue_length"] == 1
    assert "queue is full" in refused.value.reason
    holder.release()
    assert first.done.wait(5) and first.permit is not None
    first.permit.release()
    assert made.summary()["controls"]["test_jobs"]["refused"] == {"queue_full": 1}


def test_a_zero_length_queue_refuses_instead_of_waiting():
    made = controller(policy_version(1, tests=1), max_queue_length=0)
    holder = acquire(made, step="holder")
    with pytest.raises(AdmissionRefused) as refused:
        acquire(made, step="refused")
    assert refused.value.outcome == "queue_full"
    holder.release()
    acquire(made, step="after").release()


def test_a_wait_longer_than_its_bound_is_refused_and_leaves_the_queue():
    made = controller(policy_version(1, tests=1), max_wait_seconds=0.2)
    holder = acquire(made, step="holder")
    started = time.monotonic()
    with pytest.raises(AdmissionRefused) as refused:
        acquire(made, step="late")
    elapsed = time.monotonic() - started
    assert refused.value.outcome == "wait_exceeded"
    assert 0.2 <= elapsed < 2
    assert refused.value.record["admission_wait_seconds"] >= 0.2
    assert queued(made) == 0
    holder.release()
    acquire(made, step="next").release()


def test_cancellation_ends_a_wait_and_never_blocks_those_behind_it():
    made = controller(policy_version(1, tests=1))
    holder = acquire(made, step="holder")
    cancel = threading.Event()
    head = Waiter(made, "head", cancel=cancel)
    head.start()
    wait_for(lambda: queued(made) == 1)
    behind = Waiter(made, "behind")
    behind.start()
    wait_for(lambda: queued(made) == 2)
    cancel.set()
    assert head.done.wait(5)
    assert head.refused.outcome == "cancelled" and head.permit is None
    holder.release()
    assert behind.done.wait(5) and behind.permit is not None
    behind.permit.release()
    assert (queued(made), made.open_permits()) == (0, 0)


def test_the_run_deadline_ends_a_wait():
    made = controller(policy_version(1, tests=1), max_wait_seconds=5)
    holder = acquire(made, step="holder")
    with pytest.raises(AdmissionRefused) as refused:
        acquire(made, step="late", deadline=time.monotonic() + 0.1)
    assert refused.value.outcome == "run_deadline"
    assert "time limit" in refused.value.reason
    holder.release()
    assert made.open_permits() == 0


def test_cancelled_or_expired_work_is_never_admitted_even_with_free_permits():
    made = controller(policy_version(1, tests=4))
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(AdmissionRefused) as cancelled:
        acquire(made, step="cancelled", cancel=cancel)
    with pytest.raises(AdmissionRefused) as expired:
        acquire(made, step="expired", deadline=time.monotonic() - 1)
    assert (cancelled.value.outcome, expired.value.outcome) == ("cancelled", "run_deadline")
    assert made.open_permits() == 0


def test_lowering_a_limit_never_stops_admitted_work_and_affects_new_admissions_only():
    made = controller(policy_version(1, tests=3))
    held = [acquire(made, step=f"held{i}") for i in range(3)]
    assert made.apply(policy_version(2, tests=1))
    assert active(made) == 3  # nothing admitted was stopped
    waiter = Waiter(made, "new")
    waiter.start()
    wait_for(lambda: queued(made) == 1)
    held[0].release()
    held[1].release()
    # Two of three returned: one is still active, which is the new limit, so it still waits.
    assert not waiter.done.wait(0.2)
    held[2].release()
    assert waiter.done.wait(5)
    record = waiter.permit.snapshot()
    assert (record["policy_version_at_arrival"], record["policy_version"]) == (2, 2)
    assert record["limit"] == 1
    waiter.permit.release()


def test_removing_a_limit_admits_waiters_in_order_and_restores_the_baseline():
    made = controller(policy_version(1, tests=1))
    holder = acquire(made, step="holder")
    waiters = []
    for index in range(3):
        waiter = Waiter(made, f"w{index}")
        waiter.start()
        wait_for(lambda count=index + 1: queued(made) == count)
        waiters.append(waiter)
    assert made.apply(policy_version(2, action="restore", restored_from=0))
    for waiter in waiters:
        assert waiter.done.wait(5) and waiter.permit is not None
        record = waiter.permit.snapshot()
        assert (record["policy_version_at_arrival"], record["policy_version"]) == (1, 2)
        assert (record["limit_at_arrival"], record["limit"]) == (1, None)
        # No limit was applied at the decision, so none is reported as enforced.
        assert record["enforced"] is False
        assert "sets no test_jobs limit" in record["not_enforced_reason"]
    assert active(made) == 4  # the baseline has no limit
    for permit in [holder, *(waiter.permit for waiter in waiters)]:
        permit.release()
    history = made.summary()["policy_history"]
    assert [(entry["version"], entry["action"]) for entry in history] == [
        (1, "set"),
        (2, "restore"),
    ]
    assert history[1]["restored_from"] == 0


def test_the_inference_limit_never_applies_to_a_service_the_policy_does_not_name():
    # The policy limits Ollama; this runner sends model requests to vLLM.
    made = controller(policy_version(1, inference=1, tests=1), inference_service="vllm")
    first = acquire(made, "inference_requests", "first")
    second = acquire(made, "inference_requests", "second")
    record = second.snapshot()
    assert (record["enforced"], record["limit"]) == (False, None)
    assert "ollama" in record["not_enforced_reason"] and "vllm" in record["not_enforced_reason"]
    # The test-job limit names the runner's own service, so it still applies.
    tests = acquire(made, "test_jobs", "tests")
    assert tests.snapshot()["enforced"] is True
    with pytest.raises(AdmissionRefused):
        acquire(made, "test_jobs", "refused", deadline=time.monotonic() + 0.05)
    for permit in (first, second, tests):
        permit.release()
    controls = made.summary()["controls"]
    assert controls["inference_requests"]["enforced"] is False
    assert controls["test_jobs"]["enforced"] is True


def test_release_is_idempotent_and_claimed_work_returns_its_own_permit():
    made = controller(policy_version(1, inference=1))
    permit = acquire(made, "inference_requests", "call")
    permit.release()
    permit.release()
    assert active(made, "inference_requests") == 0
    assert permit.claim() is False  # a returned permit can never be used again

    claimed = acquire(made, "inference_requests", "abandoned")
    assert claimed.claim() is True
    claimed.release_unless_claimed()  # the waiter gave up; the work still runs
    assert active(made, "inference_requests") == 1
    snapshot = claimed.snapshot()
    assert (snapshot["released"], snapshot["held_by_unfinished_work"]) == (False, True)
    assert not made.wait_until_idle(0.05)
    claimed.release()  # the work ended
    assert made.wait_until_idle(1)
    assert claimed.snapshot()["hold_seconds"] >= 0

    unclaimed = acquire(made, "inference_requests", "never-started")
    unclaimed.release_unless_claimed()
    assert unclaimed.snapshot()["released"] is True
    assert made.open_permits() == 0


def test_an_admission_record_correlates_policy_step_resource_and_clock():
    made = controller(policy_version(4, tests=1), max_queue_length=3)
    permit = acquire(made, step="verify-tests")
    permit.release()
    record = permit.snapshot()
    assert {
        key: record[key]
        for key in ("control", "step_id", "span_id", "resource_id", "service_id", "outcome")
    } == {
        "control": "test_jobs",
        "step_id": "verify-tests",
        "span_id": "span-verify-tests",
        "resource_id": "resource-test_jobs",
        "service_id": "agentwatch-agent-runner",
        "outcome": "admitted",
    }
    assert (record["controller_id"], record["clock_id"]) == (made.controller_id, CLOCK.clock_id)
    assert record["policy_version"] == record["policy_version_at_arrival"] == 4
    assert record["arrived_tick_ns"] <= record["admitted_tick_ns"] <= record["released_tick_ns"]
    # Waits and holds are differences of the recorded ticks, the ones coverage subtracts.
    assert record["admission_wait_seconds"] == (
        (record["admitted_tick_ns"] - record["arrived_tick_ns"]) / 1e9
    )
    assert record["hold_seconds"] == (record["released_tick_ns"] - record["admitted_tick_ns"]) / 1e9
    assert (record["max_queue_length"], record["released"]) == (3, True)
    summary = made.summary()
    assert (summary["enforcement_scope"], summary["pid"]) == ("runner_process", made.pid)
    assert "not global" in summary["scope_note"]


@pytest.mark.parametrize(
    ("policy", "action"),
    [
        (policy_version(0), "baseline"),
        (policy_version(3), "disable"),
        (policy_version(4, action="restore", restored_from=0), "restore"),
    ],
)
def test_a_policy_without_a_limit_is_never_reported_as_enforced(policy, action):
    made = controller(policy)
    for control in ("inference_requests", "test_jobs"):
        permit = acquire(made, control, f"step-{control}")
        record = permit.snapshot()
        permit.release()
        assert (record["limit"], record["enforced"]) == (None, False)
        assert record["not_enforced_reason"] == (
            f"Policy version {policy.version} ({action}) sets no {control} limit"
        )
        state = made.summary()["controls"][control]
        assert (state["enforced"], state["limit"]) == (False, None)
    applied = made.summary()["policy_history"][-1]["controls"]["test_jobs"]
    assert (applied["configured_limit"], applied["enforced"]) == (None, False)


def test_a_refusal_measures_its_wait_between_its_own_ticks():
    made = controller(policy_version(1, tests=1), max_queue_length=0)
    holder = acquire(made, step="holder")
    with pytest.raises(AdmissionRefused) as refused:
        acquire(made, step="refused")
    record = refused.value.record
    assert record["admission_wait_seconds"] == (
        (record["refused_tick_ns"] - record["arrived_tick_ns"]) / 1e9
    )
    holder.release()


def test_a_policy_failure_refuses_waiters_and_new_arrivals_but_keeps_held_permits():
    made = controller(policy_version(1, tests=1))
    holder = acquire(made, step="holder")
    waiter = Waiter(made, "waiter")
    waiter.start()
    wait_for(lambda: queued(made) == 1)
    made.fail("PolicyError: Active policy went from version 3 back to 1")
    # Refused at once, long before its 600 s wait bound, and never admitted.
    assert waiter.done.wait(5) and waiter.permit is None
    assert waiter.refused.outcome == "policy_unavailable"
    assert "went from version 3 back to 1" in waiter.refused.reason
    with pytest.raises(AdmissionRefused) as late:
        acquire(made, step="late")
    assert late.value.outcome == "policy_unavailable"
    # Work already admitted keeps its permit until it ends, like lowering a limit.
    assert active(made) == 1
    holder.release()
    assert made.open_permits() == 0 and queued(made) == 0
    summary = made.summary()
    assert "back to 1" in summary["policy_failure"]
    assert summary["controls"]["test_jobs"]["refused"] == {"policy_unavailable": 2}
    # The controller stays stopped: no later read or version brings admission back.
    with pytest.raises(PolicyError, match="policy failure"):
        made.ensure_policy(FakePolicySource(policy_version(2, tests=1)), NO_SLEEP)
    with pytest.raises(PolicyError, match="policy failure"):
        made.apply(policy_version(2, tests=1))


def test_a_policy_that_cannot_be_enforced_faithfully_is_refused():
    current = policy_version(3, tests=2)
    made = controller(current)
    assert made.apply(current.model_copy(deep=True)) is False  # an exact repeat changes nothing
    with pytest.raises(PolicyError, match="changed content"):
        made.apply(policy_version(3, tests=1))
    with pytest.raises(PolicyError, match="back to"):
        made.apply(policy_version(2, tests=1))
    other = policy_version(4).model_copy(update={"target": "another-agent"})
    with pytest.raises(PolicyError, match="target"):
        made.apply(other)
    unnamed = policy_version(4).model_copy(update={"controlled_services": {}})
    with pytest.raises(PolicyError, match="no controlled service"):
        made.apply(unnamed)
    assert made.policy_version == 3  # a refused version never replaces the one in force


def test_admission_needs_a_policy_and_the_governed_service():
    made = AdmissionController(inference_service_id="ollama")
    with pytest.raises(RuntimeError, match="No policy"):
        acquire(made)
    made.apply(policy_version(0))
    with pytest.raises(ValueError, match="admission is for service"):
        acquire(made, "inference_requests", service_id="vllm")
    with pytest.raises(ValueError, match="Unknown admission control"):
        acquire(made, "retrieval", service_id=None)


@pytest.mark.parametrize(
    "bounds",
    [
        {"max_queue_length": -1},
        {"max_queue_length": 1025},
        {"max_queue_length": True},
        {"max_queue_length": 2.0},
        {"max_wait_seconds": 0},
        {"max_wait_seconds": float("inf")},
        {"max_wait_seconds": float("nan")},
        {"max_wait_seconds": 3601},
        {"max_wait_seconds": True},
    ],
)
def test_queue_bounds_outside_their_range_are_rejected(bounds):
    with pytest.raises(ValueError):
        QueueBounds(**bounds)


def test_many_waiters_under_churn_are_all_admitted_in_order_without_leaks():
    made = controller(policy_version(1, inference=2), max_queue_length=64)
    order, lock = [], threading.Lock()

    def work(index):
        permit = acquire(made, "inference_requests", f"s{index}")
        with lock:
            order.append(permit.snapshot()["arrived_tick_ns"])
        threading.Event().wait(0.002)
        permit.release()

    threads = [threading.Thread(target=work, args=(i,)) for i in range(40)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(10)
    summary = made.summary()["controls"]["inference_requests"]
    assert summary["admitted"] == 40 and not summary["refused"]
    assert made.open_permits() == 0
    assert summary["longest_wait_seconds"] < 5


def test_ensure_policy_reads_once_and_retries_only_transient_failures():
    source = FakePolicySource(policy_version(2, tests=1))
    source.failures = [URLError("connection refused")]
    made = AdmissionController(inference_service_id="ollama")
    made.ensure_policy(source, NO_SLEEP)
    made.ensure_policy(source, NO_SLEEP)
    assert (made.policy_version, source.reads) == (2, 2)  # one failure, one read applied

    refused = FakePolicySource(policy_version(1))
    refused.failures = [HTTPError("http://policy", 404, "Unsupported policy target", {}, None)]
    other = AdmissionController(inference_service_id="ollama")
    with pytest.raises(Exception, match="404"):
        other.ensure_policy(refused, NO_SLEEP)
    assert other.policy_version is None and refused.reads == 1


def test_the_watcher_applies_new_versions_and_keeps_the_last_one_through_outages():
    source = FakePolicySource(policy_version(1, tests=1))
    made = controller(policy_version(1, tests=1))
    with PolicyWatcher(source, made, poll_seconds=0.01) as watcher:
        source.policy = policy_version(2, tests=3)
        wait_for(lambda: made.policy_version == 2)
        source.failures = [URLError("down")] * 3
        wait_for(lambda: watcher.poll_failures == 3)
        # An unreachable policy service never becomes "no limits".
        assert made.summary()["controls"]["test_jobs"]["limit"] == 3
        source.policy = policy_version(3, tests=2)
        wait_for(lambda: made.policy_version == 3)
    summary = watcher.summary()
    assert summary["poll_failures"] == 3 and "down" in summary["last_error"]
    assert summary["stopped_by_error"] is None and watcher.error is None


@pytest.mark.parametrize(
    "failure",
    [
        HTTPError("http://policy", 404, "Unsupported policy target", {}, None),
        "backwards",
    ],
)
def test_the_watcher_stops_on_an_error_it_cannot_retry_and_keeps_it(failure):
    source = FakePolicySource(policy_version(5, tests=1))
    made = controller(policy_version(5, tests=1))
    if failure == "backwards":
        source.policy = policy_version(4, tests=9)
    else:
        source.failures = [failure]
    with PolicyWatcher(source, made, poll_seconds=0.01) as watcher:
        wait_for(lambda: watcher.error is not None)
        reads = source.reads
        threading.Event().wait(0.05)
        assert source.reads == reads  # polling stopped
    assert made.policy_version == 5
    assert watcher.summary()["stopped_by_error"]
    if failure == "backwards":
        assert isinstance(watcher.error, PolicyError)
    # The watcher's failure stops admission at once, not at the runs' next checkpoint.
    assert made.policy_failure is not None
    with pytest.raises(AdmissionRefused) as refused:
        acquire(made, step="after-failure")
    assert refused.value.outcome == "policy_unavailable"


def test_the_watcher_rejects_a_poll_interval_that_is_not_positive():
    made = controller()
    for poll in (0, -1, float("inf"), True):
        with pytest.raises(ValueError):
            PolicyWatcher(FakePolicySource(policy_version(0)), made, poll_seconds=poll)

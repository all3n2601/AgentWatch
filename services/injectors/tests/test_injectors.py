"""Injector bounds, lifecycle and cleanup, with real child processes and threads.

The model server here is a loopback test double; its responses are synthetic.
"""

import math
import threading

import pytest
from agentwatch_injectors import base, processes, retrieval
from agentwatch_injectors.base import (
    MAX_INJECTION_SECONDS,
    BoundedInjector,
    InjectorCleanupError,
    InjectorError,
)
from agentwatch_injectors.cpu import CpuSpinInjector, _spin, available_cpus
from agentwatch_injectors.inference import CompetingInferenceInjector, loopback_url
from agentwatch_injectors.injection_set import InjectionCancelled, InjectionSet
from agentwatch_injectors.processes import halt_children, new_group, start_children
from agentwatch_injectors.retrieval import IndexQueryFloodInjector
from injector_test_targets import FakeOllama, TestClock, ignore_stop, never_ready, wait_until


@pytest.fixture
def clock():
    return TestClock()


@pytest.fixture
def quick_grace(monkeypatch):
    # Escalation still runs every step; it just waits less at each one.
    monkeypatch.setattr(processes, "STOP_GRACE_SECONDS", 0.2)
    monkeypatch.setattr(processes, "KILL_GRACE_SECONDS", 2.0)
    monkeypatch.setattr(retrieval, "STOP_GRACE_SECONDS", 0.2)


def stopped(injector: BoundedInjector) -> bool:
    return "stopped_tick_ns" in injector.log()


@pytest.mark.parametrize("seconds", [0, -1, math.inf, math.nan, True, MAX_INJECTION_SECONDS + 1])
def test_duration_must_be_a_finite_bound_under_the_cap(clock, seconds):
    with pytest.raises(ValueError, match="max_duration_seconds"):
        CpuSpinInjector(clock, seconds, workers=1)


@pytest.mark.parametrize("workers", [0, available_cpus() + 1, 1.0, True])
def test_cpu_workers_are_bounded_by_the_available_cpus(clock, workers):
    with pytest.raises(ValueError, match="workers"):
        CpuSpinInjector(clock, 5, workers=workers)


def test_cpu_injector_is_stopped_by_its_duration_bound(clock):
    injector = CpuSpinInjector(clock, 0.5, workers=1)
    injector.start()
    try:
        assert wait_until(lambda: stopped(injector))
    finally:
        log = injector.stop()
    assert log["stop_reason"] == "duration_bound"
    assert log["clock_id"] == "test-clock"
    active_seconds = (log["stop_requested_tick_ns"] - log["active_tick_ns"]) / 1e9
    assert 0.5 <= active_seconds < 5
    assert log["cleanup"] == {
        "confirmed": True,
        "children": [{"stopped_by": "exited", "exitcode": 0}],
    }
    activity = log["activity"]
    assert activity["workers_started"] == activity["workers_reported"] == 1
    assert activity["hash_rounds"] > 0 and activity["worker_cpu_seconds"] > 0
    assert activity["workers_stopped_by_backstop"] == 0
    assert log["bounds"] == {
        "max_duration_seconds": 0.5,
        "workers": 1,
        "available_cpus": available_cpus(),
        "backstop_seconds": 20.5,
    }


def test_stop_is_idempotent_and_keeps_the_first_reason(clock):
    injector = CpuSpinInjector(clock, 30, workers=1)
    injector.start()
    first = injector.stop("cancelled")
    assert injector.stop("requested") == first
    assert first["stop_reason"] == "cancelled"
    with pytest.raises(InjectorError, match="already started"):
        injector.start()


def test_stop_before_start_is_refused(clock):
    with pytest.raises(InjectorError, match="never started"):
        CpuSpinInjector(clock, 5, workers=1).stop()


def test_context_manager_stops_on_failure(clock):
    injector = CpuSpinInjector(clock, 30, workers=1)
    with pytest.raises(RuntimeError, match="boom"), injector:
        raise RuntimeError("boom")
    log = injector.log()
    assert log["stop_reason"] == "failure"
    assert log["cleanup"]["confirmed"] is True


def spawned(target, backstop_seconds, name):
    group = new_group()
    start_children(group, target, [()], backstop_seconds, name)
    return group


def test_a_child_stops_itself_at_its_backstop_without_a_supervisor():
    group = spawned(_spin, 0.3, "backstop-test")
    child = group.children[0].process
    child.join(15)
    assert not child.is_alive()
    reports, cleanup = halt_children(group)
    assert reports[0]["stopped_by_backstop"] is True
    assert cleanup == {
        "confirmed": True,
        "children": [{"stopped_by": "exited_before_stop", "exitcode": 0}],
    }


def test_a_child_that_ignores_stop_and_sigterm_is_killed(quick_grace):
    group = spawned(ignore_stop, 30, "stubborn-test")
    reports, cleanup = halt_children(group)
    assert reports == [None]  # missing, never an invented report
    assert cleanup["confirmed"] is True
    assert cleanup["children"][0]["stopped_by"] == "killed"
    assert cleanup["children"][0]["exitcode"] < 0


def test_workers_that_never_apply_load_fail_the_start(monkeypatch):
    monkeypatch.setattr(processes, "READY_TIMEOUT_SECONDS", 1.0)
    group = new_group()
    with pytest.raises(InjectorError, match="did not start within 1 s"):
        start_children(group, never_ready, [()], 30, "never-ready-test")
    # The started child stays in the group, so halting it reports it.
    _reports, cleanup = halt_children(group)
    assert cleanup["confirmed"] is True
    [child] = cleanup["children"]
    # Under load the child may still be exiting when it is halted.
    assert child["stopped_by"] in {"exited", "exited_before_stop"}


def test_a_start_that_times_out_reports_every_worker_it_started(clock, monkeypatch):
    # Regression: a failed start once discarded its children's cleanup and logged an empty,
    # confirmed cleanup even when a worker had been started.
    monkeypatch.setattr(processes, "READY_TIMEOUT_SECONDS", 0.0)
    injector = CpuSpinInjector(clock, 30, workers=1)
    with pytest.raises(InjectorError, match="did not start"):
        injector.start()
    log = injector.log()
    assert log["stop_reason"] == "failure"
    [child] = log["cleanup"]["children"]  # the started worker is reported, not dropped
    # A worker still booting when the start gave up is terminated: also a confirmed stop.
    assert child["stopped_by"] in {"exited", "exited_before_stop", "terminated", "killed"}
    assert log["cleanup"]["confirmed"] is True


def test_a_failed_start_whose_cleanup_is_unconfirmed_raises_cleanup_error(clock, monkeypatch):
    def failing_start(group, *_args):
        group.children.append(object())  # a worker that was started, then not ready
        raise InjectorError("workers did not start")

    def cannot_confirm(group):
        return [None], {"confirmed": False, "children": [{"stopped_by": "still_running"}]}

    monkeypatch.setattr("agentwatch_injectors.cpu.start_children", failing_start)
    monkeypatch.setattr("agentwatch_injectors.cpu.halt_children", cannot_confirm)
    injector = CpuSpinInjector(clock, 30, workers=1)
    with pytest.raises(InjectorCleanupError) as raised:
        injector.start()
    assert isinstance(raised.value.__cause__, InjectorError)
    with pytest.raises(InjectorCleanupError):
        injector.stop()


def test_the_bound_counts_from_when_load_is_applied(clock):
    injector = CpuSpinInjector(clock, 30, workers=1)
    injector.start()
    log = injector.stop()
    assert log["started_tick_ns"] <= log["active_tick_ns"] <= log["stop_requested_tick_ns"]
    assert log["activity"]["hash_rounds"] > 0  # the worker ran before start() returned


def test_unconfirmed_cleanup_raises_on_every_stop(clock, monkeypatch):
    def cannot_confirm(group):
        for child in group.children:
            child.process.kill()
            child.process.join(5)
        return [None], {"confirmed": False, "children": [{"stopped_by": "still_running"}]}

    monkeypatch.setattr("agentwatch_injectors.cpu.halt_children", cannot_confirm)
    injector = CpuSpinInjector(clock, 30, workers=1)
    injector.start()
    with pytest.raises(InjectorCleanupError):
        injector.stop()
    with pytest.raises(InjectorCleanupError):
        injector.stop()
    assert injector.log()["cleanup"]["confirmed"] is False


def test_a_failed_launch_is_cleaned_up_and_raised(clock, monkeypatch):
    def failing_start(*_args):
        raise OSError("cannot spawn")

    monkeypatch.setattr("agentwatch_injectors.cpu.start_children", failing_start)
    injector = CpuSpinInjector(clock, 30, workers=1)
    with pytest.raises(OSError, match="cannot spawn"):
        injector.start()
    log = injector.log()
    assert log["stop_reason"] == "failure"
    assert log["errors"] == ["start: OSError: cannot spawn"]
    assert log["cleanup"]["confirmed"] is True


@pytest.mark.parametrize(
    "url",
    [
        "http://10.0.0.5:11434",
        "http://ollama.internal:11434",
        "http://0.0.0.0:11434",
        "ftp://127.0.0.1",
        "127.0.0.1:11434",
    ],
)
def test_inference_injector_only_targets_loopback_servers(url):
    with pytest.raises(ValueError):
        loopback_url(url)


@pytest.mark.parametrize(
    "url", ["http://127.0.0.1:11435", "http://localhost:11434/", "http://[::1]:11434"]
)
def test_loopback_servers_are_accepted(url):
    assert loopback_url(url) == url.rstrip("/")


@pytest.mark.parametrize(
    ("setting", "value"),
    [("concurrency", 5), ("num_predict", 257), ("request_timeout_seconds", 121), ("num_ctx", 0)],
)
def test_inference_load_is_bounded(clock, setting, value):
    settings = {"url": "http://127.0.0.1:1", "model": "m", "num_ctx": 4096} | {setting: value}
    with pytest.raises(ValueError, match=setting):
        CompetingInferenceInjector(clock, 5, **settings)


def test_competing_requests_use_the_runs_model_and_context(clock):
    with FakeOllama() as server:
        injector = CompetingInferenceInjector(
            clock, 0.6, url=server.url, model="qwen3:1.7b", num_ctx=4096, concurrency=2
        )
        injector.start()
        try:
            assert wait_until(lambda: stopped(injector))
        finally:
            log = injector.stop()
    activity = log["activity"]
    assert log["stop_reason"] == "duration_bound"
    assert activity["report"] == "received"
    assert activity["requests_completed"] > 0 and activity["requests_failed"] == 0
    assert activity["generated_tokens"] == 5 * activity["requests_completed"]
    payload = server.payloads[0]
    assert payload["model"] == "qwen3:1.7b"
    assert payload["options"] == {"temperature": 0, "num_predict": 128, "num_ctx": 4096}
    assert payload["stream"] is False
    assert log["target"]["endpoint_host"] == "127.0.0.1"
    assert log["cleanup"]["confirmed"] is True


def test_without_a_context_size_none_is_sent(clock):
    injector = CompetingInferenceInjector(
        clock, 5, url="http://127.0.0.1:1", model="m", num_ctx=None
    )
    assert "num_ctx" not in injector.payload()["options"]


def test_failed_requests_are_counted_not_hidden(clock):
    with FakeOllama(status=500) as server:
        injector = CompetingInferenceInjector(
            clock, 0.6, url=server.url, model="m", num_ctx=None, concurrency=1
        )
        injector.start()
        try:
            assert wait_until(lambda: stopped(injector))
        finally:
            log = injector.stop()
    activity = log["activity"]
    assert activity["requests_completed"] == 0
    assert activity["requests_failed"] >= 1
    assert activity["failures_by_type"] == {"GenerateRejected": activity["requests_failed"]}
    assert activity["requests_sent"] >= activity["requests_failed"]


def test_an_unreachable_model_server_fails_the_start(clock, monkeypatch):
    # Ready means a request is on the wire, so a refused connection is never "active" load.
    monkeypatch.setattr(processes, "READY_TIMEOUT_SECONDS", 1.0)
    injector = CompetingInferenceInjector(
        clock, 30, url="http://127.0.0.1:1", model="m", num_ctx=None, concurrency=1
    )
    with pytest.raises(InjectorError, match="did not start"):
        injector.start()
    log = injector.log()
    assert log["stop_reason"] == "failure"
    assert log["cleanup"]["confirmed"] is True
    assert len(log["cleanup"]["children"]) == 1
    assert log["activity"].get("requests_sent", 0) == 0  # nothing ever reached a server


def test_stopping_abandons_in_flight_requests_and_ends_the_child(clock):
    with FakeOllama(hold=threading.Event()) as server:
        injector = CompetingInferenceInjector(
            clock, 30, url=server.url, model="m", num_ctx=None, concurrency=2
        )
        injector.start()
        try:
            for _ in range(2):
                assert server.held.acquire(timeout=15)
        finally:
            log = injector.stop()
    activity = log["activity"]
    assert activity["requests_started"] == activity["requests_abandoned_in_flight"] == 2
    assert activity["requests_completed"] == 0
    assert log["cleanup"]["children"] == [{"stopped_by": "exited", "exitcode": 0}]


VLLM_COMPLETION = {
    "choices": [{"index": 0, "message": {"content": "one"}, "finish_reason": "length"}],
    "usage": {"completion_tokens": 7},
}


def test_competing_vllm_requests_use_the_chat_api_and_count_tokens(clock):
    with FakeOllama(reply=VLLM_COMPLETION) as server:
        injector = CompetingInferenceInjector(
            clock, 0.6, url=server.url, model="m@1", num_ctx=None, concurrency=2, server="vllm"
        )
        injector.start()
        try:
            assert wait_until(lambda: stopped(injector))
        finally:
            log = injector.stop()
    activity = log["activity"]
    assert log["method"] == "competing-openai-chat-v1"
    assert log["target"]["server"] == "vllm"
    assert activity["requests_completed"] > 0 and activity["requests_failed"] == 0
    assert activity["generated_tokens"] == 7 * activity["requests_completed"]
    assert set(server.paths) == {"/v1/chat/completions"}
    payload = server.payloads[0]
    assert payload["max_tokens"] == 128 and payload["stream"] is False
    assert payload["messages"][0]["role"] == "user"
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}
    assert log["cleanup"]["confirmed"] is True


@pytest.mark.parametrize(
    "reply",
    [
        {"choices": []},
        {"choices": [{"message": {}, "finish_reason": None}]},
        {"error": "overloaded"},
    ],
)
def test_unfinished_vllm_completions_count_as_failures(clock, reply):
    with FakeOllama(reply=reply) as server:
        injector = CompetingInferenceInjector(
            clock, 0.6, url=server.url, model="m", num_ctx=None, concurrency=1, server="vllm"
        )
        injector.start()
        try:
            assert wait_until(lambda: stopped(injector))
        finally:
            log = injector.stop()
    activity = log["activity"]
    assert activity["requests_completed"] == 0 and activity["requests_failed"] >= 1
    assert activity["failures_by_type"] == {"GenerateRejected": activity["requests_failed"]}


def test_a_vllm_injector_refuses_a_context_size(clock):
    with pytest.raises(ValueError, match="num_ctx does not apply"):
        CompetingInferenceInjector(
            clock, 5, url="http://127.0.0.1:1", model="m", num_ctx=4096, server="vllm"
        )


def test_an_unknown_model_server_is_refused(clock):
    with pytest.raises(ValueError, match="server must be one of"):
        CompetingInferenceInjector(
            clock, 5, url="http://127.0.0.1:1", model="m", num_ctx=None, server="tgi"
        )


def test_a_vllm_injector_only_loads_loopback_servers(clock):
    with pytest.raises(ValueError, match="loopback"):
        CompetingInferenceInjector(
            clock, 5, url="http://10.0.0.5:8000", model="m", num_ctx=None, server="vllm"
        )


def test_retrieval_flood_searches_until_its_bound(clock):
    searched = []
    injector = IndexQueryFloodInjector(
        clock, 0.3, search=searched.append, queries=["mean", "median"], threads=2
    )
    injector.start()
    try:
        assert wait_until(lambda: stopped(injector))
    finally:
        log = injector.stop()
    assert log["stop_reason"] == "duration_bound"
    assert log["activity"]["searches"] > 0
    assert log["activity"]["search_errors"] == []
    assert log["activity"]["threads_running_at_stop"] == 2
    assert set(searched) == {"mean", "median"}
    assert log["cleanup"] == {"confirmed": True, "threads_still_running": 0}


def test_retrieval_search_errors_are_recorded(clock):
    def failing(_query):
        raise KeyError("index gone")

    injector = IndexQueryFloodInjector(clock, 30, search=failing, queries=["q"], threads=1)
    with pytest.raises(InjectorError, match="searches failed"):
        injector.start()  # load never ran, so the start fails
    log = injector.log()
    assert log["stop_reason"] == "failure"
    assert log["activity"]["search_errors"] == ["KeyError: 'index gone'"]
    assert log["activity"]["threads_running_at_stop"] == 0  # it died before the stop request
    assert log["activity"]["searches"] == 0


def test_a_retrieval_thread_that_cannot_stop_is_never_reported_stopped(clock, quick_grace):
    release = threading.Event()
    entered = threading.Event()

    calls = []

    def blocking(query):
        calls.append(query)
        if len(calls) > 1:  # the first search completes, so the start succeeds
            entered.set()
            release.wait(30)

    injector = IndexQueryFloodInjector(clock, 30, search=blocking, queries=["q"], threads=1)
    injector.start()
    try:
        assert entered.wait(10)
        with pytest.raises(InjectorCleanupError):
            injector.stop()
        assert injector.log()["cleanup"] == {"confirmed": False, "threads_still_running": 1}
    finally:
        release.set()


def test_a_retrieval_start_whose_first_search_hangs_raises_cleanup_error(clock, monkeypatch):
    monkeypatch.setattr(retrieval, "READY_TIMEOUT_SECONDS", 0.3)
    monkeypatch.setattr(retrieval, "STOP_GRACE_SECONDS", 0.2)
    release = threading.Event()
    injector = IndexQueryFloodInjector(
        clock, 30, search=lambda _query: release.wait(30), queries=["q"], threads=1
    )
    try:
        with pytest.raises(InjectorCleanupError) as raised:
            injector.start()
        assert isinstance(raised.value.__cause__, InjectorError)
        assert injector.log()["cleanup"] == {"confirmed": False, "threads_still_running": 1}
    finally:
        release.set()


@pytest.mark.parametrize(
    "queries",
    [[], [""], ["x" * 201], ["q"] * 65, [3]],
    ids=["none", "blank", "long", "many", "type"],
)
def test_retrieval_queries_are_validated(clock, queries):
    with pytest.raises(ValueError, match="queries"):
        IndexQueryFloodInjector(clock, 5, search=print, queries=queries)


class RecordingInjector(BoundedInjector):
    """Test double: does no work, records its lifecycle."""

    resource_class = "test"
    method = "test-double"

    def __init__(self, clock, kind, fail_start=False, confirm=True):
        super().__init__(clock, 30)
        self.kind = kind
        self.fail_start = fail_start
        self.confirm = confirm

    def bounds(self):
        return {}

    def target(self):
        return {}

    def _launch(self):
        if self.fail_start:
            raise OSError("start failed")

    def _halt(self):
        return {"work": 1}, {"confirmed": self.confirm}


def test_a_set_stops_started_injectors_when_a_later_one_fails(clock):
    first = RecordingInjector(clock, "cpu")
    second = RecordingInjector(clock, "inference", fail_start=True)
    with pytest.raises(OSError, match="start failed"), InjectionSet(clock, [first, second]):
        pytest.fail("the body must not run")
    assert first.log()["stop_reason"] == "failure"
    assert second.log()["stop_reason"] == "failure"


def test_a_set_stops_every_injector_and_raises_unconfirmed_cleanup(clock):
    leaky = RecordingInjector(clock, "cpu", confirm=False)
    healthy = RecordingInjector(clock, "inference")
    injection = InjectionSet(clock, [leaky, healthy])
    with pytest.raises(InjectorCleanupError), injection:
        pass
    assert healthy.log()["stop_reason"] == "requested"
    log = injection.log()
    assert log["cleanup_confirmed"] is False
    assert [entry["injector"] for entry in log["injections"]] == ["cpu", "inference"]


def test_cancellation_interrupts_the_ramp(clock):
    cancel = threading.Event()
    cancel.set()
    injector = RecordingInjector(clock, "cpu")
    with (
        pytest.raises(InjectionCancelled),
        InjectionSet(clock, [injector], ramp_seconds=30, cancel=cancel),
    ):
        pytest.fail("the body must not run")
    assert injector.log()["stop_reason"] == "cancelled"


def test_a_set_records_the_ramp_in_its_clock(clock):
    injection = InjectionSet(clock, [RecordingInjector(clock, "cpu")], ramp_seconds=0.05)
    with injection:
        pass
    log = injection.log()
    started = log["injections"][0]["started_tick_ns"]
    assert log["ramp_completed_tick_ns"] - started >= 0.05e9
    assert log["cleanup_confirmed"] is True


def test_a_clean_set_runs_no_injectors(clock):
    injection = InjectionSet(clock, [], ramp_seconds=5)
    with injection:
        pass
    assert injection.log()["injections"] == [] and injection.log()["cleanup_confirmed"]


@pytest.mark.parametrize(
    "problem", ["duplicate", "foreign clock", "ramp"], ids=["duplicate", "clock", "ramp"]
)
def test_set_configuration_is_validated(clock, problem):
    injectors = [RecordingInjector(clock, "cpu")]
    ramp = 0.0
    if problem == "duplicate":
        injectors.append(RecordingInjector(clock, "cpu"))
    elif problem == "foreign clock":
        injectors.append(RecordingInjector(TestClock("other"), "inference"))
    else:
        ramp = 31.0
    with pytest.raises(ValueError):
        InjectionSet(clock, injectors, ramp_seconds=ramp)


def test_stop_reasons_are_validated(clock):
    injector = RecordingInjector(clock, "cpu")
    injector.start()
    with pytest.raises(ValueError, match="stop reason"):
        injector.stop("whenever")
    assert injector.stop()["stop_reason"] == "requested"
    assert base.STOP_REASONS == ("requested", "duration_bound", "failure", "cancelled")

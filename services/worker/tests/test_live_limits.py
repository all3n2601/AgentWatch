"""Exercise chunked uploads, deadlines and disconnects without real-time sleeps."""

import asyncio

import pytest
from agentwatch_worker import live_limits


@pytest.mark.parametrize("mode,status", [("chunks", 413), ("deadline", 408), ("disconnect", None)])
def test_live_upload_boundary(monkeypatch, mode, status):
    messages = []
    called = []
    queue = [
        {"type": "http.request", "body": b"x" * 600_000, "more_body": True},
        {"type": "http.request", "body": b"x" * 600_000, "more_body": False},
    ]
    if mode == "disconnect":
        queue = [{"type": "http.disconnect"}]
    if mode == "deadline":
        monkeypatch.setattr(live_limits, "BODY_TIMEOUT_SECONDS", 0)

    async def downstream(scope, receive, send):
        called.append(True)

    async def receive():
        if mode == "deadline":
            await asyncio.Event().wait()
        return queue.pop(0)

    async def send(message):
        messages.append(message)

    asyncio.run(
        live_limits.LiveBodyLimit(downstream)(
            {"type": "http", "path": "/v1/live/runs", "method": "POST"}, receive, send
        )
    )
    assert called == []
    if status is None:
        assert messages == []
    else:
        assert messages[0]["status"] == status


def test_chunked_body_is_replayed_exactly():
    queue = [
        {"type": "http.request", "body": b'{"records":', "more_body": True},
        {"type": "http.request", "body": b"[]}", "more_body": False},
    ]
    received = []

    async def downstream(scope, receive, send):
        received.append(await receive())

    async def receive():
        return queue.pop(0)

    async def send(message):
        raise AssertionError("Middleware must not send its own response for valid bodies")

    asyncio.run(
        live_limits.LiveBodyLimit(downstream)(
            {"type": "http", "path": "/v1/live/runs", "method": "POST"}, receive, send
        )
    )
    assert received == [{"type": "http.request", "body": b'{"records":[]}', "more_body": False}]


@pytest.mark.parametrize("length,status", [(b"1048577", 413), (b"-1", 400), (b"invalid", 400)])
def test_content_length_rejection_does_not_request_body(length, status):
    messages = []

    async def downstream(scope, receive, send):
        raise AssertionError("Rejected body must not reach the endpoint")

    async def receive():
        raise AssertionError("Do not request body/100 Continue for a rejected length")

    async def send(message):
        messages.append(message)

    asyncio.run(
        live_limits.LiveBodyLimit(downstream)(
            {
                "type": "http",
                "path": "/v1/live/runs",
                "method": "POST",
                "headers": [(b"content-length", length), (b"expect", b"100-continue")],
            },
            receive,
            send,
        )
    )
    assert messages[0]["status"] == status

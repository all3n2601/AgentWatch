"""Bound live request memory and upload time before JSON parsing."""

import asyncio

from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

MAX_BODY_BYTES = 1_048_576
BODY_TIMEOUT_SECONDS = 10


class LiveBodyLimit:
    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if (
            scope["type"] != "http"
            or not scope["path"].startswith("/v1/live/")
            or (scope["method"] not in {"POST", "PUT", "PATCH"})
        ):
            await self.app(scope, receive, send)
            return
        body = bytearray()
        try:
            async with asyncio.timeout(BODY_TIMEOUT_SECONDS):
                while True:
                    message = await receive()
                    if message["type"] == "http.disconnect":
                        return
                    body.extend(message.get("body", b""))
                    if len(body) > MAX_BODY_BYTES:
                        await JSONResponse({"detail": "Live request exceeds 1 MiB"}, 413)(
                            scope, receive, send
                        )
                        return
                    if not message.get("more_body", False):
                        break
        except TimeoutError:
            await JSONResponse({"detail": "Live request body timed out"}, 408)(scope, receive, send)
            return

        async def replay() -> Message:
            return {"type": "http.request", "body": bytes(body), "more_body": False}

        await self.app(scope, replay, send)

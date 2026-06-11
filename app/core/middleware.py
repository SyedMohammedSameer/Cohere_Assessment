"""Request-context ASGI middleware.

Assigns each request a correlation id (honoring an inbound `X-Request-ID` if
present), makes it available to every log line for the duration of the request,
echoes it back in the response header, and logs request completion with status
and latency. Implemented as plain ASGI rather than `BaseHTTPMiddleware` so the
request-id context variable is shared with the endpoint and its downstream
calls, not isolated in a separate task.
"""

import logging
import time
import uuid

from starlette.datastructures import Headers, MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.core.logging import request_id_var

logger = logging.getLogger("app.request")


class RequestContextMiddleware:
    """Set up request-id correlation and log request completion."""

    def __init__(self, app: ASGIApp) -> None:
        """Store the wrapped ASGI application."""
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Wrap a single ASGI request with correlation and access logging."""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request_id = Headers(scope=scope).get("x-request-id") or str(uuid.uuid4())
        token = request_id_var.set(request_id)
        start = time.perf_counter()
        status_code = 500

        async def send_with_request_id(message: Message) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
                MutableHeaders(scope=message).append("x-request-id", request_id)
            await send(message)

        try:
            await self.app(scope, receive, send_with_request_id)
        finally:
            latency_ms = round((time.perf_counter() - start) * 1000, 1)
            logger.info(
                "request.completed",
                extra={
                    "event": "request.completed",
                    "method": scope["method"],
                    "path": scope["path"],
                    "status_code": status_code,
                    "latency_ms": latency_ms,
                },
            )
            request_id_var.reset(token)

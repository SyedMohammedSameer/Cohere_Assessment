"""In-process fixed-window rate limiting (ASGI middleware).

Limits requests per client per minute, keyed by the `X-API-Key` header when
present, otherwise the client host. Disabled when `RATE_LIMIT_PER_MINUTE` is 0.
This is a single-process limiter suited to one instance; a horizontally scaled
deployment would use a shared store (for example Redis) so the budget is global.
"""

import json
import logging
import time

from starlette.datastructures import Headers
from starlette.types import ASGIApp, Receive, Scope, Send

from app.core.config import get_settings

logger = logging.getLogger("app.core.rate_limit")

_EXEMPT_PATHS = frozenset({"/health"})


class RateLimitMiddleware:
    """Reject requests that exceed a per-client per-minute budget."""

    def __init__(self, app: ASGIApp) -> None:
        """Store the wrapped app and read the configured limit once."""
        self.app = app
        self._limit = get_settings().rate_limit_per_minute
        self._window = -1
        self._counts: dict[str, int] = {}

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Allow, or reject with 429, a single request."""
        if self._limit <= 0 or scope["type"] != "http" or scope["path"] in _EXEMPT_PATHS:
            await self.app(scope, receive, send)
            return

        if self._over_limit(self._client_key(scope)):
            await self._reject(send)
            return
        await self.app(scope, receive, send)

    def _client_key(self, scope: Scope) -> str:
        """Identify the client by API key header, falling back to host."""
        api_key = Headers(scope=scope).get("x-api-key")
        if api_key:
            return f"key:{api_key}"
        client = scope.get("client")
        return f"ip:{client[0]}" if client else "ip:unknown"

    def _over_limit(self, key: str) -> bool:
        """Increment the client's count for the current minute window.

        Counters are dropped when the window rolls over, so the map only ever
        holds the current minute's active clients rather than growing forever.
        """
        window = int(time.time() // 60)
        if window != self._window:
            self._window = window
            self._counts.clear()
        count = self._counts.get(key, 0) + 1
        self._counts[key] = count
        return count > self._limit

    async def _reject(self, send: Send) -> None:
        """Send a 429 response with the standard error envelope."""
        logger.warning("rate limit exceeded", extra={"event": "rate_limited"})
        body = json.dumps(
            {"error_code": "rate_limited", "detail": "Rate limit exceeded. Try again shortly."}
        ).encode()
        await send(
            {
                "type": "http.response.start",
                "status": 429,
                "headers": [(b"content-type", b"application/json")],
            }
        )
        await send({"type": "http.response.body", "body": body})

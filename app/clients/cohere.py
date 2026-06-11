"""Async wrapper around the Cohere Chat (v2) API.

This is the single place that talks to Cohere. It owns timeouts, retries with
exponential backoff on transient failures, translation of SDK errors into the
app's domain exceptions, and per-call observability (latency and token usage).
The rest of the app depends on the small, provider-neutral types defined here
(`ChatMessage`, `ChatResult`, `ToolCall`, `Citation`), not on the Cohere SDK
directly, which keeps the provider swappable and call sites easy to mock.

The message types mirror the Cohere v2 representation closely enough to carry
single-turn chat, multi-turn conversations, and tool-calling exchanges, which
is also the shape we persist as conversation history in a later phase.
"""

import asyncio
import logging
import time
from collections.abc import AsyncIterator
from typing import Any

import cohere
import httpx
from cohere.core.api_error import ApiError
from pydantic import BaseModel

from app.core.config import Settings
from app.core.exceptions import (
    CohereAuthError,
    CohereError,
    CohereTimeoutError,
    CohereUnavailableError,
)

logger = logging.getLogger(__name__)

# HTTP statuses worth retrying: rate limiting and transient server faults.
_RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})


class ToolCall(BaseModel):
    """A tool invocation requested by the model."""

    id: str
    name: str
    arguments: str  # JSON-encoded arguments, as returned by the model.


class Citation(BaseModel):
    """A span of the answer grounded in one or more retrieved sources."""

    start: int | None = None
    end: int | None = None
    text: str | None = None
    source_ids: list[str] = []


class ChatMessage(BaseModel):
    """A single message in a Cohere v2 conversation.

    Covers every role: `system`, `user`, `assistant` (including tool-call
    turns), and `tool` (results). `content` is a plain string for ordinary
    turns or a list of content blocks (for example tool-result documents).
    """

    role: str
    content: str | list[dict[str, Any]] | None = None
    tool_calls: list[ToolCall] | None = None
    tool_plan: str | None = None
    tool_call_id: str | None = None


class ChatResult(BaseModel):
    """Normalized result of a chat call, decoupled from the SDK response."""

    text: str
    model: str
    finish_reason: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    tool_calls: list[ToolCall] = []
    tool_plan: str | None = None
    citations: list[Citation] = []


class StreamTextDelta(BaseModel):
    """An incremental chunk of generated answer text from a streaming call."""

    text: str


class StreamResult(BaseModel):
    """Terminal event of a streaming call carrying the assembled result."""

    result: ChatResult


def _is_retryable(exc: BaseException) -> bool:
    """True for timeouts, connection errors, and retryable HTTP statuses."""
    if isinstance(exc, httpx.TimeoutException | httpx.ConnectError | httpx.RemoteProtocolError):
        return True
    if isinstance(exc, ApiError):
        return exc.status_code in _RETRYABLE_STATUS
    return False


class CohereClient:
    """Thin, resilient async client for the Cohere Chat API."""

    def __init__(
        self,
        client: cohere.AsyncClientV2,
        model: str,
        max_attempts: int,
        retry_min_s: float,
        retry_max_s: float,
    ) -> None:
        """Wrap an SDK client; injected so tests can pass a mock."""
        self._client = client
        self._model = model
        self._max_attempts = max_attempts
        self._retry_min_s = retry_min_s
        self._retry_max_s = retry_max_s

    @classmethod
    def from_settings(cls, settings: Settings) -> "CohereClient":
        """Build a client from application settings."""
        sdk_client = cohere.AsyncClientV2(
            api_key=settings.cohere_api_key,
            timeout=settings.cohere_timeout_s,
        )
        return cls(
            client=sdk_client,
            model=settings.cohere_model,
            max_attempts=settings.cohere_max_attempts,
            retry_min_s=settings.cohere_retry_min_s,
            retry_max_s=settings.cohere_retry_max_s,
        )

    async def aclose(self) -> None:
        """Close the SDK's HTTP pool, best effort.

        The SDK exposes no public close, so we reach the wrapped httpx client
        defensively and ignore failures.
        """
        httpx_client = _nested(self._client, "_client_wrapper", "httpx_client", "httpx_client")
        aclose = getattr(httpx_client, "aclose", None)
        if aclose is not None:
            try:
                await aclose()
            except Exception:  # pragma: no cover - shutdown is best effort
                logger.debug("cohere client close skipped", exc_info=True)

    async def chat_stream(
        self,
        messages: list[ChatMessage],
        tools: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[StreamTextDelta | StreamResult]:
        """Stream a chat call, yielding text deltas then a terminal result.

        This is the single model-call path: the buffered `/chat` endpoint drains
        it too (see `ChatService.answer`), so both share one tool loop. Yields a
        `StreamTextDelta` per text chunk, then one `StreamResult` with the
        assembled reply (tool calls, usage, finish reason, citations).

        A stream cannot be replayed once tokens are emitted, so retries cover
        establishment only: a transient failure before the first token is retried
        with backoff; one after it is raised as a domain exception.
        """
        payload = _to_payload(messages)
        attempt = 0
        while True:
            attempt += 1
            text_parts: list[str] = []
            tool_plan_parts: list[str] = []
            tool_calls: dict[int, dict[str, Any]] = {}
            citations: list[Citation] = []
            finish_reason: str | None = None
            input_tokens: int | None = None
            output_tokens: int | None = None
            emitted = False
            start = time.perf_counter()

            try:
                # The SDK accepts message and tool dicts at runtime; we pass our
                # provider-neutral dicts rather than constructing SDK types.
                stream = self._client.chat_stream(
                    model=self._model,
                    messages=payload,  # type: ignore[arg-type]
                    tools=tools,  # type: ignore[arg-type]
                )
                async for event in stream:
                    event_type = getattr(event, "type", None)
                    if event_type == "content-delta":
                        text = _nested(event, "delta", "message", "content", "text")
                        if text:
                            text_parts.append(text)
                            emitted = True
                            yield StreamTextDelta(text=text)
                    elif event_type == "tool-plan-delta":
                        # The model's reasoning before a tool call.
                        plan = _nested(event, "delta", "message", "tool_plan")
                        if plan:
                            tool_plan_parts.append(plan)
                    elif event_type == "tool-call-start":
                        _begin_tool_call(tool_calls, event)
                    elif event_type == "tool-call-delta":
                        _extend_tool_call(tool_calls, event)
                    elif event_type == "citation-start":
                        raw = _nested(event, "delta", "message", "citations")
                        parsed = _parse_stream_citation(raw)
                        if parsed is not None:
                            citations.append(parsed)
                    elif event_type == "message-end":
                        finish_reason = _nested(event, "delta", "finish_reason")
                        input_tokens, output_tokens = _usage_tokens(
                            _nested(event, "delta", "usage")
                        )
            except (httpx.TimeoutException, httpx.ConnectError, httpx.RemoteProtocolError) as exc:
                if not emitted and attempt < self._max_attempts:
                    await self._backoff(attempt)
                    continue
                raise CohereTimeoutError("The language model request timed out.") from exc
            except ApiError as exc:
                if not emitted and attempt < self._max_attempts and _is_retryable(exc):
                    logger.warning(
                        "cohere.chat_stream retrying",
                        extra={
                            "event": "cohere.retry",
                            "status_code": exc.status_code,
                            "attempt": attempt,
                        },
                    )
                    await self._backoff(attempt)
                    continue
                raise self._translate_api_error(exc) from exc

            result = ChatResult(
                text="".join(text_parts),
                model=self._model,
                finish_reason=finish_reason,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                tool_calls=[
                    ToolCall(id=call["id"], name=call["name"], arguments="".join(call["args"]))
                    for call in tool_calls.values()
                    if call["id"] and call["name"]
                ],
                tool_plan="".join(tool_plan_parts) or None,
                citations=citations,
            )
            logger.info(
                "cohere.chat_stream completed",
                extra={
                    "event": "cohere.chat_stream",
                    "model": result.model,
                    "latency_ms": round((time.perf_counter() - start) * 1000, 1),
                    "input_tokens": result.input_tokens,
                    "output_tokens": result.output_tokens,
                    "finish_reason": result.finish_reason,
                    "tool_calls": len(result.tool_calls),
                },
            )
            yield StreamResult(result=result)
            return

    async def _backoff(self, attempt: int) -> None:
        """Sleep with exponential backoff before retrying stream establishment."""
        delay = min(self._retry_max_s, self._retry_min_s * (2 ** (attempt - 1)))
        await asyncio.sleep(delay)

    def _translate_api_error(self, exc: ApiError) -> CohereError:
        """Map an SDK API error to a domain exception."""
        if exc.status_code in (401, 403):
            logger.error(
                "cohere.chat auth failure",
                extra={"event": "cohere.error", "status_code": exc.status_code},
            )
            return CohereAuthError("The language model is misconfigured.")
        logger.error(
            "cohere.chat upstream failure",
            extra={"event": "cohere.error", "status_code": exc.status_code},
        )
        return CohereUnavailableError("The language model is currently unavailable.")


def _to_payload(messages: list[ChatMessage]) -> list[dict[str, Any]]:
    """Convert domain messages into Cohere v2 message dicts."""
    payload: list[dict[str, Any]] = []
    for message in messages:
        item: dict[str, Any] = {"role": message.role}
        if message.content is not None:
            item["content"] = message.content
        if message.tool_calls:
            item["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": call.arguments},
                }
                for call in message.tool_calls
            ]
        if message.tool_plan is not None:
            item["tool_plan"] = message.tool_plan
        if message.tool_call_id is not None:
            item["tool_call_id"] = message.tool_call_id
        payload.append(item)
    return payload


def _parse_stream_citation(citation: object) -> Citation | None:
    """Map a streamed citation object into the app's `Citation` type."""
    if citation is None:
        return None
    sources = getattr(citation, "sources", None) or []
    source_ids = [sid for sid in (getattr(s, "id", None) for s in sources) if sid]
    return Citation(
        start=getattr(citation, "start", None),
        end=getattr(citation, "end", None),
        text=getattr(citation, "text", None),
        source_ids=source_ids,
    )


def _usage_tokens(usage: object) -> tuple[int | None, int | None]:
    """Read input and output token counts from a usage object, if present."""
    tokens = getattr(usage, "tokens", None)
    if tokens is None:
        return None, None
    input_tokens = getattr(tokens, "input_tokens", None)
    output_tokens = getattr(tokens, "output_tokens", None)
    return (
        int(input_tokens) if input_tokens is not None else None,
        int(output_tokens) if output_tokens is not None else None,
    )


def _nested(obj: object, *attrs: str) -> Any:
    """Follow a chain of attributes, returning None if any link is missing."""
    for attr in attrs:
        if obj is None:
            return None
        obj = getattr(obj, attr, None)
    return obj


def _begin_tool_call(tool_calls: dict[int, dict[str, Any]], event: object) -> None:
    """Record the start of a streamed tool call at its index."""
    index = getattr(event, "index", 0) or 0
    call = _nested(event, "delta", "message", "tool_calls")
    tool_calls[index] = {
        "id": getattr(call, "id", None),
        "name": _nested(call, "function", "name"),
        "args": [_nested(call, "function", "arguments") or ""],
    }


def _extend_tool_call(tool_calls: dict[int, dict[str, Any]], event: object) -> None:
    """Append an argument fragment to a streamed tool call."""
    index = getattr(event, "index", 0) or 0
    fragment = _nested(event, "delta", "message", "tool_calls", "function", "arguments")
    if fragment and index in tool_calls:
        tool_calls[index]["args"].append(fragment)

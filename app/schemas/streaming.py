"""Server-sent event (SSE) types and serialization for streaming chat.

These are the events emitted over `POST /chat/stream`. Each maps to a named SSE
event whose `data` is the JSON body. The sequence for a grounded answer is:
zero or more `tool_call` and `sources` events, then a run of `token` events, and
finally one `done` event. Failures after the stream has started arrive as a
single `error` event (failures before it, such as a bad key or unknown
conversation, are ordinary HTTP errors).
"""

from typing import Any

from pydantic import BaseModel


class TokenChunk(BaseModel):
    """An incremental piece of the answer text."""

    text: str


class ToolCallStatus(BaseModel):
    """Notice that a tool is being invoked, for a progress indicator."""

    tool: str
    query: str


class SourcesChunk(BaseModel):
    """The Wikipedia sources retrieved so far."""

    sources: list[dict[str, Any]]


class StreamDone(BaseModel):
    """Terminal event with the conversation id and answer metadata."""

    conversation_id: str
    response: str
    model: str
    finish_reason: str | None = None
    usage: dict[str, Any] | None = None
    sources: list[dict[str, Any]] = []
    citations: list[dict[str, Any]] = []


class StreamError(BaseModel):
    """An error raised after streaming began."""

    error_code: str
    detail: str


StreamEvent = TokenChunk | ToolCallStatus | SourcesChunk | StreamDone | StreamError

_EVENT_NAMES: dict[type, str] = {
    TokenChunk: "token",
    ToolCallStatus: "tool_call",
    SourcesChunk: "sources",
    StreamDone: "done",
    StreamError: "error",
}


def format_sse(event: StreamEvent) -> str:
    """Serialize a stream event into the SSE wire format.

    The result is an `event:` line naming the event, a `data:` line with the JSON
    body, and a blank line terminator.
    """
    return f"event: {_EVENT_NAMES[type(event)]}\ndata: {event.model_dump_json()}\n\n"

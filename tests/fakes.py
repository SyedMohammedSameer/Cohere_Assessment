"""Reusable test doubles and builders.

These let the tests run without network access or a real API key by standing in
for the Cohere and Wikipedia clients and, at the HTTP layer, for the whole chat
service. They are duck-typed against the real classes' public methods.
"""

import json
from typing import Any

from app.clients.cohere import (
    ChatMessage,
    ChatResult,
    Citation,
    StreamResult,
    StreamTextDelta,
    ToolCall,
)
from app.schemas.streaming import SourcesChunk, TokenChunk, ToolCallStatus
from app.services.chat import AnswerDoneEvent, AnswerResult, RetrievedSource


def make_tool_call(query: str, call_id: str = "call-1", name: str = "search_wikipedia") -> ToolCall:
    """Build a tool call with JSON-encoded `{"query": ...}` arguments."""
    return ToolCall(id=call_id, name=name, arguments=json.dumps({"query": query}))


def make_chat_result(
    text: str = "",
    *,
    tool_calls: list[ToolCall] | None = None,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    finish_reason: str = "COMPLETE",
    citations: list[Citation] | None = None,
    model: str = "test-model",
) -> ChatResult:
    """Build a `ChatResult` for scripting the fake Cohere client."""
    return ChatResult(
        text=text,
        model=model,
        finish_reason=finish_reason,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        tool_calls=tool_calls or [],
        citations=citations or [],
    )


class ScriptedCohereClient:
    """Returns pre-scripted `ChatResult` objects in order and records calls."""

    def __init__(self, results: list[ChatResult]) -> None:
        self._results = list(results)
        self.calls: list[dict[str, Any]] = []

    async def chat_stream(
        self,
        messages: list[ChatMessage],
        tools: list[dict[str, Any]] | None = None,
    ):
        """Stream the next scripted result: text in two chunks, then the result."""
        self.calls.append({"messages": messages, "tools": tools})
        result = self._results.pop(0)
        if result.text and not result.tool_calls:
            middle = max(1, len(result.text) // 2)
            yield StreamTextDelta(text=result.text[:middle])
            yield StreamTextDelta(text=result.text[middle:])
        yield StreamResult(result=result)


class StubWikipediaClient:
    """Returns canned articles, or raises a provided error, and records queries."""

    def __init__(self, articles: list[Any] | None = None, error: Exception | None = None) -> None:
        self._articles = articles or []
        self._error = error
        self.queries: list[str] = []

    async def search(self, query: str, limit: int | None = None) -> list[Any]:
        """Record the query and return canned articles or raise the error."""
        self.queries.append(query)
        if self._error is not None:
            raise self._error
        return self._articles


class FakeChatService:
    """Stands in for `ChatService` at the HTTP layer; records replayed history."""

    def __init__(
        self,
        *,
        text: str = "Answer",
        sources: list[RetrievedSource] | None = None,
        tool_invocations: int = 1,
    ) -> None:
        self._text = text
        self._sources = sources or []
        self._tool_invocations = tool_invocations
        self.calls: list[dict[str, Any]] = []

    def _result(self, turn: int) -> AnswerResult:
        citations = (
            [Citation(start=0, end=6, text=self._text[:6], source_ids=[self._sources[0].id])]
            if self._sources
            else []
        )
        return AnswerResult(
            text=f"{self._text} {turn}",
            model="test-model",
            finish_reason="COMPLETE",
            input_tokens=10 * turn,
            output_tokens=2 * turn,
            sources=self._sources,
            citations=citations,
            tool_invocations=self._tool_invocations,
            latency_ms=1.5 * turn,
        )

    async def answer(self, query: str, history: list[ChatMessage] | None = None) -> AnswerResult:
        """Record the query and replayed history, return a deterministic result."""
        self.calls.append({"query": query, "history": list(history or [])})
        return self._result(len(self.calls))

    async def answer_stream(self, query: str, history: list[ChatMessage] | None = None):
        """Stream a deterministic answer: optional tool/sources, tokens, then done."""
        self.calls.append({"query": query, "history": list(history or [])})
        result = self._result(len(self.calls))
        if self._sources:
            yield ToolCallStatus(tool="search_wikipedia", query=query)
            yield SourcesChunk(sources=[s.model_dump() for s in self._sources])
        for word in result.text.split():
            yield TokenChunk(text=word + " ")
        yield AnswerDoneEvent(result=result)

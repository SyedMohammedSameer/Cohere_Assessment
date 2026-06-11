"""Chat orchestration with Wikipedia tool calling.

Runs the multi-turn tool loop: the model is offered a Wikipedia search tool, and
when it requests a search we execute it, feed the results back as a `tool`
message, and call the model again. This repeats until the model returns a final
answer or a safety bound is reached, at which point we make one tool-free call
to force an answer.

The orchestrator depends only on the provider-neutral `CohereClient` and
`WikipediaClient`, so it is straightforward to unit test with both clients
mocked.
"""

import json
import logging
import time
from collections.abc import AsyncIterator
from typing import Any

from pydantic import BaseModel

from app.clients.cohere import (
    ChatMessage,
    ChatResult,
    Citation,
    CohereClient,
    StreamTextDelta,
    ToolCall,
)
from app.clients.wikipedia import WikipediaClient
from app.core.exceptions import WikipediaError
from app.schemas.streaming import SourcesChunk, TokenChunk, ToolCallStatus

logger = logging.getLogger(__name__)

_TOOL_NAME = "search_wikipedia"

# Steers the model to ground in Wikipedia without trapping it into refusing when
# a search returns off-topic results. It may fall back on the conversation and
# its own knowledge, which keeps multi-turn follow-ups answerable.
_SYSTEM_PROMPT = (
    "You are a helpful, accurate assistant with a search_wikipedia tool. Use it to "
    "look up or verify real-world facts, people, places, events, dates, and concepts, "
    "and ground your answer in the results when they are relevant, citing them. You "
    "may also draw on the ongoing conversation and your own knowledge, especially when "
    "the search results are unhelpful or off-topic. Prefer answering the question "
    "accurately and concisely; only say you are unsure when you genuinely cannot "
    "determine the answer."
)

# Cohere v2 tool definition advertised to the model on every chat call.
WIKIPEDIA_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": _TOOL_NAME,
            "description": (
                "Search Wikipedia for factual, up-to-date information about people, "
                "places, events, dates, and concepts. Returns the top matching "
                "article titles and snippets."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search terms to look up on Wikipedia.",
                    }
                },
                "required": ["query"],
            },
        },
    }
]


class RetrievedSource(BaseModel):
    """A Wikipedia article surfaced to the model during grounding."""

    id: str
    title: str
    url: str
    snippet: str | None = None


class AnswerResult(BaseModel):
    """Outcome of an orchestrated chat turn, including grounding metadata."""

    text: str
    model: str
    finish_reason: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    sources: list[RetrievedSource] = []
    citations: list[Citation] = []
    tool_invocations: int = 0
    latency_ms: float | None = None


class AnswerDoneEvent(BaseModel):
    """Internal terminal event of the streaming loop, carrying the full result."""

    result: AnswerResult


def _document(doc_id: str, data: dict[str, str]) -> dict[str, Any]:
    """Build a Cohere v2 document tool-result block."""
    return {"type": "document", "document": {"id": doc_id, "data": data}}


def _tool_error_content(message: str) -> list[dict[str, Any]]:
    """Build a tool result that reports a failure to the model."""
    return [_document("error", {"error": message})]


def _tool_query(call: ToolCall) -> str:
    """Best-effort extraction of the search query from a tool call, for status."""
    try:
        return str(json.loads(call.arguments).get("query", ""))
    except (json.JSONDecodeError, AttributeError, TypeError):
        return ""


class ChatService:
    """Orchestrates grounded chat using the Cohere and Wikipedia clients."""

    def __init__(
        self,
        cohere_client: CohereClient,
        wikipedia_client: WikipediaClient,
        max_iterations: int,
    ) -> None:
        """Initialize the orchestrator."""
        self._cohere = cohere_client
        self._wikipedia = wikipedia_client
        self._max_iterations = max_iterations

    async def answer(
        self,
        query: str,
        history: list[ChatMessage] | None = None,
    ) -> AnswerResult:
        """Answer a query (buffered), grounding it in Wikipedia via tool calling.

        Drains `answer_stream` so the streaming and non-streaming endpoints share
        a single tool loop and one set of behavior.
        """
        final: AnswerResult | None = None
        async for event in self.answer_stream(query, history=history):
            if isinstance(event, AnswerDoneEvent):
                final = event.result
        assert final is not None, "answer_stream always yields a final AnswerDoneEvent"
        return final

    def _build_result(
        self,
        result: Any,
        usage: "_UsageAccumulator",
        sources: dict[str, RetrievedSource],
        tool_invocations: int,
        start: float,
    ) -> AnswerResult:
        """Assemble the final answer with accumulated usage and timing."""
        return AnswerResult(
            text=result.text,
            model=result.model,
            finish_reason=result.finish_reason,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            sources=list(sources.values()),
            citations=result.citations,
            tool_invocations=tool_invocations,
            latency_ms=(time.perf_counter() - start) * 1000,
        )

    async def answer_stream(
        self,
        query: str,
        history: list[ChatMessage] | None = None,
    ) -> AsyncIterator[TokenChunk | ToolCallStatus | SourcesChunk | AnswerDoneEvent]:
        """Answer a query as a stream of events, grounding it in Wikipedia.

        Yields `ToolCallStatus` and `SourcesChunk` while searching, `TokenChunk`
        for each piece of the answer as it is generated, and a final
        `AnswerDoneEvent` with the assembled result and grounding metadata.
        """
        start = time.perf_counter()
        messages: list[ChatMessage] = [ChatMessage(role="system", content=_SYSTEM_PROMPT)]
        messages.extend(history or [])
        messages.append(ChatMessage(role="user", content=query))

        sources: dict[str, RetrievedSource] = {}
        usage = _UsageAccumulator()
        tool_invocations = 0

        for iteration in range(self._max_iterations):
            round_result: ChatResult | None = None
            async for event in self._cohere.chat_stream(messages, tools=WIKIPEDIA_TOOLS):
                if isinstance(event, StreamTextDelta):
                    yield TokenChunk(text=event.text)
                else:
                    round_result = event.result
            assert round_result is not None, "chat_stream always yields a StreamResult"
            usage.add(round_result.input_tokens, round_result.output_tokens)

            if not round_result.tool_calls:
                yield AnswerDoneEvent(
                    result=self._build_result(round_result, usage, sources, tool_invocations, start)
                )
                return

            logger.info(
                "tool loop iteration",
                extra={
                    "event": "tool.loop",
                    "iteration": iteration,
                    "tool_calls": len(round_result.tool_calls),
                },
            )
            messages.append(
                ChatMessage(
                    role="assistant",
                    tool_calls=round_result.tool_calls,
                    tool_plan=round_result.tool_plan,
                )
            )
            for call in round_result.tool_calls:
                tool_invocations += 1
                yield ToolCallStatus(tool=call.name, query=_tool_query(call))
                content, retrieved = await self._run_tool(call)
                for source in retrieved:
                    sources[source.id] = source
                if retrieved:
                    yield SourcesChunk(sources=[s.model_dump() for s in sources.values()])
                messages.append(ChatMessage(role="tool", tool_call_id=call.id, content=content))

        logger.warning(
            "tool loop hit safety bound; forcing a tool-free answer",
            extra={"event": "tool.loop_exhausted", "max_iterations": self._max_iterations},
        )
        round_result = None
        async for event in self._cohere.chat_stream(messages, tools=None):
            if isinstance(event, StreamTextDelta):
                yield TokenChunk(text=event.text)
            else:
                round_result = event.result
        assert round_result is not None, "chat_stream always yields a StreamResult"
        usage.add(round_result.input_tokens, round_result.output_tokens)
        yield AnswerDoneEvent(
            result=self._build_result(round_result, usage, sources, tool_invocations, start)
        )

    async def _run_tool(self, call: ToolCall) -> tuple[list[dict[str, Any]], list[RetrievedSource]]:
        """Execute a single tool call and return its result content.

        Failures degrade gracefully: instead of raising, we return a tool result
        describing the problem so the model can adjust or answer without it.
        """
        if call.name != _TOOL_NAME:
            logger.warning(
                "model requested unknown tool",
                extra={"event": "tool.unknown", "tool": call.name},
            )
            return _tool_error_content(f"Unknown tool '{call.name}'."), []

        try:
            arguments = json.loads(call.arguments)
            search_query = arguments["query"]
        except (json.JSONDecodeError, KeyError, TypeError):
            logger.warning(
                "invalid tool arguments",
                extra={"event": "tool.invalid_args", "call_id": call.id},
            )
            return _tool_error_content("Tool arguments were invalid."), []

        logger.info(
            "tool call dispatched",
            extra={"event": "tool.call", "tool": call.name, "query": search_query},
        )
        try:
            articles = await self._wikipedia.search(search_query)
        except WikipediaError:
            return _tool_error_content("Wikipedia search is currently unavailable."), []

        if not articles:
            no_results = f"No Wikipedia articles found for '{search_query}'."
            return [_document("0", {"result": no_results})], []

        content: list[dict[str, Any]] = []
        retrieved: list[RetrievedSource] = []
        for article in articles:
            doc_id = str(article.id)
            content.append(
                _document(
                    doc_id,
                    {"title": article.title, "snippet": article.snippet, "url": article.url},
                )
            )
            retrieved.append(
                RetrievedSource(
                    id=doc_id,
                    title=article.title,
                    url=article.url,
                    snippet=article.snippet,
                )
            )
        return content, retrieved


class _UsageAccumulator:
    """Sums token usage across the multiple model calls in one tool loop."""

    def __init__(self) -> None:
        self.input_tokens = 0
        self.output_tokens = 0

    def add(self, input_tokens: int | None, output_tokens: int | None) -> None:
        """Add one call's token counts, treating missing counts as zero."""
        self.input_tokens += input_tokens or 0
        self.output_tokens += output_tokens or 0

"""Unit tests for the Cohere wrapper: payload, retries, and stream assembly."""

from types import SimpleNamespace

import httpx
import pytest
from cohere.core.api_error import ApiError

from app.clients.cohere import (
    ChatMessage,
    CohereClient,
    StreamResult,
    StreamTextDelta,
    ToolCall,
    _is_retryable,
    _to_payload,
)
from app.core.exceptions import CohereAuthError, CohereTimeoutError, CohereUnavailableError

MESSAGES = [ChatMessage(role="user", content="hi")]


def make_client(sdk_client, *, max_attempts: int = 3) -> CohereClient:
    """Build a CohereClient around a fake SDK client with no real backoff."""
    return CohereClient(
        client=sdk_client,
        model="test-model",
        max_attempts=max_attempts,
        retry_min_s=0.0,
        retry_max_s=0.0,
    )


def _event(event_type: str, delta: object) -> SimpleNamespace:
    return SimpleNamespace(type=event_type, index=0, delta=delta)


def content(text: str) -> SimpleNamespace:
    return _event(
        "content-delta",
        SimpleNamespace(message=SimpleNamespace(content=SimpleNamespace(text=text))),
    )


def tool_start() -> SimpleNamespace:
    return _event(
        "tool-call-start",
        SimpleNamespace(
            message=SimpleNamespace(
                tool_calls=SimpleNamespace(
                    id="tc1", function=SimpleNamespace(name="search_wikipedia", arguments='{"q')
                )
            )
        ),
    )


def tool_delta() -> SimpleNamespace:
    return _event(
        "tool-call-delta",
        SimpleNamespace(
            message=SimpleNamespace(
                tool_calls=SimpleNamespace(function=SimpleNamespace(arguments='uery":"x"}'))
            )
        ),
    )


def tool_plan(text: str) -> SimpleNamespace:
    return _event("tool-plan-delta", SimpleNamespace(message=SimpleNamespace(tool_plan=text)))


def citation() -> SimpleNamespace:
    return _event(
        "citation-start",
        SimpleNamespace(
            message=SimpleNamespace(
                citations=SimpleNamespace(
                    start=0, end=4, text="Buzz", sources=[SimpleNamespace(id="440")]
                )
            )
        ),
    )


def message_end(input_tokens: int = 12, output_tokens: int = 4) -> SimpleNamespace:
    return _event(
        "message-end",
        SimpleNamespace(
            finish_reason="COMPLETE",
            usage=SimpleNamespace(
                tokens=SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens)
            ),
        ),
    )


class _StreamingSdkClient:
    """Fake SDK whose chat_stream replays attempts; an Exception in a list raises."""

    def __init__(self, attempts: list[list]):
        self._attempts = list(attempts)
        self.call_count = 0

    async def chat_stream(self, model, messages, tools=None):
        self.call_count += 1
        for event in self._attempts.pop(0):
            if isinstance(event, Exception):
                raise event
            yield event


def test_to_payload_renders_all_roles():
    messages = [
        ChatMessage(role="system", content="be helpful"),
        ChatMessage(role="user", content="hi"),
        ChatMessage(
            role="assistant",
            tool_calls=[ToolCall(id="tc1", name="search_wikipedia", arguments='{"query":"x"}')],
            tool_plan="searching",
        ),
        ChatMessage(role="tool", tool_call_id="tc1", content=[{"type": "document"}]),
    ]
    payload = _to_payload(messages)

    assert payload[0] == {"role": "system", "content": "be helpful"}
    assert payload[2]["tool_calls"][0]["function"]["name"] == "search_wikipedia"
    assert payload[2]["tool_plan"] == "searching"
    assert payload[3]["tool_call_id"] == "tc1"


def test_is_retryable_predicate():
    assert _is_retryable(ApiError(status_code=503, body="x")) is True
    assert _is_retryable(ApiError(status_code=429, body="x")) is True
    assert _is_retryable(ApiError(status_code=400, body="x")) is False
    assert _is_retryable(httpx.TimeoutException("t")) is True
    assert _is_retryable(ValueError("nope")) is False


async def test_chat_stream_assembles_text_tool_calls_usage_and_citations():
    events = [
        tool_plan("I'll search "),
        tool_plan("Wikipedia."),
        content("Buzz "),
        content("Aldrin."),
        tool_start(),
        tool_delta(),
        citation(),
        message_end(),
    ]
    client = make_client(_StreamingSdkClient([events]))

    collected = [event async for event in client.chat_stream(MESSAGES)]

    deltas = [event.text for event in collected if isinstance(event, StreamTextDelta)]
    assert deltas == ["Buzz ", "Aldrin."]

    final = next(event for event in collected if isinstance(event, StreamResult)).result
    assert final.text == "Buzz Aldrin."
    assert final.finish_reason == "COMPLETE"
    assert final.input_tokens == 12 and final.output_tokens == 4
    assert final.tool_calls[0].name == "search_wikipedia"
    assert final.tool_calls[0].arguments == '{"query":"x"}'
    assert final.tool_plan == "I'll search Wikipedia."  # captured, not streamed
    assert final.citations[0].source_ids == ["440"]
    assert final.citations[0].text == "Buzz"


async def test_chat_stream_retries_establishment_then_succeeds():
    sdk = _StreamingSdkClient(
        [[ApiError(status_code=503, body="x")], [content("Hi."), message_end()]]
    )
    collected = [event async for event in make_client(sdk).chat_stream(MESSAGES)]

    final = next(event for event in collected if isinstance(event, StreamResult)).result
    assert final.text == "Hi."
    assert sdk.call_count == 2


async def test_chat_stream_does_not_retry_after_first_token():
    sdk = _StreamingSdkClient([[content("Partial "), ApiError(status_code=503, body="x")]])
    with pytest.raises(CohereUnavailableError):
        [event async for event in make_client(sdk).chat_stream(MESSAGES)]
    assert sdk.call_count == 1  # already emitted, so no replay


async def test_chat_stream_translates_auth_error_without_retry():
    sdk = _StreamingSdkClient([[ApiError(status_code=401, body="bad key")]])
    with pytest.raises(CohereAuthError):
        [event async for event in make_client(sdk).chat_stream(MESSAGES)]
    assert sdk.call_count == 1


async def test_chat_stream_translates_timeout():
    sdk = _StreamingSdkClient([[httpx.TimeoutException("slow")], [httpx.TimeoutException("slow")]])
    with pytest.raises(CohereTimeoutError):
        [event async for event in make_client(sdk, max_attempts=2).chat_stream(MESSAGES)]
    assert sdk.call_count == 2

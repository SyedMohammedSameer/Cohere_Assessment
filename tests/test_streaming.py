"""Integration tests for the streaming chat endpoint."""

import json

from app.services.chat import RetrievedSource
from tests.fakes import FakeChatService


def parse_sse(body: str) -> list[tuple[str, dict]]:
    """Parse an SSE response body into (event_name, data) pairs."""
    events: list[tuple[str, dict]] = []
    for block in body.strip().split("\n\n"):
        if not block.strip():
            continue
        lines = block.split("\n")
        name = next(line[len("event: ") :] for line in lines if line.startswith("event: "))
        data = next(line[len("data: ") :] for line in lines if line.startswith("data: "))
        events.append((name, json.loads(data)))
    return events


def test_chat_stream_streams_tokens_and_persists(client):
    response = client.post("/chat/stream", json={"query": "hi"})
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]

    events = parse_sse(response.text)
    names = [name for name, _ in events]
    assert "token" in names
    assert names[-1] == "done"

    tokens = "".join(data["text"] for name, data in events if name == "token")
    done = next(data for name, data in events if name == "done")
    assert tokens.strip() == done["response"]
    assert done["usage"] == {"input_tokens": 10, "output_tokens": 2}

    # The streamed turn was persisted under the returned conversation id.
    history = client.get("/history").json()
    assert history["total"] == 1
    assert history["conversations"][0]["id"] == done["conversation_id"]


def test_chat_stream_emits_tool_and_sources_events(client):
    client.app.state.chat_service = FakeChatService(
        sources=[
            RetrievedSource(
                id="440",
                title="Buzz Aldrin",
                url="https://en.wikipedia.org/?curid=440",
                snippet="second man on the Moon",
            )
        ]
    )
    events = parse_sse(client.post("/chat/stream", json={"query": "who?"}).text)
    names = [name for name, _ in events]

    assert names.index("tool_call") < names.index("sources") < names.index("done")
    sources_event = next(data for name, data in events if name == "sources")
    assert sources_event["sources"][0]["title"] == "Buzz Aldrin"

    # The done event carries citations for parity with the non-streaming endpoint.
    done = next(data for name, data in events if name == "done")
    assert done["citations"][0]["source_ids"] == ["440"]


def test_chat_stream_unknown_conversation_is_404_before_stream(client):
    response = client.post("/chat/stream", json={"query": "hi", "conversation_id": "missing"})
    assert response.status_code == 404
    assert response.json()["error_code"] == "conversation_not_found"


def test_chat_stream_emits_error_event_on_failure(client):
    class BoomStream:
        async def answer_stream(self, query, history=None):
            if False:  # pragma: no cover - makes this an async generator
                yield
            raise RuntimeError("kaboom")

    client.app.state.chat_service = BoomStream()
    events = parse_sse(client.post("/chat/stream", json={"query": "hi"}).text)

    assert events[-1][0] == "error"
    assert events[-1][1]["error_code"] == "internal_error"

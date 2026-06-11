"""Integration tests for the API endpoints.

These run the real app, conversation service, repository, and database; only the
chat service (the model side) is faked, so the full HTTP and persistence path is
exercised without network access.
"""

from app.services.chat import RetrievedSource
from tests.fakes import FakeChatService


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "version": "0.1.0"}


def test_chat_creates_conversation_and_returns_answer(client):
    response = client.post("/chat", json={"query": "Who walked second on the moon?"})
    assert response.status_code == 200
    body = response.json()
    assert body["conversation_id"]
    assert body["response"] == "Answer 1"
    assert body["usage"] == {"input_tokens": 10, "output_tokens": 2}


def test_chat_rejects_blank_query(client):
    response = client.post("/chat", json={"query": "   "})
    assert response.status_code == 422
    # Validation errors use the same envelope as other errors.
    assert response.json()["error_code"] == "validation_error"
    assert "detail" in response.json()


def test_chat_continues_conversation_and_replays_history(client, fake_chat_service):
    first = client.post("/chat", json={"query": "Who was second?"}).json()
    cid = first["conversation_id"]

    second = client.post("/chat", json={"query": "And who was first?", "conversation_id": cid})
    assert second.status_code == 200
    assert second.json()["conversation_id"] == cid

    # The second turn replays the first turn's user and assistant messages.
    replayed = fake_chat_service.calls[1]["history"]
    assert [(m.role, m.content) for m in replayed] == [
        ("user", "Who was second?"),
        ("assistant", "Answer 1"),
    ]


def test_chat_unknown_conversation_returns_404(client):
    response = client.post("/chat", json={"query": "x", "conversation_id": "missing"})
    assert response.status_code == 404
    assert response.json() == {
        "error_code": "conversation_not_found",
        "detail": "Conversation 'missing' was not found.",
    }


def test_history_lists_conversations_with_turns(client):
    client.post("/chat", json={"query": "first question"})
    client.post("/chat", json={"query": "second question"})

    body = client.get("/history").json()
    assert body["total"] == 2
    assert body["limit"] == 20
    assert len(body["conversations"]) == 2
    queries = {c["turns"][0]["query"] for c in body["conversations"]}
    assert queries == {"first question", "second question"}


def test_history_pagination(client):
    for _ in range(3):
        client.post("/chat", json={"query": "q"})

    body = client.get("/history", params={"limit": 1, "offset": 0}).json()
    assert body["total"] == 3
    assert body["limit"] == 1
    assert len(body["conversations"]) == 1


def test_conversation_detail_returns_rich_record(client, fake_chat_service):
    cid = client.post("/chat", json={"query": "Who walked second?"}).json()["conversation_id"]

    body = client.get(f"/conversations/{cid}").json()
    assert body["id"] == cid
    turn = body["turns"][0]
    assert turn["query"] == "Who walked second?"
    assert turn["response"] == "Answer 1"
    assert turn["usage"] == {"input_tokens": 10, "output_tokens": 2}
    assert turn["tool_invocations"] == 1
    assert turn["latency_ms"] == 1.5


def test_conversation_detail_unknown_returns_404(client):
    response = client.get("/conversations/missing")
    assert response.status_code == 404
    assert response.json()["error_code"] == "conversation_not_found"


def test_chat_surfaces_grounding_sources(client):
    # Re-fake with sources to confirm they flow through to the response.
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
    body = client.post("/chat", json={"query": "Who was second?"}).json()
    assert body["sources"][0]["title"] == "Buzz Aldrin"
    assert body["sources"][0]["url"] == "https://en.wikipedia.org/?curid=440"

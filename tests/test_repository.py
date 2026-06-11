"""Unit tests for the conversation repository against a real SQLite session."""

from app.db.repository import ConversationRepository

TURN = {
    "finish_reason": "COMPLETE",
    "input_tokens": 10,
    "output_tokens": 2,
    "tool_invocations": 1,
    "latency_ms": 12.5,
    "sources": [{"id": "440", "title": "Buzz Aldrin", "url": "u", "snippet": "s"}],
    "citations": [{"start": 0, "end": 4, "text": "Buzz", "source_ids": ["440"]}],
}


async def test_create_and_get_conversation(db_session):
    repo = ConversationRepository(db_session)
    conversation_id = await repo.create_conversation("test-model", "public")

    fetched = await repo.get_conversation(conversation_id, "public")
    assert fetched is not None
    assert fetched.id == conversation_id
    assert fetched.model == "test-model"
    assert fetched.owner == "public"
    assert fetched.messages == []


async def test_get_missing_conversation_returns_none(db_session):
    repo = ConversationRepository(db_session)
    assert await repo.get_conversation("does-not-exist", "public") is None


async def test_get_conversation_is_scoped_by_owner(db_session):
    repo = ConversationRepository(db_session)
    conversation_id = await repo.create_conversation("test-model", "owner-a")

    # The owner can read it; another owner cannot (no existence leak).
    assert await repo.get_conversation(conversation_id, "owner-a") is not None
    assert await repo.get_conversation(conversation_id, "owner-b") is None


async def test_add_turn_persists_messages_in_sequence(db_session):
    repo = ConversationRepository(db_session)
    conversation_id = await repo.create_conversation("test-model", "public")

    await repo.add_turn(conversation_id, query="q1", answer="a1", **TURN)
    await repo.add_turn(conversation_id, query="q2", answer="a2", **TURN)

    fetched = await repo.get_conversation(conversation_id, "public")
    messages = fetched.messages
    assert [m.sequence for m in messages] == [0, 1, 2, 3]
    assert [(m.role, m.content) for m in messages] == [
        ("user", "q1"),
        ("assistant", "a1"),
        ("user", "q2"),
        ("assistant", "a2"),
    ]

    # Rich record lives on the assistant rows only.
    assistant = messages[1]
    assert assistant.input_tokens == 10
    assert assistant.latency_ms == 12.5
    assert assistant.sources[0]["title"] == "Buzz Aldrin"
    assert messages[0].input_tokens is None


async def test_list_conversations_scoped_and_paginated(db_session):
    repo = ConversationRepository(db_session)
    for _ in range(3):
        await repo.create_conversation("test-model", "owner-a")
    await repo.create_conversation("test-model", "owner-b")

    page, total = await repo.list_conversations("owner-a", limit=2, offset=0)
    assert total == 3
    assert len(page) == 2

    page2, total2 = await repo.list_conversations("owner-a", limit=2, offset=2)
    assert total2 == 3
    assert len(page2) == 1

    # owner-b sees only its own single conversation.
    _, total_b = await repo.list_conversations("owner-b", limit=10, offset=0)
    assert total_b == 1

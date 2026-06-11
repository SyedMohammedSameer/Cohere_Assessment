"""Data-access layer for conversations and messages.

The only place that issues queries. Methods flush within the caller's session
but do not commit; the caller owns the transaction. Reads and writes are scoped
by `owner` so one caller cannot see or extend another's conversations.

`add_turn` operates by conversation id rather than a live ORM instance, so a turn
can be persisted in a short transaction separate from the one that loaded
history, keeping a connection from being held across the slow external calls.
"""

import datetime as dt
import uuid
from typing import Any

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.db.models import Conversation, Message


class ConversationRepository:
    """Reads and writes conversation history, scoped by owner."""

    def __init__(self, session: AsyncSession) -> None:
        """Operate within the given async session."""
        self._session = session

    async def create_conversation(self, model: str, owner: str) -> str:
        """Create an empty conversation and return its generated id."""
        conversation = Conversation(id=str(uuid.uuid4()), owner=owner, model=model)
        self._session.add(conversation)
        await self._session.flush()
        return conversation.id

    async def get_conversation(self, conversation_id: str, owner: str) -> Conversation | None:
        """Return the owner's conversation with messages loaded, else None.

        None is returned both when the id does not exist and when it belongs to
        another owner, so the response never reveals which.
        """
        result = await self._session.execute(
            select(Conversation)
            .where(Conversation.id == conversation_id, Conversation.owner == owner)
            .options(selectinload(Conversation.messages))
        )
        return result.scalar_one_or_none()

    async def list_conversations(
        self, owner: str, limit: int, offset: int
    ) -> tuple[list[Conversation], int]:
        """Return a page of the owner's conversations (newest first) and the total."""
        total = await self._session.scalar(
            select(func.count()).select_from(Conversation).where(Conversation.owner == owner)
        )
        result = await self._session.execute(
            select(Conversation)
            .where(Conversation.owner == owner)
            .options(selectinload(Conversation.messages))
            .order_by(Conversation.created_at.desc(), Conversation.id.desc())
            .limit(limit)
            .offset(offset)
        )
        return list(result.scalars().all()), int(total or 0)

    async def add_turn(
        self,
        conversation_id: str,
        *,
        query: str,
        answer: str,
        finish_reason: str | None,
        input_tokens: int | None,
        output_tokens: int | None,
        tool_invocations: int | None,
        latency_ms: float | None,
        sources: list[dict[str, Any]],
        citations: list[dict[str, Any]],
    ) -> None:
        """Append a user query and the assistant answer, with the turn's rich record.

        The metric and grounding fields ride on the assistant message; the
        conversation's `updated_at` is bumped to reflect the new activity.
        """
        current_max = await self._session.scalar(
            select(func.max(Message.sequence)).where(Message.conversation_id == conversation_id)
        )
        next_sequence = (current_max if current_max is not None else -1) + 1
        self._session.add_all(
            [
                Message(
                    conversation_id=conversation_id,
                    sequence=next_sequence,
                    role="user",
                    content=query,
                ),
                Message(
                    conversation_id=conversation_id,
                    sequence=next_sequence + 1,
                    role="assistant",
                    content=answer,
                    finish_reason=finish_reason,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    tool_invocations=tool_invocations,
                    latency_ms=latency_ms,
                    sources=sources,
                    citations=citations,
                ),
            ]
        )
        await self._session.execute(
            update(Conversation)
            .where(Conversation.id == conversation_id)
            .values(updated_at=dt.datetime.now(dt.timezone.utc))
        )
        await self._session.flush()

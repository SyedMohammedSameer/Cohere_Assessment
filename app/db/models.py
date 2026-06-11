"""SQLAlchemy ORM models for conversation history.

A conversation is an ordered sequence of messages. We persist the durable
user and assistant turns (the content needed to replay a multi-turn
conversation), and attach the rich per-turn record (token usage, latency,
grounding sources, and citations) to the assistant message that concludes each
turn. The transient tool-call scaffolding exchanged within a single turn is not
persisted; its outcome is summarized by `sources` and `tool_invocations`.
"""

import datetime as dt

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.engine import Base


class Conversation(Base):
    """A single multi-turn conversation."""

    __tablename__ = "conversations"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    # Owner of the conversation, used to scope reads and writes per API key.
    # "public" when authentication is disabled.
    owner: Mapped[str] = mapped_column(String(64), index=True, default="public")
    model: Mapped[str] = mapped_column(String(100))
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    messages: Mapped[list["Message"]] = relationship(
        back_populates="conversation",
        order_by="Message.sequence",
        cascade="all, delete-orphan",
    )


class Message(Base):
    """One turn in a conversation: a user query or an assistant answer.

    The metric and grounding columns are populated only on assistant messages;
    they are null on user messages.
    """

    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    conversation_id: Mapped[str] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE"), index=True
    )
    sequence: Mapped[int] = mapped_column(Integer)
    role: Mapped[str] = mapped_column(String(20))
    content: Mapped[str] = mapped_column(Text)
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    # Rich record, set on assistant turns only.
    finish_reason: Mapped[str | None] = mapped_column(String(50), nullable=True)
    input_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    output_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    tool_invocations: Mapped[int | None] = mapped_column(Integer, nullable=True)
    latency_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    sources: Mapped[list | None] = mapped_column(JSON, nullable=True)
    citations: Mapped[list | None] = mapped_column(JSON, nullable=True)

    conversation: Mapped["Conversation"] = relationship(back_populates="messages")

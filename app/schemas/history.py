"""Response schemas for the history endpoints."""

import datetime as dt

from pydantic import BaseModel, Field

from app.schemas.chat import Citation, Source, UsageInfo


class TurnView(BaseModel):
    """One user query and its assistant answer, with the rich per-turn record."""

    query: str = Field(description="The user's input for this turn.")
    response: str = Field(description="The model's answer for this turn.")
    created_at: dt.datetime = Field(description="When the answer was stored.")
    finish_reason: str | None = Field(default=None, description="Why generation stopped.")
    usage: UsageInfo | None = Field(default=None, description="Token usage for the turn.")
    tool_invocations: int | None = Field(
        default=None, description="Tool calls executed during the turn."
    )
    latency_ms: float | None = Field(default=None, description="Turn latency, in milliseconds.")
    sources: list[Source] = Field(default_factory=list, description="Wikipedia sources used.")
    citations: list[Citation] = Field(default_factory=list, description="Answer-span citations.")


class ConversationView(BaseModel):
    """A conversation and its ordered turns."""

    id: str = Field(description="Conversation identifier.")
    model: str = Field(description="Model used for the conversation.")
    created_at: dt.datetime = Field(description="When the conversation started.")
    updated_at: dt.datetime = Field(description="When the conversation was last updated.")
    turns: list[TurnView] = Field(default_factory=list, description="Turns, oldest first.")


class HistoryResponse(BaseModel):
    """Paginated history of conversations and their turns."""

    total: int = Field(description="Total conversations stored.")
    limit: int = Field(description="Page size used.")
    offset: int = Field(description="Number of conversations skipped.")
    conversations: list[ConversationView] = Field(
        default_factory=list, description="Conversations, newest first."
    )

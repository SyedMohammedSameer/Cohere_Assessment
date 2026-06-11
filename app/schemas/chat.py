"""Request and response schemas for the chat endpoint."""

from typing import Annotated

from pydantic import BaseModel, Field, StringConstraints

# Trimmed, non-empty, length-bounded user input. Validation lives in the schema
# so the route handler can assume clean input.
QueryStr = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=8000),
]


class ChatRequest(BaseModel):
    """Body for `POST /chat`."""

    query: QueryStr = Field(description="The user's natural-language query.")
    conversation_id: str | None = Field(
        default=None,
        description="Existing conversation to continue. Omit to start a new one.",
    )


class UsageInfo(BaseModel):
    """Token usage, summed across every model call made to answer the query."""

    input_tokens: int | None = Field(default=None, description="Tokens in the prompts.")
    output_tokens: int | None = Field(default=None, description="Tokens generated.")


class Source(BaseModel):
    """A Wikipedia article used to ground the answer."""

    id: str = Field(description="Identifier for cross-referencing citations.")
    title: str = Field(description="Article title.")
    url: str = Field(description="Link to the article.")
    snippet: str | None = Field(default=None, description="Matching excerpt from the article.")


class Citation(BaseModel):
    """A span of the answer attributed to one or more sources."""

    start: int | None = Field(default=None, description="Start offset in the answer text.")
    end: int | None = Field(default=None, description="End offset in the answer text.")
    text: str | None = Field(default=None, description="The cited span of answer text.")
    source_ids: list[str] = Field(
        default_factory=list, description="Identifiers of the sources supporting the span."
    )


class ChatResponse(BaseModel):
    """Successful response from `POST /chat`."""

    conversation_id: str = Field(
        description="Conversation this turn belongs to; pass it back to continue."
    )
    response: str = Field(description="The model's answer text.")
    model: str = Field(description="The Cohere model that produced the answer.")
    finish_reason: str | None = Field(
        default=None, description="Why generation stopped, for example 'COMPLETE'."
    )
    usage: UsageInfo | None = Field(default=None, description="Token usage, when reported.")
    sources: list[Source] = Field(
        default_factory=list, description="Wikipedia articles used to ground the answer."
    )
    citations: list[Citation] = Field(
        default_factory=list, description="Spans of the answer attributed to sources."
    )

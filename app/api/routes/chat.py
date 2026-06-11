"""Chat route.

Takes a user query, runs the Wikipedia-grounded tool loop within a new or
existing conversation, persists the turn, and returns the answer with its
sources. The handler only orchestrates at the HTTP boundary: validation lives in
the schema, the tool loop in the chat service, and persistence in the
conversation service.
"""

from typing import Annotated

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from app.api.dependencies import PrincipalDep, get_conversation_service
from app.schemas.chat import ChatRequest, ChatResponse, Citation, Source, UsageInfo
from app.schemas.streaming import format_sse
from app.services.conversation import ConversationService, ConversationTurn

router = APIRouter(tags=["chat"])

ConversationServiceDep = Annotated[ConversationService, Depends(get_conversation_service)]


@router.post("/chat", response_model=ChatResponse)
async def chat(
    payload: ChatRequest, service: ConversationServiceDep, owner: PrincipalDep
) -> ChatResponse:
    """Answer a user query, grounded in Wikipedia, within a conversation."""
    turn = await service.respond(payload.query, payload.conversation_id, owner)
    return _to_response(turn)


@router.post("/chat/stream")
async def chat_stream(
    payload: ChatRequest, service: ConversationServiceDep, owner: PrincipalDep
) -> StreamingResponse:
    """Answer a query as a server-sent event stream.

    Emits `tool_call` and `sources` events while grounding, `token` events as the
    answer is generated, and a final `done` event with the conversation id and
    metadata. Validation errors (auth, unknown conversation) are returned as
    ordinary HTTP errors before the stream begins.
    """
    events = await service.start_stream(payload.query, payload.conversation_id, owner)

    async def event_source():
        async for event in events:
            yield format_sse(event)

    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _to_response(turn: ConversationTurn) -> ChatResponse:
    """Map the orchestrator's domain result onto the API response schema."""
    result = turn.result
    return ChatResponse(
        conversation_id=turn.conversation_id,
        response=result.text,
        model=result.model,
        finish_reason=result.finish_reason,
        usage=UsageInfo(
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
        ),
        sources=[
            Source(id=s.id, title=s.title, url=s.url, snippet=s.snippet) for s in result.sources
        ],
        citations=[
            Citation(start=c.start, end=c.end, text=c.text, source_ids=c.source_ids)
            for c in result.citations
        ],
    )

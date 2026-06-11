"""History routes.

`GET /history` returns a paginated list of past conversations and their turns;
`GET /conversations/{id}` returns a single conversation. Both read through the
repository and map stored rows onto response schemas; the handlers stay thin.
"""

from typing import Annotated

from fastapi import APIRouter, Query

from app.api.dependencies import PrincipalDep, RepositoryDep
from app.core.config import get_settings
from app.core.exceptions import ConversationNotFoundError
from app.db.models import Conversation
from app.schemas.chat import Citation, Source, UsageInfo
from app.schemas.history import ConversationView, HistoryResponse, TurnView

router = APIRouter(tags=["history"])


@router.get("/history", response_model=HistoryResponse)
async def get_history(
    repository: RepositoryDep,
    owner: PrincipalDep,
    limit: Annotated[int | None, Query(ge=1, le=100)] = None,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> HistoryResponse:
    """Return a paginated history of the caller's conversations, newest first."""
    page_size = limit or get_settings().history_page_size
    conversations, total = await repository.list_conversations(
        owner=owner, limit=page_size, offset=offset
    )
    return HistoryResponse(
        total=total,
        limit=page_size,
        offset=offset,
        conversations=[_to_view(conversation) for conversation in conversations],
    )


@router.get("/conversations/{conversation_id}", response_model=ConversationView)
async def get_conversation(
    conversation_id: str, repository: RepositoryDep, owner: PrincipalDep
) -> ConversationView:
    """Return a single conversation owned by the caller, and its turns."""
    conversation = await repository.get_conversation(conversation_id, owner)
    if conversation is None:
        raise ConversationNotFoundError(f"Conversation '{conversation_id}' was not found.")
    return _to_view(conversation)


def _to_view(conversation: Conversation) -> ConversationView:
    """Map a stored conversation onto its response schema, pairing turns."""
    turns: list[TurnView] = []
    pending_query: str | None = None
    for message in conversation.messages:
        if message.role == "user":
            pending_query = message.content
            continue
        turns.append(
            TurnView(
                query=pending_query or "",
                response=message.content,
                created_at=message.created_at,
                finish_reason=message.finish_reason,
                usage=UsageInfo(
                    input_tokens=message.input_tokens,
                    output_tokens=message.output_tokens,
                ),
                tool_invocations=message.tool_invocations,
                latency_ms=message.latency_ms,
                sources=[Source(**source) for source in (message.sources or [])],
                citations=[Citation(**citation) for citation in (message.citations or [])],
            )
        )
        pending_query = None
    return ConversationView(
        id=conversation.id,
        model=conversation.model,
        created_at=conversation.created_at,
        updated_at=conversation.updated_at,
        turns=turns,
    )

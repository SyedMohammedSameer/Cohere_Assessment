"""FastAPI dependency providers.

Centralizing providers here keeps route handlers thin and gives tests a single,
well-known seam to override (for example, swapping the conversation service for
one built on mocked clients via `app.dependency_overrides`).
"""

from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.auth import get_principal
from app.core.config import get_settings
from app.db.repository import ConversationRepository
from app.services.conversation import ConversationService

# Owner id for the current caller (see app.core.auth), used to scope history.
PrincipalDep = Annotated[str, Depends(get_principal)]


async def get_session(request: Request) -> AsyncIterator[AsyncSession]:
    """Yield a request-scoped database session for read endpoints.

    Commits on success and rolls back on any exception. Used by the history
    endpoints; the chat path manages its own short sessions in the service so a
    connection is not held across the model call.
    """
    session_factory = request.app.state.session_factory
    async with session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


SessionDep = Annotated[AsyncSession, Depends(get_session)]


def get_conversation_repository(session: SessionDep) -> ConversationRepository:
    """Provide a repository bound to the request-scoped session."""
    return ConversationRepository(session)


RepositoryDep = Annotated[ConversationRepository, Depends(get_conversation_repository)]


def get_conversation_service(request: Request) -> ConversationService:
    """Assemble the conversation service for a request.

    The stateless chat orchestrator is shared from app state; the service is
    given the session factory so it can open its own short transactions around
    the model call.
    """
    settings = get_settings()
    return ConversationService(
        chat_service=request.app.state.chat_service,
        session_factory=request.app.state.session_factory,
        model=settings.cohere_model,
        max_history_messages=settings.max_history_messages,
    )

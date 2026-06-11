"""Conversation coordination: persistence around the tool loop.

Sits between the HTTP layer and the stateless `ChatService`. For each request
it loads prior turns (for a continuing conversation) or starts a new one, runs
the grounded tool loop, persists the new turn, and returns the answer with its
conversation id.

Persistence is split into short transactions that bracket the model call: prior
turns are read in one transaction, the slow Cohere and Wikipedia calls run with
no database connection held, and the new turn is written in a second
transaction. This keeps a connection from being tied up across multi-second
external I/O. Only the most recent messages are replayed to the model, bounding
prompt tokens and context-window risk for long conversations.
"""

import logging
from collections.abc import AsyncIterator

from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.clients.cohere import ChatMessage
from app.core.exceptions import AppError, ConversationNotFoundError
from app.db.repository import ConversationRepository
from app.schemas.streaming import StreamDone, StreamError, StreamEvent
from app.services.chat import AnswerDoneEvent, AnswerResult, ChatService

logger = logging.getLogger(__name__)


class ConversationTurn(BaseModel):
    """The result of handling one chat request."""

    conversation_id: str
    result: AnswerResult


class ConversationService:
    """Coordinates history persistence around the grounded chat loop."""

    def __init__(
        self,
        chat_service: ChatService,
        session_factory: async_sessionmaker[AsyncSession],
        model: str,
        max_history_messages: int,
    ) -> None:
        """Initialize the service."""
        self._chat = chat_service
        self._session_factory = session_factory
        self._model = model
        self._max_history_messages = max_history_messages

    async def respond(
        self, query: str, conversation_id: str | None, owner: str
    ) -> ConversationTurn:
        """Answer a query within a new or existing conversation."""
        history = await self._load_history(conversation_id, owner)

        result = await self._chat.answer(query, history=history)

        async with self._session_factory() as session:
            repository = ConversationRepository(session)
            target_id = conversation_id or await repository.create_conversation(self._model, owner)
            await self._persist_turn(repository, target_id, query, result)
            await session.commit()

        logger.info(
            "conversation turn stored",
            extra={
                "event": "conversation.turn",
                "conversation_id": target_id,
                "tool_invocations": result.tool_invocations,
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
                "latency_ms": round(result.latency_ms or 0.0, 1),
            },
        )
        return ConversationTurn(conversation_id=target_id, result=result)

    async def start_stream(
        self, query: str, conversation_id: str | None, owner: str
    ) -> AsyncIterator[StreamEvent]:
        """Validate the request and return a stream of chat events.

        History is loaded (and ownership checked) before any event is produced,
        so a missing conversation surfaces as a normal HTTP error rather than
        mid-stream. The returned generator streams the answer and persists the
        turn once it completes.
        """
        history = await self._load_history(conversation_id, owner)
        return self._run_stream(query, conversation_id, owner, history)

    async def _run_stream(
        self,
        query: str,
        conversation_id: str | None,
        owner: str,
        history: list[ChatMessage],
    ) -> AsyncIterator[StreamEvent]:
        """Stream the answer, persist the completed turn, then emit `done`.

        Persistence runs only after the answer fully streams, so if the client
        disconnects mid-stream no partial turn is written (and no database
        connection is held during streaming). A `terminated` flag distinguishes a
        clean finish or handled error from such an abandonment.
        """
        final: AnswerResult | None = None
        terminated = False
        try:
            try:
                async for event in self._chat.answer_stream(query, history=history):
                    if isinstance(event, AnswerDoneEvent):
                        final = event.result
                        break
                    yield event
            except AppError as exc:
                yield StreamError(error_code=exc.error_code, detail=exc.message)
                terminated = True
                return
            except Exception:
                logger.exception("streaming chat failed", extra={"event": "stream.error"})
                yield StreamError(
                    error_code="internal_error", detail="An unexpected error occurred."
                )
                terminated = True
                return

            if final is None:
                yield StreamError(error_code="internal_error", detail="No answer was produced.")
                terminated = True
                return

            try:
                async with self._session_factory() as session:
                    repository = ConversationRepository(session)
                    target_id = conversation_id or await repository.create_conversation(
                        self._model, owner
                    )
                    await self._persist_turn(repository, target_id, query, final)
                    await session.commit()
            except Exception:
                logger.exception("streaming persist failed", extra={"event": "stream.error"})
                yield StreamError(
                    error_code="internal_error", detail="Failed to save the conversation."
                )
                terminated = True
                return

            logger.info(
                "conversation turn stored",
                extra={
                    "event": "conversation.turn",
                    "conversation_id": target_id,
                    "tool_invocations": final.tool_invocations,
                    "input_tokens": final.input_tokens,
                    "output_tokens": final.output_tokens,
                    "latency_ms": round(final.latency_ms or 0.0, 1),
                },
            )
            terminated = True
            yield StreamDone(
                conversation_id=target_id,
                response=final.text,
                model=final.model,
                finish_reason=final.finish_reason,
                usage={"input_tokens": final.input_tokens, "output_tokens": final.output_tokens},
                sources=[source.model_dump() for source in final.sources],
                citations=[citation.model_dump() for citation in final.citations],
            )
        finally:
            if not terminated:
                logger.info(
                    "stream abandoned before completion",
                    extra={"event": "stream.abandoned"},
                )

    async def _load_history(self, conversation_id: str | None, owner: str) -> list[ChatMessage]:
        """Load and window the replay history for a continuing conversation."""
        if conversation_id is None:
            return []

        async with self._session_factory() as session:
            conversation = await ConversationRepository(session).get_conversation(
                conversation_id, owner
            )
        if conversation is None:
            raise ConversationNotFoundError(f"Conversation '{conversation_id}' was not found.")

        messages = [
            ChatMessage(role=message.role, content=message.content)
            for message in conversation.messages
        ]
        # Replay only the most recent messages to bound prompt size. Drop a
        # leading assistant turn (possible when the window cuts mid-turn) so the
        # replayed history begins with a user message, as the Chat API expects
        # after the system prompt.
        windowed = messages[-self._max_history_messages :]
        if windowed and windowed[0].role == "assistant":
            windowed = windowed[1:]
        return windowed

    @staticmethod
    async def _persist_turn(
        repository: ConversationRepository,
        conversation_id: str,
        query: str,
        result: AnswerResult,
    ) -> None:
        """Write one completed turn and its rich record."""
        await repository.add_turn(
            conversation_id,
            query=query,
            answer=result.text,
            finish_reason=result.finish_reason,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            tool_invocations=result.tool_invocations,
            latency_ms=result.latency_ms,
            sources=[source.model_dump() for source in result.sources],
            citations=[citation.model_dump() for citation in result.citations],
        )

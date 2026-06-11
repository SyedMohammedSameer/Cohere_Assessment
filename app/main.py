"""Application entry point.

Builds the FastAPI application via a factory so configuration, logging, the
Cohere client lifecycle, and exception handling are set up in one place and the
app can be constructed cleanly in tests. The module also exposes a module-level
`app` for `uvicorn app.main:app`.
"""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app import __version__
from app.api.routes import chat, health, history
from app.clients.cohere import CohereClient
from app.clients.wikipedia import WikipediaClient
from app.core.config import get_settings
from app.core.exceptions import AppError
from app.core.logging import configure_logging
from app.core.middleware import RequestContextMiddleware
from app.core.rate_limit import RateLimitMiddleware
from app.db.engine import create_engine, create_session_factory, init_models
from app.schemas.error import ErrorResponse
from app.services.chat import ChatService

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Manage startup and shutdown of shared resources.

    Creates the external clients, the chat orchestrator, and the database engine
    once at startup so all requests share connection pools, and exposes them on
    `app.state` for dependency injection. The database schema is initialized, and
    the Wikipedia HTTP pool and the engine are closed on shutdown.
    """
    settings = get_settings()
    cohere_client = CohereClient.from_settings(settings)
    wikipedia_client = WikipediaClient.from_settings(settings)
    app.state.chat_service = ChatService(
        cohere_client=cohere_client,
        wikipedia_client=wikipedia_client,
        max_iterations=settings.max_tool_iterations,
    )

    engine = create_engine(settings.database_url)
    await init_models(engine)
    app.state.session_factory = create_session_factory(engine)

    logger.info("Chat service and database ready (model=%s)", settings.cohere_model)
    try:
        yield
    finally:
        await wikipedia_client.aclose()
        await cohere_client.aclose()
        await engine.dispose()


def register_exception_handlers(app: FastAPI) -> None:
    """Register handlers that render domain errors as a consistent envelope."""

    @app.exception_handler(AppError)
    async def handle_app_error(_: Request, exc: AppError) -> JSONResponse:
        body = ErrorResponse(error_code=exc.error_code, detail=exc.message)
        return JSONResponse(status_code=exc.status_code, content=body.model_dump())

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        # Render request-validation failures in the same envelope as other
        # errors, instead of FastAPI's default list-shaped body.
        errors = exc.errors()
        if errors:
            location = ".".join(str(part) for part in errors[0].get("loc", ()) if part != "body")
            message = errors[0].get("msg", "Invalid request.")
            detail = f"{location}: {message}" if location else message
        else:
            detail = "Invalid request."
        body = ErrorResponse(error_code="validation_error", detail=detail)
        return JSONResponse(status_code=422, content=body.model_dump())

    @app.exception_handler(Exception)
    async def handle_unexpected_error(_: Request, exc: Exception) -> JSONResponse:
        # Catch-all so unexpected failures return our envelope, not a bare 500,
        # and the internal detail is logged rather than leaked to the client.
        logger.exception("unhandled error", extra={"event": "unhandled_error"})
        body = ErrorResponse(error_code="internal_error", detail="An unexpected error occurred.")
        return JSONResponse(status_code=500, content=body.model_dump())


def create_app() -> FastAPI:
    """Construct and configure the FastAPI application."""
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_format)

    app = FastAPI(
        title="Cohere Chat App",
        version=__version__,
        summary="Chat over the Cohere v2 API with a Wikipedia tool and history.",
        lifespan=lifespan,
    )

    # Added inner-first: RequestContext wraps RateLimit so a rate-limited
    # response still gets a request id stamped and logged.
    app.add_middleware(RateLimitMiddleware)
    app.add_middleware(RequestContextMiddleware)
    app.include_router(health.router)
    app.include_router(chat.router)
    app.include_router(history.router)
    register_exception_handlers(app)

    logger.info("Application initialized (version=%s)", __version__)
    return app


app = create_app()

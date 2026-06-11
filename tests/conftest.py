"""Shared pytest fixtures.

A real API key is never required: COHERE_API_KEY is set to a dummy value so
config validation passes, and the Cohere and Wikipedia clients are never
actually called (tests inject fakes). Each test using the HTTP client or the
database gets an isolated SQLite file under a temporary directory.
"""

import os
from contextlib import contextmanager

# Set before any import that reads settings, so config validation passes.
os.environ.setdefault("COHERE_API_KEY", "test-key")

import pytest  # noqa: E402
import pytest_asyncio  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy.ext.asyncio import AsyncSession  # noqa: E402

from app.core.config import get_settings  # noqa: E402
from app.db.engine import create_engine, create_session_factory, init_models  # noqa: E402
from app.main import create_app  # noqa: E402
from tests.fakes import FakeChatService  # noqa: E402


@pytest.fixture
def fake_chat_service() -> FakeChatService:
    """A fresh fake chat service for HTTP-layer tests."""
    return FakeChatService()


@pytest.fixture
def client(tmp_path, monkeypatch, fake_chat_service) -> TestClient:
    """A TestClient backed by an isolated DB and a fake chat service.

    The real conversation service, repository, and database run; only the model
    side (the chat service) is faked, so persistence and history are exercised
    end to end without network.
    """
    monkeypatch.setenv("COHERE_API_KEY", "test-key")
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{tmp_path}/test.db")
    get_settings.cache_clear()

    app = create_app()
    with TestClient(app) as test_client:
        app.state.chat_service = fake_chat_service
        yield test_client
    get_settings.cache_clear()


@pytest.fixture
def client_factory(tmp_path, monkeypatch, fake_chat_service):
    """Build a TestClient with extra settings (auth, rate limit, windowing).

    Returns a context manager taking environment overrides (and an optional
    `raise_server_exceptions` flag), each yielding a client over its own
    isolated database and the shared fake chat service.
    """
    counter = {"n": 0}

    @contextmanager
    def _make(raise_server_exceptions: bool = True, **env: str):
        counter["n"] += 1
        monkeypatch.setenv("COHERE_API_KEY", "test-key")
        monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{tmp_path}/c{counter['n']}.db")
        for key, value in env.items():
            monkeypatch.setenv(key, value)
        get_settings.cache_clear()

        app = create_app()
        with TestClient(app, raise_server_exceptions=raise_server_exceptions) as test_client:
            app.state.chat_service = fake_chat_service
            yield test_client
        get_settings.cache_clear()

    return _make


@pytest_asyncio.fixture
async def db_session(tmp_path) -> AsyncSession:
    """An async session against an isolated, initialized SQLite database."""
    engine = create_engine(f"sqlite+aiosqlite:///{tmp_path}/repo.db")
    await init_models(engine)
    session_factory = create_session_factory(engine)
    async with session_factory() as session:
        yield session
    await engine.dispose()

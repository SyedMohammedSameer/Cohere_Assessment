"""Async database engine, session factory, and schema bootstrap.

Uses SQLAlchemy 2.0 with an async engine (SQLite via aiosqlite by default). The
engine and session factory are created once at startup and stored on app state;
requests get a short-lived `AsyncSession` via the `get_session` dependency.

`init_models` creates tables from the ORM metadata. That is appropriate for this
assessment; a production system would manage schema with migrations (Alembic).
"""

from sqlalchemy import event
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""


def create_engine(database_url: str) -> AsyncEngine:
    """Create the async SQLAlchemy engine.

    For SQLite, enables write-ahead logging so reads do not block during a write,
    a busy timeout to ride out brief lock contention, and foreign-key
    enforcement. `pool_pre_ping` validates pooled connections before use, which
    matters once the URL points at a networked database like Postgres.
    """
    engine = create_async_engine(database_url, future=True, pool_pre_ping=True)

    if engine.dialect.name == "sqlite":

        @event.listens_for(engine.sync_engine, "connect")
        def _set_sqlite_pragmas(dbapi_connection: object, _record: object) -> None:
            cursor = dbapi_connection.cursor()  # type: ignore[attr-defined]
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA synchronous=NORMAL")
            cursor.execute("PRAGMA busy_timeout=5000")
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

    return engine


def create_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    """Build a session factory bound to the engine."""
    return async_sessionmaker(engine, expire_on_commit=False)


async def init_models(engine: AsyncEngine) -> None:
    """Create all tables defined on the ORM metadata if they do not exist."""
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)

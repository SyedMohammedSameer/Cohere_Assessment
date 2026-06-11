"""Application configuration.

Settings are loaded from environment variables and an optional `.env` file via
pydantic-settings, which gives us typed, validated config rather than ad hoc
`os.getenv` calls. `get_settings` is cached so the environment is read once per
process and the same instance is reused across requests.
"""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed application settings sourced from the environment and `.env`.

    Environment variable names are matched case-insensitively, so
    `COHERE_API_KEY` populates `cohere_api_key`.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Required: the service cannot serve chat without it, so we fail fast at
    # startup rather than on the first request. `min_length=1` rejects an empty
    # value left in `.env`.
    cohere_api_key: str = Field(min_length=1)

    # Cohere Chat model.
    cohere_model: str = "command-a-03-2025"

    # Per-request timeout, in seconds, for calls to the Cohere API.
    cohere_timeout_s: float = 30.0

    # Total attempts (initial try plus retries) for transient Cohere failures.
    cohere_max_attempts: int = 3

    # Exponential backoff bounds, in seconds, between Cohere retries.
    cohere_retry_min_s: float = 0.5
    cohere_retry_max_s: float = 8.0

    # Maximum rounds of tool calls before forcing a final, tool-free answer.
    # Bounds the multi-turn tool loop so a misbehaving model cannot loop forever.
    max_tool_iterations: int = 5

    # MediaWiki search endpoint. English Wikipedia by default; overridable to
    # point at another language or a mirror.
    wikipedia_api_url: str = "https://en.wikipedia.org/w/api.php"

    # Number of search hits to retrieve per Wikipedia query. A handful gives the
    # model enough material to recover when its first query is imprecise.
    wikipedia_search_limit: int = 5

    # Per-request timeout, in seconds, for the Wikipedia API.
    wikipedia_timeout_s: float = 10.0

    # Total attempts (initial try plus retries) for transient Wikipedia failures.
    wikipedia_max_attempts: int = 3

    # Exponential backoff bounds, in seconds, between Wikipedia retries.
    wikipedia_retry_min_s: float = 0.25
    wikipedia_retry_max_s: float = 4.0

    # Most recent messages replayed to the model on a multi-turn request. Bounds
    # prompt tokens, cost, and context-window risk for long conversations.
    max_history_messages: int = 20

    # Wikipedia search cache. TTL of 0 disables caching; max entries bounds the
    # in-memory footprint (oldest entries are evicted first).
    wikipedia_cache_ttl_s: float = 300.0
    wikipedia_cache_max_entries: int = 512

    # Async SQLAlchemy database URL. SQLite by default; swap for Postgres
    # (postgresql+asyncpg://...) in production without code changes.
    database_url: str = "sqlite+aiosqlite:///./history.db"

    # Default page size for the history endpoint.
    history_page_size: int = 20

    # Comma-separated API keys accepted in the `X-API-Key` header. When empty,
    # authentication is disabled and all requests share the "public" owner.
    api_keys: str = ""

    # Per-key (or per-client) request budget per minute. 0 disables rate limiting.
    rate_limit_per_minute: int = 0

    # Logging verbosity for the stdlib logging root configuration.
    log_level: str = "INFO"

    # Log output format: "json" for structured logs (production), "text" for a
    # human-readable single line (local development).
    log_format: str = "json"

    @property
    def api_key_set(self) -> frozenset[str]:
        """Configured API keys as a set; empty means authentication is off."""
        return frozenset(key.strip() for key in self.api_keys.split(",") if key.strip())


@lru_cache
def get_settings() -> Settings:
    """Return the cached application settings instance."""
    # Fields are populated from the environment, so no arguments are passed.
    return Settings()  # type: ignore[call-arg]

"""Wikipedia search tool, backed by the MediaWiki search API.

Exposes a single async `search` call with the same production posture as the
Cohere client: a request timeout and retries with exponential backoff on
transient failures. Hard failures raise `WikipediaError`, which the chat
orchestrator catches so the model can degrade gracefully rather than the whole
request failing.

MediaWiki API reference: https://www.mediawiki.org/wiki/API:Search
"""

import html
import logging
import re
import time

import httpx
from pydantic import BaseModel

from app.core.cache import TTLCache
from app.core.config import Settings
from app.core.exceptions import WikipediaError
from app.core.resilience import make_retrying

logger = logging.getLogger(__name__)

_RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})
_HTML_TAG_RE = re.compile(r"<[^>]+>")

# MediaWiki asks API clients to identify themselves with a descriptive
# User-Agent. See https://meta.wikimedia.org/wiki/User-Agent_policy
_USER_AGENT = "cohere-chat-app/0.1 (Cohere Chat assessment; +https://github.com/cohere-chat-app)"


class WikiArticle(BaseModel):
    """A single Wikipedia search hit."""

    id: int
    title: str
    snippet: str
    url: str


def _is_retryable(exc: BaseException) -> bool:
    """Return whether an HTTP failure is transient and worth retrying."""
    if isinstance(exc, httpx.TimeoutException | httpx.ConnectError | httpx.RemoteProtocolError):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in _RETRYABLE_STATUS
    return False


def _clean_snippet(raw: str) -> str:
    """Strip the HTML markup and entities MediaWiki returns in snippets."""
    return html.unescape(_HTML_TAG_RE.sub("", raw)).strip()


class WikipediaClient:
    """Async client for the MediaWiki search API."""

    def __init__(
        self,
        client: httpx.AsyncClient,
        api_url: str,
        search_limit: int,
        max_attempts: int,
        retry_min_s: float,
        retry_max_s: float,
        cache: TTLCache[list[WikiArticle]] | None = None,
    ) -> None:
        """Initialize the client."""
        self._client = client
        self._api_url = api_url
        self._search_limit = search_limit
        self._max_attempts = max_attempts
        self._retry_min_s = retry_min_s
        self._retry_max_s = retry_max_s
        self._cache = cache

    @classmethod
    def from_settings(cls, settings: Settings) -> "WikipediaClient":
        """Build a client from application settings."""
        client = httpx.AsyncClient(
            timeout=settings.wikipedia_timeout_s,
            headers={"User-Agent": _USER_AGENT},
        )
        return cls(
            client=client,
            api_url=settings.wikipedia_api_url,
            search_limit=settings.wikipedia_search_limit,
            max_attempts=settings.wikipedia_max_attempts,
            retry_min_s=settings.wikipedia_retry_min_s,
            retry_max_s=settings.wikipedia_retry_max_s,
            cache=TTLCache(
                ttl_s=settings.wikipedia_cache_ttl_s,
                max_entries=settings.wikipedia_cache_max_entries,
            ),
        )

    async def aclose(self) -> None:
        """Close the underlying HTTP connection pool."""
        await self._client.aclose()

    async def search(self, query: str, limit: int | None = None) -> list[WikiArticle]:
        """Search Wikipedia and return the top matching articles."""
        srlimit = limit or self._search_limit
        cache_key = f"{srlimit}:{query.strip().lower()}"
        if self._cache is not None:
            cached = self._cache.get(cache_key)
            if cached is not None:
                logger.info(
                    "wikipedia.search cache hit",
                    extra={
                        "event": "wikipedia.cache_hit",
                        "query": query,
                        "results": len(cached),
                    },
                )
                return cached

        params: dict[str, str | int] = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": srlimit,
            "srprop": "snippet",
            "format": "json",
        }

        start = time.perf_counter()
        try:
            data = await self._get_json(params)
        except (httpx.HTTPError, httpx.HTTPStatusError) as exc:
            logger.error(
                "wikipedia.search failed",
                extra={"event": "wikipedia.error", "query": query, "error": str(exc)},
            )
            raise WikipediaError("Wikipedia search is currently unavailable.") from exc

        articles = _parse_results(data, srlimit)
        if self._cache is not None:
            self._cache.set(cache_key, articles)
        logger.info(
            "wikipedia.search completed",
            extra={
                "event": "wikipedia.search",
                "query": query,
                "results": len(articles),
                "latency_ms": round((time.perf_counter() - start) * 1000, 1),
            },
        )
        return articles

    async def _get_json(self, params: dict[str, str | int]) -> dict:
        """Issue the GET request with retries and return the parsed JSON body."""
        retrying = make_retrying(
            max_attempts=self._max_attempts,
            wait_min_s=self._retry_min_s,
            wait_max_s=self._retry_max_s,
            predicate=_is_retryable,
            logger=logger,
        )
        async for attempt in retrying:
            with attempt:
                response = await self._client.get(self._api_url, params=params)
                response.raise_for_status()
                return response.json()
        # Unreachable: tenacity re-raises once attempts are exhausted.
        raise WikipediaError("Wikipedia search exhausted its retries.")  # pragma: no cover


def _parse_results(data: dict, limit: int) -> list[WikiArticle]:
    """Map a MediaWiki search payload into `WikiArticle` objects."""
    hits = data.get("query", {}).get("search", [])
    articles: list[WikiArticle] = []
    for hit in hits[:limit]:
        page_id = hit.get("pageid")
        title = hit.get("title")
        if page_id is None or title is None:
            continue
        articles.append(
            WikiArticle(
                id=page_id,
                title=title,
                snippet=_clean_snippet(hit.get("snippet", "")),
                # curid resolves regardless of title punctuation or spacing.
                url=f"https://en.wikipedia.org/?curid={page_id}",
            )
        )
    return articles

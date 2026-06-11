"""Unit tests for the Wikipedia search client, using a mocked HTTP transport."""

import httpx
import pytest

from app.clients.wikipedia import WikipediaClient, _clean_snippet
from app.core.cache import TTLCache
from app.core.exceptions import WikipediaError

API_URL = "https://en.wikipedia.org/w/api.php"


def make_client(handler: httpx.MockTransport, *, max_attempts: int = 3) -> WikipediaClient:
    """Build a WikipediaClient whose transport is the given mock handler."""
    return WikipediaClient(
        client=httpx.AsyncClient(transport=handler),
        api_url=API_URL,
        search_limit=3,
        max_attempts=max_attempts,
        retry_min_s=0.0,
        retry_max_s=0.0,
    )


def search_payload(hits: list[dict]) -> dict:
    """Wrap search hits in the MediaWiki response envelope."""
    return {"query": {"search": hits}}


def test_clean_snippet_strips_tags_and_unescapes_entities():
    raw = 'was the <span class="searchmatch">second</span> person &amp; pilot'
    assert _clean_snippet(raw) == "was the second person & pilot"


async def test_search_parses_results():
    payload = search_payload(
        [{"pageid": 440, "title": "Buzz Aldrin", "snippet": "second person on the Moon"}]
    )
    client = make_client(httpx.MockTransport(lambda r: httpx.Response(200, json=payload)))

    articles = await client.search("buzz aldrin")

    assert len(articles) == 1
    article = articles[0]
    assert article.id == 440
    assert article.title == "Buzz Aldrin"
    assert article.snippet == "second person on the Moon"
    assert article.url == "https://en.wikipedia.org/?curid=440"
    await client._client.aclose()


async def test_search_sends_expected_query_params():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(dict(request.url.params))
        return httpx.Response(200, json=search_payload([]))

    client = make_client(httpx.MockTransport(handler))
    await client.search("apollo 11", limit=5)

    assert captured["action"] == "query"
    assert captured["list"] == "search"
    assert captured["srsearch"] == "apollo 11"
    assert captured["srlimit"] == "5"
    assert captured["format"] == "json"
    await client._client.aclose()


async def test_search_returns_empty_when_no_hits():
    handler = httpx.MockTransport(lambda r: httpx.Response(200, json=search_payload([])))
    client = make_client(handler)
    assert await client.search("nonexistent topic") == []
    await client._client.aclose()


async def test_search_retries_transient_status_then_succeeds():
    attempts = {"n": 0}
    payload = search_payload([{"pageid": 1, "title": "Test", "snippet": "x"}])

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        if attempts["n"] == 1:
            return httpx.Response(503)
        return httpx.Response(200, json=payload)

    client = make_client(httpx.MockTransport(handler))
    articles = await client.search("test")

    assert attempts["n"] == 2
    assert articles[0].title == "Test"
    await client._client.aclose()


async def test_search_raises_after_exhausting_retries():
    attempts = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        return httpx.Response(500)

    client = make_client(httpx.MockTransport(handler), max_attempts=3)
    with pytest.raises(WikipediaError):
        await client.search("test")

    assert attempts["n"] == 3
    await client._client.aclose()


async def test_search_wraps_timeout_in_wikipedia_error():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.TimeoutException("slow")

    client = make_client(httpx.MockTransport(handler), max_attempts=2)
    with pytest.raises(WikipediaError):
        await client.search("test")
    await client._client.aclose()


async def test_search_serves_repeat_from_cache():
    calls = {"n": 0}
    payload = search_payload([{"pageid": 1, "title": "Test", "snippet": "x"}])

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(200, json=payload)

    client = WikipediaClient(
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        api_url=API_URL,
        search_limit=3,
        max_attempts=3,
        retry_min_s=0.0,
        retry_max_s=0.0,
        cache=TTLCache(ttl_s=100, max_entries=10),
    )
    first = await client.search("buzz aldrin")
    second = await client.search("buzz aldrin")

    assert calls["n"] == 1  # second call served from cache, no HTTP request
    assert first == second
    await client._client.aclose()

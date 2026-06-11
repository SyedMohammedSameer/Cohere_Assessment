"""Unit tests for the TTL cache."""

from app.core.cache import TTLCache


def test_set_and_get():
    cache: TTLCache[int] = TTLCache(ttl_s=100, max_entries=10)
    cache.set("a", 1)
    assert cache.get("a") == 1
    assert cache.get("missing") is None


def test_disabled_when_ttl_zero():
    cache: TTLCache[int] = TTLCache(ttl_s=0, max_entries=10)
    cache.set("a", 1)
    assert cache.get("a") is None


def test_expires_after_ttl(monkeypatch):
    clock = {"now": 1000.0}
    monkeypatch.setattr("app.core.cache.time.monotonic", lambda: clock["now"])
    cache: TTLCache[int] = TTLCache(ttl_s=10, max_entries=10)
    cache.set("a", 1)
    assert cache.get("a") == 1

    clock["now"] += 11
    assert cache.get("a") is None


def test_evicts_oldest_when_full():
    cache: TTLCache[int] = TTLCache(ttl_s=100, max_entries=2)
    cache.set("a", 1)
    cache.set("b", 2)
    cache.set("c", 3)
    assert cache.get("a") is None
    assert cache.get("b") == 2
    assert cache.get("c") == 3


def test_get_refreshes_recency():
    cache: TTLCache[int] = TTLCache(ttl_s=100, max_entries=2)
    cache.set("a", 1)
    cache.set("b", 2)
    assert cache.get("a") == 1  # touch "a" so "b" becomes the oldest
    cache.set("c", 3)
    assert cache.get("b") is None
    assert cache.get("a") == 1
    assert cache.get("c") == 3

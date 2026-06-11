"""Bounded in-memory TTL cache with LRU eviction.

Single-process only; a multi-instance deployment would use a shared cache such
as Redis. A TTL of zero disables caching.
"""

import time
from collections import OrderedDict
from typing import Generic, TypeVar

V = TypeVar("V")


class TTLCache(Generic[V]):
    """A bounded, time-to-live cache keyed by string."""

    def __init__(self, ttl_s: float, max_entries: int) -> None:
        """Build a cache; a ttl_s of zero or less disables caching entirely."""
        self._ttl = ttl_s
        self._max_entries = max_entries
        self._store: OrderedDict[str, tuple[float, V]] = OrderedDict()

    def get(self, key: str) -> V | None:
        """Return the cached value for a key, or None if absent or expired."""
        if self._ttl <= 0:
            return None
        entry = self._store.get(key)
        if entry is None:
            return None
        expires_at, value = entry
        if expires_at < time.monotonic():
            del self._store[key]
            return None
        self._store.move_to_end(key)
        return value

    def set(self, key: str, value: V) -> None:
        """Store a value under a key, evicting the oldest entry if full."""
        if self._ttl <= 0 or self._max_entries <= 0:
            return
        self._store[key] = (time.monotonic() + self._ttl, value)
        self._store.move_to_end(key)
        while len(self._store) > self._max_entries:
            self._store.popitem(last=False)

"""Thin wrapper around redis-py for Tom_and_Jerry.

This module owns:
  - The default connection config (matches scripts/verify_redis.py)
  - A small RedisClient abstraction so the rest of the codebase doesn't
    import redis-py directly. That lets us swap in a fake for unit tests
    without touching call sites.

Keep this file MINIMAL. Project-specific operations (key schemas,
encoding) live in l1_store.py. This file only knows about connections.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Protocol


# Defaults — kept in sync with scripts/verify_redis.py
DEFAULT_HOST = "localhost"
DEFAULT_PORT = 6380
DEFAULT_DB = 1
DEFAULT_NAMESPACE = "tj:"


@dataclass(frozen=True, slots=True)
class RedisConfig:
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    db: int = DEFAULT_DB
    namespace: str = DEFAULT_NAMESPACE


class RedisLike(Protocol):
    """The subset of Redis operations we use.

    Anything implementing this Protocol can stand in for real Redis —
    used for the in-memory fake during unit tests.
    """
    def set(self, key: str, value: str, ex: int | None = None) -> Any: ...
    def get(self, key: str) -> str | None: ...
    def delete(self, *keys: str) -> int: ...
    def incrby(self, key: str, amount: int = 1) -> int: ...
    def incrbyfloat(self, key: str, amount: float) -> float: ...
    def hset(self, key: str, field: str | None = None,
             value: str | None = None,
             mapping: dict | None = None) -> int: ...
    def hget(self, key: str, field: str) -> str | None: ...
    def hgetall(self, key: str) -> dict: ...
    def hincrby(self, key: str, field: str, amount: int = 1) -> int: ...
    def hincrbyfloat(self, key: str, field: str, amount: float) -> float: ...
    def expire(self, key: str, seconds: int) -> bool: ...
    def scan_iter(self, match: str | None = None,
                  count: int | None = None) -> Iterable[str]: ...
    def ping(self) -> bool: ...


class RedisClient:
    """Wraps a real redis.Redis with our config defaults.

    The wrapper exists so call sites depend on `RedisClient` (our class)
    rather than `redis.Redis` (third-party). When we want to swap to a
    Postgres backend for Phase 4, or to a fake for tests, we don't have
    to grep the codebase for `import redis`.
    """

    def __init__(self, config: RedisConfig | None = None,
                 client: RedisLike | None = None):
        """If `client` is provided, use it directly (handy for injecting
        a fake). Otherwise construct a real redis-py client from config.
        """
        self.config = config or RedisConfig()
        if client is not None:
            self._client = client
        else:
            import redis
            self._client = redis.Redis(
                host=self.config.host,
                port=self.config.port,
                db=self.config.db,
                decode_responses=True,
                socket_connect_timeout=5,
            )

    # Direct passthroughs — keep the surface area small so swapping
    # backends later is straightforward.
    def set(self, key, value, ex=None): return self._client.set(key, value, ex=ex)
    def get(self, key): return self._client.get(key)
    def delete(self, *keys): return self._client.delete(*keys)
    def incrby(self, key, amount=1): return self._client.incrby(key, amount)
    def incrbyfloat(self, key, amount): return self._client.incrbyfloat(key, amount)
    def hset(self, key, field=None, value=None, mapping=None):
        # redis-py's hset signature varies between versions; normalize
        if mapping is not None:
            return self._client.hset(key, mapping=mapping)
        return self._client.hset(key, field, value)
    def hget(self, key, field): return self._client.hget(key, field)
    def hgetall(self, key): return self._client.hgetall(key)
    def hincrby(self, key, field, amount=1):
        return self._client.hincrby(key, field, amount)
    def hincrbyfloat(self, key, field, amount):
        return self._client.hincrbyfloat(key, field, amount)
    def expire(self, key, seconds): return self._client.expire(key, seconds)
    def scan_iter(self, match=None, count=None):
        return self._client.scan_iter(match=match, count=count)
    def ping(self): return self._client.ping()

    def ns(self, *parts: str) -> str:
        """Build a namespaced key: ns('l1', 'noise', '42') → 'tj:l1:noise:42'."""
        return self.config.namespace + ":".join(str(p) for p in parts)


# ---- in-memory fake for unit tests ------------------------------------

class FakeRedis:
    """Drop-in replacement for redis-py that lives entirely in memory.

    Implements only the operations RedisClient passes through. Used in
    unit tests so we can verify L1 semantics without a Redis dependency.

    NOT thread-safe — fine for tests, would break in production.
    """

    def __init__(self):
        self._strings: dict[str, str] = {}
        self._hashes: dict[str, dict[str, str]] = {}
        self._expires: dict[str, float] = {}   # unused in tests but kept for API symmetry

    def set(self, key, value, ex=None):
        self._strings[str(key)] = str(value)
        return True

    def get(self, key):
        return self._strings.get(str(key))

    def delete(self, *keys):
        n = 0
        for k in keys:
            k = str(k)
            if k in self._strings:
                del self._strings[k]
                n += 1
            if k in self._hashes:
                del self._hashes[k]
                n += 1
        return n

    def incrby(self, key, amount=1):
        cur = int(self._strings.get(str(key), "0"))
        cur += amount
        self._strings[str(key)] = str(cur)
        return cur

    def incrbyfloat(self, key, amount):
        cur = float(self._strings.get(str(key), "0"))
        cur += amount
        self._strings[str(key)] = repr(cur)
        return cur

    def hset(self, key, field=None, value=None, mapping=None):
        key = str(key)
        if key not in self._hashes:
            self._hashes[key] = {}
        added = 0
        if mapping is not None:
            for f, v in mapping.items():
                if f not in self._hashes[key]:
                    added += 1
                self._hashes[key][str(f)] = str(v)
        else:
            if field not in self._hashes[key]:
                added += 1
            self._hashes[key][str(field)] = str(value)
        return added

    def hget(self, key, field):
        return self._hashes.get(str(key), {}).get(str(field))

    def hgetall(self, key):
        return dict(self._hashes.get(str(key), {}))

    def hincrby(self, key, field, amount=1):
        key = str(key)
        field = str(field)
        if key not in self._hashes:
            self._hashes[key] = {}
        cur = int(self._hashes[key].get(field, "0")) + amount
        self._hashes[key][field] = str(cur)
        return cur

    def hincrbyfloat(self, key, field, amount):
        key = str(key)
        field = str(field)
        if key not in self._hashes:
            self._hashes[key] = {}
        cur = float(self._hashes[key].get(field, "0")) + amount
        self._hashes[key][field] = repr(cur)
        return cur

    def expire(self, key, seconds):
        # Fake — tests don't depend on actual expiry
        return True

    def scan_iter(self, match=None, count=None):
        keys = list(self._strings.keys()) + list(self._hashes.keys())
        if match:
            import fnmatch
            keys = [k for k in keys if fnmatch.fnmatchcase(k, match)]
        return iter(keys)

    def ping(self):
        return True

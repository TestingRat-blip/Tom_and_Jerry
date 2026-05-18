"""Verify the dedicated Redis instance is reachable and configured correctly.

Run this AFTER `docker compose up -d` to confirm everything is wired up
before writing any L1 code. The script:

  1. Connects to the configured Redis host/port.
  2. Pings the server.
  3. Round-trips a test key in the Tom_and_Jerry namespace.
  4. Confirms a dedicated DB is being used (not the default DB 0 — see below).
  5. Cleans up after itself.

Usage:
    python -m scripts.verify_redis
    python -m scripts.verify_redis --host localhost --port 6380 --db 1

If this script passes, the L1 system has working infrastructure beneath it.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


# Default config — kept in one place so the L1 code can import the same constants
DEFAULT_HOST = "localhost"
DEFAULT_PORT = 6380
DEFAULT_DB = 1   # 0 is reserved for general use; 1 is Tom_and_Jerry L1.
NAMESPACE_PREFIX = "tj:"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Verify Tom_and_Jerry Redis setup.")
    p.add_argument("--host", default=DEFAULT_HOST)
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument("--db", type=int, default=DEFAULT_DB,
                   help="Redis DB number to claim for Tom_and_Jerry.")
    return p.parse_args(argv)


def step(n: int, msg: str) -> None:
    print(f"  [{n}] {msg} ... ", end="", flush=True)


def ok(detail: str = "") -> None:
    print(f"OK  {detail}".rstrip())


def fail(msg: str) -> None:
    print(f"FAIL\n  → {msg}")
    sys.exit(1)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    print(f"Verifying Tom_and_Jerry Redis at {args.host}:{args.port} db={args.db}\n")

    # 1. Import
    step(1, "Importing redis-py")
    try:
        import redis
    except ImportError:
        fail("redis-py is not installed. Run `pip install redis>=5.0`.")
    ok(f"redis-py {redis.__version__}")

    # 2. Connect
    step(2, "Opening connection")
    try:
        client = redis.Redis(
            host=args.host, port=args.port, db=args.db,
            socket_connect_timeout=5,
            decode_responses=True,
        )
    except Exception as e:
        fail(f"could not construct client: {e}")
    ok()

    # 3. Ping
    step(3, "PING")
    try:
        result = client.ping()
        if not result:
            fail("PING returned a falsy value")
    except redis.exceptions.ConnectionError as e:
        fail(
            f"connection refused: {e}\n"
            f"  → is the container running? Try: docker compose up -d"
        )
    except Exception as e:
        fail(f"unexpected error: {e}")
    ok("PONG")

    # 4. Round-trip a test key
    step(4, "SET/GET round trip")
    test_key = f"{NAMESPACE_PREFIX}verify:{int(time.time())}"
    test_value = "tom_and_jerry_phase3_handshake"
    try:
        client.set(test_key, test_value, ex=30)  # 30s TTL, self-cleaning
        retrieved = client.get(test_key)
        if retrieved != test_value:
            fail(f"set {test_value!r} but got {retrieved!r}")
    except Exception as e:
        fail(f"round-trip failed: {e}")
    ok(f"key={test_key!r}")

    # 5. Confirm we're on the expected DB
    step(5, "Verify DB assignment")
    try:
        info = client.client_info()
        actual_db = info.get("db", 0)
        if int(actual_db) != args.db:
            fail(f"expected DB {args.db}, but client reports DB {actual_db}")
    except Exception as e:
        fail(f"could not read client info: {e}")
    ok(f"db={actual_db}")

    # 6. Check we have some elbow room
    step(6, "Server INFO")
    try:
        info = client.info(section="memory")
        used = info.get("used_memory_human", "?")
        max_mem = info.get("maxmemory_human", "unlimited")
    except Exception as e:
        fail(f"INFO failed: {e}")
    ok(f"used={used}, max={max_mem}")

    # 7. Confirm namespace is sane
    step(7, "Namespace scan")
    try:
        keys = list(client.scan_iter(match=f"{NAMESPACE_PREFIX}*", count=100))
    except Exception as e:
        fail(f"SCAN failed: {e}")
    ok(f"{len(keys)} existing {NAMESPACE_PREFIX!r} keys (including this verify key)")

    # 8. Clean up our test key
    step(8, "Cleanup")
    try:
        client.delete(test_key)
    except Exception as e:
        fail(f"DELETE failed: {e}")
    ok()

    print(f"\n  All checks passed. Tom_and_Jerry can use Redis at:")
    print(f"    host = {args.host}")
    print(f"    port = {args.port}")
    print(f"    db   = {args.db}")
    print(f"    namespace = {NAMESPACE_PREFIX!r}\n")


if __name__ == "__main__":
    main(sys.argv[1:])

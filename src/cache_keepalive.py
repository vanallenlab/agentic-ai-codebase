"""Keep the Anthropic prompt cache alive while a tool runs.

The conversation prefix is cached with the 5-minute TTL. A tool call that runs longer
(marker peaks, cNMF, MACS3 on a big cohort) would let it expire, and the next request would
write the whole conversation into the cache again (1.25x input price instead of a cheap read).
While the main thread waits on the worker, a background thread re-sends the last request with
max_tokens=0 about every 4 minutes: a cache read refreshes the entry's timer and nothing is
generated. Pings stop after `client.cache_keepalive_minutes` (default 30) of tool runtime.

Usage:
    with CacheKeepalive(state):
        resp = state.runtime.call(...)
"""

from __future__ import annotations

import threading
import time

from src.logging_utils import log

PING_AFTER_S = 240.0    # entry lifetime is 300 s from the start of the last request that touched it
CHECK_EVERY_S = 5.0


class CacheKeepalive:
    def __init__(self, state):
        self.state = state
        self.client = getattr(state, "client", None)
        self.usages: list = []
        self.errors = 0
        self._stop = threading.Event()
        self._thread = None

    def _enabled(self) -> bool:
        c = self.client
        return (c is not None and getattr(c, "provider", "") == "anthropic"
                and getattr(c, "prompt_caching", False)
                and float(getattr(c, "cache_keepalive_minutes", 0) or 0) > 0
                and c.seconds_since_cache_touch() is not None)

    def _run(self):
        deadline = time.monotonic() + float(self.client.cache_keepalive_minutes) * 60
        while not self._stop.wait(CHECK_EVERY_S):
            if time.monotonic() > deadline:
                return
            age = self.client.seconds_since_cache_touch()
            if age is None or age < PING_AFTER_S:
                continue
            try:
                usage = self.client.cache_keepalive()
                if usage is not None:
                    self.usages.append(usage)
            except Exception as e:  # never disturb the tool call; the next request just re-writes
                self.errors += 1
                if self.errors <= 2:
                    log(f"   ⚠️ cache keep-alive ping failed: {type(e).__name__}: {str(e)[:200]}")
                if self.errors >= 3:
                    return

    def __enter__(self):
        if self._enabled():
            self._thread = threading.Thread(target=self._run, name="cache-keepalive", daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc):
        if self._thread is not None:
            self._stop.set()
            self._thread.join(timeout=120)   # a ping in flight finishes (it is short)
        if self.usages:
            tokens = getattr(self.state, "tokens", None)
            if tokens is not None and hasattr(tokens, "add_keepalive"):
                for u in self.usages:
                    tokens.add_keepalive(u)
            read = sum(getattr(u, "cache_read_input_tokens", 0) or 0 for u in self.usages)
            log(f"   ♻️ kept the prompt cache alive: {len(self.usages)} ping(s), {read:,} cached tokens read")
        return False

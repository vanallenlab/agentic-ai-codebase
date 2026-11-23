"""Length-prefixed JSON framing between the harness and the tool worker.

Frame = 8-byte big-endian payload length + UTF-8 JSON payload. Used over the
worker's stdin/stdout, which works identically for a local subprocess and for
`docker/podman run -i`.
"""

from __future__ import annotations

import json
import os
import select
import struct
import time
from typing import Any, Optional

_HEADER = struct.Struct(">Q")
MAX_FRAME = 1 << 31  # 2 GiB sanity limit


class WorkerDied(Exception):
    """The worker closed its end of the pipe (crashed, was killed, or exited)."""


class WorkerTimeout(Exception):
    """No complete response from the worker before the deadline."""


def encode(obj: Any) -> bytes:
    payload = json.dumps(obj, default=str).encode("utf-8")
    return _HEADER.pack(len(payload)) + payload


def write_frame(fd: int, obj: Any):
    data = encode(obj)
    view = memoryview(data)
    while view:
        try:
            n = os.write(fd, view)
        except (BrokenPipeError, OSError) as e:
            raise WorkerDied(f"pipe closed while sending: {e}") from e
        view = view[n:]


def _read_exact(fd: int, n: int, deadline: Optional[float]) -> bytes:
    chunks, remaining = [], n
    while remaining:
        if deadline is not None:
            wait = deadline - time.monotonic()
            if wait <= 0:
                raise WorkerTimeout("timed out waiting for worker")
            ready, _, _ = select.select([fd], [], [], min(wait, 1.0))
            if not ready:
                continue
        chunk = os.read(fd, min(remaining, 1 << 20))
        if not chunk:
            raise WorkerDied("worker closed the pipe")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def read_frame(fd: int, timeout: Optional[float] = None) -> Any:
    """Read one frame. timeout=None blocks indefinitely (but the select loop
    still wakes every second so signals such as SIGTERM are delivered)."""
    deadline = None if timeout is None else time.monotonic() + timeout
    if deadline is None:
        # Block, but in 1s slices so Python-level signal handlers run promptly.
        while True:
            ready, _, _ = select.select([fd], [], [], 1.0)
            if ready:
                break
    (length,) = _HEADER.unpack(_read_exact(fd, _HEADER.size, deadline))
    if length > MAX_FRAME:
        raise WorkerDied(f"invalid frame length {length}")
    return json.loads(_read_exact(fd, length, deadline).decode("utf-8"))

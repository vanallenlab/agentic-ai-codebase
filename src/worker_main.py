"""
Tool worker process: `python <repo>/src/worker_main.py` (cwd = run output dir)

Speaks the framed-JSON protocol (src/worker_protocol.py) on its ORIGINAL
stdin/stdout. Immediately on start, fd 1 is redirected to stderr, so anything
tools, libraries or agent code print (including from C extensions) goes to the
console instead of corrupting the protocol stream.

Requests:  {"op": "init"|"call"|"checkpoint"|"info"|"finalize"|"ping"|"shutdown", ...}
Responses: {"ok": true, ...result, "logs": "<worker log lines>"}
           {"ok": false, "error": "...", "traceback": "...", "logs": "..."}
"""

from __future__ import annotations

import io
import os
import sys
import traceback


def _setup_fds():
    proto_in = os.dup(0)
    proto_out = os.dup(1)
    devnull = os.open(os.devnull, os.O_RDONLY)
    os.dup2(devnull, 0)      # tools must not read the protocol stream
    os.close(devnull)
    os.dup2(2, 1)            # all stdout (Python + C level) -> stderr
    sys.stdout = os.fdopen(1, "w", buffering=1, closefd=False)
    sys.stdin = open(os.devnull)
    return proto_in, proto_out


def _apply_memory_limit():
    """Optional address-space cap (subprocess mode) so a runaway allocation
    raises MemoryError in the worker instead of invoking the kernel OOM killer."""
    gb = os.environ.get("M3A_WORKER_MEMORY_LIMIT_GB")
    if not gb:
        return
    try:
        import resource
        limit = int(float(gb) * (1 << 30))
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    except Exception as e:  # noqa: BLE001
        print(f"[worker] could not set memory limit: {e}", file=sys.stderr)


def main():
    proto_in, proto_out = _setup_fds()
    _apply_memory_limit()

    # Run as a script, sys.path[0] is src/ itself, which would let src/config.py
    # etc. shadow other modules. Replace it with the repo root (cwd is the run's
    # output dir, which must not be importable either).
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.dirname(here)
    sys.path[:] = [p for p in sys.path if p and os.path.abspath(p) not in (here, os.getcwd())]
    sys.path.insert(0, repo)

    from src.harness_guard import TerminationRequested, install_signal_handlers
    from src.logging_utils import logger
    from src.worker_protocol import WorkerDied, read_frame, write_frame

    install_signal_handlers()
    log_buf = io.StringIO()
    logger.set_log_file(log_buf)  # collected per request and returned to the harness

    runtime = None

    def _drain_logs() -> str:
        text = log_buf.getvalue()
        log_buf.seek(0)
        log_buf.truncate(0)
        return text

    write_frame(proto_out, {"ok": True, "event": "ready", "pid": os.getpid()})

    while True:
        try:
            req = read_frame(proto_in)
        except (WorkerDied, TerminationRequested, KeyboardInterrupt):
            break
        op = req.get("op")
        try:
            if op == "shutdown":
                write_frame(proto_out, {"ok": True, "logs": _drain_logs()})
                break
            if op == "ping":
                result = {}
            else:
                if runtime is None:
                    from src.worker_runtime import ToolRuntime
                    runtime = ToolRuntime()
                handler = {"init": runtime.init, "call": runtime.call,
                           "checkpoint": runtime.checkpoint, "info": runtime.info,
                           "finalize": runtime.finalize}.get(op)
                if handler is None:
                    raise ValueError(f"unknown op {op!r}")
                result = handler(req)
            write_frame(proto_out, {"ok": True, **result, "logs": _drain_logs()})
        except (TerminationRequested, KeyboardInterrupt):
            break
        except BaseException as e:  # noqa: BLE001 - report everything, keep serving
            write_frame(proto_out, {
                "ok": False,
                "error": f"{type(e).__name__}: {e}",
                "traceback": traceback.format_exc(),
                "logs": _drain_logs(),
            })
            if isinstance(e, SystemExit):
                break


if __name__ == "__main__":
    main()

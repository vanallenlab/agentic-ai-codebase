"""
Harness robustness utilities.

Everything in here exists so that a bad tool call or bad agent-written code
cannot (a) crash the run, (b) silently corrupt shared data, or (c) lose work.

Contents:
- TerminationRequested / install_signal_handlers: turn SIGTERM/SIGHUP into an
  exception so `finally` blocks (final saves, notebook, wandb finish) run.
- ToolTimeout / time_limit: per-tool-call wall-clock limit (SIGALRM based).
- shape_tool_output: keep tool output within a size the API will accept,
  saving the full text to disk when it has to be shortened.
- prepare_image_for_model: downscale plots to the model's native resolution
  (lossless w.r.t. what the model actually sees) so many-image requests
  never hit the API's per-image dimension limit.
- fingerprint_state / diff_fingerprints: cheap detection of changes to the
  shared data objects (used to warn the model when a FAILED call left data
  partially modified, and to decide what needs checkpointing).
- checkpoint_state: atomic per-step h5ad checkpoints of modified data.
- RunRecorder: append-only JSONL transcript of every request, response and
  tool execution, written as it happens (survives crashes / kills).
"""

from __future__ import annotations

import base64
import contextlib
import io
import json
import os
import signal
import threading
import time
from pathlib import Path
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Termination & timeouts
# ---------------------------------------------------------------------------


class TerminationRequested(BaseException):
    """Raised when the process receives SIGTERM/SIGHUP (e.g. cluster preemption).

    Subclasses BaseException (not Exception) so neither agent-written code nor
    tool code with a broad ``except Exception`` can swallow it.
    """


class ToolTimeout(BaseException):
    """Raised inside a tool call that exceeded its wall-clock limit.

    BaseException so ``except Exception: pass`` loops in agent code cannot
    swallow it. The executor catches it and reports it to the model as an error.
    """


def install_signal_handlers():
    """Convert SIGTERM / SIGHUP into TerminationRequested so cleanup runs."""

    def _handler(signum, _frame):
        raise TerminationRequested(f"Received signal {signal.Signals(signum).name}")

    for sig in (signal.SIGTERM, getattr(signal, "SIGHUP", None)):
        if sig is None:
            continue
        try:
            signal.signal(sig, _handler)
        except (ValueError, OSError):
            pass  # not in main thread / unsupported platform


@contextlib.contextmanager
def time_limit(seconds: Optional[float]):
    """Raise ToolTimeout in the main thread after `seconds` (None/0 = no limit).

    Note: a long-running C extension call is interrupted only when control
    returns to the Python interpreter. This covers infinite Python loops and
    most long pipelines; it cannot preempt a single stuck C call. Full
    preemption needs an out-of-process worker (see README / container plan).
    """
    if not seconds or seconds <= 0 or threading.current_thread() is not threading.main_thread():
        yield
        return

    def _handler(_signum, _frame):
        raise ToolTimeout(f"Tool call exceeded the {seconds:.0f}s time limit and was interrupted.")

    previous = signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


# ---------------------------------------------------------------------------
# Tool output shaping
# ---------------------------------------------------------------------------

DEFAULT_TOOL_OUTPUT_MAX_CHARS = 100_000  # ~25k tokens; well under any API limit


def shape_tool_output(text: str, output_dir: Path, exec_id: Any,
                      max_chars: Optional[int] = DEFAULT_TOOL_OUTPUT_MAX_CHARS) -> tuple[str, Optional[str]]:
    """Return (text_for_model, full_output_path_or_None).

    Output up to `max_chars` is passed through unchanged. Anything larger
    (e.g. printing a 1M-row DataFrame) would otherwise make the request fail,
    so the full text is written to disk and the model gets the head and tail
    with an explicit marker telling it where the rest is.
    """
    if text is None:
        text = ""
    if not max_chars or len(text) <= max_chars:
        return text, None

    out_dir = Path(output_dir) / "tool_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    full_path = out_dir / f"exec_{exec_id}.txt"
    full_path.write_text(text)

    head = int(max_chars * 0.7)
    tail = max_chars - head
    omitted = len(text) - head - tail
    shaped = (
        text[:head]
        + f"\n\n[... OUTPUT TRUNCATED: {omitted:,} of {len(text):,} characters omitted. "
          f"Full output saved to {full_path} (read it with run_python_repl if needed) ...]\n\n"
        + text[-tail:]
    )
    return shaped, str(full_path)


# ---------------------------------------------------------------------------
# Images
# ---------------------------------------------------------------------------

def model_max_image_edge(model: str) -> int:
    """Long-edge resolution the model natively processes (larger images are
    downscaled server-side anyway). Claude 4.7+ is high-res (2576 px); other
    models are standard (1568 px). Both are below the 3000 px limit that applies
    once a request contains more than 20 images."""
    m = (model or "").lower()
    standard = ("claude-3", "claude-4-", "opus-4-5", "opus-4-6", "sonnet-4-5",
                "sonnet-4-6", "haiku-4-5", "opus-4-1", "opus-4-2", "sonnet-4-2")
    if any(s in m for s in standard) or "claude" not in m:
        return 1568
    return 2576


def prepare_image_for_model(path: str, max_edge: int = 1568) -> tuple[str, str]:
    """Return (base64_data, media_type) for an image, downscaled to `max_edge`.

    The original file on disk is untouched (full resolution stays in output/).
    """
    try:
        from PIL import Image
        with Image.open(path) as im:
            if max(im.size) > max_edge:
                im = im.copy()
                im.thumbnail((max_edge, max_edge), Image.LANCZOS)
                if im.mode not in ("RGB", "RGBA", "L", "LA", "P"):
                    im = im.convert("RGBA")
                buf = io.BytesIO()
                im.save(buf, format="PNG", optimize=True)
                return base64.b64encode(buf.getvalue()).decode("utf-8"), "image/png"
    except Exception:
        pass  # fall back to raw bytes
    with open(path, "rb") as f:
        data = f.read()
    ext = os.path.splitext(path)[1].lower()
    media = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".gif": "image/gif",
             ".webp": "image/webp"}.get(ext, "image/png")
    return base64.b64encode(data).decode("utf-8"), media


# ---------------------------------------------------------------------------
# Data fingerprints (change detection)
# ---------------------------------------------------------------------------

# state attribute -> name the agent sees
DATA_SLOTS = {
    "adata_rna": "adata_rna",
    "adata_atac": "adata_atac",
    "adata_combined": "adata_combined",
    "cnmf_usage_df": "usage_df",
    "cnmf_spectra_scores_df": "spectra_scores_df",
    "cnmf_top_genes_df": "top_genes_df",
}


def _matrix_signature(X) -> Optional[str]:
    if X is None:
        return None
    try:
        import numpy as np
        import scipy.sparse as sp
        if sp.issparse(X):
            s = float(np.asarray(X.data, dtype=np.float64).sum()) if X.nnz else 0.0
            return f"{X.shape}|{X.dtype}|nnz={X.nnz}|sum={s:.10g}"
        arr = np.asarray(X)
        s = float(np.nansum(arr, dtype=np.float64)) if arr.size else 0.0
        return f"{arr.shape}|{arr.dtype}|sum={s:.10g}"
    except Exception as e:  # backed / exotic matrices
        return f"{type(X).__name__}|unhashable:{type(e).__name__}"


def _frame_signature(df) -> Optional[str]:
    if df is None:
        return None
    try:
        import pandas as pd
        h = int(pd.util.hash_pandas_object(df, index=True).sum())
        return f"{df.shape}|{h}"
    except Exception:
        try:
            return f"{df.shape}|cols={list(df.columns)}"
        except Exception:
            return repr(type(df))


def _value_signature(v, _depth: int = 0):
    """Exact content signature of an arbitrary .uns value (nested dicts/lists,
    arrays incl. structured ones, DataFrames, scalars). Small, so hashed fully."""
    import hashlib
    import numpy as np
    import pandas as pd
    import scipy.sparse as sp
    if _depth > 20:
        return f"<depth>{type(v).__name__}"
    try:
        if v is None or isinstance(v, (bool, int, float, str, bytes, np.generic)):
            return repr(v)
        if isinstance(v, dict):
            return {str(k): _value_signature(x, _depth + 1) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [_value_signature(x, _depth + 1) for x in v]
        if isinstance(v, (pd.DataFrame, pd.Series)):
            return _frame_signature(v.to_frame() if isinstance(v, pd.Series) else v)
        if sp.issparse(v):
            v = v.tocsr()
            h = hashlib.blake2b(digest_size=16)
            for part in (v.data, v.indices, v.indptr):
                h.update(np.ascontiguousarray(part).tobytes())
            return f"sparse{v.shape}|{v.dtype}|{h.hexdigest()}"
        if isinstance(v, np.ndarray):
            if v.dtype.hasobject:
                data = repr(v.tolist()).encode()
            else:
                data = np.ascontiguousarray(v).tobytes()
            return f"{v.shape}|{v.dtype}|{hashlib.blake2b(data, digest_size=16).hexdigest()}"
        return f"{type(v).__name__}|{hashlib.blake2b(repr(v).encode(), digest_size=16).hexdigest()}"
    except Exception as e:  # noqa: BLE001 - fingerprinting must never fail a call
        return f"{type(v).__name__}|unhashable:{type(e).__name__}"


def _anndata_fingerprint(ad) -> dict:
    fp = {"_id": id(ad)}
    try:
        fp["shape"] = (int(ad.n_obs), int(ad.n_vars))
        fp["X"] = _matrix_signature(getattr(ad, "X", None))
        fp["obs"] = _frame_signature(ad.obs)
        fp["var"] = _frame_signature(ad.var)
        fp["obs_columns"] = tuple(map(str, ad.obs.columns))
        fp["var_columns"] = tuple(map(str, ad.var.columns))
        for attr in ("layers", "obsm", "varm", "obsp", "varp"):
            m = getattr(ad, attr, None)
            fp[attr] = {str(k): (_frame_signature(m[k]) if hasattr(m[k], "columns")
                                 else _matrix_signature(m[k]))
                        for k in m.keys()} if m is not None else {}
        fp["uns_keys"] = tuple(sorted(map(str, ad.uns.keys())))
        fp["uns"] = {str(k): _value_signature(v) for k, v in ad.uns.items()}
    except Exception as e:
        fp["error"] = f"{type(e).__name__}: {e}"
    return fp


def fingerprint_state(state) -> dict:
    """Fingerprint every mutable shared data object on the agent state."""
    out = {}
    for attr in DATA_SLOTS:
        obj = getattr(state, attr, None)
        if obj is None:
            out[attr] = None
        elif hasattr(obj, "n_obs"):
            out[attr] = _anndata_fingerprint(obj)
        else:
            out[attr] = {"_id": id(obj), "frame": _frame_signature(obj)}
    return out


def diff_fingerprints(before: dict, after: dict) -> dict[str, list[str]]:
    """Return {state_attr: [human-readable changes]} for slots that changed."""
    changes: dict[str, list[str]] = {}
    for attr in DATA_SLOTS:
        b, a = before.get(attr), after.get(attr)
        if b == a:
            continue
        items: list[str] = []
        if b is None or a is None:
            items.append("object created" if b is None else "object removed")
        else:
            if b.get("_id") != a.get("_id"):
                items.append("object replaced with a new object")
            if b.get("shape") != a.get("shape") and "shape" in a:
                items.append(f"shape {b.get('shape')} -> {a.get('shape')}")
            if b.get("X") != a.get("X") and "X" in a:
                items.append(".X values changed")
            for key in ("obs", "var"):
                bc, ac = set(b.get(f"{key}_columns", ())), set(a.get(f"{key}_columns", ()))
                if ac - bc:
                    items.append(f".{key} columns added: {sorted(ac - bc)}")
                if bc - ac:
                    items.append(f".{key} columns removed: {sorted(bc - ac)}")
                if not (ac ^ bc) and b.get(key) != a.get(key):
                    items.append(f".{key} values changed")
            for key in ("layers", "obsm", "varm", "obsp", "varp"):
                bd, ad_ = b.get(key, {}) or {}, a.get(key, {}) or {}
                added = sorted(set(ad_) - set(bd))
                removed = sorted(set(bd) - set(ad_))
                modified = sorted(k for k in set(bd) & set(ad_) if bd[k] != ad_[k])
                if added:
                    items.append(f".{key} added: {added}")
                if removed:
                    items.append(f".{key} removed: {removed}")
                if modified:
                    items.append(f".{key} modified: {modified}")
            bu, au = set(b.get("uns_keys", ())), set(a.get("uns_keys", ()))
            if au - bu:
                items.append(f".uns keys added: {sorted(au - bu)}")
            if bu - au:
                items.append(f".uns keys removed: {sorted(bu - au)}")
            bv, av = b.get("uns", {}) or {}, a.get("uns", {}) or {}
            uns_mod = sorted(k for k in set(bv) & set(av) if bv[k] != av[k])
            if uns_mod:
                items.append(f".uns modified: {uns_mod}")
            if "frame" in a and b.get("frame") != a.get("frame"):
                items.append("values changed")
            if not items:
                items.append("modified")
        changes[attr] = items
    return changes


def format_changes(changes: dict[str, list[str]]) -> str:
    return "; ".join(f"{DATA_SLOTS.get(k, k)}: {', '.join(v)}" for k, v in changes.items())


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------

def checkpoint_state(state, output_dir: Path, attrs) -> dict[str, str]:
    """Atomically write the given data slots to output_dir/checkpoints/.

    Returns {attr: path_or_error}. Never raises (a failed checkpoint must not
    kill the run); errors are returned so the caller can log them.
    """
    ckpt_dir = Path(output_dir) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for attr in attrs:
        obj = getattr(state, attr, None)
        if obj is None:
            continue
        is_adata = hasattr(obj, "n_obs")
        final = ckpt_dir / (f"{attr}.h5ad" if is_adata else f"{attr}.csv")
        tmp = final.with_name(final.name + ".tmp")
        try:
            if is_adata:
                obj.write_h5ad(tmp)
            else:
                obj.to_csv(tmp)
            os.replace(tmp, final)
            results[attr] = str(final)
        except BaseException as e:  # noqa: BLE001 - incl. PyO3 panics
            if isinstance(e, (KeyboardInterrupt, TerminationRequested)):
                raise
            with contextlib.suppress(Exception):
                tmp.unlink()
            results[attr] = f"ERROR {type(e).__name__}: {e}"
    return results


# ---------------------------------------------------------------------------
# Append-only run transcript
# ---------------------------------------------------------------------------

def _strip_image_bytes(obj):
    """Replace base64 image payloads with a pointer to the file on disk."""
    if isinstance(obj, list):
        return [_strip_image_bytes(x) for x in obj]
    if isinstance(obj, dict):
        if obj.get("type") == "input_image":
            return {"type": "input_image", "path": obj.get("path", "<unknown>")}
        if obj.get("type") == "image":
            return {"type": "image", "path": obj.get("path", "<omitted>")}
        return {k: _strip_image_bytes(v) for k, v in obj.items()}
    return obj


class RunRecorder:
    """Append-only JSONL logs, flushed + fsynced per record.

    - conversation.jsonl : every request sent to and response received from the model
    - steps.jsonl        : every tool execution record (full args, result, error)
    Image bytes are replaced with the path of the saved PNG.
    """

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self._files = {}

    def _write(self, name: str, record: dict):
        try:
            f = self._files.get(name)
            if f is None:
                f = open(self.output_dir / name, "a")
                self._files[name] = f
            record = {"ts": time.time(), **record}
            f.write(json.dumps(_strip_image_bytes(record), default=str) + "\n")
            f.flush()
            os.fsync(f.fileno())
        except Exception as e:  # logging must never kill the run
            print(f"⚠️ RunRecorder could not write {name}: {e}")

    def request(self, turn: int, messages: list):
        self._write("conversation.jsonl", {"kind": "request", "turn": turn, "messages": messages})

    def response(self, turn: int, response):
        raw = getattr(response, "raw_response", None)
        raw_dump = None
        if raw is not None and hasattr(raw, "model_dump"):
            try:
                raw_dump = raw.model_dump(mode="json")
            except Exception:
                raw_dump = None
        self._write("conversation.jsonl", {
            "kind": "response", "turn": turn, "id": getattr(response, "id", None),
            "stop_reason": getattr(response, "stop_reason", None),
            "output": raw_dump if raw_dump is not None else getattr(response, "output", None),
        })

    def event(self, kind: str, **fields):
        self._write("conversation.jsonl", {"kind": kind, **fields})

    def step(self, record: dict):
        self._write("steps.jsonl", record)

    def close(self):
        for f in self._files.values():
            with contextlib.suppress(Exception):
                f.close()
        self._files = {}

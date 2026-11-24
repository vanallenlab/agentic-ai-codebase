"""
Tool runtime: owns the shared data objects and executes tool calls.

This is everything that touches the (large, mutable) analysis data or runs
agent-written code. It runs inside the *worker* process (local subprocess or
container, see worker_main.py / worker_client.py) so that a crash, OOM kill or
hang in a tool cannot take down the harness. It can also run in-process
(execution.mode: inprocess) for debugging.

The harness side (tool_executor.py) does argument validation, the telemetry
hook, output shaping, image encoding and logging; it sends only validated
calls here.
"""

from __future__ import annotations

import contextlib
import os
import random
import re
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.harness_guard import (  # noqa: E402
    DATA_SLOTS, TerminationRequested, ToolTimeout, checkpoint_state, diff_fingerprints,
    fingerprint_state, format_changes, time_limit,
)
from src.logging_utils import log  # noqa: E402
from src.tool_metadata import get_output_path_params, is_plotting_tool  # noqa: E402
from src.tool_registry import execute_tool  # noqa: E402

# Tools that run agent-written code
AGENT_CODE_TOOLS = frozenset({"run_python_repl", "run_terminal"})

# Post-hoc audit of agent code (flags only; nothing is blocked by these).
_AUDIT_PATTERNS = {
    "image_read": re.compile(
        r"Image\.open|imread|imageio|\bcv2\b|tesseract|easyocr|\bocr\b|base64|read_bytes"
        r"|open\([^)\n]*\.(?:png|jpe?g|gif|bmp|tiff?|svg|pdf)['\"]", re.I),
    "llm_access": re.compile(
        r"\banthropic\b|\bopenai\b|api\.anthropic\.com|api\.openai\.com|_API_KEY|/environ\b"
        r"|\bclaude-|\bgpt-\d", re.I),
}


def audit_agent_code(name: str, args: dict) -> list[str]:
    code = str(args.get("command", "")) if name in AGENT_CODE_TOOLS else ""
    return [flag for flag, rx in _AUDIT_PATTERNS.items() if code and rx.search(code)]


_IMAGE_READ_BLOCKED = ("Reading image files is disabled in this run: plots are saved to disk, but this is a "
                       "text-only configuration and their contents are not available to you.")


@contextlib.contextmanager
def _block_image_reads(active: bool):
    """Text-only runs: make the standard image readers fail inside agent code."""
    if not active:
        yield
        return
    import matplotlib.image as mpimg
    from PIL import Image

    def _blocked(*_a, **_k):
        raise PermissionError(_IMAGE_READ_BLOCKED)

    saved = (Image.open, mpimg.imread)
    Image.open, mpimg.imread = _blocked, _blocked
    try:
        yield
    finally:
        Image.open, mpimg.imread = saved


# Tools that accept literature_blacklist (injected from config to exclude source study)
LITERATURE_TOOLS = frozenset({
    "query_pubmed",
    "query_arxiv",
    "query_scholar",
    "search_google",
    "extract_url_content",
    "extract_pdf_content",
    "fetch_supplementary_info_from_doi",
    "advanced_web_search_claude",
})


@dataclass
class DataState:
    """The shared analysis data (lives only in the worker)."""
    adata_rna: Any = None
    adata_atac: Any = None
    adata_atac_backed: Any = None
    adata_combined: Any = None
    data_dir: Optional[Path] = None
    cnmf_usage_df: Any = None
    cnmf_spectra_scores_df: Any = None
    cnmf_top_genes_df: Any = None
    literature_blacklist: Optional[dict] = None
    dirty_slots: set = field(default_factory=set)


class ToolRuntime:
    """Loads data and executes tool calls. All methods return JSON-able dicts."""

    def __init__(self):
        self.state = DataState()
        self.output_dir: Optional[Path] = None
        self.tool_timeout_s: Optional[float] = None
        self.track_data_changes = True
        self.rollback_on_failure = True
        self.block_image_reads = False
        self._secrets: dict[str, str] = {}
        self.generation = 0
        # attr -> on-disk copy of the slot as of the last checkpoint (None = slot was empty)
        self.restore_sources: dict[str, Optional[str]] = {}

    # ------------------------------------------------------------------ init
    def init(self, req: dict) -> dict:
        """Load data. `checkpoints` maps state attr -> checkpoint file to load
        instead of the original input (used when restarting after a crash)."""
        import scanpy as sc

        self.output_dir = Path(req["output_dir"]).resolve()
        self.tool_timeout_s = req.get("tool_timeout_s") or None
        self.track_data_changes = req.get("track_data_changes", True)
        self.rollback_on_failure = req.get("rollback_on_failure", True)
        self.block_image_reads = not req.get("send_images", True)
        self._secrets = dict(req.get("secrets") or {})
        self.generation = int(req.get("generation", 0))
        self.state.literature_blacklist = req.get("literature_blacklist")

        seed = req.get("seed")
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Distinct REPL plot names per worker incarnation (no overwrite after restart)
        from src.tools_external import support_tools
        support_tools._repl_file_prefix = f"w{self.generation}_" if self.generation else ""

        data = req.get("data", {}) or {}
        ckpt = req.get("checkpoints", {}) or {}

        def _src(attr, key):
            return ckpt.get(attr) or data.get(key)

        path = _src("adata_rna", "scrna_data_path")
        if path:
            self.state.adata_rna = sc.read_h5ad(path)
            log(f"✓ Loaded scRNA data from {path}: {self.state.adata_rna.n_obs} cells, "
                f"{self.state.adata_rna.n_vars} genes")
        path = _src("adata_atac", "scatac_data_path")
        if path:
            self.state.adata_atac = sc.read_h5ad(path)
            log(f"✓ Loaded scATAC data from {path}: {self.state.adata_atac.n_obs} cells, "
                f"{self.state.adata_atac.n_vars} peaks")
        path = ckpt.get("adata_combined")
        if path:
            self.state.adata_combined = sc.read_h5ad(path)
            log(f"✓ Loaded combined data from {path}")
        path = data.get("scatac_snap_path")
        if path:
            import snapatac2 as snap
            from src.tools_external.support_tools import ReadOnlyAnnDataProxy
            self.state.adata_atac_backed = ReadOnlyAnnDataProxy(snap.read_dataset(path, mode="r"))
            log(f"✓ Loaded scATAC backed dataset from {path}: {self.state.adata_atac_backed.n_obs} cells, "
                f"{self.state.adata_atac_backed.n_vars} peaks (read-only proxy)")
        if data.get("data_dir"):
            p = Path(data["data_dir"])
            if not p.exists():
                raise FileNotFoundError(f"data_dir not found: {p}")
            self.state.data_dir = p
            log(f"✓ Data directory: {p} ({len(list(p.iterdir()))} items)")
        for attr, key in (("cnmf_usage_df", "cnmf_usage_path"),
                          ("cnmf_spectra_scores_df", "cnmf_spectra_scores_path"),
                          ("cnmf_top_genes_df", "cnmf_top_genes_path")):
            path = _src(attr, key)
            if path:
                setattr(self.state, attr, pd.read_csv(path, index_col=0))
                log(f"✓ Loaded {attr} from {path}: {getattr(self.state, attr).shape}")

        # Where each slot's last saved state lives (used to roll back failed calls)
        input_keys = {"adata_rna": "scrna_data_path", "adata_atac": "scatac_data_path",
                      "adata_combined": None, "cnmf_usage_df": "cnmf_usage_path",
                      "cnmf_spectra_scores_df": "cnmf_spectra_scores_path",
                      "cnmf_top_genes_df": "cnmf_top_genes_path"}
        self.restore_sources = {attr: (ckpt.get(attr) or (data.get(key) if key else None))
                                for attr, key in input_keys.items()}
        self.state.dirty_slots = set()
        return {"info": self.info()["info"]}

    def info(self, _req: dict = None) -> dict:
        out = {}
        for key, attr in (("rna", "adata_rna"), ("atac", "adata_atac"),
                          ("combined", "adata_combined"), ("atac_backed", "adata_atac_backed")):
            obj = getattr(self.state, attr, None)
            if obj is not None:
                try:
                    out[key] = {"n_obs": int(obj.n_obs), "n_vars": int(obj.n_vars)}
                except BaseException:  # noqa: BLE001 - closed/invalid backed object
                    pass
        return {"info": out}

    # ------------------------------------------------------------------ call
    def call(self, req: dict) -> dict:
        """Execute one (already validated) tool call."""
        name, args = req["name"], req.get("args") or {}
        exec_id = req.get("exec_id", 0)
        st = self.state
        output_dir = self.output_dir

        exec_args = _prepare_args(args, output_dir, tool_name=name, state=st)
        args_preview = {k: (f"<AnnData {v.n_obs}x{v.n_vars}>" if hasattr(v, "n_obs") else
                            f"<DataFrame {v.shape}>" if isinstance(v, pd.DataFrame) else v)
                        for k, v in exec_args.items()}
        log(f"    📥 Args: {args_preview}")

        before = _safe(fingerprint_state, st, what="data fingerprint") if self.track_data_changes else None
        dirty_before = set(st.dirty_slots)
        figures_before = set(plt.get_fignums())
        result, error_text, notes, saved_plots = None, None, [], []

        # PyO3 converts most Rust panics to PanicException (BaseException), caught here.
        agent_code = name in AGENT_CODE_TOOLS
        audit_flags = audit_agent_code(name, args)
        if audit_flags:
            log(f"   🔎 Audit flags: {audit_flags}")
        try:
            with time_limit(self.tool_timeout_s), self._tool_env(agent_code), \
                    _block_image_reads(self.block_image_reads and name == "run_python_repl"):
                result = execute_tool(name, **exec_args)
            if _looks_like_error_string(result):
                error_text = str(result)
        except (KeyboardInterrupt, TerminationRequested):
            raise
        except BaseException as e:  # noqa: BLE001 - incl. SystemExit, PanicException, ToolTimeout
            error_text = _format_exception(e)
            log(f"   ❌ FAILED [{type(e).__name__}]")
            log("      " + "".join(traceback.format_exception(type(e), e, e.__traceback__)).replace("\n", "\n      "))

        # AnnData returned -> route to state slot by tool name prefix
        if error_text is None and hasattr(result, "n_obs"):
            if name.startswith("scRNA_scATAC_"):
                attr, label = "adata_combined", "genes/peaks"
            elif name.startswith("scATAC_"):
                attr, label = "adata_atac", "peaks"
            elif name.startswith("scRNA_"):
                attr, label = "adata_rna", "genes"
            else:
                attr, label = "adata_combined", "features"
            prev = getattr(st, attr, None)
            setattr(st, attr, result)
            msg = f"{result.n_obs} cells, {result.n_vars} {label}"
            if prev is not None and hasattr(prev, "n_obs") and \
                    (prev.n_obs, prev.n_vars) != (result.n_obs, result.n_vars):
                msg += (f" [harness] {DATA_SLOTS.get(attr, attr)} replaced: was "
                        f"{prev.n_obs} x {prev.n_vars}, now {result.n_obs} x {result.n_vars}.")
            result = msg

        if name == "run_python_repl":
            notes += _safe(_sync_repl_namespace, st, what="REPL namespace sync") or []

        saved_plots += _safe(_save_new_figures, output_dir, figures_before, exec_id,
                             tool_name=name, what="figure capture") or []
        if name == "run_python_repl":
            from src.tools_external.support_tools import get_repl_saved_plots
            for path in _safe(get_repl_saved_plots, what="REPL plot list") or []:
                if Path(path).is_file():
                    saved_plots.append(path)

        data_changes, rolled_back = {}, False
        if before is not None:
            after = _safe(fingerprint_state, st, what="data fingerprint")
            if after is not None:
                data_changes = diff_fingerprints(before, after)
                if data_changes:
                    log(f"   🧬 Data changes: {format_changes(data_changes)}")
                    if error_text is None:
                        st.dirty_slots |= set(data_changes)
                    else:
                        reason = "disabled (rollback_on_failure: false)"
                        if self.rollback_on_failure:
                            rolled_back, reason = self._rollback(sorted(data_changes), dirty_before)
                        if rolled_back:
                            log(f"   ↩️ Rolled back: {', '.join(sorted(data_changes))}")
                            notes.append(
                                "[harness] This call FAILED after modifying shared data "
                                f"({format_changes(data_changes)}). Those changes were ROLLED BACK: "
                                f"{', '.join(DATA_SLOTS[a] for a in sorted(data_changes))} "
                                "is exactly as it was before this call. Other variables you "
                                "created in the REPL during this call are unaffected."
                            )
                        else:
                            st.dirty_slots |= set(data_changes)
                            log(f"   ⚠️ Rollback not possible: {reason}")
                            notes.append(
                                "[harness] WARNING: this call FAILED but had already modified shared data "
                                f"before failing: {format_changes(data_changes)}. It could not be rolled "
                                f"back ({reason}). The data is now in a partially modified state; check "
                                "it before re-running steps that are not idempotent (e.g. log1p, "
                                "normalization, scaling)."
                            )

        success = error_text is None
        if success:
            body = "" if result is None else str(result)
        else:
            body = error_text
        return {
            "success": success,
            "body": body,
            "notes": notes,
            "plots": saved_plots,
            "data_changes": (format_changes(data_changes) + (" [rolled back]" if rolled_back else "")
                             if data_changes else ""),
            "rolled_back": rolled_back,
            "audit_flags": audit_flags,
            "args_preview": {k: (v if isinstance(v, (str, int, float, bool, type(None))) else str(v))
                             for k, v in args_preview.items()},
        }

    @contextlib.contextmanager
    def _tool_env(self, agent_code: bool):
        """LLM credentials are in os.environ only while a harness tool runs,
        never while agent-written code runs."""
        if agent_code or not self._secrets:
            yield
            return
        saved = {k: os.environ.get(k) for k in self._secrets}
        os.environ.update(self._secrets)
        try:
            yield
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    # ------------------------------------------------------------ persistence
    def checkpoint(self, _req: dict = None) -> dict:
        """Write slots modified since the last checkpoint (atomic)."""
        slots = sorted(self.state.dirty_slots)
        if not slots:
            return {"results": {}}
        results = checkpoint_state(self.state, self.output_dir, slots)
        failed = set()
        for attr in slots:
            res = results.get(attr)
            if res is None:            # slot is now empty -> nothing to write
                self.restore_sources[attr] = None
            elif res.startswith("ERROR"):
                failed.add(attr)
            else:
                self.restore_sources[attr] = res
        self.state.dirty_slots = failed
        return {"results": results}

    def _rollback(self, attrs: list[str], dirty_before: set) -> tuple[bool, str]:
        """Restore the given slots to their last checkpoint, all or nothing.

        Valid only for slots that had no unsaved changes before this call (then
        the checkpoint IS the pre-call state). Everything is loaded before
        anything is swapped, so a failed load leaves the state untouched.
        """
        stale = [a for a in attrs if a in dirty_before]
        if stale:
            return False, (f"{', '.join(DATA_SLOTS[a] for a in stale)} had unsaved changes from an "
                           "earlier step (checkpointing disabled or a checkpoint failed)")
        loaded = {}
        try:
            for attr in attrs:
                src = self.restore_sources.get(attr)
                if src is None:
                    loaded[attr] = None
                elif str(src).endswith(".h5ad"):
                    import anndata
                    loaded[attr] = anndata.read_h5ad(src)
                else:
                    loaded[attr] = pd.read_csv(src, index_col=0)
        except (KeyboardInterrupt, TerminationRequested):
            raise
        except BaseException as e:  # noqa: BLE001
            return False, f"reloading the saved copy failed: {type(e).__name__}: {e}"

        from src.tools_external import support_tools
        ns = support_tools._persistent_namespace
        for attr, obj in loaded.items():
            setattr(self.state, attr, obj)
            varname = DATA_SLOTS[attr]
            if obj is None:
                ns.pop(varname, None)
            else:
                ns[varname] = obj
        import gc
        gc.collect()
        return True, ""

    def finalize(self, _req: dict = None) -> dict:
        """Write final result_*.h5ad files (atomic) and close the backed dataset."""
        saved, errors = {}, {}
        for attr, fname in (("adata_rna", "result_rna.h5ad"), ("adata_atac", "result_atac.h5ad"),
                            ("adata_combined", "result_combined.h5ad")):
            obj = getattr(self.state, attr, None)
            if obj is None:
                continue
            out = self.output_dir / fname
            tmp = out.with_name(out.name + ".tmp")
            try:
                obj.write_h5ad(tmp)
                os.replace(tmp, out)
                saved[attr] = str(out)
            except BaseException as e:  # noqa: BLE001
                if isinstance(e, KeyboardInterrupt):
                    raise
                errors[attr] = f"{type(e).__name__}: {e}"
        info = self.info()["info"]
        if self.state.adata_atac_backed is not None:
            try:
                self.state.adata_atac_backed.harness_close()
            except BaseException as e:  # noqa: BLE001
                errors["adata_atac_backed"] = f"close failed: {e}"
        return {"saved": saved, "errors": errors, "info": info}


# ---------------------------------------------------------------------------
# Helpers (moved from tool_executor)
# ---------------------------------------------------------------------------

def _safe(fn, *a, what: str = "", **kw):
    """Run bookkeeping so that its failure can never fail the tool or the run."""
    try:
        return fn(*a, **kw)
    except (KeyboardInterrupt, TerminationRequested):
        raise
    except Exception as e:
        log(f"   ⚠️ {what or getattr(fn, '__name__', 'bookkeeping')} failed: {type(e).__name__}: {e}")
        return None


def _save_new_figures(output_dir, figures_before: set, exec_id, tool_name: str = None) -> list[str]:
    """Save matplotlib figures created during the call. Filenames carry the
    (run-unique) exec id, so nothing is overwritten, including across restarts."""
    saved = []
    new_figures = set(plt.get_fignums()) - figures_before
    for k, fig_num in enumerate(sorted(new_figures), 1):
        fig = plt.figure(fig_num)
        try:
            title = None
            if fig._suptitle:
                title = fig._suptitle.get_text().strip()
            elif fig.axes and fig.axes[0].get_title():
                title = fig.axes[0].get_title().strip()
            stem = ("".join(c if c.isalnum() or c in "-_" else "_" for c in title)[:50]
                    if title else (tool_name or "plot"))
            plot_file = Path(output_dir) / f"{stem}_e{exec_id}_{k}.png"
            fig.savefig(plot_file, dpi=150, bbox_inches="tight")
            saved.append(str(plot_file))
        except Exception as e:
            log(f"   ⚠️ Could not save figure {fig_num}: {e}")
        finally:
            plt.close(fig)
    return saved


def _sync_repl_namespace(state: DataState) -> list[str]:
    """Make the data state follow REPL rebinding of dataset names (see docs in
    tool_executor history): adopt new datasets, restore non-dataset rebinding."""
    from src.tools_external import support_tools as st
    ns = st._persistent_namespace
    notes = []
    for attr, varname in DATA_SLOTS.items():
        if varname not in ns:
            continue
        obj, cur = ns[varname], getattr(state, attr, None)
        if obj is cur:
            continue
        is_adata_slot = attr.startswith("adata")
        valid = hasattr(obj, "n_obs") if is_adata_slot else isinstance(obj, pd.DataFrame)
        if valid:
            if getattr(obj, "is_view", False):
                obj = obj.copy()
                ns[varname] = obj
            setattr(state, attr, obj)
            shape = f"{obj.n_obs} x {obj.n_vars}" if is_adata_slot else f"{obj.shape}"
            notes.append(f"[harness] `{varname}` was reassigned in the REPL; all tools now use "
                         f"the new object ({shape}).")
        elif cur is not None:
            ns[varname] = cur
            notes.append(f"[harness] `{varname}` was reassigned to a {type(obj).__name__}; the dataset "
                         f"object was restored. Use a different variable name for other values.")
    backed = getattr(state, "adata_atac_backed", None)
    if backed is not None and ns.get("adata_atac_backed") is not backed:
        ns["adata_atac_backed"] = backed
        notes.append("[harness] `adata_atac_backed` is read-only and cannot be reassigned; restored.")
    return notes


def _format_exception(e: BaseException) -> str:
    """Error text for the model: message + the innermost traceback frames."""
    from src.tools_external.support_tools import ReplExecutionError
    if isinstance(e, (ReplExecutionError, ToolTimeout)):
        return str(e)
    msg = str(e) or "(no message)"
    tail = "".join(traceback.format_exception(type(e), e, e.__traceback__)[-6:])
    if len(tail) > 4000:
        tail = tail[-4000:]
    return f"{type(e).__name__}: {msg}\n\nTraceback (most recent call last, innermost frames):\n{tail}"


def _looks_like_error_string(result) -> bool:
    """Many tools signal failure by returning a string starting with 'Error'."""
    return isinstance(result, str) and result.lstrip().lower().startswith(("error:", "error -", "error "))


def _prepare_args(args: dict, output_dir: Path, tool_name: str, state: DataState) -> dict:
    """Inject data objects and redirect output paths.

    Data objects are injected unconditionally; execute_tool() filters kwargs by
    function signature so each tool only receives what it declares.
    """
    exec_args = dict(args)
    exec_args.pop("dataset", None)  # Legacy arg from LLM; no longer used

    if tool_name in LITERATURE_TOOLS and state.literature_blacklist is not None:
        exec_args["literature_blacklist"] = state.literature_blacklist

    if state.adata_rna is not None:
        exec_args["adata"] = state.adata_rna       # scRNA tools use "adata"
        exec_args["adata_rna"] = state.adata_rna   # multimodal + REPL use "adata_rna"
    if state.adata_atac is not None:
        exec_args["adata_atac"] = state.adata_atac
    if state.adata_atac_backed is not None:
        exec_args["adata_atac_backed"] = state.adata_atac_backed
    if state.adata_combined is not None:
        exec_args["adata_combined"] = state.adata_combined
    if state.data_dir is not None:
        exec_args["data_dir"] = str(state.data_dir)
    if state.cnmf_usage_df is not None:
        exec_args["usage_df"] = state.cnmf_usage_df
    if state.cnmf_spectra_scores_df is not None:
        exec_args["spectra_scores_df"] = state.cnmf_spectra_scores_df
    if state.cnmf_top_genes_df is not None:
        exec_args["top_genes_df"] = state.cnmf_top_genes_df

    output_path_params = get_output_path_params(tool_name) if tool_name else set()
    if "output_dir" in output_path_params and "output_dir" not in args:
        exec_args["output_dir"] = str(output_dir)
    if "show" in args or (tool_name and is_plotting_tool(tool_name)):
        exec_args["show"] = False
    for path_key in output_path_params:
        if path_key in exec_args and exec_args[path_key]:
            original = str(exec_args[path_key])
            if not original.startswith(str(output_dir)):
                basename = Path(original).name or original.replace("./", "").replace("/", "_")
                exec_args[path_key] = str(Path(output_dir) / basename)
    return exec_args

"""Tool execution logic for the agent."""

import base64
import difflib
import json
import time
import traceback
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

from src.tool_registry import execute_tool
from src.hooks import pre_tool_call_hook, REASONING_FIELDS, RETRY_REASONING_FIELDS
from src.logging_utils import log
from src.wandb_utils import log_execution, log_plot
from src.tool_metadata import is_plotting_tool, get_output_path_params
from src.harness_guard import (
    DATA_SLOTS, TerminationRequested, ToolTimeout, diff_fingerprints, fingerprint_state,
    format_changes, prepare_image_for_model, shape_tool_output, time_limit,
)

# Arguments some models send that the harness drops on purpose (legacy).
IGNORED_MODEL_ARGS = frozenset({"dataset"})

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


class ToolCallWrapper:
    """Wrapper for tool call data from unified response format."""
    __slots__ = ('name', 'call_id', 'arguments', 'type')

    def __init__(self, data: dict):
        self.name = data.get("name")
        self.call_id = data.get("call_id")
        self.arguments = data.get("arguments", "{}")
        self.type = "function_call"


def _encode_image_to_base64(image_path: str) -> str:
    """Convert image file to base64 string for Responses API."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _save_new_figures(state, output_dir, figures_before: set, tool_name: str = None) -> list[dict]:
    """
    Save any matplotlib figures that were created during tool execution.

    Filenames always carry the monotonic plot id, so two figures with the same
    title (or from the same tool) never overwrite each other.

    Returns:
        list[dict]: List of dicts with a 'path' key for each saved plot
    """
    saved_plots = []
    figures_after = set(plt.get_fignums())
    new_figures = figures_after - figures_before

    for fig_num in sorted(new_figures):
        fig = plt.figure(fig_num)
        plot_id = state.get_next_plot_id()  # Monotonic ID ensures unique filenames
        try:
            fig_title = None
            if fig._suptitle:
                fig_title = fig._suptitle.get_text().strip()
            elif fig.axes and fig.axes[0].get_title():
                fig_title = fig.axes[0].get_title().strip()

            if fig_title:
                stem = "".join(c if c.isalnum() or c in "-_" else "_" for c in fig_title)[:50]
            elif tool_name:
                stem = tool_name
            else:
                stem = "plot"
            plot_file = Path(output_dir) / f"{stem}_{plot_id:03d}.png"

            fig.savefig(plot_file, dpi=150, bbox_inches="tight")
            saved_plots.append({"path": str(plot_file)})
            log_plot(str(plot_file), state.step_count)
        except Exception as e:
            log(f"   ⚠️ Could not save figure {fig_num}: {e}")
        finally:
            plt.close(fig)

    return saved_plots


def _validate_args(name: str, args: dict, state) -> str | None:
    """Check the model's arguments against the schema it was given.

    Returns an error message for the model, or None if valid. Unknown parameter
    names are reported (instead of being silently dropped) so the model can fix
    a typo on its next turn; missing required parameters likewise.
    """
    schemas = getattr(state, "tool_schemas", None) or {}
    if not schemas:
        return None  # no schema info (e.g. used outside run_agent) -> skip
    if name not in schemas:
        available = ", ".join(sorted(schemas))
        return f"Unknown tool '{name}'. Available tools: {available}"
    schema = schemas[name] or {}
    props = schema.get("properties", {}) or {}
    required = schema.get("required", []) or []

    def _describe(p):
        sch = props[p]
        t = sch.get("type") or "|".join(x.get("type", "?") for x in sch.get("anyOf", [])) or "any"
        return f"{p} ({t}{', required' if p in required else ''})"

    problems = []
    unknown = sorted(k for k in args if k not in props and k not in IGNORED_MODEL_ARGS)
    if unknown:
        hints = []
        for k in unknown:
            close = difflib.get_close_matches(k, list(props), n=1)
            hints.append(f"'{k}'" + (f" (did you mean '{close[0]}'?)" if close else ""))
        problems.append(f"unknown parameter(s): {', '.join(hints)}")
    missing = [p for p in required if p not in args]
    if missing:
        problems.append(f"missing required parameter(s): {', '.join(repr(m) for m in missing)}")
    if not problems:
        return None
    valid = ", ".join(_describe(p) for p in props) or "(none)"
    return (f"Invalid arguments for tool '{name}': {'; '.join(problems)}. "
            f"The tool was NOT run. Valid parameters: {valid}")


def _sync_repl_namespace(state) -> list[str]:
    """After a REPL call, make the harness state follow what the agent did.

    - If the agent rebound a dataset name to a new AnnData/DataFrame
      (e.g. ``adata_rna = adata_rna[mask].copy()``), the harness adopts the new
      object so every later tool and REPL call sees it (previously the rebinding
      was silently undone on the next call).
    - If the agent rebound a dataset name to something that is not a dataset
      (e.g. ``adata_rna = None``), the dataset is restored and the agent told.
    Returns notes for the model.
    """
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
    # The backed dataset is never replaceable
    backed = getattr(state, "adata_atac_backed", None)
    if backed is not None and ns.get("adata_atac_backed") is not backed:
        ns["adata_atac_backed"] = backed
        notes.append("[harness] `adata_atac_backed` is read-only and cannot be reassigned; restored.")
    return notes


def _format_exception(e: BaseException) -> str:
    """Error text for the model: message + the innermost traceback frames."""
    from src.tools_external.support_tools import ReplExecutionError
    if isinstance(e, ReplExecutionError):
        return str(e)  # already contains the agent-code traceback + partial output
    if isinstance(e, ToolTimeout):
        return str(e)
    msg = str(e) or "(no message)"
    tb_lines = traceback.format_exception(type(e), e, e.__traceback__)
    tail = "".join(tb_lines[-6:])
    if len(tail) > 4000:
        tail = tail[-4000:]
    return f"{type(e).__name__}: {msg}\n\nTraceback (most recent call last, innermost frames):\n{tail}"


def _looks_like_error_string(result) -> bool:
    """Many tools signal failure by returning a string starting with 'Error'."""
    return isinstance(result, str) and result.lstrip().lower().startswith(("error:", "error -", "error "))


def _safe(fn, *a, what: str = "", **kw):
    """Run bookkeeping so that its failure can never fail the tool or the run."""
    try:
        return fn(*a, **kw)
    except (KeyboardInterrupt, TerminationRequested):
        raise
    except Exception as e:
        log(f"   ⚠️ {what or getattr(fn, '__name__', 'bookkeeping')} failed: {type(e).__name__}: {e}")
        return None


def execute_tool_call(tc, state, output_dir, token_info=None):
    """
    Execute a single tool call and return the output dict(s).

    Robustness contract:
    - Never raises for anything the agent did (bad JSON, unknown tool, invalid
      args, exceptions/SystemExit/timeouts in tool or agent code). Only
      KeyboardInterrupt and TerminationRequested propagate.
    - Errors are returned to the model with status "error" and recorded as
      failures (success=False) everywhere.
    - Bookkeeping (plots, wandb, JSONL) is isolated from the tool's success.

    Returns:
        list: [function_call_output dict] plus a user message with any plots.
    """
    name = tc.name
    exec_id = state.get_next_exec_id()
    t0 = time.monotonic()
    output_dir = Path(output_dir)

    # --- 1. Parse arguments (malformed JSON must not crash the loop) ---
    raw_args = tc.arguments
    pre_error = None
    if isinstance(raw_args, dict):
        args = raw_args
    else:
        try:
            args = json.loads(raw_args) if raw_args else {}
        except (json.JSONDecodeError, TypeError) as e:
            args = {}
            pre_error = (f"Could not parse the tool arguments as JSON ({e}). "
                         f"The tool was NOT run. Raw arguments: {str(raw_args)[:2000]}")
    if pre_error is None and not isinstance(args, dict):
        pre_error = (f"Tool arguments must be a JSON object, got {type(args).__name__}. "
                     f"The tool was NOT run.")
        args = {}
    if pre_error is None:
        pre_error = _validate_args(name, args, state)

    # Detect retry
    recent_failures = state.get_recent_failures(name)
    is_retry = len(recent_failures) > 0
    retry_count = len(recent_failures)

    args_preview_str = ', '.join(f'{k}={v}' for k, v in list(args.items())[:2])
    retry_indicator = f" 🔄 RETRY #{retry_count + 1}" if is_retry else ""
    log(f"🔧 {name}({args_preview_str}){retry_indicator}")
    if is_retry and recent_failures:
        _safe(_log_retry_changes, args, recent_failures[-1].get("args", {}) or {}, what="retry diff")

    # --- 2. Post-hoc reasoning hook (observability; skipped for rejected calls) ---
    reasoning = None
    if pre_error is None and getattr(state, "capture_telemetry", True):
        last_failure = recent_failures[-1] if recent_failures else None
        reasoning = _safe(
            pre_tool_call_hook,
            client=state.client,
            tool_name=name,
            tool_args=args,
            previous_response_id=state.previous_response_id,
            pending_tool_calls=[tc],
            is_retry=is_retry,
            previous_error=last_failure.get("error") if last_failure else None,
            previous_args=last_failure.get("args") if last_failure else None,
            task=state.task,
            execution_history=state.execution_history,
            what="reasoning hook",
        )
    reasoning_data, intent = _extract_reasoning(reasoning)

    # --- 3. Execute ---
    saved_plots: list[dict] = []
    notes: list[str] = []
    result = None
    error_text = None
    args_preview = {}
    data_changes = {}

    if pre_error is not None:
        error_text = pre_error
        log(f"   ❌ REJECTED {error_text}")
    else:
        exec_args = _prepare_args(args, output_dir, tool_name=name,
                                  adata_rna=state.adata_rna,
                                  adata_atac=state.adata_atac,
                                  adata_atac_backed=state.adata_atac_backed,
                                  adata_combined=state.adata_combined,
                                  data_dir=state.data_dir,
                                  literature_blacklist=getattr(state, "literature_blacklist", None),
                                  cnmf_usage_df=getattr(state, "cnmf_usage_df", None),
                                  cnmf_spectra_scores_df=getattr(state, "cnmf_spectra_scores_df", None),
                                  cnmf_top_genes_df=getattr(state, "cnmf_top_genes_df", None))
        args_preview = {k: (f"<AnnData {v.n_obs}x{v.n_vars}>" if hasattr(v, 'n_obs') else
                            f"<DataFrame {v.shape}>" if isinstance(v, pd.DataFrame) else v)
                        for k, v in exec_args.items()}
        log(f"    📥 Args: {args_preview}")

        track = getattr(state, "track_data_changes", True)
        before = _safe(fingerprint_state, state, what="data fingerprint") if track else None
        figures_before = set(plt.get_fignums())

        # PyO3 converts most Rust panics to PanicException (BaseException), caught here.
        # True SIGABRT (double-panic, panic=abort) is uncatchable in-process.
        try:
            with time_limit(getattr(state, "tool_timeout_s", None)):
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
            prev = getattr(state, attr, None)
            setattr(state, attr, result)
            msg = f"{result.n_obs} cells, {result.n_vars} {label}"
            if prev is not None and hasattr(prev, "n_obs") and \
                    (prev.n_obs, prev.n_vars) != (result.n_obs, result.n_vars):
                msg += (f" [harness] {DATA_SLOTS.get(attr, attr)} replaced: was "
                        f"{prev.n_obs} x {prev.n_vars}, now {result.n_obs} x {result.n_vars}.")
            result = msg

        # REPL: adopt rebinding of dataset names
        if name == "run_python_repl":
            notes += _safe(_sync_repl_namespace, state, what="REPL namespace sync") or []

        # Plots: matplotlib figures left open + those the REPL saved itself
        saved_plots += _safe(_save_new_figures, state, output_dir, figures_before,
                             tool_name=name, what="figure capture") or []
        if name == "run_python_repl":
            from src.tools_external.support_tools import get_repl_saved_plots
            for path in _safe(get_repl_saved_plots, what="REPL plot list") or []:
                if Path(path).is_file():
                    saved_plots.append({"path": path})
                    _safe(log_plot, path, state.step_count, what="wandb plot")

        # Detect data changes (and warn the model if a FAILED call left data modified)
        if track and before is not None:
            after = _safe(fingerprint_state, state, what="data fingerprint")
            if after is not None:
                data_changes = diff_fingerprints(before, after)
                if data_changes:
                    state.dirty_slots |= set(data_changes)
                    log(f"   🧬 Data changes: {format_changes(data_changes)}")
                    if error_text is not None:
                        notes.append(
                            "[harness] WARNING: this call FAILED but had already modified shared data "
                            f"before failing: {format_changes(data_changes)}. The data is now in a "
                            "partially modified state; check it before re-running steps that are "
                            "not idempotent (e.g. log1p, normalization, scaling)."
                        )

    # --- 4. Build the text the model sees ---
    success = error_text is None
    if success:
        body = "" if result is None else str(result)
        if saved_plots:
            plot_info = ", ".join(f"📊 {p['path']}" for p in saved_plots)
            body = f"{body}\n{plot_info}" if body else plot_info
    else:
        body = error_text
    if notes:
        body = (body + "\n\n" if body else "") + "\n".join(notes)
    shaped, full_output_path = shape_tool_output(
        body, output_dir, exec_id, getattr(state, "tool_output_max_chars", None))
    if full_output_path:
        log(f"   📄 Output {len(body):,} chars; full text saved to {full_output_path}")

    if success:
        log(f"   ✅ {body}")
        output = json.dumps({"status": "success", "result": shaped})
    else:
        log(f"   ❌ {body}" if pre_error is None else "")
        output = json.dumps({"status": "error", "error": shaped})

    duration = round(time.monotonic() - t0, 3)

    # --- 5. Record (isolated from the tool outcome) ---
    def _record():
        state.add_execution(
            name, args, json.dumps(args_preview or args, default=str), intent, reasoning_data,
            success, body, token_info=token_info, exec_id=exec_id,
            error=None if success else body,
            model_text=getattr(state, "last_turn_text", ""),
            model_thinking=getattr(state, "last_turn_thinking", ""),
            data_changes=format_changes(data_changes) if data_changes else "",
            duration_s=duration,
            full_output_path=full_output_path or "",
            plots=[p["path"] for p in saved_plots],
        )
    _safe(_record, what="execution record")
    if state.execution_history:
        entry = state.execution_history[-1]
        _safe(log_execution, entry, what="wandb execution row")
        if getattr(state, "recorder", None) is not None:
            _safe(state.recorder.step, entry, what="steps.jsonl")

    # --- 6. Return function output + any generated images for vision ---
    outputs = [{"type": "function_call_output", "call_id": tc.call_id, "output": output}]
    if saved_plots:
        image_content = [{
            "type": "input_text",
            "text": f"[GENERATED PLOTS]: The tool generated {len(saved_plots)} plot(s). Please analyze them:",
        }]
        max_edge = getattr(state, "image_max_edge", 1568)
        for plot in saved_plots:
            encoded = _safe(prepare_image_for_model, plot["path"], max_edge, what="image encode")
            if encoded is None:
                continue
            b64, media = encoded
            image_content.append({
                "type": "input_image",
                "image_url": f"data:{media};base64,{b64}",
                "path": plot["path"],  # harness-only; stripped before sending
            })
        if len(image_content) > 1:
            outputs.append({"type": "message", "role": "user", "content": image_content})
            log(f"   👁️ {len(image_content) - 1} plot(s) sent to model for visual inspection")

    return outputs


def _log_retry_changes(new_args: dict, old_args: dict):
    """Log what changed between retry attempts."""
    changed = []
    for k, v in new_args.items():
        if k in old_args and str(old_args[k]) != str(v):
            changed.append(f"{k}: {old_args[k]} → {v}")
        elif k not in old_args:
            changed.append(f"{k}: (new) {v}")
    if changed:
        log(f"    📝 Changed params: {', '.join(changed)}")


def _extract_reasoning(reasoning: dict) -> tuple[dict, str]:
    """Extract reasoning data and intent from hook response."""
    reasoning_data = {}
    intent = None
    
    if reasoning:
        if reasoning.get("_is_retry"):
            for field in RETRY_REASONING_FIELDS:
                reasoning_data[field] = reasoning.get(field, 'N/A')
            intent = f"Retry: {reasoning.get('error_diagnosis', 'N/A')}"
        else:
            for field in REASONING_FIELDS:
                reasoning_data[field] = reasoning.get(field, 'N/A')
            intent = reasoning.get('intent', 'N/A')
    
    return reasoning_data, intent


def _prepare_args(args: dict, output_dir: Path, tool_name: str = None,
                   adata_rna=None, adata_atac=None, adata_atac_backed=None,
                   adata_combined=None, data_dir=None, literature_blacklist=None,
                   cnmf_usage_df=None, cnmf_spectra_scores_df=None, cnmf_top_genes_df=None) -> dict:
    """Prepare args for tool execution (inject adata, redirect paths).
    
    Data objects are injected unconditionally; execute_tool() filters
    kwargs by function signature so each tool only receives what it declares:
    - scRNA tools  (``adata``)             → adata_rna
    - scATAC tools (``adata_atac``)        → adata_atac
    - scATAC tools (``adata_atac_backed``) → adata_atac_backed (snapatac2 backed, read-only)
    - Multimodal   (``adata_rna`` + ``adata_atac``) → both
    - REPL         (``adata_rna``, ``adata_atac``, ``adata_atac_backed``, ``adata_combined``, ``data_dir``)
    - Literature   (``literature_blacklist``) → from config, to exclude source study
    Uses tool_metadata registry to determine:
    - Which tools need output_dir passed
    - Which tools are plotting tools (need show=False)
    - Which parameter names are output paths (need redirection)
    """
    exec_args = args.copy()
    exec_args.pop("dataset", None)  # Legacy arg from LLM; no longer used
    
    # Inject literature blacklist for search/fetch tools (so agent cannot shortcut via source paper)
    if tool_name in LITERATURE_TOOLS and literature_blacklist is not None:
        exec_args["literature_blacklist"] = literature_blacklist
    
    # Inject all available data — execute_tool filters kwargs by function signature,
    # so each tool only receives the parameters it actually declares.
    # scRNA tools accept "adata", scATAC tools accept "adata_atac",
    # multimodal tools accept "adata_rna" + "adata_atac".
    if adata_rna is not None:
        exec_args["adata"] = adata_rna       # scRNA tools use "adata"
        exec_args["adata_rna"] = adata_rna   # multimodal + REPL use "adata_rna"
    if adata_atac is not None:
        exec_args["adata_atac"] = adata_atac
    if adata_atac_backed is not None:
        exec_args["adata_atac_backed"] = adata_atac_backed
    if adata_combined is not None:
        exec_args["adata_combined"] = adata_combined
    if data_dir is not None:
        exec_args["data_dir"] = str(data_dir)
    if cnmf_usage_df is not None:
        exec_args["usage_df"] = cnmf_usage_df
    if cnmf_spectra_scores_df is not None:
        exec_args["spectra_scores_df"] = cnmf_spectra_scores_df
    if cnmf_top_genes_df is not None:
        exec_args["top_genes_df"] = cnmf_top_genes_df

    # Get output path parameters for this tool from metadata registry
    output_path_params = get_output_path_params(tool_name) if tool_name else set()
    
    # If tool has output_dir in its output path params, pass it
    if "output_dir" in output_path_params and "output_dir" not in args:
        exec_args["output_dir"] = str(output_dir)
    
    # If tool is registered as a plotting tool OR has 'show' parameter, set show=False
    # so figures stay in memory for auto-saving
    if "show" in args or (tool_name and is_plotting_tool(tool_name)):
        exec_args["show"] = False
    
    # Redirect output paths using metadata-defined parameter names
    for path_key in output_path_params:
        if path_key in exec_args and exec_args[path_key]:
            original = str(exec_args[path_key])
            if not original.startswith(str(output_dir)):
                basename = Path(original).name or original.replace("./", "").replace("/", "_")
                exec_args[path_key] = str(output_dir / basename)
    
    return exec_args


def parse_response(response, state) -> tuple[list, str]:
    """
    Parse API response into tool calls, final text, and reasoning.
    Handles both unified dict format and raw object format.
    
    Returns:
        tuple: (tool_calls list, final_text or None)
    """
    tool_calls = []
    final_text = None
    texts, thoughts = [], []
    
    for item in response.output:
        # Handle dict format (unified response — Anthropic and converted formats)
        if isinstance(item, dict):
            item_type = item.get("type")
            
            if item_type == "message":
                content = item.get("content", [])
                for c in content:
                    if isinstance(c, dict) and c.get("type") == "text":
                        final_text = c.get("text")
                        texts.append(final_text or "")
            
            elif item_type == "reasoning":
                summary = item.get("summary")
                if summary:
                    thoughts.append(str(summary))
                    state.add_reasoning(summary)
                    log(f"💭 {summary}\n")
            
            elif item_type == "function_call":
                tool_calls.append(ToolCallWrapper(item))
        
        # Handle object format (raw Azure OpenAI response)
        elif hasattr(item, "type"):
            if item.type == "message":
                for c in item.content:
                    if hasattr(c, "text"):
                        final_text = c.text
                        texts.append(final_text or "")
            
            elif item.type == "reasoning":
                summary = getattr(item, "summary", None)
                if summary:
                    thoughts.append(str(summary))
                    state.add_reasoning(summary)
                    log(f"💭 {summary}\n")
            
            elif item.type == "function_call":
                tool_calls.append(item)
    
    state.last_turn_text = "\n".join(t for t in texts if t)
    state.last_turn_thinking = "\n".join(thoughts)
    return tool_calls, final_text

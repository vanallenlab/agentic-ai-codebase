"""Tool execution — harness side.

Validates the model's tool call, captures post-hoc telemetry, sends the call
to the tool runtime (a separate worker process by default, see
worker_client.py), then shapes the output for the model, attaches plots and
records everything. The data and the tool code never run in this process
unless execution.mode is "inprocess".
"""

import difflib
import json
import time
from pathlib import Path

from src.hooks import pre_tool_call_hook, REASONING_FIELDS, RETRY_REASONING_FIELDS
from src.logging_utils import log
from src.wandb_utils import log_execution, log_plot
from src.harness_guard import TerminationRequested, prepare_image_for_model, shape_tool_output

# Arguments some models send that the harness drops on purpose (legacy).
IGNORED_MODEL_ARGS = frozenset({"dataset"})


class ToolCallWrapper:
    """Wrapper for tool call data from unified response format."""
    __slots__ = ('name', 'call_id', 'arguments', 'type')

    def __init__(self, data: dict):
        self.name = data.get("name")
        self.call_id = data.get("call_id")
        self.arguments = data.get("arguments", "{}")
        self.type = "function_call"


def _safe(fn, *a, what: str = "", **kw):
    """Run bookkeeping so that its failure can never fail the tool or the run."""
    try:
        return fn(*a, **kw)
    except (KeyboardInterrupt, TerminationRequested):
        raise
    except Exception as e:
        log(f"   ⚠️ {what or getattr(fn, '__name__', 'bookkeeping')} failed: {type(e).__name__}: {e}")
        return None


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


def execute_tool_call(tc, state, output_dir, token_info=None):
    """
    Execute a single tool call and return the output dict(s).

    Robustness contract:
    - Never raises for anything the agent did (bad JSON, unknown tool, invalid
      args, exceptions/SystemExit/timeouts in tool or agent code, a crashed or
      OOM-killed worker). Only KeyboardInterrupt, TerminationRequested and
      WorkerFatal (worker cannot be restarted) propagate.
    - Errors are returned to the model with status "error" and recorded as
      failures (success=False) everywhere.

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

    # --- 3. Execute in the tool runtime (worker) ---
    if pre_error is not None:
        log(f"   ❌ REJECTED {pre_error}")
        resp = {"success": False, "body": pre_error, "notes": [], "plots": [],
                "data_changes": "", "args_preview": args}
    else:
        resp = state.runtime.call(name, args, exec_id)

    success = bool(resp.get("success"))
    plot_paths = [p for p in (resp.get("plots") or []) if Path(p).is_file()]
    for p in plot_paths:
        _safe(log_plot, p, state.step_count, what="wandb plot")
    state.plot_count += len(plot_paths)

    # --- 4. Build the text the model sees ---
    body = resp.get("body") or ""
    if success and plot_paths:
        plot_info = ", ".join(f"📊 {p}" for p in plot_paths)
        body = f"{body}\n{plot_info}" if body else plot_info
    notes = resp.get("notes") or []
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
        if pre_error is None:
            log(f"   ❌ {body}")
        output = json.dumps({"status": "error", "error": shaped})

    duration = round(time.monotonic() - t0, 3)

    # --- 5. Record (isolated from the tool outcome) ---
    args_preview = resp.get("args_preview") or args

    def _record():
        state.add_execution(
            name, args, json.dumps(args_preview, default=str), intent, reasoning_data,
            success, body, token_info=token_info, exec_id=exec_id,
            error=None if success else body,
            model_text=getattr(state, "last_turn_text", ""),
            model_thinking=getattr(state, "last_turn_thinking", ""),
            data_changes=resp.get("data_changes", ""),
            duration_s=duration,
            full_output_path=full_output_path or "",
            plots=plot_paths,
            worker_restarted=bool(resp.get("worker_restarted")),
        )
    _safe(_record, what="execution record")
    if state.execution_history:
        entry = state.execution_history[-1]
        _safe(log_execution, entry, what="wandb execution row")
        if getattr(state, "recorder", None) is not None:
            _safe(state.recorder.step, entry, what="steps.jsonl")

    # --- 6. Return function output + any generated images for vision ---
    outputs = [{"type": "function_call_output", "call_id": tc.call_id, "output": output}]
    if plot_paths:
        image_content = [{
            "type": "input_text",
            "text": f"[GENERATED PLOTS]: The tool generated {len(plot_paths)} plot(s). Please analyze them:",
        }]
        max_edge = getattr(state, "image_max_edge", 1568)
        for path in plot_paths:
            encoded = _safe(prepare_image_for_model, path, max_edge, what="image encode")
            if encoded is None:
                continue
            b64, media = encoded
            image_content.append({
                "type": "input_image",
                "image_url": f"data:{media};base64,{b64}",
                "path": path,  # harness-only; stripped before sending
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

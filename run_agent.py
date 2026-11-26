#!/usr/bin/env python3
"""Scientific Data Analysis Agent."""

import argparse
import json
import os
import random
import sys
import uuid
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

import wandb
import weave

from src.tool_registry import get_tools, list_categories
from src.config import (load_config, MODEL_CONTEXT_LIMITS, check_leakage_guard,
                        resolve_literature_blacklist)
from src.llm_client import create_llm_client_from_config, RequestTooLargeError, LLMProviderError
from src.logging_utils import logger, log
from src.interactive import interactive_prompt, handle_user_question, prompt_for_continuation
from src.wandb_utils import log_tokens, finalize, print_summary, log_execution, save_live_files
from src.harness_guard import (
    RunRecorder, TerminationRequested, checkpoint_state, install_signal_handlers,
    model_max_image_edge, DEFAULT_TOOL_OUTPUT_MAX_CHARS,
)
from src.agent_state import AgentState
from src.tool_executor import execute_tool_call, parse_response
from src.agent_control import set_agent_state
from src.notebook_generator import generate_verification_notebook
from src.worker_client import make_runtime_client, WorkerFatal


def init_services(config: dict, task: str, user_name: str = None):
    """Initialize wandb and weave logging. Returns wandb_run or None.
    
    The full config dict is passed directly to wandb so every setting
    from the YAML is logged. A single `logging.enabled` flag controls
    both wandb and weave together.
    """
    logging_cfg = config.get("logging", {})
    enable_logging = logging_cfg.get("enabled", True)
    
    wandb_run = None
    if enable_logging:
        log_project = logging_cfg.get("project", "agentic-ai-pilot-1")
        log_entity = logging_cfg.get("entity")
        run_name = logging_cfg.get("run_name") or f"agent_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Log the full config + task to wandb (everything from the YAML)
        config_dict = dict(config)  # shallow copy so we don't mutate the original
        config_dict["task"] = task
        if user_name:
            config_dict["user"] = user_name
        
        try:
            wandb_run = wandb.init(
                project=log_project, 
                entity=log_entity, 
                name=run_name,
                group=user_name,  # Groups runs by user in wandb dashboard (collapsible)
                config=config_dict,
            )
            logger.set_wandb_run(wandb_run)
            print(f"📊 Wandb run initialized: {wandb_run.url}" + (f" (group: {user_name})" if user_name else ""))
        except Exception as e:
            if logging_cfg.get("required", False):
                raise
            wandb_run = None
            print(f"⚠️ wandb init failed ({type(e).__name__}: {e}); continuing with local logs only "
                  f"(conversation.jsonl / steps.jsonl / agent.log). Set logging.required: true to make this fatal.")
        
        # Initialize weave (uses same project/entity as wandb)
        weave_full_project = f"{log_entity}/{log_project}" if log_entity else log_project
        try:
            weave.init(weave_full_project)
            print(f"🔍 Weave tracing initialized for project: {weave_full_project}")
        except Exception as e:
            print(f"⚠️ weave init failed ({type(e).__name__}: {e}); continuing without weave tracing.")
    
    return wandb_run


def load_tools(tool_category: str, provider: str = "openai", exclude_tools=None) -> list:
    """Load and format tools for a specific LLM provider.

    Tools are loaded from three sources:
    1. User-specified category/categories (REQUIRED via config, can be comma-separated)
    2. Literature search tools (always loaded for research capabilities)
    3. Support tools (run_python_repl)

    Args:
        tool_category: Exact category name(s) to load, comma-separated
                       (e.g., 'scselected_scrna' or 'scselected_scrna, scselected_scatac')
        provider: LLM provider for tool formatting ("openai" / "azure_openai" or "anthropic")

    Returns:
        list: Tool definitions formatted for the specified LLM provider
    """
    tools = []
    tool_names_seen = set()

    def _add_tool(tool: dict):
        name = tool["name"]
        if name not in tool_names_seen:
            tools.append(tool)
            tool_names_seen.add(name)

    # 1. User-specified tool categories
    if tool_category:
        categories = [cat.strip() for cat in tool_category.split(',')]
        for category in categories:
            for t in get_tools(category, provider=provider):
                _add_tool(t)

    # 2. Literature search tools (always loaded)
    for t in get_tools("external_literature", provider=provider):
        _add_tool(t)

    # 3. Support tools (run_python_repl, run_terminal for pip install)
    for t in get_tools("external_support_tools", provider=provider):
        if t["name"] in ("run_python_repl", "run_terminal"):
            _add_tool(t)

    # 4. Config `exclude_tools`: never offered to the model (unknown-tool calls are rejected)
    if exclude_tools:
        excluded = set(exclude_tools)
        tools = [t for t in tools if t["name"] not in excluded]

    return tools


def handle_interactive(state, outputs) -> tuple[str, any]:
    """Handle interactive mode prompts after each tool execution.
    
    Returns:
        tuple: (action, data) where:
            - action="continue": data=outputs (proceed with agent loop)
            - action="abort": data=None (stop execution)
            - action="feedback": data=outputs with feedback appended
    """
    while True:
        action, data = interactive_prompt()
        
        if action == "abort":
            log("\n⛔ User aborted execution")
            return action, None
        elif action == "continue":
            return action, outputs
        elif action == "question" and data:
            last_tool = state.execution_history[-1]["name"] if state.execution_history else None
            answer = handle_user_question(state.client, data, state.task,
                                        state.execution_history, last_tool)
            state.add_question(data, answer)
            log_execution(state.execution_history[-1])
            if state.recorder is not None:
                state.recorder.step(state.execution_history[-1])
            continue  # Prompt again after answering
        elif action == "feedback" and data:
            log(f"💬 User feedback: {data}")
            state.add_feedback(data)
            log_execution(state.execution_history[-1])
            if state.recorder is not None:
                state.recorder.step(state.execution_history[-1])
            outputs.append({"type": "message", "role": "user", 
                        "content": f"[USER FEEDBACK]: {data}"})
            return action, outputs
        else:
            return "continue", outputs


def _set_seeds(seed):
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    log(f"🎲 Random seed: {seed} (python, numpy; also applied in the tool worker)")


def run_agent(config_path: str, user_name: str = None):
    """Run the agent loop.

    Loads config from the YAML file, reads the task, and runs the agent.
    Everything is driven by the config file — the only external input is
    user_name (CLI-only, for wandb grouping). Used by both run_agent.py (HAI)
    and run_agent_htan.py (HTAN; cNMF inputs are loaded when configured).

    Robustness settings (all optional, in the config YAML):
        tool_timeout_s: 7200          # per tool call wall-clock limit; 0/null = none
        tool_output_max_chars: 100000 # larger outputs: head+tail to model, full text to disk
        track_data_changes: true      # detect/report changes to shared data per call
        checkpoint_data: true         # write modified data to output/checkpoints after each step
        rollback_on_failure: true     # a FAILED call's changes to shared data are undone (reloaded from the last checkpoint)
        send_images: true             # false = text-only: plots are saved to disk but never shown to the model
        seed: 0                       # python/numpy seed; null = don't seed
        llm.max_tokens / llm.thinking / llm.effort / llm.max_retries /
        llm.retry_total_seconds / llm.max_images_per_request

    Returns:
        dict: {"end_reason": str, "output_dir": str}. The data lives in the tool
        worker; final objects are written to output_dir/result_*.h5ad.
    """
    install_signal_handlers()  # SIGTERM/SIGHUP -> clean shutdown with saves

    # Load config
    config = load_config(config_path)

    # Validate and load task
    task_file = config.get("task_file")
    if not task_file:
        raise ValueError("'task_file' is required in config file.")
    task_path = Path(task_file)
    if not task_path.exists():
        raise FileNotFoundError(f"Task file not found: {task_path}")
    task = task_path.read_text().strip()
    if not task:
        raise ValueError(f"Task file is empty: {task_path}")

    # Optional dataset description (included in system prompt with the task)
    dataset_path = config.get("dataset_path")
    dataset_text = ""
    if dataset_path:
        dataset_file = Path(dataset_path)
        if dataset_file.exists():
            dataset_text = dataset_file.read_text().strip()
            log(f"✓ Loaded dataset description from {dataset_path}")
        else:
            log(f"⚠️ dataset_path not found: {dataset_path}")

    if not config.get("tool_category"):
        raise ValueError(
            "'tool_category' is required in config file. "
            "Run with --list-tools to see available categories."
        )

    # Unpack config
    scrna_data_path = config.get("scrna_data_path")
    scatac_data_path = config.get("scatac_data_path")
    scatac_snap_path = config.get("scatac_snap_path")
    data_dir = config.get("data_dir")
    cnmf_usage_path = config.get("cnmf_usage_path")
    cnmf_spectra_scores_path = config.get("cnmf_spectra_scores_path")
    cnmf_top_genes_path = config.get("cnmf_top_genes_path")
    tool_category = config.get("tool_category")
    output_dir = config.get("output_dir")
    output_base_dir = config.get("output_base_dir", "output")
    reasoning_effort = config.get("reasoning_effort", "medium")
    interactive = config.get("interactive", False)
    capture_telemetry = config.get("capture_telemetry", True)
    checkpoint_data = config.get("checkpoint_data", True)
    literature_blacklist = resolve_literature_blacklist(config)
    check_leakage_guard(config, literature_blacklist)
    # Setup output directory (absolute, so agent code changing cwd can't redirect writes)
    if output_dir is None and config.get("anonymous_run_dir"):
        # The worker's cwd is the run directory, so its path must not name the cohort or task:
        # runs live in <anonymous_run_root>/<random id>; output_base_dir gets a pointer file.
        run_id = uuid.uuid4().hex[:12]
        output_dir = Path(config.get("anonymous_run_root", "output/runs")) / run_id
        pointer_dir = Path(output_base_dir) / (user_name or "")
        pointer_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        (pointer_dir / f"run_{stamp}_{run_id}.json").write_text(json.dumps({
            "run_dir": str(output_dir.resolve()), "run_id": run_id, "config": str(config_path),
            "task_file": str(task_file), "started": datetime.now().isoformat(timespec="seconds"),
        }, indent=1) + "\n")
    elif output_dir is None:
        base_dir = Path(output_base_dir)
        run_folder = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        if user_name:
            output_dir = base_dir / user_name / run_folder
        else:
            output_dir = base_dir / run_folder
    else:
        output_dir = Path(output_dir)
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize wandb/weave logging
    init_services(config, task, user_name=user_name)

    # Start file logging
    logger.set_log_file(open(output_dir / "agent.log", "w"))
    log(f"# Agent Log - {datetime.now()}\n")
    _set_seeds(config.get("seed", 0))

    # Initialize LLM client
    client, model = create_llm_client_from_config(config)
    log(f"🤖 LLM Provider: {client.provider} | Model: {model}")

    # Initialize state (client lives on state — single place everything flows from)
    # Known limit: model table first, then what the client learned (OpenRouter catalogue)
    context_limit = MODEL_CONTEXT_LIMITS.get(model) or getattr(client, "context_window", None)
    if client.provider == "openrouter":
        log(f"🔀 OpenRouter routing: {client.openrouter_routing}")
    if context_limit is None:
        if client.provider == "anthropic":
            context_limit = 200_000
        else:
            context_limit = 128_000
        log(f"⚠️ Model '{model}' not in MODEL_CONTEXT_LIMITS, using default: {context_limit:,}")

    if literature_blacklist:
        log("📚 Literature blacklist: " + ", ".join(f"{len(v)} {k}" for k, v in literature_blacklist.items()))
    state = AgentState(
        output_dir=output_dir,
        task=task,
        model=model,
        context_limit=context_limit,
        client=client,
        interactive=interactive,
        literature_blacklist=literature_blacklist,
        capture_telemetry=capture_telemetry,
    )
    state.tool_timeout_s = config.get("tool_timeout_s", 7200) or None
    state.tool_output_max_chars = config.get("tool_output_max_chars", DEFAULT_TOOL_OUTPUT_MAX_CHARS) or None
    state.track_data_changes = config.get("track_data_changes", True)
    state.image_max_edge = int(config.get("image_max_edge") or model_max_image_edge(model))
    if config.get("send_images", True) and getattr(client, "accepts_images", None) is False:
        log(f"🖼️ {model} does not accept images: plots are saved but not sent (send_images -> false)")
        config["send_images"] = False
    state.send_images = bool(config.get("send_images", True))
    if not state.send_images:
        log("🖼️ send_images: false - plots are saved but never shown to the model")
    state.recorder = RunRecorder(output_dir)
    set_agent_state(state)
    save_live_files(output_dir)

    # Start the tool worker; it loads the data and owns it for the whole run.
    # The harness process never holds the datasets (unless execution.mode=inprocess).
    def _abs(p):
        return str(Path(p).resolve()) if p else None
    init_payload = {
        "output_dir": str(output_dir),
        "data": {k: v for k, v in {
            "scrna_data_path": _abs(scrna_data_path),
            "scatac_data_path": _abs(scatac_data_path),
            "scatac_snap_path": _abs(scatac_snap_path),
            "data_dir": _abs(data_dir),
            "cnmf_usage_path": _abs(cnmf_usage_path),
            "cnmf_spectra_scores_path": _abs(cnmf_spectra_scores_path),
            "cnmf_top_genes_path": _abs(cnmf_top_genes_path),
        }.items() if v},
        "literature_blacklist": literature_blacklist,
        "tool_timeout_s": state.tool_timeout_s,
        "track_data_changes": state.track_data_changes,
        "rollback_on_failure": bool(config.get("rollback_on_failure", True)),
        "send_images": bool(config.get("send_images", True)),
        "seed": config.get("seed", 0),
    }
    state.runtime = make_runtime_client(config, init_payload, output_dir,
                                        repo_dir=Path(__file__).resolve().parent,
                                        tool_timeout_s=state.tool_timeout_s)
    data_info = state.runtime.start()
    log(f"✓ Tool worker ready ({(config.get('execution') or {}).get('mode', 'subprocess')}): {data_info}")
    if logger.wandb_run:
        try:
            flat = {f"n_obs_{k}": v["n_obs"] for k, v in data_info.items()}
            flat.update({f"n_vars_{k}": v["n_vars"] for k, v in data_info.items()})
            wandb.config.update(flat)
        except Exception as e:
            log(f"⚠️ wandb config update failed: {e}")

    # Load tools
    tools = load_tools(tool_category, provider=state.client.provider,
                       exclude_tools=config.get("exclude_tools"))
    if config.get("exclude_tools"):
        log(f"🚫 Tools excluded by config: {', '.join(config['exclude_tools'])}")
    state.tool_schemas = {t["name"]: (t.get("input_schema") or t.get("parameters") or {}) for t in tools}
    log(f"🧬 {len(tools)} tools loaded | Reasoning: {reasoning_effort}" +
        (" | 🎯 Interactive" if interactive else "") + "\n")
    if interactive:
        log(f"📢 Interactive mode: You'll be prompted after each tool execution\n")
    # Build system prompt: instructions first, then optional dataset context
    system_content = task
    if dataset_text:
        system_content = f"{task}\n\n---\n\n{dataset_text}"
    log(f"## INSTRUCTIONS\n{task}\n")

    if logger.wandb_run:
        try:
            wandb.config.update({"n_tools_loaded": len(tools), "context_limit": state.context_limit, "model": model})
        except Exception as e:
            log(f"⚠️ wandb config update failed: {e}")

    log("## Execution\n")

    # Track whether we entered the main loop (for try/finally)
    entered_main_loop = False
    stop_reason_for_run = "completed"

    messages = [
        {"type": "message", "role": "system", "content": system_content},
        {"type": "message", "role": "user", "content": "Please begin the analysis."}
    ]

    # Helper to count images in messages (for token tracking)
    def count_images_in_messages(msgs: list) -> int:
        count = 0
        for msg in msgs:
            if isinstance(msg, dict) and msg.get("type") == "message":
                content = msg.get("content", [])
                if isinstance(content, list):
                    count += sum(1 for c in content if isinstance(c, dict) and c.get("type") == "input_image")
        return count

    def _api_error_types():
        errs = []
        try:
            import anthropic
            errs.append(anthropic.APIError)
        except ImportError:
            pass
        try:
            import openai
            errs.append(openai.APIError)
        except ImportError:
            pass
        errs.append(LLMProviderError)
        return tuple(errs)

    api_errors = _api_error_types()
    turn = 0
    consecutive_truncations = 0

    # =========================================================================
    # Main agent loop (wrapped in try/finally so finalize always runs)
    # =========================================================================
    try:
        entered_main_loop = True
        while True:
            images_in_request = count_images_in_messages(messages)
            turn += 1
            state.recorder.request(turn, messages)

            log(f"🔗 Calling {state.client.provider} API" + (
                f" (previous_response_id: {state.previous_response_id})" if state.client.keeps_server_state else ""))
            try:
                response = state.client.create(
                    messages=messages,
                    tools=tools,
                    reasoning_effort=reasoning_effort,
                    store=True,
                    parallel_tool_calls=False,
                    previous_response_id=state.previous_response_id if state.client.keeps_server_state else None,
                )
            except RequestTooLargeError as e:
                log(f"\n🛑 {e}")
                state.recorder.event("run_stopped", reason="request_too_large", error=str(e))
                stop_reason_for_run = "request_too_large"
                break
            except api_errors as e:
                # Retryable errors were already retried (SDK + client backoff). What is
                # left is not recoverable without changing history, so stop cleanly:
                # the finally block saves data, notebook and logs.
                status = getattr(e, "status_code", None)
                log(f"\n🛑 LLM API error{f' (HTTP {status})' if status else ''}: {type(e).__name__}: {e}")
                state.recorder.event("run_stopped", reason="api_error", status=status,
                                     error=f"{type(e).__name__}: {e}")
                stop_reason_for_run = f"api_error:{status or type(e).__name__}"
                break

            state.previous_response_id = response.id
            state.recorder.response(turn, response)
            log(f"🔗 Response {response.id} (stop_reason: {response.stop_reason})")

            # Token tracking
            token_info = state.tokens.process_usage(response.usage, images_sent=images_in_request)
            log(state.tokens.format_log_line(token_info))
            log_tokens(token_info, state.tokens)

            # Parse response into tool calls and/or final text
            tool_calls, final_text = parse_response(response, state)

            # --- Refusal: record as a failure, do not treat as a finished analysis ---
            if response.stop_reason == "refusal":
                log(f"\n🛑 Model refused (stop_reason=refusal). Text: {final_text}")
                state.add_model_response(final_text, token_info, stop_reason="refusal",
                                         name="MODEL_REFUSAL", success=False)
                log_execution(state.execution_history[-1])
                state.recorder.step(state.execution_history[-1])
                if interactive:
                    action, new_instructions = prompt_for_continuation()
                    if action == "exit" or not new_instructions:
                        break
                    messages = [{"type": "message", "role": "user", "content": new_instructions}]
                    continue
                stop_reason_for_run = "refusal"
                break

            # --- Output limit hit ---
            if response.stop_reason == "max_tokens":
                consecutive_truncations += 1
                if tool_calls:
                    # The (last) tool call may be cut off mid-arguments: do not run it.
                    log("⚠️ Response hit the output token limit while writing a tool call; not executing it.")
                    messages = [{
                        "type": "function_call_output", "call_id": tc.call_id,
                        "output": json.dumps({"status": "error", "error": (
                            "Your response hit the maximum output length while writing this tool call, "
                            "so it may be incomplete and was NOT run. Re-issue it; if it contains a lot "
                            "of code, split it into smaller calls.")}),
                    } for tc in tool_calls]
                    state.add_model_response(final_text, token_info, stop_reason="max_tokens",
                                             name="MODEL_TRUNCATED", success=False)
                    log_execution(state.execution_history[-1])
                    state.recorder.step(state.execution_history[-1])
                    continue
                if consecutive_truncations <= 5:
                    log("⚠️ Response hit the output token limit; asking the model to continue.")
                    state.add_model_response(final_text, token_info, stop_reason="max_tokens",
                                             name="MODEL_TRUNCATED", success=False)
                    log_execution(state.execution_history[-1])
                    state.recorder.step(state.execution_history[-1])
                    messages = [{"type": "message", "role": "user", "content": (
                        "[harness] Your previous response was cut off because it reached the maximum "
                        "output length. Continue exactly where you left off.")}]
                    continue
            else:
                consecutive_truncations = 0

            # --- Server-side pause: just continue the turn ---
            if response.stop_reason == "pause_turn" and not tool_calls:
                messages = []
                continue

            # --- No tool calls: model is done (or waiting for user) ---
            if not tool_calls:
                log(f"## Result\n{final_text}")

                # Log the final model response to wandb
                state.add_model_response(final_text, token_info, stop_reason=response.stop_reason)
                log_execution(state.execution_history[-1])
                state.recorder.step(state.execution_history[-1])

                if interactive:
                    action, new_instructions = prompt_for_continuation()
                    if action == "exit" or not new_instructions:
                        log("\n👋 User ended the session")
                        break
                    log(f"\n## New Instructions\n{new_instructions}\n")
                    log("## Execution\n")
                    messages = [{"type": "message", "role": "user", "content": new_instructions}]
                    continue
                else:
                    break

            # --- Tool execution ---
            state.step_count += 1
            log(f"### Step {state.step_count}\n")

            # Execute all tool calls (one per turn: parallel tool use is disabled for
            # every provider, but every returned call is still answered)
            outputs = []
            for tc in tool_calls:
                outputs.extend(execute_tool_call(tc, state, output_dir, token_info))

            # Checkpoint any data this step modified (atomic; failures only logged)
            if checkpoint_data:
                results = state.runtime.checkpoint()
                for attr, res in results.items():
                    if str(res).startswith("ERROR") or attr == "_error":
                        log(f"   ⚠️ Checkpoint of {attr} failed: {res}")
                    else:
                        log(f"   💾 Checkpointed {attr} -> {res}")
                if results:
                    state.recorder.event("checkpoint", step=state.step_count, results=results)
            
            # Interactive mode: prompt user after execution
            if interactive:
                action, data = handle_interactive(state, outputs)
                if action == "abort":
                    stop_reason_for_run = "user_abort"
                    break
                elif action in ("continue", "feedback"):
                    outputs = data if isinstance(data, list) else outputs

            messages = outputs
            log("")

    except WorkerFatal as e:
        stop_reason_for_run = "worker_fatal"
        log(f"\n🛑 Tool worker could not be restarted: {e}")
        _safe_event(state, "run_stopped", reason=stop_reason_for_run, error=str(e))
    except (KeyboardInterrupt, TerminationRequested) as e:
        stop_reason_for_run = f"interrupted:{type(e).__name__}"
        log(f"\n🛑 Run interrupted ({type(e).__name__}: {e}); saving outputs...")
        _safe_event(state, "run_stopped", reason=stop_reason_for_run, error=str(e))
        raise
    except Exception as e:
        stop_reason_for_run = f"harness_error:{type(e).__name__}"
        log(f"\n❌ Unhandled exception in main loop: {type(e).__name__}: {e}")
        import traceback as _tb
        log(_tb.format_exc())
        _safe_event(state, "run_stopped", reason=stop_reason_for_run, error=str(e))
        raise  # Re-raise after finally block runs

    finally:
        if not entered_main_loop:
            return {"end_reason": "not_started", "output_dir": str(output_dir)}

        # ==================================================================
        # Finalize (always runs, even on crash / SIGTERM). Every stage is
        # guarded so one failure cannot prevent the others.
        # ==================================================================
        log(f"\n🏁 Run ended: {stop_reason_for_run}")
        _safe_event(state, "run_end", reason=stop_reason_for_run, steps=state.step_count)
        try:
            print_summary(state.step_count, state.execution_history, state.tokens)
        except Exception as e:
            log(f"⚠️ Could not print summary: {e}")
        if logger.wandb_run:
            try:
                wandb.summary["run_end_reason"] = stop_reason_for_run
            except Exception:
                pass

        adata_info = {}
        if state.runtime is not None:
            try:
                fin = state.runtime.finalize()
                for attr, path in (fin.get("saved") or {}).items():
                    log(f"💾 Saved {attr}: {path}")
                for attr, err in (fin.get("errors") or {}).items():
                    log(f"⚠️ Could not save {attr}: {err}")
                adata_info = fin.get("info") or {}
            except Exception as e:
                log(f"⚠️ Final save failed: {type(e).__name__}: {e}")
            try:
                state.runtime.shutdown()
            except Exception as e:
                log(f"⚠️ Worker shutdown failed: {e}")
        
        # Generate verification notebook
        try:
            notebook_path = generate_verification_notebook(
                output_dir=output_dir,
                task=task,
                execution_history=state.execution_history,
                interaction_history=state.interaction_history,
                adata_info=adata_info if adata_info else None,
            )
            log(f"📓 Verification notebook: {notebook_path}")
        except Exception as e:
            log(f"⚠️ Could not generate notebook: {e}")

        log(f"📁 Output: {output_dir}")
        if state.recorder is not None:
            state.recorder.close()
        logger.close()

        try:
            finalize(output_dir, state.step_count, state.plot_count,
                     state.execution_history, state.interaction_history, state.tokens,
                     config_path=config_path, task_path=str(task_path))
        except Exception as e:
            print(f"⚠️ wandb finalize failed: {e}")

    return {"end_reason": stop_reason_for_run, "output_dir": str(output_dir)}


def _safe_event(state, kind, **fields):
    try:
        if state is not None and state.recorder is not None:
            state.recorder.event(kind, **fields)
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(
        description="Run the analysis agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n  python run_agent.py --config config/scatac_pilot_v2.yaml\n"
            "  python run_agent.py --config config/scatac_pilot_v2.yaml --name sjohri"
    )
    parser.add_argument("--config", "-c", required=True, help="Path to YAML config file")
    parser.add_argument("--name", "-n", type=str, default=None,
                       help="Your name (used to group runs in wandb dashboard)")
    parser.add_argument("--list-tools", action="store_true", help="List tool categories")
    
    args = parser.parse_args()
    
    if args.list_tools:
        for cat in sorted(list_categories()):
            print(f"  {cat}")
        return
    
    result = run_agent(args.config, user_name=args.name)
    end_reason = (result or {}).get("end_reason", "completed")
    if end_reason not in ("completed", "user_abort"):
        # Non-zero exit so batch schedulers can see the run did not finish normally
        sys.exit(f"Run ended early: {end_reason}")


if __name__ == "__main__":
    main()

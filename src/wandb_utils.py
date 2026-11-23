"""Wandb logging utilities for the agent."""

import wandb
from src.logging_utils import logger, log


def log_tokens(token_info: dict, tracker: "TokenTracker"):
    """Log token usage metrics to wandb.
    
    Args:
        token_info: Dict from TokenTracker.process_usage() with per-step metrics
        tracker: TokenTracker instance with cumulative state
    """
    if not logger.wandb_run:
        return
    
    metrics = {
        # Per-step token metrics
        "tokens/input": token_info["input_tokens"],
        "tokens/output": token_info["output_tokens"],
        "tokens/reasoning": token_info["reasoning_tokens"],
        "tokens/completion": token_info["completion_tokens"],
        # Cumulative totals
        "tokens/cumulative_input": tracker.cumulative_input,
        "tokens/cumulative_output": tracker.cumulative_output,
        "tokens/cumulative_total": tracker.cumulative_input + tracker.cumulative_output,
        # Context saturation
        "context/saturation_pct": token_info["saturation_pct"],
        "context/peak_saturation_pct": tracker.peak_saturation_pct,
        "context/limit": tracker.context_limit,
    }
    
    # Add image metrics if images were sent
    if token_info.get("images_sent", 0) > 0 or tracker.total_images_sent > 0:
        metrics.update({
            "tokens/image_estimate": token_info.get("image_tokens_estimate", 0),
            "tokens/images_sent": token_info.get("images_sent", 0),
            "tokens/cached": token_info.get("cached_tokens", 0),
            "tokens/cumulative_image_estimate": tracker.cumulative_image_tokens,
            "tokens/cumulative_images_sent": tracker.total_images_sent,
        })
    
    try:
        wandb.log(metrics)
    except Exception as e:
        log(f"⚠️ wandb log_tokens failed: {e}")


_execution_rows = []  # Simple list of rows

# All columns - including flattened reasoning fields.
# NOTE: the first 18 columns keep their original names/order (downstream analysis
# reads tables/execution.csv by column name); new columns are appended at the end.
# Values are logged in FULL (no truncation).
_COLUMNS = [
    "step", "exec_id", "tool", "args", "success", "error",
    "input_tokens", "output_tokens", "reasoning_tokens", "completion_tokens", "saturation_pct",
    # Flattened reasoning fields (post-hoc, produced by the observability hook)
    "intent", "alternatives_considered", "reason_for_choice", "assumptions", "expected_outcome",
    # Flattened retry reasoning fields  
    "error_diagnosis", "parameters_changed",
    # --- added by harness hardening ---
    "result",            # full tool result / model response text (as the model saw it)
    "model_text",        # the agent's own text in the turn that issued this call
    "model_thinking",    # the agent's own thinking in that turn (if returned)
    "stop_reason",
    "data_changes",      # changes detected in shared data objects by this call
    "duration_s",
    "full_output_path",  # set when the tool output was too large and saved to disk
    "reasoning_source",  # "posthoc_hook" for the intent/... fields above
]


def _s(v) -> str:
    return "" if v is None else str(v)


def log_execution(entry: dict):
    """Log execution to wandb table at each step (full values, no truncation)."""
    if not logger.wandb_run:
        return
    try:
        reasoning = entry.get("reasoning", {}) or {}
        _execution_rows.append([
            entry.get("step", 0),
            entry.get("exec_id", 0),
            entry.get("name", "unknown"),
            _s(entry.get("args_preview", "")),
            entry.get("success", True),
            _s(entry.get("error", "")),
            entry.get("input_tokens", 0),
            entry.get("output_tokens", 0),
            entry.get("reasoning_tokens", 0),
            entry.get("completion_tokens", 0),
            entry.get("saturation_pct", 0),
            _s(reasoning.get("intent", "")),
            _s(reasoning.get("alternatives_considered", "")),
            _s(reasoning.get("reason_for_choice", "")),
            _s(reasoning.get("assumptions", "")),
            _s(reasoning.get("expected_outcome", "")),
            _s(reasoning.get("error_diagnosis", "")),
            _s(reasoning.get("parameters_changed", "")),
            _s(entry.get("result", "")),
            _s(entry.get("model_text", "")),
            _s(entry.get("model_thinking", "")),
            _s(entry.get("stop_reason", "")),
            _s(entry.get("data_changes", "")),
            entry.get("duration_s", None),
            _s(entry.get("full_output_path", "")),
            "posthoc_hook" if reasoning else "",
        ])
        # Re-log the full table each step (survives crashes; downstream reads the
        # latest table version as the complete record).
        table = wandb.Table(columns=_COLUMNS, data=_execution_rows, allow_mixed_types=True)
        wandb.log({"execution": table})
    except Exception as e:  # telemetry must never kill the run
        log(f"⚠️ wandb log_execution failed: {e}")


def save_live_files(output_dir, names=("conversation.jsonl", "steps.jsonl", "agent.log")):
    """Stream append-only run logs to wandb as they grow (policy='live')."""
    if not logger.wandb_run:
        return
    from pathlib import Path
    for name in names:
        try:
            path = Path(output_dir) / name
            path.touch(exist_ok=True)
            wandb.save(str(path), base_path=str(output_dir), policy="live")
        except Exception as e:
            log(f"⚠️ wandb.save({name}) failed: {e}")


def log_plot(plot_path: str, step: int):
    """Log a plot image to wandb."""
    if not logger.wandb_run:
        return
    from pathlib import Path
    plot_file = Path(plot_path)
    try:
        if plot_file.exists():
            wandb.log({f"plots/{plot_file.stem}": wandb.Image(str(plot_file))})
    except Exception as e:  # telemetry must never kill the run
        log(f"⚠️ wandb log_plot failed: {e}")


def finalize(output_dir, step_count: int, plot_count: int,
             execution_history: list, interaction_history: list, tracker: "TokenTracker",
             config_path: str = None, task_path: str = None):
    """Finalize wandb run - log summary metrics, tables, and artifacts.
    
    Args:
        output_dir: Output directory containing logs, plots, notebooks
        config_path: Path to config YAML file (for reproducibility)
        task_path: Path to task.txt file (for reproducibility)
        
    Note: .h5ad files are NOT uploaded (too large) - saved locally in output_dir.
    """
    if not logger.wandb_run:
        return
    
    # Calculate stats from execution history
    total_calls = len(execution_history)
    failures = sum(1 for h in execution_history if not h.get("success", True))
    
    # Summary metrics
    wandb.summary.update({
        "total_steps": step_count,
        "total_tool_calls": total_calls,
        "total_failures": failures,
        "success_rate": (total_calls - failures) / total_calls if total_calls > 0 else 1.0,
        "total_plots": plot_count,
        # Token totals
        "tokens/total_input": tracker.cumulative_input,
        "tokens/total_output": tracker.cumulative_output,
        "tokens/total_reasoning": tracker.cumulative_reasoning,
        "tokens/total_consumed": tracker.cumulative_input + tracker.cumulative_output,
        # Context
        "context/peak_saturation_pct": tracker.peak_saturation_pct,
        "context/limit": tracker.context_limit,
        "context/high_saturation_warning": tracker.peak_saturation_pct > 80,
    })
    
    # Image token summary (if any images were sent)
    if tracker.total_images_sent > 0:
        wandb.summary.update({
            "tokens/total_image_estimate": tracker.cumulative_image_tokens,
            "tokens/total_images_sent": tracker.total_images_sent,
        })
    
    # Note: execution table is already maintained incrementally by log_execution().
    # No need to rebuild it here — avoids storage duplication.
    
    # Log interaction history table (only produced at end-of-run)
    _log_interaction_table(interaction_history)
    
    # Log artifacts (including config for reproducibility, excluding .h5ad)
    _log_artifacts(output_dir, config_path, task_path)
    
    # Finish wandb run
    wandb.finish()
    logger.set_wandb_run(None)
    print(f"📊 Wandb run finished")


def _log_interaction_table(interaction_history: list):
    """Log user interactions as a wandb Table."""
    if not interaction_history:
        return
    
    columns = ["step", "type", "content", "response", "affects_context"]
    table = wandb.Table(columns=columns)
    for h in interaction_history:
        table.add_data(
            h.get("step", 0),
            h.get("type", "unknown"),
            h.get("content", "") or "",
            h.get("response", h.get("reason", "")) or "",
            h.get("affects_context", False),
        )
    wandb.log({"user_interactions": table})
    
    # Interaction summary
    wandb.summary["interactions/total"] = len(interaction_history)
    wandb.summary["interactions/questions"] = sum(1 for h in interaction_history if h.get("type") == "question")
    wandb.summary["interactions/feedbacks"] = sum(1 for h in interaction_history if h.get("type") == "feedback")


def _log_artifacts(output_dir, config_path: str = None, task_path: str = None):
    """Log output files as wandb Artifacts.
    
    Includes config, task, logs, plots, and notebooks.
    Note: .h5ad files are excluded (too large) - saved locally in output_dir.
    """
    from pathlib import Path
    
    artifact = wandb.Artifact(f"agent_output_{logger.wandb_run.id}", type="output")
    
    # Config files (for reproducibility)
    if config_path and Path(config_path).exists():
        artifact.add_file(str(config_path), name="config.yaml")
    if task_path and Path(task_path).exists():
        artifact.add_file(str(task_path), name="task.txt")
    
    # Agent log
    log_file = output_dir / "agent.log"
    if log_file.exists():
        artifact.add_file(str(log_file))
    
    # Plots (all images the run produced, not only plot_*.png)
    for pattern in ("*.png", "*.jpg", "*.jpeg", "*.svg", "*.pdf", "*.html"):
        for plot_file in output_dir.glob(pattern):
            artifact.add_file(str(plot_file))

    # Full append-only transcripts and oversized tool outputs
    for name in ("conversation.jsonl", "steps.jsonl"):
        f = output_dir / name
        if f.exists():
            artifact.add_file(str(f))
    tool_outputs = output_dir / "tool_outputs"
    if tool_outputs.is_dir():
        artifact.add_dir(str(tool_outputs), name="tool_outputs")
    
    # Notebooks
    for nb_file in output_dir.glob("*.ipynb"):
        artifact.add_file(str(nb_file))
    
    wandb.log_artifact(artifact)


def print_summary(step_count: int, execution_history: list, tracker: "TokenTracker"):
    """Print execution summary to console/log."""
    total_calls = len(execution_history)
    failures = [h for h in execution_history if not h.get("success", True)]
    
    log(f"\n## Summary")
    log(f"📊 Steps: {step_count} | Tool calls: {total_calls} | Failures: {len(failures)}")
    
    # Context usage
    log(f"\n### Context Usage")
    total_consumed = tracker.cumulative_input + tracker.cumulative_output
    log(f"   📏 Total tokens consumed: {total_consumed:,}")
    log(f"      Input:  {tracker.cumulative_input:,}")
    log(f"      Output: {tracker.cumulative_output:,}")
    
    if tracker.cumulative_reasoning > 0:
        reasoning_pct = (tracker.cumulative_reasoning / tracker.cumulative_output * 100) if tracker.cumulative_output > 0 else 0
        log(f"        🧠 Reasoning:  {tracker.cumulative_reasoning:,} ({reasoning_pct:.1f}% of output)")
        log(f"        💬 Completion: {tracker.cumulative_completion:,}")
    
    if tracker.total_images_sent > 0:
        image_pct = (tracker.cumulative_image_tokens / tracker.cumulative_input * 100) if tracker.cumulative_input > 0 else 0
        log(f"   🖼️ Images sent to model: {tracker.total_images_sent}")
        log(f"      Estimated image tokens: ~{tracker.cumulative_image_tokens:,} ({image_pct:.1f}% of input)")
    
    log(f"   📈 Peak context saturation: {tracker.peak_saturation_pct:.1f}% of {tracker.context_limit:,} token limit")
    if tracker.peak_saturation_pct > 80:
        log(f"   ⚠️  Context saturation exceeded 80% - consider summarization or context pruning")
    
    # Saturation progression
    if len(execution_history) > 1:
        saturations = [f"{h.get('saturation_pct', 0):.0f}%" for h in execution_history if h.get('saturation_pct', 0) > 0]
        if saturations:
            log(f"   📉 Saturation progression: " + " → ".join(saturations))
    
    # Errors
    if failures:
        log(f"\n### Errors Encountered")
        for f in failures:
            first_line = (str(f.get('error') or 'unknown').strip().splitlines() or ['unknown'])[-1]
            log(f"   Step {f['step']}: {f['name']} - {first_line[:200]}")
    
    return total_calls, len(failures)
